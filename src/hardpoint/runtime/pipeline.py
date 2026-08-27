"""The linear runtime shape (INSTRUCTIONS.md §6.1, **[LOCKED]**).

Implements ARCHITECTURE.md §11.1 and ADR-003. One of exactly two runtime shapes:
everything linear is a ``Pipeline``, everything with a feedback loop is a
``ControlLoop``. Two shapes are teachable; ten are not.

## What the pipeline does per step, in this order

1. Check the deadline.
2. Check the budget.
3. Open a ``hardpoint.step`` span with ``step.name`` and ``step.type``.
4. Invoke the step.
5. Record latency and usage.
6. Collect degradations, and apply the failure policy on exception.

**Nothing else.** No hooks, no middleware, no event bus, no plugin lifecycle
(INSTRUCTIONS.md §13.5). Every one of those looks like a small addition and each
makes the answer to "what actually ran" require reading a registry instead of
reading the composition.

## Branching lives in Python, not here

Conditionals, loops and fan-out are written in ordinary Python inside a step or
a composed callable. That is a feature: the debugger works, the stack trace
means something, and reading the code tells you the control flow. A pipeline
that could branch would need a language to express the branch in, and that
language would have no debugger (ADR-005).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import anyio

from hardpoint.core.config.snapshot import ConfigSnapshot
from hardpoint.core.context import Budget, CacheHandle, Deadline, RunContext, UsageAccumulator
from hardpoint.core.models import Degradation
from hardpoint.core.types import JsonValue
from hardpoint.observability.metrics import NoOpMetricSink
from hardpoint.observability.tracing import NoOpTracer
from hardpoint.runtime.step import (
    FailurePolicy,
    PipelineInfo,
    Step,
    StepInfo,
    step_type_name,
    unwrap,
)

if TYPE_CHECKING:
    from hardpoint.core.ports import MetricSink, Tracer

__all__ = ["Pipeline", "PipelineRun"]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class PipelineRun:
    """What one execution produced, beyond the output value itself.

    Returned by :meth:`Pipeline.run_detailed` for callers that need the
    degradations and the per-step usage -- the service layer building an
    ``Answer``, and ``ask --explain``. :meth:`Pipeline.__call__` returns just the
    value, because that is what a step composition wants.

    Args:
        value: The final step's output.
        degradations: Everything any step reported, in the order reported.
        usage: Consumption attributed per step.
        trace_id: The trace this run belongs to, when tracing was enabled.
    """

    __slots__ = ("degradations", "trace_id", "usage", "value")

    def __init__(
        self,
        value: Any,
        degradations: tuple[Degradation, ...],
        usage: UsageAccumulator,
        trace_id: str | None,
    ) -> None:
        self.value = value
        self.degradations = degradations
        self.usage = usage
        self.trace_id = trace_id

    def __repr__(self) -> str:
        """Render the degradation count rather than the payload."""
        return (
            f"PipelineRun(value={type(self.value).__name__}, degradations={len(self.degradations)})"
        )


class Pipeline(Generic[TIn, TOut]):
    """A sequence of steps, executed in order.

    Args:
        name: The pipeline's name, recorded on every span and in the run
            manifest.
        steps: The steps, in execution order. Each step's output is the next
            step's input.
        on_step_failure: Per-step failure policy, keyed by step name. Absent
            names default to ``FailurePolicy.FAIL``.

    Raises:
        ValueError: If two steps share a name, or if ``on_step_failure`` names a
            step the pipeline does not contain. Both are typos that would
            otherwise be discovered as mis-attributed usage or as a failure
            policy that silently never applied.
    """

    __slots__ = ("_policies", "_steps", "name")

    def __init__(
        self,
        name: str,
        steps: Sequence[Step[Any, Any]],
        on_step_failure: Mapping[str, FailurePolicy] | None = None,
    ) -> None:
        self.name = name
        self._steps = tuple(steps)
        self._policies = dict(on_step_failure or {})

        names = [step.name for step in self._steps]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Pipeline {name!r} has steps sharing a name: {', '.join(duplicates)}. "
                f"Step names key usage attribution, span attributes and failure "
                f"policies, so they must be unique within a pipeline."
            )

        unknown = sorted(set(self._policies) - set(names))
        if unknown:
            raise ValueError(
                f"Pipeline {name!r} was given a failure policy for steps it does "
                f"not contain: {', '.join(unknown)}. Known steps: {', '.join(names)}."
            )

    # ----------------------------------------------------------------- #
    # Execution                                                         #
    # ----------------------------------------------------------------- #

    async def __call__(self, data: TIn, ctx: RunContext) -> TOut:
        """Run every step in order and return the final output.

        Raises:
            BudgetExceeded: If the deadline passed or a budget limit was
                reached before a step could start.
            BaseException: Whatever a step raised, when its failure policy is
                ``FAIL``.
        """
        run = await self.run_detailed(data, ctx)
        result: TOut = run.value
        return result

    async def run_detailed(self, data: TIn, ctx: RunContext) -> PipelineRun:
        """Run every step and return the output together with what happened.

        Args:
            data: The first step's input.
            ctx: The run context. Its usage accumulator is written to.

        Returns:
            The final value, the collected degradations, the usage and the
            trace id.
        """
        degradations: list[Degradation] = []
        current: Any = data
        trace_id: str | None = None

        pipeline_attrs: dict[str, JsonValue] = {
            "pipeline.name": self.name,
            "pipeline.steps": len(self._steps),
        }
        async with ctx.tracer.span("hardpoint.pipeline", **pipeline_attrs) as pipeline_span:
            trace_id = pipeline_span.trace_id

            for step in self._steps:
                # 1 and 2: deadline first, then budget. A run that is out of time
                # should say so, rather than reporting whichever budget it
                # happened to exhaust while running late.
                ctx.check_limits(step=step.name)

                # 3: one span per step, with both the name and the type. Two
                # RerankStep instances share a type and differ by name, and a
                # trace needs to answer "which step" and "what kind".
                step_attrs: dict[str, JsonValue] = {
                    "step.name": step.name,
                    "step.type": step_type_name(step),
                }
                async with ctx.tracer.span("hardpoint.step", **step_attrs) as span:
                    started = time.perf_counter()
                    try:
                        # 4: invoke.
                        returned = await step(current, ctx)
                    except Exception as exc:
                        # 5: latency is recorded even for a failed step, because
                        # "which step was slow before it fell over" is the first
                        # question asked about a timeout.
                        self._record_latency(ctx, step.name, started)
                        span.record_exception(exc)
                        span.set_status("error", type(exc).__name__)

                        # 6: apply the failure policy.
                        if self._policy_for(step.name) is FailurePolicy.FAIL:
                            raise
                        degradations.append(
                            Degradation(
                                step=step.name,
                                reason="step_skipped",
                                detail=(
                                    f"{type(exc).__name__}: {exc}. The step's input was "
                                    f"passed through unchanged."
                                ),
                            )
                        )
                        continue

                    self._record_latency(ctx, step.name, started)
                    span.set_status("ok")

                    # 6: collect what the step reported.
                    current, reported = unwrap(returned)
                    degradations.extend(reported)
                    if reported:
                        span.set_attribute("step.degradations", len(reported))

            pipeline_span.set_attribute("pipeline.degradations", len(degradations))

        return PipelineRun(
            value=current,
            degradations=tuple(degradations),
            usage=ctx.usage,
            trace_id=trace_id,
        )

    def run_sync(
        self,
        data: TIn,
        *,
        config: ConfigSnapshot | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        run_id: str = "sync",
    ) -> TOut:
        """Run the pipeline from synchronous code, building a ``RunContext``.

        The facade for scripts, notebooks and the CLI (ADR-007). The library is
        async-first because RAG workloads are IO-bound and concurrent; wrapping
        async in sync is a few lines, and the reverse is a rewrite.

        Do not call this from inside a running event loop -- ``anyio.run``
        refuses, and correctly so.

        Args:
            data: The first step's input.
            config: The resolved configuration. Defaults to an empty snapshot.
            tracer: Defaults to :class:`NoOpTracer`.
            metrics: Defaults to :class:`NoOpMetricSink`.
            budget: Defaults to unbounded.
            deadline: Defaults to none.
            run_id: Identifier for this run.

        Returns:
            The final step's output.
        """

        async def main() -> TOut:
            ctx = RunContext(
                run_id=run_id,
                tracer=tracer or NoOpTracer(),
                metrics=metrics or NoOpMetricSink(),
                deadline=deadline or Deadline.none(),
                budget=budget or Budget(),
                cache=CacheHandle(),
                config=config or ConfigSnapshot(env="local", data={}),
                usage=UsageAccumulator(),
                extras={},
            )
            return await self(data, ctx)

        return anyio.run(main)

    # ----------------------------------------------------------------- #
    # Introspection                                                     #
    # ----------------------------------------------------------------- #

    def describe(self) -> PipelineInfo:
        """Return the pipeline's static structure."""
        return PipelineInfo(
            name=self.name,
            steps=tuple(
                StepInfo(
                    name=step.name,
                    type=step_type_name(step),
                    on_failure=self._policy_for(step.name),
                )
                for step in self._steps
            ),
        )

    def explain(self) -> str:
        """Render the static structure as readable text.

        Static, not an execution report: this answers "what would run, in what
        order" without running anything. The per-request report that answers
        "why did it answer that" is ``ask --explain``.
        """
        return self.describe().render()

    @property
    def steps(self) -> tuple[Step[Any, Any], ...]:
        """The steps, in execution order."""
        return self._steps

    def _policy_for(self, step_name: str) -> FailurePolicy:
        return self._policies.get(step_name, FailurePolicy.FAIL)

    @staticmethod
    def _record_latency(ctx: RunContext, step_name: str, started: float) -> None:
        """Attribute wall time to a step.

        Latency is the pipeline's to record, because only it knows when the step
        began. Tokens and cost are the step's, because only it knows what it
        called.
        """
        ctx.usage.record(step_name, latency_ms=(time.perf_counter() - started) * 1000.0)

    def __repr__(self) -> str:
        """Render the name and the step names, in order."""
        return f"Pipeline({self.name!r}, steps=[{', '.join(s.name for s in self._steps)}])"
