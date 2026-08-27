"""The pipeline. **[LOCKED]** INSTRUCTIONS.md §6.1.

The locked requirement is that the pipeline does exactly six things per step and
nothing else. That is asserted two ways: by observing the six happen in order,
and by asserting the class has no hook, middleware or event-bus surface for a
seventh to attach to.

The rest covers the behaviour a pipeline exists to provide: deadline and budget
checked *before* a step rather than after, degradations collected rather than
logged, and a failure policy that can keep an optional stage from failing a
request.
"""

from __future__ import annotations

import time
from typing import Any

import anyio
import pytest

from hardpoint.core.context import Budget, Deadline
from hardpoint.core.errors import BudgetExceeded, TransientError
from hardpoint.core.models import Degradation
from hardpoint.runtime import FailurePolicy, Pipeline, StepResult, as_step
from hardpoint.testing import RecordingTracer, build_run_context


class Append:
    """A step that appends its name to a list, so order is observable."""

    def __init__(self, name: str, *, cost_usd: float | None = 0.0) -> None:
        self.name = name
        self.calls = 0
        self._cost = cost_usd

    async def __call__(self, data: list[str], ctx: Any) -> list[str]:
        self.calls += 1
        ctx.usage.record(self.name, calls=1, prompt_tokens=10, cost_usd=self._cost)
        return [*data, self.name]


class Boom:
    """A step that always raises."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.calls = 0
        self._error = error or TransientError("upstream 503")

    async def __call__(self, data: Any, ctx: Any) -> Any:
        self.calls += 1
        raise self._error


class Degrading:
    """A step that succeeds but reports that it reduced what it did."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def __call__(self, data: list[str], ctx: Any) -> StepResult[list[str]]:
        return StepResult(
            value=[*data, self.name],
            degradations=(Degradation(step=self.name, reason="rerank_skipped", detail="no model"),),
        )


# --------------------------------------------------------------------------- #
# The six things, and only the six  [LOCKED]                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_pipeline_does_the_six_documented_things_in_order() -> None:
    """**[LOCKED]** deadline, budget, span, invoke, record, collect.

    Observed rather than inspected: the span exists with both attributes, usage
    carries a latency the pipeline measured, and the degradation the step
    reported arrived on the run.
    """
    tracer = RecordingTracer()
    ctx = build_run_context(tracer=tracer)
    pipeline: Pipeline[list[str], list[str]] = Pipeline(
        "qa", [Append("retrieve"), Degrading("rerank"), Append("generate")]
    )

    run = await pipeline.run_detailed([], ctx)

    assert run.value == ["retrieve", "rerank", "generate"]

    # 3: a span per step, nested under one pipeline span, carrying name and type.
    assert [span.name for span in tracer.roots] == ["hardpoint.pipeline"]
    steps = tracer.find("hardpoint.step")
    assert [s.attributes["step.name"] for s in steps] == ["retrieve", "rerank", "generate"]
    assert [s.attributes["step.type"] for s in steps] == ["Append", "Degrading", "Append"]

    # 5: latency is the pipeline's to record; tokens and cost are the step's.
    usage = ctx.usage.snapshot()
    assert set(usage.by_step) == {"retrieve", "rerank", "generate"}
    assert usage.by_step["retrieve"].latency_ms > 0
    assert usage.by_step["retrieve"].prompt_tokens == 10

    # 6: degradations are collected, not logged.
    assert [d.reason for d in run.degradations] == ["rerank_skipped"]


def test_the_pipeline_has_no_hook_middleware_or_event_bus_surface() -> None:
    """**Forbidden shortcut §13.5.**

    Each of those looks like a small addition, and each makes "what actually
    ran" require reading a registry instead of reading the composition.
    """
    surface = {name for name in dir(Pipeline) if not name.startswith("_")}
    forbidden = {
        name
        for name in surface
        if any(
            word in name.lower()
            for word in ("hook", "middleware", "listener", "event", "plugin", "subscribe", "emit")
        )
    }
    assert not forbidden, f"the pipeline grew an extension surface: {forbidden}"


def test_the_pipeline_public_surface_is_small() -> None:
    """Pinned, so growing it is a deliberate act rather than a drift."""
    assert {name for name in dir(Pipeline) if not name.startswith("_")} == {
        "describe",
        "explain",
        "name",
        "run_detailed",
        "run_sync",
        "steps",
    }


# --------------------------------------------------------------------------- #
# Limits are checked before a step, not after                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_an_expired_deadline_stops_the_pipeline_before_the_first_step() -> None:
    """Checked before, so an over-budget run stops at a boundary.

    Checking after would let the run do the expensive thing and *then* report
    that it should not have.
    """
    step = Append("retrieve")
    ctx = build_run_context(deadline=Deadline(at=time.monotonic() - 1))
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [step])

    with pytest.raises(BudgetExceeded) as exc_info:
        await pipeline([], ctx)

    assert step.calls == 0, "the step must not have run"
    assert exc_info.value.limit == "deadline_s"
    assert exc_info.value.step == "retrieve"


@pytest.mark.anyio
async def test_a_budget_stops_the_pipeline_between_steps() -> None:
    """The first step's spend is what trips the limit for the second."""
    first, second = Append("first", cost_usd=0.10), Append("second")
    ctx = build_run_context(budget=Budget(max_cost_usd=0.05))
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [first, second])

    with pytest.raises(BudgetExceeded) as exc_info:
        await pipeline([], ctx)

    assert first.calls == 1
    assert second.calls == 0
    assert exc_info.value.limit == "max_cost_usd"
    assert exc_info.value.step == "second"


@pytest.mark.anyio
async def test_a_run_within_its_limits_completes() -> None:
    ctx = build_run_context(budget=Budget(max_cost_usd=1.0), deadline=Deadline.in_seconds(30))
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [Append("a"), Append("b")])
    assert await pipeline([], ctx) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Failure policy                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_failing_step_raises_by_default() -> None:
    """FAIL is the default because most steps produce what the rest depends on."""
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [Append("a"), Boom("b"), Append("c")])
    tail = pipeline.steps[2]

    with pytest.raises(TransientError):
        await pipeline([], build_run_context())
    assert tail.calls == 0  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_a_skipped_step_passes_its_input_through_and_records_a_degradation() -> None:
    """Availability-preserving degradation for an optional quality stage.

    A reranker that timed out should leave the order alone and say so, not fail
    the request (ARCHITECTURE.md §18.2).
    """
    ctx = build_run_context()
    pipeline: Pipeline[list[str], list[str]] = Pipeline(
        "qa",
        [Append("retrieve"), Boom("rerank"), Append("generate")],
        on_step_failure={"rerank": FailurePolicy.SKIP},
    )

    run = await pipeline.run_detailed([], ctx)

    assert run.value == ["retrieve", "generate"], "the failed step's input passed through"
    assert len(run.degradations) == 1
    degradation = run.degradations[0]
    assert degradation.step == "rerank"
    assert degradation.reason == "step_skipped"
    assert "TransientError" in degradation.detail


@pytest.mark.anyio
async def test_a_failed_step_still_records_its_latency() -> None:
    """ "Which step was slow before it fell over" is the first question asked."""
    ctx = build_run_context()
    pipeline: Pipeline[Any, Any] = Pipeline(
        "qa", [Boom("slow")], on_step_failure={"slow": FailurePolicy.SKIP}
    )
    await pipeline.run_detailed([], ctx)
    assert ctx.usage.snapshot().by_step["slow"].latency_ms >= 0


@pytest.mark.anyio
async def test_a_failed_step_marks_its_span_as_an_error() -> None:
    tracer = RecordingTracer()
    pipeline: Pipeline[Any, Any] = Pipeline(
        "qa", [Boom("b")], on_step_failure={"b": FailurePolicy.SKIP}
    )
    await pipeline.run_detailed([], build_run_context(tracer=tracer))

    span = tracer.find("hardpoint.step")[0]
    assert span.status == ("error", "TransientError")
    assert [type(exc).__name__ for exc in span.exceptions] == ["TransientError"]


@pytest.mark.anyio
async def test_cancellation_is_never_swallowed_by_a_skip_policy() -> None:
    """A deadline must actually stop the run, whatever the failure policy says.

    ``SKIP`` catches ``Exception``; ``CancelledError`` derives from
    ``BaseException`` and therefore passes through. Asserted because a policy
    that swallowed cancellation would make deadlines advisory.
    """
    started = anyio.Event()

    async def hang(data: Any, ctx: Any) -> Any:
        started.set()
        await anyio.sleep(10)
        return data

    pipeline: Pipeline[Any, Any] = Pipeline(
        "qa",
        [as_step(hang, name="hang")],
        on_step_failure={"hang": FailurePolicy.SKIP},
    )

    with anyio.move_on_after(0.05) as scope:
        await pipeline([], build_run_context())

    assert scope.cancelled_caught, "the run must have been cancelled, not skipped"


# --------------------------------------------------------------------------- #
# Construction-time checks                                                    #
# --------------------------------------------------------------------------- #


def test_duplicate_step_names_are_rejected() -> None:
    """Names key usage attribution, spans and failure policies.

    Two steps sharing one would silently merge their usage and make a failure
    policy apply to both.
    """
    with pytest.raises(ValueError, match="sharing a name"):
        Pipeline("qa", [Append("retrieve"), Append("retrieve")])


def test_a_failure_policy_for_an_unknown_step_is_rejected() -> None:
    """A typo here would produce a policy that silently never applied."""
    with pytest.raises(ValueError, match="does not contain") as exc_info:
        Pipeline("qa", [Append("retrieve")], on_step_failure={"reranker": FailurePolicy.SKIP})
    assert "retrieve" in str(exc_info.value), "the message lists the steps that do exist"


def test_an_empty_pipeline_is_allowed() -> None:
    """A generated project starts from one and adds steps."""
    assert Pipeline("empty", []).describe().steps == ()


# --------------------------------------------------------------------------- #
# Introspection                                                               #
# --------------------------------------------------------------------------- #


def test_describe_reports_the_static_structure() -> None:
    pipeline: Pipeline[Any, Any] = Pipeline(
        "qa",
        [Append("retrieve"), Boom("rerank")],
        on_step_failure={"rerank": FailurePolicy.SKIP},
    )
    info = pipeline.describe()

    assert info.name == "qa"
    assert [s.name for s in info.steps] == ["retrieve", "rerank"]
    assert [s.type for s in info.steps] == ["Append", "Boom"]
    assert info.steps[1].on_failure is FailurePolicy.SKIP


def test_explain_renders_the_order_and_the_non_default_policies() -> None:
    """Static structure. The per-request report is ``ask --explain``."""
    pipeline: Pipeline[Any, Any] = Pipeline(
        "qa",
        [Append("retrieve"), Boom("rerank")],
        on_step_failure={"rerank": FailurePolicy.SKIP},
    )
    rendered = pipeline.explain()

    assert "Pipeline 'qa' (2 steps)" in rendered
    assert " 1. retrieve" in rendered
    assert "on_failure=skip" in rendered
    assert rendered.count("on_failure") == 1, "the default policy is not printed"


def test_repr_lists_the_steps_in_order() -> None:
    pipeline: Pipeline[Any, Any] = Pipeline("qa", [Append("a"), Append("b")])
    assert repr(pipeline) == "Pipeline('qa', steps=[a, b])"


# --------------------------------------------------------------------------- #
# The sync facade                                                             #
# --------------------------------------------------------------------------- #


def test_run_sync_builds_a_context_and_runs() -> None:
    """The facade for scripts, notebooks and the CLI (ADR-007)."""
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [Append("a"), Append("b")])
    assert pipeline.run_sync([]) == ["a", "b"]


def test_run_sync_defaults_to_the_no_op_tracer_not_a_testing_fake() -> None:
    """``hardpoint.testing`` must never appear on a production path."""
    from hardpoint.observability import NoOpMetricSink, NoOpTracer

    seen: dict[str, Any] = {}

    async def capture(data: Any, ctx: Any) -> Any:
        seen["tracer"] = ctx.tracer
        seen["metrics"] = ctx.metrics
        return data

    Pipeline("qa", [as_step(capture, name="capture")]).run_sync(None)

    assert isinstance(seen["tracer"], NoOpTracer)
    assert isinstance(seen["metrics"], NoOpMetricSink)


def test_run_sync_accepts_a_budget_and_a_deadline() -> None:
    """Two steps, because the budget is checked *before* each one.

    With a single step the check runs before anything has been spent, so
    nothing trips -- which is the correct behaviour and the reason the first
    version of this test was wrong.
    """
    pipeline: Pipeline[list[str], list[str]] = Pipeline(
        "qa", [Append("spend", cost_usd=1.0), Append("next")]
    )
    with pytest.raises(BudgetExceeded) as exc_info:
        pipeline.run_sync([], budget=Budget(max_cost_usd=0.01), deadline=Deadline.in_seconds(5))
    assert exc_info.value.step == "next"


def test_a_budget_cannot_stop_the_first_step_before_it_spends_anything() -> None:
    """Stated, because it is the natural consequence of checking before a step.

    A budget bounds what a run *may* spend, and the first step is what discovers
    the cost. Refusing to start it would make any budget below one step's cost
    reject every request.
    """
    step = Append("expensive", cost_usd=1000.0)
    pipeline: Pipeline[list[str], list[str]] = Pipeline("qa", [step])
    assert pipeline.run_sync([], budget=Budget(max_cost_usd=0.01)) == ["expensive"]
    assert step.calls == 1
