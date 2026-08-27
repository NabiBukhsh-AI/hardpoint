"""The ``Step`` contract and the machinery a pipeline needs around it.

Implements ARCHITECTURE.md §9.1. A step is one transformation with a declared
input type and a declared output type. That is all it is.

## What a step does not own

Retries, timeouts, tracing, budget enforcement and cross-step state. Every one
of those is supplied by the runtime, which is why a step can be tested by
calling it with a fake ``RunContext`` and asserting on the result -- no mocking
framework, no harness.

## Typed rather than universal

A single untyped ``Component.run(input) -> output`` looks elegant on a
whiteboard and destroys IDE support in practice (ARCHITECTURE.md §6.2). Generic
``Step[TIn, TOut]`` costs more to write and pays for itself the first time a
composition error is caught at the point of composition rather than at request
time.

## How a step reports that it degraded

By returning :class:`StepResult` instead of a bare value. A step that has
nothing to report returns the value directly and never sees this class, so the
simple case stays simple; a reranker that gave up returns its input plus a
``Degradation`` and the pipeline collects it (ADR-012).

This is deliberately not a mutable sink on ``RunContext``. Degradation is part
of what a step *returned*, so it travels with the return value.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from hardpoint.core.models import Degradation

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = [
    "FailurePolicy",
    "PipelineInfo",
    "Step",
    "StepInfo",
    "StepResult",
    "as_step",
    "step_type_name",
    "unwrap",
]

# Variance is declared, and the names carry the ruff-mandated suffixes: a step
# accepts anything its input type covers and returns something its output type
# covers, which is what makes a `Step[Query, Answer]` usable where a
# `Step[Query, object]` is expected.
TIn_contra = TypeVar("TIn_contra", contravariant=True)
TOut_co = TypeVar("TOut_co", covariant=True)
T = TypeVar("T")


class Step(Protocol[TIn_contra, TOut_co]):
    """One transformation, from a declared input type to a declared output type.

    The primary user extension point, and the only one most projects need: write
    a class with a ``name`` and an ``async __call__``, and pass the instance.
    No base class, no registration, no import of hardpoint into the module that
    defines it.

    Attributes:
        name: How this step appears in traces, usage attribution and failure
            policies. Unique within a pipeline.

    Invariants an implementation must hold:

    - Pure with respect to ``ctx``. It may read the budget and open spans; it
      must never mutate another step's data.
    - Cancellable at every ``await``. A step that shields itself from
      cancellation defeats the deadline.
    - Never swallows ``CancelledError``.
    """

    name: str

    async def __call__(self, data: TIn_contra, ctx: RunContext) -> TOut_co:
        """Transform the input."""
        ...


class FailurePolicy(StrEnum):
    """What a pipeline does when a step raises.

    Attributes:
        FAIL: Re-raise. The default, and correct for any step whose output the
            rest of the pipeline depends on.
        SKIP: Pass the step's input through as its output and record a
            ``Degradation``. Correct for an optional quality stage -- reranking,
            compression -- where a degraded answer beats no answer
            (ARCHITECTURE.md §18.2).

    ``SKIP`` only makes sense for a step whose input and output types are the
    same, because passing the input through *is* the fallback. Nothing can check
    that statically from a name-keyed mapping, so it is stated here and the
    pipeline says so in the error if the following step then fails.
    """

    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class StepResult(Generic[T]):
    """A step's output together with anything it wants to report about the run.

    Returned instead of a bare value by a step that degraded. The pipeline
    unwraps it: ``value`` continues to the next step, ``degradations`` are
    collected onto the answer.

    Args:
        value: What the next step receives.
        degradations: What was skipped or reduced, and why.
    """

    value: T
    degradations: tuple[Degradation, ...] = ()

    def __repr__(self) -> str:
        """Render the degradation count rather than the whole payload."""
        return (
            f"StepResult(value={type(self.value).__name__}, degradations={len(self.degradations)})"
        )


def unwrap(result: object) -> tuple[Any, tuple[Degradation, ...]]:
    """Split a step's return value into its payload and its degradations.

    Args:
        result: Whatever the step returned.

    Returns:
        ``(value, degradations)``. A bare value yields an empty degradation
        tuple, which is why a step that never degrades never has to know
        :class:`StepResult` exists.
    """
    if isinstance(result, StepResult):
        return result.value, result.degradations
    return result, ()


def step_type_name(step: object) -> str:
    """Return the implementation type of a step, for the ``step.type`` span attribute.

    Distinct from ``step.name``: two ``RerankStep`` instances in one pipeline
    have different names and the same type, and a trace needs both to answer
    "which step" and "what kind of step".
    """
    return type(step).__name__


class _FunctionStep:
    """Adapts a plain async function to the ``Step`` protocol."""

    __slots__ = ("_func", "name")

    def __init__(self, func: Callable[[Any, RunContext], Awaitable[Any]], name: str) -> None:
        self._func = func
        self.name = name

    async def __call__(self, data: Any, ctx: RunContext) -> Any:
        """Invoke the wrapped function."""
        return await self._func(data, ctx)

    def __repr__(self) -> str:
        """Render the step name and the function it wraps."""
        return f"as_step({self._func.__qualname__!r}, name={self.name!r})"


def as_step(
    func: Callable[[Any, RunContext], Awaitable[Any]], *, name: str | None = None
) -> Step[Any, Any]:
    """Turn an async function into a ``Step``.

    A step needs a ``name``, and a function has ``__name__`` rather than
    ``name``, so a bare function does not satisfy the protocol. This closes that
    gap, because "can it be a function instead of a class?" should usually be
    answered yes (INSTRUCTIONS.md §17.3).

    Args:
        func: An ``async def f(data, ctx)``.
        name: The step's name. Defaults to the function's ``__name__``.

    Returns:
        A ``Step`` delegating to the function.

    Example:
        >>> async def normalise(query: str, ctx: RunContext) -> str:
        ...     return query.strip().lower()
        >>> step = as_step(normalise)
        >>> step.name
        'normalise'
    """
    resolved = name or str(getattr(func, "__name__", None) or "step")
    return _FunctionStep(func, resolved)


class StepInfo(BaseModel):
    """One step's static description, for ``Pipeline.describe``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: str
    on_failure: FailurePolicy = FailurePolicy.FAIL


class PipelineInfo(BaseModel):
    """A pipeline's static structure.

    What ``hardpoint components list --resolved`` and the generated project's
    startup log print, so that "is retry actually on" and "what runs in what
    order" are answerable without reading the composition code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    steps: tuple[StepInfo, ...] = field(default=())

    def render(self) -> str:
        """Render the structure as an ordered, readable list."""
        lines = [f"Pipeline {self.name!r} ({len(self.steps)} steps)"]
        width = max((len(s.name) for s in self.steps), default=0)
        lines.extend(
            f"  {index + 1:>2}. {info.name:<{width}}  {info.type}"
            + (
                f"  [on_failure={info.on_failure.value}]"
                if info.on_failure is not FailurePolicy.FAIL
                else ""
            )
            for index, info in enumerate(self.steps)
        )
        return "\n".join(lines)
