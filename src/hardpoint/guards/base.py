"""Guards: checks with a configured action, placed in the pipeline as steps.

ARCHITECTURE.md §19: guards are steps returning ``GuardResult(action, reason,
evidence)``, and their placement is explicit in the composition so the order is
readable. A guard decides *whether* something violated its rule; the action
configured on it decides *what happens*:

- ``allow`` -- nothing, even on a violation (useful to trial a guard).
- ``flag`` -- continue unchanged, and record a ``Degradation`` saying so.
- ``redact`` -- continue with a modified value, and record a ``Degradation``.
  **Never a silent change** (INSTRUCTIONS.md §7): the degradation names what
  was removed.
- ``block`` -- on output, return the guard's refusal with ``blocked=True``; on
  input, raise ``GuardViolation`` because there is no answer to return yet.
- ``retry`` -- on output, generate once more with the violation as feedback,
  through :class:`GuardedGenerate`; a second failure blocks.

This is a gate, not a sandbox, and not a claim of prevention (ARCHITECTURE.md
§18.3): the injection heuristic in particular defaults to ``flag``, because false
positives on legitimate content are common and a guard that blocks by default
teaches people to switch it off.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.errors import GuardViolation
from hardpoint.core.models import Answer, Degradation
from hardpoint.core.types import JsonValue
from hardpoint.runtime.step import StepResult

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.generation.generate import Generate
    from hardpoint.retrieval.retrievers import Assembled

__all__ = [
    "DEFAULT_REFUSAL",
    "GuardAction",
    "GuardCheck",
    "GuardResult",
    "GuardedGenerate",
    "InputGuard",
    "OutputGuard",
]

GuardAction = Literal["allow", "flag", "redact", "block", "retry"]

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)

DEFAULT_REFUSAL = "This answer was withheld because it did not pass a safety check."
"""A neutral placeholder. The product's wording belongs in the generated project."""


class GuardResult(BaseModel):
    """What one guard concluded.

    Args:
        guard: The guard's name.
        action: What to do: ``allow`` when there was no violation, otherwise the
            action the guard was configured with.
        reason: A stable machine-readable code, for example ``"ungrounded"``.
        detail: Human-readable explanation.
        evidence: What the decision rested on -- the matched pattern, the
            unsupported sentences, the validation error.
        replacement: The modified value, for ``redact``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    guard: str
    action: GuardAction = "allow"
    reason: str = ""
    detail: str = ""
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    replacement: str | None = None

    @property
    def violated(self) -> bool:
        """Whether the guard found a violation, whatever it was told to do about it."""
        return bool(self.reason)


class GuardCheck(Protocol[T_contra]):
    """One rule. Returns ``GuardResult`` with ``action="allow"`` when it passes.

    Attributes:
        name: How the guard appears in degradations and spans.
        action: What to do on a violation.
    """

    name: str
    action: GuardAction

    async def check(self, value: T_contra, ctx: RunContext) -> GuardResult:
        """Evaluate the rule against a value."""
        ...


async def _run_check(check: GuardCheck[T], value: T, ctx: RunContext) -> GuardResult:
    """Run one check inside a ``hardpoint.guard`` span, and count a violation."""
    async with ctx.tracer.span("hardpoint.guard", **{"guard.name": check.name}) as span:
        result = await check.check(value, ctx)
        span.set_attribute("guard.action", result.action)
        span.set_attribute("guard.reason", result.reason)
        if result.violated:
            ctx.metrics.increment(
                "hardpoint.degradations", step=check.name, reason=f"guard_{result.reason}"
            )
        return result


def _degradation(step: str, result: GuardResult, *, severity: str = "warn") -> Degradation:
    return Degradation(
        step=step,
        reason=f"guard_{result.action}:{result.guard}:{result.reason}",
        detail=result.detail,
        severity="info" if severity == "info" else "warn",
    )


class InputGuard:
    """Runs checks against the pipeline's input -- the query -- in order.

    Args:
        checks: The checks, run in the order given.
        name: The step's name.
    """

    def __init__(self, checks: Sequence[GuardCheck[str]], *, name: str = "input_guards") -> None:
        self.name = name
        self.checks = tuple(checks)

    async def __call__(self, data: str, ctx: RunContext) -> StepResult[str]:
        """Check the query.

        Raises:
            GuardViolation: For a violation whose action is ``block``.
        """
        current = data
        degradations: list[Degradation] = []
        for check in self.checks:
            result = await _run_check(check, current, ctx)
            if result.action == "block":
                raise GuardViolation(
                    f"The request was blocked by {result.guard}: {result.detail}",
                    guard=result.guard,
                    reason=result.reason,
                    step=self.name,
                    run_id=ctx.run_id,
                    remedy=(
                        "Rephrase the request. If this block is wrong, set the guard's "
                        "`action` to `flag` in `guards.input` to record instead of block."
                    ),
                )
            if result.action == "redact" and result.replacement is not None:
                current = result.replacement
                degradations.append(_degradation(self.name, result))
            elif result.action in ("flag", "retry") and result.violated:
                degradations.append(_degradation(self.name, result))
        return StepResult(current, tuple(degradations))

    def __repr__(self) -> str:
        """Render the check names, in order."""
        return f"InputGuard([{', '.join(c.name for c in self.checks)}])"


class OutputGuard:
    """Runs checks against the ``Answer``, in order.

    A ``retry`` action here behaves as ``block``: regenerating needs the
    generation step, which is what :class:`GuardedGenerate` wraps.

    Args:
        checks: The checks, run in the order given.
        refusal: What a blocked answer says. Product voice, so the generated
            project supplies it.
        name: The step's name.
    """

    def __init__(
        self,
        checks: Sequence[GuardCheck[Answer]],
        *,
        refusal: str | None = None,
        name: str = "output_guards",
    ) -> None:
        self.name = name
        self.checks = tuple(checks)
        self.refusal = refusal or DEFAULT_REFUSAL

    async def __call__(self, data: Answer, ctx: RunContext) -> StepResult[Answer]:
        """Check the answer, returning it -- modified, refused or as it was."""
        answer, degradations = await self.apply(data, ctx)
        return StepResult(answer, tuple(degradations))

    async def evaluate(self, answer: Answer, ctx: RunContext) -> list[GuardResult]:
        """Run every check without acting on the results."""
        return [await _run_check(check, answer, ctx) for check in self.checks]

    async def apply(self, answer: Answer, ctx: RunContext) -> tuple[Answer, list[Degradation]]:
        """Run the checks and act on them."""
        degradations: list[Degradation] = []
        for check in self.checks:
            result = await _run_check(check, answer, ctx)
            if result.action in ("block", "retry"):
                degradations.append(_degradation(self.name, result))
                return self.blocked(answer), degradations
            if result.action == "redact" and result.replacement is not None:
                answer = answer.model_copy(update={"text": result.replacement})
                degradations.append(_degradation(self.name, result))
            elif result.action == "flag" and result.violated:
                degradations.append(_degradation(self.name, result))
        return answer, degradations

    def blocked(self, answer: Answer) -> Answer:
        """The refusal that replaces a blocked answer. Citations are dropped with it."""
        return answer.model_copy(update={"text": self.refusal, "blocked": True, "citations": []})

    def __repr__(self) -> str:
        """Render the check names, in order."""
        return f"OutputGuard([{', '.join(c.name for c in self.checks)}])"


class GuardedGenerate:
    """Generation with output guards that may ask for one regeneration.

    ARCHITECTURE.md §18.2: when a guard's action is ``retry``, generate once
    more with the violation appended as feedback; a second failure blocks.
    Other actions behave exactly as in :class:`OutputGuard`.

    Args:
        generate: The generation step.
        guards: The output checks.
        refusal: What a blocked answer says.
        name: The step's name.
    """

    def __init__(
        self,
        generate: Generate,
        guards: Sequence[GuardCheck[Answer]],
        *,
        refusal: str | None = None,
        name: str = "generate",
    ) -> None:
        self.name = name
        self.generate = generate
        self.output = OutputGuard(guards, refusal=refusal, name=name)

    @property
    def streamable(self) -> bool:
        """Whether text can be streamed before the guards have seen it.

        Only when no guard could change or withhold it: ``allow`` and ``flag``
        leave the text as it is, so the answer is checked after streaming and a
        flag is still recorded. ``block``, ``redact`` and ``retry`` could each
        withdraw text already shown, so those wait for the check.
        """
        return all(check.action in ("allow", "flag") for check in self.output.checks)

    async def stream_events(
        self, data: Assembled, ctx: RunContext
    ) -> AsyncIterator[str | StepResult[Answer]]:
        """Stream when :attr:`streamable`, then yield the checked result, last."""
        if not self.streamable:
            yield await self(data, ctx)
            return
        answer: Answer | None = None
        async for event in self.generate.stream_events(data, ctx):
            if isinstance(event, Answer):
                answer = event
            else:
                yield event
        if answer is None:  # pragma: no cover - stream_events always ends with an Answer
            raise RuntimeError("Generate.stream_events ended without an Answer")
        final, applied = await self.output.apply(answer, ctx)
        yield StepResult(final, tuple(applied))

    async def __call__(self, data: Assembled, ctx: RunContext) -> StepResult[Answer]:
        """Generate, check, and regenerate once if a guard asks."""
        answer = await self.generate(data, ctx)
        results = await self.output.evaluate(answer, ctx)
        retry = next((r for r in results if r.action == "retry"), None)
        degradations: list[Degradation] = []
        if retry is not None:
            degradations.append(_degradation(self.name, retry, severity="info"))
            feedback = (
                f"Your previous answer was rejected: {retry.detail} "
                f"Answer again, following every instruction."
            )
            answer = await self.generate.regenerate(data, ctx, feedback=feedback)

        final, applied = await self.output.apply(answer, ctx)
        return StepResult(final, (*degradations, *applied))

    def __repr__(self) -> str:
        """Render the generation step and its guards."""
        return f"GuardedGenerate({self.generate!r}, {self.output!r})"
