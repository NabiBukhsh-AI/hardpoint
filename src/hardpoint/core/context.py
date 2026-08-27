"""``RunContext`` and the per-run machinery it carries (**[LOCKED]**).

Implements INSTRUCTIONS.md §5.4 and ARCHITECTURE.md §11.2.

## The frozen scope

``RunContext`` has exactly nine attributes and a test asserts the set. This is
not fussiness. In the first draft it was accumulating helpers and drifting
toward a god object -- the failure mode where every new feature adds a field,
every step depends on the whole context, and nothing can be tested in isolation
(ARCHITECTURE.md §6.3). Adding an attribute now requires architectural review,
and the test is what makes that requirement real rather than aspirational.

Anything a step needs that is not on this list is passed to the step explicitly,
at construction time or as its input. That is the whole mechanism.

## Cancellation is structural, not a flag

There is no ``cancelled`` boolean. An exceeded deadline must actually stop an
in-flight provider call, and a flag cannot do that: nothing polls it while
awaiting a socket. Cancellation goes through anyio cancel scopes, and
:meth:`Deadline.as_anyio_deadline` is how a deadline enters one.

## ``extras`` belongs to the user

``extras`` exists so a caller can thread request-scoped data through their own
steps. **The library never reads it.** A test scans the source to keep that
true, because the moment the library starts depending on a key in ``extras``,
``extras`` has become an untyped part of the contract.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Self

import anyio

from hardpoint.core.errors import BudgetExceeded
from hardpoint.core.models import StepUsage, Usage

if TYPE_CHECKING:
    from hardpoint.core.config.schema import RequestBudgetConfig
    from hardpoint.core.config.snapshot import ConfigSnapshot
    from hardpoint.core.ports import CacheBackend, MetricSink, Tracer

__all__ = [
    "Budget",
    "CacheHandle",
    "Deadline",
    "RunContext",
    "UsageAccumulator",
]

CacheScope = Literal["request", "shared"]
"""Which cache a read or write addresses."""


@dataclass(frozen=True)
class Deadline:
    """An absolute point in time by which a run must finish.

    Absolute rather than a duration, so that it propagates correctly: a step
    three levels deep inherits the time that is actually left, not a fresh copy
    of the original allowance.

    Args:
        at: A ``time.monotonic()`` timestamp, or ``None`` for no deadline.
            Monotonic rather than wall-clock, so an NTP correction or a daylight
            saving change cannot make a deadline pass or recede.
    """

    at: float | None = None

    @classmethod
    def none(cls) -> Deadline:
        """Return a deadline that never expires."""
        return cls(at=None)

    @classmethod
    def in_seconds(cls, seconds: float | None) -> Deadline:
        """Return a deadline that many seconds from now, or none for ``None``."""
        return cls(at=None if seconds is None else time.monotonic() + seconds)

    def remaining_s(self) -> float | None:
        """Return the seconds left, never negative, or ``None`` for no deadline.

        A step calling a provider passes this as the request timeout, so that
        the provider call cannot outlive the run that asked for it.
        """
        if self.at is None:
            return None
        return max(0.0, self.at - time.monotonic())

    def expired(self) -> bool:
        """Return whether the deadline has passed."""
        return self.at is not None and time.monotonic() >= self.at

    def earliest(self, other: Deadline) -> Deadline:
        """Return whichever of two deadlines comes first.

        A step may impose its own timeout, but it can only ever *tighten* the
        run's deadline. Combining this way makes that structural rather than a
        rule somebody has to remember.
        """
        if self.at is None:
            return other
        if other.at is None:
            return self
        return self if self.at <= other.at else other

    def as_anyio_deadline(self) -> float:
        """Return this deadline on anyio's clock, for ``CancelScope(deadline=...)``.

        anyio measures time with ``anyio.current_time()``, which is the event
        loop's clock and is not guaranteed to be the same reading as
        ``time.monotonic()``. Passing a ``time.monotonic()`` value straight into
        a cancel scope would therefore be subtly wrong. Converting through the
        remaining duration at the point of use is correct on both clocks.

        Usage::

            with anyio.CancelScope(deadline=ctx.deadline.as_anyio_deadline()):
                await provider.call()

        Returns:
            An absolute anyio timestamp. ``math.inf`` when there is no deadline,
            which is what ``CancelScope`` already treats as unbounded.

        Raises:
            RuntimeError: If called outside an async context, where anyio has no
                clock to read.
        """
        remaining = self.remaining_s()
        if remaining is None:
            return float("inf")
        return anyio.current_time() + remaining

    def check(self, *, step: str | None = None, run_id: str | None = None) -> None:
        """Raise if the deadline has passed.

        The pipeline calls this before each step, so an over-budget run stops at
        a step boundary with a clear error rather than midway through one.

        Raises:
            BudgetExceeded: Naming the deadline as the limit that was passed.
        """
        if not self.expired():
            return
        # `at` is not None here: expired() is false for an unbounded deadline.
        overdue = time.monotonic() - (self.at or 0.0)
        raise BudgetExceeded(
            f"The run passed its deadline {overdue:.3f}s ago, before this step could start.",
            limit="deadline_s",
            limit_value=0.0,
            observed=overdue,
            step=step,
            run_id=run_id,
            remedy=(
                "Raise `budgets.request.deadline_s`, reduce the work in the "
                "pipeline, or lower `retrieval.top_k` so retrieval returns sooner."
            ),
        )


@dataclass(frozen=True)
class Budget:
    """Ceilings on what one run may consume.

    Every limit defaults to ``None``, meaning unbounded. A default ceiling would
    silently truncate a legitimate workload; the generated project sets real
    numbers.

    Args:
        max_cost_usd: Total spend across every provider call.
        max_llm_calls: Number of language model invocations.
        max_tokens: Prompt, completion and embedding tokens combined.
        max_steps: Number of pipeline steps that may execute.
    """

    max_cost_usd: float | None = None
    max_llm_calls: int | None = None
    max_tokens: int | None = None
    max_steps: int | None = None

    @classmethod
    def from_config(cls, config: RequestBudgetConfig) -> Budget:
        """Build a budget from the ``budgets.request`` configuration block."""
        return cls(
            max_cost_usd=config.max_cost_usd,
            max_llm_calls=config.max_llm_calls,
            max_tokens=config.max_tokens,
            max_steps=config.max_steps,
        )

    def check(self, usage: Usage, *, step: str | None = None, run_id: str | None = None) -> None:
        """Raise if usage has passed any limit.

        An unknown cost never trips the cost limit. ``cost_usd=None`` means
        unpriced, not free, and refusing to run because a model is missing from
        the pricing table would be the wrong failure.

        Args:
            usage: Consumption so far.
            step: The step about to run, for the error.
            run_id: The run, for the error.

        Raises:
            BudgetExceeded: Naming the limit, its value, and what was observed.
        """
        checks: tuple[tuple[str, float | None, float], ...] = (
            ("max_cost_usd", self.max_cost_usd, usage.total_cost_usd or 0.0),
            ("max_llm_calls", self.max_llm_calls, usage.total_calls),
            ("max_tokens", self.max_tokens, usage.total_tokens),
            ("max_steps", self.max_steps, len(usage.by_step)),
        )
        for name, limit, observed in checks:
            if limit is None or observed <= limit:
                continue
            raise BudgetExceeded(
                f"The run exceeded its {name} budget: {observed} used against a limit of {limit}.",
                limit=name,
                limit_value=float(limit),
                observed=float(observed),
                step=step,
                run_id=run_id,
                remedy=(
                    f"Raise `budgets.request.{name}` if the work genuinely needs it, "
                    f"or reduce what the pipeline does per request. "
                    f"`hardpoint ask --explain` shows where the budget went."
                ),
            )


class UsageAccumulator:
    """Append-only record of what a run consumed, attributed per step.

    Mutable by design, and the one mutable thing ``RunContext`` carries: usage
    accrues as a run proceeds, and the alternative is threading a return value
    through every step signature.

    Safe to use from concurrent tasks on one event loop, because :meth:`record`
    contains no ``await`` and therefore cannot be interleaved. It is *not* safe
    across threads, and nothing in the library uses it from one.
    """

    __slots__ = ("_cost_known", "_steps")

    def __init__(self) -> None:
        self._steps: dict[str, StepUsage] = {}
        self._cost_known = True

    def record(
        self,
        step: str,
        *,
        calls: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        embed_tokens: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float | None = None,
        estimated: bool = False,
    ) -> None:
        """Add consumption to a step's tally.

        Repeated calls for one step accumulate rather than replace, so a step
        that makes three provider calls reports all three.

        Args:
            step: The step's name.
            calls: Provider calls made.
            prompt_tokens: Input tokens.
            completion_tokens: Generated tokens.
            embed_tokens: Tokens embedded.
            latency_ms: Wall time spent.
            cost_usd: Cost, or ``None`` when the model is not in the pricing
                table. ``None`` with ``calls > 0`` makes the run's total cost
                unknown rather than understated (INSTRUCTIONS.md §13.8).
            estimated: Whether the token counts came from an estimator rather
                than from the provider.
        """
        previous = self._steps.get(step, StepUsage())

        if cost_usd is None and calls > 0:
            self._cost_known = False
            merged_cost = previous.cost_usd
        elif cost_usd is None:
            merged_cost = previous.cost_usd
        else:
            merged_cost = (previous.cost_usd or 0.0) + cost_usd

        self._steps[step] = StepUsage(
            calls=previous.calls + calls,
            prompt_tokens=previous.prompt_tokens + prompt_tokens,
            completion_tokens=previous.completion_tokens + completion_tokens,
            embed_tokens=previous.embed_tokens + embed_tokens,
            latency_ms=previous.latency_ms + latency_ms,
            cost_usd=merged_cost,
            estimated=previous.estimated or estimated,
        )

    def snapshot(self) -> Usage:
        """Return an immutable view of consumption so far.

        ``total_cost_usd`` is ``None`` when any billed call had no price, rather
        than a partial sum presented as a total.
        """
        total_cost: float | None = None
        if self._cost_known:
            costs = [s.cost_usd for s in self._steps.values() if s.cost_usd is not None]
            total_cost = sum(costs) if costs else None

        return Usage(
            by_step=dict(self._steps),
            total_cost_usd=total_cost,
            total_latency_ms=sum(s.latency_ms for s in self._steps.values()),
        )

    def __repr__(self) -> str:
        """Summarise without rendering every step."""
        return f"UsageAccumulator(steps={len(self._steps)}, cost_known={self._cost_known})"


@dataclass(frozen=True)
class CacheHandle:
    """Access to the request-scoped and shared caches for one run.

    Two scopes because they invalidate differently. ``request`` lives and dies
    with one run and is where a repeated embedding within a single multi-query
    expansion is avoided. ``shared`` outlives the run and is where cache key
    correctness actually matters -- which is why every shared key includes the
    index epoch and the prompt version (ARCHITECTURE.md §22.2).

    Either backend may be ``None``, meaning that scope is not cached. The
    methods here handle that so call sites do not have to; they add no caching
    behaviour of their own beyond dispatching to a backend.

    Args:
        request: Backend scoped to this run.
        shared: Backend shared across runs.
    """

    request: CacheBackend | None = None
    shared: CacheBackend | None = None

    def _backend(self, scope: CacheScope) -> CacheBackend | None:
        return self.request if scope == "request" else self.shared

    async def get(self, key: str, *, scope: CacheScope = "shared") -> bytes | None:
        """Read a value, or ``None`` when absent or when the scope is not cached."""
        backend = self._backend(scope)
        if backend is None:
            return None
        return await backend.get(key)

    async def set(
        self, key: str, value: bytes, *, ttl_s: int | None = None, scope: CacheScope = "shared"
    ) -> None:
        """Write a value, doing nothing when the scope is not cached."""
        backend = self._backend(scope)
        if backend is not None:
            await backend.set(key, value, ttl_s)

    def enabled(self, scope: CacheScope = "shared") -> bool:
        """Return whether a scope has a backend behind it."""
        return self._backend(scope) is not None


@dataclass(frozen=True)
class RunContext:
    """Everything one run carries. **[LOCKED]** -- the attribute set is frozen.

    Nine attributes, asserted by ``test_run_context_scope_is_frozen``. Adding a
    tenth requires architectural review, because this object is depended on by
    every step in the system and each field added to it is a field every future
    step is coupled to.

    Args:
        run_id: Identifier for this run, unique and stable for its lifetime.
        tracer: Where spans go. A no-op tracer is the default, not ``None``, so
            no step has to check.
        metrics: Where counters and histograms go.
        deadline: Absolute time by which the run must finish.
        budget: Ceilings on cost, calls, tokens and steps.
        cache: Request-scoped and shared cache access.
        config: The resolved, hashed configuration. Read-only.
        usage: Append-only consumption record.
        extras: User space. **The library never reads this.**
    """

    run_id: str
    tracer: Tracer
    metrics: MetricSink
    deadline: Deadline
    budget: Budget
    cache: CacheHandle
    config: ConfigSnapshot
    usage: UsageAccumulator
    extras: Mapping[str, Any] = field(default_factory=dict)

    def check_limits(self, *, step: str | None = None) -> None:
        """Raise if the deadline has passed or a budget limit is exceeded.

        Called by the pipeline before each step. Deadline first, because a run
        that is out of time should say so rather than reporting whichever budget
        it happened to exhaust while running late.

        Raises:
            BudgetExceeded: For a passed deadline or a breached limit.
        """
        self.deadline.check(step=step, run_id=self.run_id)
        self.budget.check(self.usage.snapshot(), step=step, run_id=self.run_id)

    def with_deadline(self, deadline: Deadline) -> Self:
        """Return a copy whose deadline is the earlier of the two.

        A step may tighten the deadline; it can never extend it. Returning a
        copy rather than mutating keeps ``RunContext`` frozen and keeps the
        original valid for the caller (INSTRUCTIONS.md §13.16).
        """
        return replace(self, deadline=self.deadline.earliest(deadline))

    def __repr__(self) -> str:
        """Summarise without rendering the config, which could hold secrets."""
        return (
            f"RunContext(run_id={self.run_id!r}, config={self.config.hash[:12]!r}, "
            f"deadline={self.deadline.remaining_s()}, extras={len(self.extras)})"
        )
