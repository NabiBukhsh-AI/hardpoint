"""Retry, timeout, circuit breaking, rate limiting and fallback (**[LOCKED]**).

Implements ADR-004 and INSTRUCTIONS.md §6.1. **Retry logic exists in exactly one
place: here.** No adapter implements backoff, and a review that finds one in an
adapter rejects it (INSTRUCTIONS.md §13.1).

The reason is not tidiness. Five adapters with their own retry loops means five
subtly different backoff curves, five different opinions about what is
retryable, and no way for a user to change any of them without forking an
adapter. One implementation means retry behaviour is configurable, observable as
a span event, and correct once.

## The nesting order is load-bearing

Policies compose as a chain, and the order is not a style choice. From outermost
to innermost:

```
Fallback( CircuitBreaker( Timeout(total)( Retry( Timeout(per_attempt)( RateLimit( call ))))))
```

Each position is forced:

- **Fallback outermost.** It should fire when the primary has genuinely failed,
  which means after its retries are exhausted, not after the first attempt.
- **CircuitBreaker outside Retry.** The breaker counts failed *operations*. If
  it sat inside, a single retried call would trip it in one request.
- **Total timeout outside Retry, per-attempt timeout inside.** This is the one
  people invert. A per-attempt timeout placed outside the retry loop is not a
  per-attempt timeout at all -- it is a total timeout, and the retries it was
  meant to bound never happen because the first slow attempt consumes the whole
  budget.
- **RateLimit innermost.** It gates actual calls, so a retry attempt queues
  behind the limiter exactly as a first attempt does.

A test asserts this order by observing which policy sees what.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

import anyio

from hardpoint.core.config.schema import (
    CircuitBreakerPolicyConfig,
    PolicyConfig,
    RateLimitPolicyConfig,
    RetryableKind,
    RetryPolicyConfig,
)
from hardpoint.core.errors import (
    HardpointError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    TransientError,
)

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = [
    "BreakerState",
    "CircuitBreaker",
    "CircuitOpenError",
    "Fallback",
    "PolicyChain",
    "RateLimit",
    "Retry",
    "Timeout",
    "backoff_delay",
    "is_retryable",
]

T = TypeVar("T")

Call = Callable[[], Awaitable[T]]


def _kind_of(error: BaseException) -> RetryableKind | None:
    """Classify an error into the kinds a retry policy can be told to act on.

    Returns ``None`` for anything a retry policy must never act on, which is
    every non-provider error and every provider error the taxonomy marks as not
    retryable. An ``AuthError`` retried four times is four identical 403s and a
    quarter of the budget spent.
    """
    if isinstance(error, RateLimitedError):
        return "rate_limited"
    if isinstance(error, ProviderTimeout | TimeoutError):
        return "timeout"
    if isinstance(error, TransientError):
        return "transient"
    return None


def is_retryable(error: BaseException, kinds: Sequence[RetryableKind]) -> bool:
    """Return whether an error may be retried under a policy's configured kinds.

    Two conditions, both required. The error must classify into one of the
    configured kinds, *and* the taxonomy must agree it is retryable. Listing
    ``transient`` in configuration cannot make an ``AuthError`` retryable,
    because a configuration mistake should not be able to turn a permanent
    failure into a spend loop.
    """
    kind = _kind_of(error)
    if kind is None or kind not in kinds:
        return False
    if isinstance(error, HardpointError):
        return error.retryable
    return True


def backoff_delay(attempt: int, *, initial_s: float, maximum_s: float, jitter: float) -> float:
    """Return the delay before an attempt, using exponential backoff with full jitter.

    Full jitter -- a uniform draw across the whole window rather than a fixed
    delay plus noise -- because the alternative synchronises every client that
    failed at the same moment into retrying at the same moment, which is how a
    provider blip becomes an outage.

    Args:
        attempt: Zero-based index of the attempt about to be made. Attempt 0 is
            the first try and has no delay.
        initial_s: Base delay.
        maximum_s: Ceiling on the window.
        jitter: A value in ``[0, 1)``. Injected rather than drawn here so a test
            can pin it.

    Returns:
        Seconds to wait. Zero for the first attempt.
    """
    if attempt <= 0:
        return 0.0
    window = min(maximum_s, initial_s * (2 ** (attempt - 1)))
    return float(window * jitter)


@dataclass(frozen=True)
class Retry:
    """Retry a failing call with exponential backoff and full jitter.

    Args:
        config: Attempts, which error kinds to act on, and the backoff window.
        sleep: How to wait. Injected so a test does not spend real seconds.
        jitter: Draws the jitter factor. Injected so a test is deterministic.
    """

    config: RetryPolicyConfig = field(default_factory=RetryPolicyConfig)
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep
    jitter: Callable[[], float] = random.random

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call``, retrying per the policy.

        Every attempt after the first emits a ``retry`` span event carrying the
        attempt number, the delay and the error that caused it, so a trace shows
        whether retries happened rather than leaving it to be inferred from
        latency.

        Raises:
            BaseException: The last error, once attempts are exhausted or the
                error is not retryable.
        """
        last: BaseException | None = None

        for attempt in range(self.config.max_attempts):
            if attempt > 0:
                delay = self._delay(attempt, last)
                async with ctx.tracer.span("hardpoint.retry") as span:
                    span.add_event(
                        "retry",
                        {
                            "operation": operation,
                            "attempt": attempt,
                            "delay_s": delay,
                            "error": type(last).__name__ if last else "",
                        },
                    )
                await self.sleep(delay)

            try:
                return await call()
            except Exception as exc:
                last = exc
                if not is_retryable(exc, self.config.on):
                    raise
                if attempt == self.config.max_attempts - 1:
                    raise

        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover

    def _delay(self, attempt: int, last: BaseException | None) -> float:
        """Return the wait before an attempt, honouring a server-supplied Retry-After.

        A provider that says how long to wait knows better than any backoff
        curve, and ignoring it is how a client keeps hammering a service that
        already told it to stop.
        """
        if (
            self.config.respect_retry_after
            and isinstance(last, RateLimitedError)
            and last.retry_after_s is not None
        ):
            return min(last.retry_after_s, self.config.max_backoff_s)
        return backoff_delay(
            attempt,
            initial_s=self.config.initial_backoff_s,
            maximum_s=self.config.max_backoff_s,
            jitter=self.jitter(),
        )


@dataclass(frozen=True)
class Timeout:
    """Bound how long a call may take.

    Args:
        seconds: The limit, or ``None`` for no limit of its own.
        scope: ``per_attempt`` or ``total``. Only used in the error message; the
            distinction that matters is where the policy sits in the chain.
    """

    seconds: float | None = None
    scope: str = "total"

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call`` under a time limit.

        The run's own deadline always applies as well, and whichever is sooner
        wins: a step may tighten the deadline but can never extend it.

        Raises:
            ProviderTimeout: If the limit or the run deadline passes first.
        """
        limit = self.seconds
        remaining = ctx.deadline.remaining_s()
        if remaining is not None:
            limit = remaining if limit is None else min(limit, remaining)

        if limit is None:
            return await call()

        result: list[T] = []
        with anyio.move_on_after(limit) as scope:
            result.append(await call())

        if scope.cancelled_caught:
            raise ProviderTimeout(
                f"{operation} exceeded its {self.scope} timeout of {limit:.3f}s.",
                component=operation,
                remedy=(
                    f"Raise `policies.timeout.{self.scope}_s` for this component, or "
                    f"`budgets.request.deadline_s` if the run deadline is what ran out."
                ),
            )
        return result[0]


class BreakerState(StrEnum):
    """Where a circuit breaker currently sits.

    Attributes:
        CLOSED: Calls pass through. The normal state.
        OPEN: Calls are refused without being attempted, until the cooldown
            elapses.
        HALF_OPEN: One call is allowed through to test whether the dependency
            has recovered. Success closes the circuit; failure re-opens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ProviderError):
    """A call was refused because its circuit breaker is open.

    Retryable, because the cooldown will elapse. A caller that wants to fall
    back rather than wait should configure a fallback.
    """

    default_code = "provider.circuit_open"
    default_retryable = True


class CircuitBreaker:
    """Stop calling a dependency that is failing, for a cooldown period.

    Stateful, and deliberately so: the whole point is to remember failures
    across calls. One breaker instance guards one component, and it is shared by
    every call to that component.

    Args:
        config: Failure threshold and cooldown.
        clock: Reads the current monotonic time. Injected so a test can advance
            it without sleeping.
    """

    __slots__ = ("_clock", "_failures", "_opened_at", "_state", "config")

    def __init__(
        self,
        config: CircuitBreakerPolicyConfig | None = None,
        *,
        clock: Callable[[], float] = anyio.current_time,
    ) -> None:
        self.config = config or CircuitBreakerPolicyConfig()
        self._clock = clock
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0

    @property
    def state(self) -> BreakerState:
        """The breaker's current state, after accounting for an elapsed cooldown."""
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= (
            self.config.cooldown_s
        ):
            return BreakerState.HALF_OPEN
        return self._state

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call`` unless the circuit is open.

        Raises:
            CircuitOpenError: If the circuit is open and the cooldown has not
                elapsed.
            BaseException: Whatever the call raised, after recording the failure.
        """
        if not self.config.enabled:
            return await call()

        current = self.state
        if current is BreakerState.OPEN:
            raise CircuitOpenError(
                f"The circuit breaker for {operation!r} is open after "
                f"{self._failures} consecutive failures.",
                component=operation,
                remedy=(
                    f"Wait {self.config.cooldown_s:.0f}s for the cooldown, configure a "
                    f"fallback for this component, or raise "
                    f"`policies.circuit_breaker.failure_threshold`."
                ),
            )

        try:
            result = await call()
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.config.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()

    def _record_success(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED

    def __repr__(self) -> str:
        """Render the state and the consecutive failure count."""
        return f"CircuitBreaker(state={self.state.value}, failures={self._failures})"


class RateLimit:
    """Bound concurrency and request rate for one component, client-side.

    Stateful, like the breaker, and shared by every call to the component it
    guards. Respecting a provider's limit before it has to enforce it is
    cheaper than being rate-limited and retrying.

    Args:
        config: Concurrency ceiling and requests per second.
        sleep: How to wait. Injected for tests.
        clock: Reads the current monotonic time. Injected for tests.
    """

    __slots__ = ("_clock", "_next_slot", "_semaphore", "_sleep", "config")

    def __init__(
        self,
        config: RateLimitPolicyConfig | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        clock: Callable[[], float] = anyio.current_time,
    ) -> None:
        self.config = config or RateLimitPolicyConfig()
        self._sleep = sleep
        self._clock = clock
        self._semaphore = (
            anyio.Semaphore(self.config.max_concurrent)
            if self.config.max_concurrent is not None
            else None
        )
        self._next_slot = 0.0

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call`` once the limiter allows it."""
        if self._semaphore is None:
            await self._await_slot()
            return await call()

        async with self._semaphore:
            await self._await_slot()
            return await call()

    async def _await_slot(self) -> None:
        """Wait until the configured request rate permits another call."""
        rate = self.config.requests_per_second
        if rate is None:
            return
        interval = 1.0 / rate
        now = self._clock()
        if self._next_slot > now:
            await self._sleep(self._next_slot - now)
            now = self._clock()
        self._next_slot = max(now, self._next_slot) + interval


@dataclass(frozen=True)
class Fallback:
    """Try an alternative when the primary call fails.

    The mechanism behind model fallback, reranker degradation and provider
    outage handling (ARCHITECTURE.md §11.3). Placed outermost in the chain, so
    it fires after the primary's retries are exhausted rather than after its
    first stumble.

    Args:
        alternative: What to call instead. ``None`` disables the policy.
        on: Exception types that trigger the fallback. Defaults to any provider
            error, because a fallback that also caught programming errors would
            hide bugs behind a second provider call.
    """

    alternative: Call[object] | None = None
    on: tuple[type[BaseException], ...] = (ProviderError,)

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call``, falling back on a matching failure.

        Raises:
            BaseException: The original error when no alternative is configured
                or the error does not match ``on``; the alternative's error if
                the fallback itself fails.
        """
        if self.alternative is None:
            return await call()

        try:
            return await call()
        except self.on as exc:
            async with ctx.tracer.span("hardpoint.fallback") as span:
                span.add_event(
                    "fallback",
                    {"operation": operation, "error": type(exc).__name__},
                )
            fallen_back: T = await self.alternative()  # type: ignore[assignment]  # caller's contract
            return fallen_back


@dataclass(frozen=True)
class PolicyChain:
    """The policies applied to one component or step, composed in the fixed order.

    Built once at construction time from configuration and reused for every
    call, which matters for the stateful members: a breaker that was rebuilt per
    call would never remember a failure, and a rate limiter would never limit.

    Args:
        retry: Backoff behaviour.
        total_timeout: Bound on the whole operation, retries included.
        attempt_timeout: Bound on one attempt.
        circuit_breaker: Failure isolation. Shared, stateful.
        rate_limit: Client-side ceilings. Shared, stateful.
        fallback: Alternative to try once the primary has genuinely failed.
    """

    retry: Retry | None = None
    total_timeout: Timeout | None = None
    attempt_timeout: Timeout | None = None
    circuit_breaker: CircuitBreaker | None = None
    rate_limit: RateLimit | None = None
    fallback: Fallback | None = None

    @classmethod
    def from_config(
        cls,
        config: PolicyConfig,
        *,
        alternative: Call[object] | None = None,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        jitter: Callable[[], float] = random.random,
        clock: Callable[[], float] = anyio.current_time,
    ) -> PolicyChain:
        """Build a chain from a component's ``policies`` block.

        Args:
            config: The component's policy configuration.
            alternative: The fallback call, already constructed. Configuration
                names a component; turning that name into a callable is the
                registry's job, not this module's.
            sleep: Injected for tests.
            jitter: Injected for tests.
            clock: Injected for tests.

        Returns:
            A chain ready to wrap calls.
        """
        return cls(
            retry=Retry(config.retry, sleep=sleep, jitter=jitter)
            if config.retry.max_attempts > 1
            else None,
            total_timeout=Timeout(config.timeout.total_s, scope="total")
            if config.timeout.total_s is not None
            else None,
            attempt_timeout=Timeout(config.timeout.per_attempt_s, scope="per_attempt")
            if config.timeout.per_attempt_s is not None
            else None,
            circuit_breaker=CircuitBreaker(config.circuit_breaker, clock=clock)
            if config.circuit_breaker.enabled
            else None,
            rate_limit=RateLimit(config.rate_limit, sleep=sleep, clock=clock)
            if config.rate_limit.max_concurrent is not None
            or config.rate_limit.requests_per_second is not None
            else None,
            fallback=Fallback(alternative) if alternative is not None else None,
        )

    def describe(self) -> str:
        """Render the active policies, outermost first.

        What ``hardpoint components list --resolved`` prints, so nobody has to
        guess whether retries are on (ARCHITECTURE.md §11.3).
        """
        active = [
            name
            for name, policy in (
                ("fallback", self.fallback),
                ("circuit_breaker", self.circuit_breaker),
                ("timeout(total)", self.total_timeout),
                ("retry", self.retry),
                ("timeout(per_attempt)", self.attempt_timeout),
                ("rate_limit", self.rate_limit),
            )
            if policy is not None
        ]
        return " -> ".join(active) if active else "(none)"

    async def run(self, operation: str, call: Call[T], ctx: RunContext) -> T:
        """Invoke ``call`` through every configured policy, in the fixed order.

        Args:
            operation: What is being called, for errors and span events.
            call: The operation, as a callable so a retry can invoke it again.
            ctx: The run context, for the deadline and the tracer.

        Returns:
            Whatever the call returned.

        Raises:
            BaseException: Whatever survives the chain.
        """
        # Built inside out: `wrapped` starts as the raw call and each policy
        # wraps what came before, so the last one applied is the outermost.
        wrapped = call

        for policy in (
            self.rate_limit,
            self.attempt_timeout,
            self.retry,
            self.total_timeout,
            self.circuit_breaker,
            self.fallback,
        ):
            if policy is None:
                continue
            wrapped = _bind(policy, operation, wrapped, ctx)

        return await wrapped()


def _bind(
    policy: Retry | Timeout | CircuitBreaker | RateLimit | Fallback,
    operation: str,
    call: Call[T],
    ctx: RunContext,
) -> Call[T]:
    """Return a zero-argument callable that runs ``call`` through ``policy``."""

    async def invoke() -> T:
        return await policy.run(operation, call, ctx)

    return invoke
