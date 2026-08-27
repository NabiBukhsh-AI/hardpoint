"""Policies. **[LOCKED]** INSTRUCTIONS.md §6.1, ADR-004.

Retry lives in exactly one place, and these are the properties that make one
implementation worth having:

- **A non-retryable error is never retried.** An ``AuthError`` retried four
  times is four identical 403s and a quarter of the budget.
- **Configuration cannot make a permanent failure retryable.** Listing
  ``transient`` in ``on`` must not turn an ``AuthError`` into a spend loop.
- **The nesting order is the documented one.** Asserted by observing which
  policy sees what, because inverting per-attempt and total timeout produces a
  system that looks configured for retries and never retries.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from hardpoint.core.config.schema import (
    CircuitBreakerPolicyConfig,
    PolicyConfig,
    RateLimitPolicyConfig,
    RetryPolicyConfig,
    TimeoutPolicyConfig,
)
from hardpoint.core.errors import (
    AuthError,
    InvalidRequestError,
    ProviderTimeout,
    RateLimitedError,
    TransientError,
)
from hardpoint.runtime.policies import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    Fallback,
    PolicyChain,
    RateLimit,
    Retry,
    Timeout,
    backoff_delay,
    is_retryable,
)
from hardpoint.testing import RecordingTracer, build_run_context


class Clock:
    """A monotonic clock a test advances by hand, so nothing sleeps for real."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def failing(times: int, error: Exception, then: str = "ok") -> Callable[[], Awaitable[str]]:
    """Return a call that raises ``times`` times, then succeeds."""
    attempts = {"count": 0}

    async def call() -> str:
        attempts["count"] += 1
        if attempts["count"] <= times:
            raise error
        return then

    call.attempts = attempts  # type: ignore[attr-defined]
    return call


def no_jitter() -> float:
    """Pin the jitter factor so backoff is deterministic in tests."""
    return 1.0


# --------------------------------------------------------------------------- #
# Retryability                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TransientError("503"), True),
        (RateLimitedError("429"), True),
        (ProviderTimeout("slow"), True),
        (TimeoutError("slow"), True),
        (AuthError("401"), False),
        (InvalidRequestError("400"), False),
        (ValueError("bug"), False),
    ],
)
def test_only_retryable_kinds_are_retryable(error: Exception, expected: bool) -> None:
    assert is_retryable(error, ("rate_limited", "transient", "timeout")) is expected


def test_configuration_cannot_make_a_permanent_failure_retryable() -> None:
    """Listing every kind must not turn an ``AuthError`` into a spend loop.

    The taxonomy has the final say: a configuration mistake should never be able
    to make a client hammer an endpoint that will keep returning 403.
    """
    assert is_retryable(AuthError("401"), ("rate_limited", "transient", "timeout")) is False


def test_a_kind_left_out_of_the_policy_is_not_retried() -> None:
    """``on: [rate_limited]`` means transient failures fail fast."""
    assert is_retryable(TransientError("503"), ("rate_limited",)) is False
    assert is_retryable(RateLimitedError("429"), ("rate_limited",)) is True


# --------------------------------------------------------------------------- #
# Backoff                                                                     #
# --------------------------------------------------------------------------- #


def test_the_first_attempt_has_no_delay() -> None:
    assert backoff_delay(0, initial_s=1.0, maximum_s=30.0, jitter=1.0) == 0.0


def test_backoff_grows_exponentially_and_is_capped() -> None:
    delays = [
        backoff_delay(attempt, initial_s=1.0, maximum_s=8.0, jitter=1.0) for attempt in range(1, 6)
    ]
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_full_jitter_spreads_across_the_whole_window() -> None:
    """A uniform draw across the window, not a fixed delay plus noise.

    Fixed-plus-noise synchronises every client that failed at the same moment
    into retrying at the same moment, which turns a provider blip into an
    outage.
    """
    window = backoff_delay(3, initial_s=1.0, maximum_s=30.0, jitter=1.0)
    assert backoff_delay(3, initial_s=1.0, maximum_s=30.0, jitter=0.0) == 0.0
    assert backoff_delay(3, initial_s=1.0, maximum_s=30.0, jitter=0.5) == window / 2


# --------------------------------------------------------------------------- #
# Retry                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_retry_succeeds_after_transient_failures() -> None:
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=3), sleep=clock.sleep, jitter=no_jitter)
    call = failing(2, TransientError("503"))

    assert await retry.run("llm.generate", call, build_run_context()) == "ok"
    assert call.attempts["count"] == 3  # type: ignore[attr-defined]
    assert clock.slept == [0.5, 1.0]


@pytest.mark.anyio
async def test_retry_gives_up_after_max_attempts() -> None:
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=2), sleep=clock.sleep, jitter=no_jitter)
    call = failing(5, TransientError("503"))

    with pytest.raises(TransientError):
        await retry.run("llm.generate", call, build_run_context())
    assert call.attempts["count"] == 2  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_retry_does_not_retry_a_non_retryable_error() -> None:
    """**The property that stops a bad credential burning the budget.**"""
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=5), sleep=clock.sleep, jitter=no_jitter)
    call = failing(5, AuthError("401"))

    with pytest.raises(AuthError):
        await retry.run("llm.generate", call, build_run_context())
    assert call.attempts["count"] == 1, "an auth failure must be attempted exactly once"  # type: ignore[attr-defined]
    assert clock.slept == []


@pytest.mark.anyio
async def test_retry_honours_a_server_supplied_retry_after() -> None:
    """A provider that says how long to wait knows better than any curve."""
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=2), sleep=clock.sleep, jitter=no_jitter)
    call = failing(1, RateLimitedError("429", retry_after_s=7.5))

    assert await retry.run("llm.generate", call, build_run_context()) == "ok"
    assert clock.slept == [7.5], "the computed backoff must not override Retry-After"


@pytest.mark.anyio
async def test_retry_after_is_capped_by_max_backoff() -> None:
    """A provider asking for an hour must not stall a request that long."""
    clock = Clock()
    retry = Retry(
        RetryPolicyConfig(max_attempts=2, max_backoff_s=30.0), sleep=clock.sleep, jitter=no_jitter
    )
    call = failing(1, RateLimitedError("429", retry_after_s=3600.0))

    await retry.run("llm.generate", call, build_run_context())
    assert clock.slept == [30.0]


@pytest.mark.anyio
async def test_retry_after_can_be_ignored_by_configuration() -> None:
    clock = Clock()
    retry = Retry(
        RetryPolicyConfig(max_attempts=2, respect_retry_after=False),
        sleep=clock.sleep,
        jitter=no_jitter,
    )
    await retry.run(
        "op", failing(1, RateLimitedError("429", retry_after_s=99.0)), build_run_context()
    )
    assert clock.slept == [0.5]


@pytest.mark.anyio
async def test_every_retry_attempt_emits_a_span_event() -> None:
    """So a trace shows retries happened, rather than leaving it to be inferred."""
    tracer = RecordingTracer()
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=3), sleep=clock.sleep, jitter=no_jitter)

    await retry.run(
        "llm.generate", failing(2, TransientError("503")), build_run_context(tracer=tracer)
    )

    events = [event for span in tracer.find("hardpoint.retry") for event in span.events]
    assert [name for name, _ in events] == ["retry", "retry"]
    assert events[0][1]["attempt"] == 1
    assert events[0][1]["error"] == "TransientError"


@pytest.mark.anyio
async def test_a_single_attempt_policy_does_not_sleep() -> None:
    clock = Clock()
    retry = Retry(RetryPolicyConfig(max_attempts=1), sleep=clock.sleep, jitter=no_jitter)
    with pytest.raises(TransientError):
        await retry.run("op", failing(1, TransientError("503")), build_run_context())
    assert clock.slept == []


# --------------------------------------------------------------------------- #
# Timeout                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_timeout_raises_provider_timeout() -> None:
    import anyio

    async def slow() -> str:
        await anyio.sleep(5.0)
        return "never"

    with pytest.raises(ProviderTimeout) as exc_info:
        await Timeout(0.02, scope="per_attempt").run("llm.generate", slow, build_run_context())

    assert exc_info.value.retryable is True, "a timeout is worth retrying"
    assert exc_info.value.remedy is not None
    assert "per_attempt_s" in exc_info.value.remedy


@pytest.mark.anyio
async def test_timeout_passes_a_fast_call_through() -> None:
    async def quick() -> str:
        return "ok"

    assert await Timeout(5.0).run("op", quick, build_run_context()) == "ok"


@pytest.mark.anyio
async def test_no_configured_timeout_still_honours_the_run_deadline() -> None:
    """A step may tighten the deadline; it can never extend it."""
    import anyio

    from hardpoint.core.context import Deadline

    async def slow() -> str:
        await anyio.sleep(5.0)
        return "never"

    ctx = build_run_context(deadline=Deadline.in_seconds(0.02))
    with pytest.raises(ProviderTimeout):
        await Timeout(None).run("op", slow, ctx)


@pytest.mark.anyio
async def test_the_shorter_of_timeout_and_deadline_wins() -> None:
    import anyio

    from hardpoint.core.context import Deadline

    async def slow() -> str:
        await anyio.sleep(5.0)
        return "never"

    ctx = build_run_context(deadline=Deadline.in_seconds(0.02))
    with pytest.raises(ProviderTimeout) as exc_info:
        await Timeout(60.0).run("op", slow, ctx)
    assert "0.0" in str(exc_info.value), "the deadline, not the 60s timeout, bounded the call"


# --------------------------------------------------------------------------- #
# CircuitBreaker                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_disabled_breaker_never_interferes() -> None:
    breaker = CircuitBreaker(CircuitBreakerPolicyConfig(enabled=False))
    for _ in range(10):
        with pytest.raises(TransientError):
            await breaker.run("op", failing(1, TransientError("503")), build_run_context())
    assert breaker.state is BreakerState.CLOSED


@pytest.mark.anyio
async def test_the_breaker_opens_after_the_threshold_and_refuses_calls() -> None:
    clock = Clock()
    breaker = CircuitBreaker(
        CircuitBreakerPolicyConfig(enabled=True, failure_threshold=2, cooldown_s=30.0),
        clock=clock,
    )
    ctx = build_run_context()

    for _ in range(2):
        with pytest.raises(TransientError):
            await breaker.run("index.query", failing(1, TransientError("503")), ctx)

    assert breaker.state is BreakerState.OPEN
    call = failing(0, TransientError("503"))
    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.run("index.query", call, ctx)

    assert call.attempts["count"] == 0, "an open circuit must not attempt the call"  # type: ignore[attr-defined]
    assert exc_info.value.retryable is True
    assert exc_info.value.remedy is not None
    assert "cooldown" in exc_info.value.remedy


@pytest.mark.anyio
async def test_the_breaker_half_opens_after_the_cooldown_and_closes_on_success() -> None:
    clock = Clock()
    breaker = CircuitBreaker(
        CircuitBreakerPolicyConfig(enabled=True, failure_threshold=1, cooldown_s=10.0),
        clock=clock,
    )
    ctx = build_run_context()

    with pytest.raises(TransientError):
        await breaker.run("op", failing(1, TransientError("503")), ctx)
    assert breaker.state is BreakerState.OPEN

    clock.now += 11.0
    assert breaker.state is BreakerState.HALF_OPEN

    assert await breaker.run("op", failing(0, TransientError("503")), ctx) == "ok"
    assert breaker.state is BreakerState.CLOSED


@pytest.mark.anyio
async def test_a_success_resets_the_failure_count() -> None:
    """Otherwise a long-lived process trips the breaker on unrelated failures."""
    clock = Clock()
    breaker = CircuitBreaker(
        CircuitBreakerPolicyConfig(enabled=True, failure_threshold=3), clock=clock
    )
    ctx = build_run_context()

    with pytest.raises(TransientError):
        await breaker.run("op", failing(1, TransientError("x")), ctx)
    await breaker.run("op", failing(0, TransientError("x")), ctx)
    with pytest.raises(TransientError):
        await breaker.run("op", failing(1, TransientError("x")), ctx)

    assert breaker.state is BreakerState.CLOSED


# --------------------------------------------------------------------------- #
# RateLimit                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_an_unconfigured_rate_limit_is_transparent() -> None:
    limiter = RateLimit(RateLimitPolicyConfig())
    assert await limiter.run("op", failing(0, TransientError("x")), build_run_context()) == "ok"


@pytest.mark.anyio
async def test_the_rate_limiter_spaces_calls_by_the_configured_interval() -> None:
    clock = Clock()
    limiter = RateLimit(
        RateLimitPolicyConfig(requests_per_second=2.0), sleep=clock.sleep, clock=clock
    )
    ctx = build_run_context()

    for _ in range(3):
        await limiter.run("op", failing(0, TransientError("x")), ctx)

    assert clock.slept == [0.5, 0.5], "two calls per second means a 0.5s interval"


@pytest.mark.anyio
async def test_the_rate_limiter_bounds_concurrency() -> None:
    import anyio

    limiter = RateLimit(RateLimitPolicyConfig(max_concurrent=2))
    ctx = build_run_context()
    in_flight = {"now": 0, "peak": 0}

    async def tracked() -> str:
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await anyio.sleep(0.01)
        in_flight["now"] -= 1
        return "ok"

    async with anyio.create_task_group() as group:
        for _ in range(6):
            group.start_soon(limiter.run, "op", tracked, ctx)

    assert in_flight["peak"] <= 2


# --------------------------------------------------------------------------- #
# Fallback                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_fallback_is_not_used_when_the_primary_succeeds() -> None:
    alternative = failing(0, TransientError("x"), then="secondary")
    result = await Fallback(alternative).run(
        "llm.generate", failing(0, TransientError("x"), then="primary"), build_run_context()
    )
    assert result == "primary"
    assert alternative.attempts["count"] == 0  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_fallback_fires_on_a_provider_error() -> None:
    result = await Fallback(failing(0, TransientError("x"), then="secondary")).run(
        "llm.generate", failing(1, TransientError("503")), build_run_context()
    )
    assert result == "secondary"


@pytest.mark.anyio
async def test_fallback_does_not_swallow_a_programming_error() -> None:
    """A fallback that caught everything would hide bugs behind a second call."""
    with pytest.raises(ValueError, match="bug"):
        await Fallback(failing(0, TransientError("x"), then="secondary")).run(
            "op", failing(1, ValueError("bug")), build_run_context()
        )


@pytest.mark.anyio
async def test_fallback_emits_a_span_event() -> None:
    tracer = RecordingTracer()
    await Fallback(failing(0, TransientError("x"), then="secondary")).run(
        "llm.generate", failing(1, TransientError("503")), build_run_context(tracer=tracer)
    )
    events = [event for span in tracer.find("hardpoint.fallback") for event in span.events]
    assert [name for name, _ in events] == ["fallback"]


@pytest.mark.anyio
async def test_no_alternative_means_the_policy_is_transparent() -> None:
    with pytest.raises(TransientError):
        await Fallback(None).run("op", failing(1, TransientError("503")), build_run_context())


# --------------------------------------------------------------------------- #
# The chain, and its order  [LOCKED]                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_chain_composes_in_the_documented_order() -> None:
    """**The order is load-bearing**, so it is asserted rather than assumed.

    Proved by the observable consequence rather than by inspecting the code: the
    fallback must see a call that has *already* been retried, so the primary is
    attempted ``max_attempts`` times before the alternative runs once.
    """
    clock = Clock()
    primary = failing(99, TransientError("503"))
    alternative = failing(0, TransientError("x"), then="secondary")

    chain = PolicyChain(
        retry=Retry(RetryPolicyConfig(max_attempts=3), sleep=clock.sleep, jitter=no_jitter),
        fallback=Fallback(alternative),
    )

    assert await chain.run("llm.generate", primary, build_run_context()) == "secondary"
    assert primary.attempts["count"] == 3, "the fallback fired after retries, not before"  # type: ignore[attr-defined]
    assert alternative.attempts["count"] == 1  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_the_per_attempt_timeout_sits_inside_retry() -> None:
    """**The inversion people actually make.**

    A per-attempt timeout placed outside the retry loop is a total timeout, and
    the retries it was meant to bound never happen. Here three attempts each
    time out and each is retried, which is only possible if the timeout is
    inside.
    """
    import anyio

    clock = Clock()
    attempts = {"count": 0}

    async def slow() -> str:
        attempts["count"] += 1
        await anyio.sleep(5.0)
        return "never"

    chain = PolicyChain(
        retry=Retry(
            RetryPolicyConfig(max_attempts=3, on=("timeout",)), sleep=clock.sleep, jitter=no_jitter
        ),
        attempt_timeout=Timeout(0.01, scope="per_attempt"),
    )

    with pytest.raises(ProviderTimeout):
        await chain.run("llm.generate", slow, build_run_context())

    assert attempts["count"] == 3, "each attempt got its own timeout and was retried"


@pytest.mark.anyio
async def test_the_breaker_counts_operations_not_attempts() -> None:
    """The breaker sits outside retry, so one retried call is one failure.

    If it sat inside, a single request with three retries would trip a
    three-failure threshold on its own.
    """
    clock = Clock()
    breaker = CircuitBreaker(
        CircuitBreakerPolicyConfig(enabled=True, failure_threshold=2), clock=clock
    )
    chain = PolicyChain(
        retry=Retry(RetryPolicyConfig(max_attempts=3), sleep=clock.sleep, jitter=no_jitter),
        circuit_breaker=breaker,
    )
    ctx = build_run_context()

    with pytest.raises(TransientError):
        await chain.run("index.query", failing(99, TransientError("503")), ctx)
    assert breaker.state is BreakerState.CLOSED, "one operation, not three, has failed"

    with pytest.raises(TransientError):
        await chain.run("index.query", failing(99, TransientError("503")), ctx)
    assert breaker.state is BreakerState.OPEN


@pytest.mark.anyio
async def test_an_empty_chain_is_transparent() -> None:
    assert (
        await PolicyChain().run("op", failing(0, TransientError("x")), build_run_context()) == "ok"
    )


def test_the_chain_describes_itself_outermost_first() -> None:
    """``components list --resolved`` prints this, so nobody guesses."""
    chain = PolicyChain.from_config(
        PolicyConfig(
            retry=RetryPolicyConfig(max_attempts=3),
            timeout=TimeoutPolicyConfig(per_attempt_s=30, total_s=90),
            circuit_breaker=CircuitBreakerPolicyConfig(enabled=True),
        ),
        alternative=failing(0, TransientError("x")),
    )
    assert chain.describe() == (
        "fallback -> circuit_breaker -> timeout(total) -> retry -> timeout(per_attempt)"
    )


def test_an_unconfigured_chain_describes_itself_as_empty() -> None:
    assert PolicyChain().describe() == "(none)"


def test_from_config_omits_policies_that_are_switched_off() -> None:
    """A single-attempt retry is not a retry, and building one would be noise."""
    chain = PolicyChain.from_config(PolicyConfig(retry=RetryPolicyConfig(max_attempts=1)))
    assert chain.retry is None
    assert chain.circuit_breaker is None
    assert chain.rate_limit is None
    assert chain.fallback is None


def test_from_config_builds_what_is_configured() -> None:
    chain = PolicyChain.from_config(
        PolicyConfig(
            timeout=TimeoutPolicyConfig(per_attempt_s=30.0, total_s=90.0),
            rate_limit=RateLimitPolicyConfig(max_concurrent=4),
        )
    )
    assert chain.attempt_timeout is not None
    assert chain.attempt_timeout.seconds == 30.0
    assert chain.total_timeout is not None
    assert chain.total_timeout.seconds == 90.0
    assert chain.rate_limit is not None


def test_a_chain_holds_its_stateful_policies_rather_than_rebuilding_them() -> None:
    """A breaker rebuilt per call would never remember a failure.

    Stated as a test because "build the chain once at construction time" is the
    kind of thing a later refactor moves into the call path without noticing
    that it silently disables two policies.
    """
    config = PolicyConfig(circuit_breaker=CircuitBreakerPolicyConfig(enabled=True))
    chain = PolicyChain.from_config(config)
    assert chain.circuit_breaker is not None
    assert chain.circuit_breaker is chain.circuit_breaker


def test_policy_run_signatures_are_uniform() -> None:
    """The chain composes them by calling ``run(operation, call, ctx)`` on each."""
    import inspect

    policies: list[Any] = [
        Retry(),
        Timeout(),
        CircuitBreaker(),
        RateLimit(),
        Fallback(),
    ]
    for policy in policies:
        parameters = list(inspect.signature(policy.run).parameters)
        assert parameters == ["operation", "call", "ctx"], type(policy).__name__
