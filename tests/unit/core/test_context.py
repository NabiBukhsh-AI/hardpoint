"""``RunContext`` and its machinery. **[LOCKED]** INSTRUCTIONS.md §5.4.

The locked assertion is ``test_run_context_scope_is_frozen``, reproduced from
§5.4 verbatim. It is the mechanism that makes "adding a field requires
architectural review" real rather than a comment nobody reads.

The rest covers the behaviour that has to be right for budgets and deadlines to
mean anything: monotonic time, deadlines that only tighten, an unknown cost that
poisons a total rather than being counted as free, and a limit error that names
which limit was passed.
"""

from __future__ import annotations

import ast
import dataclasses
import time
from pathlib import Path

import anyio
import pytest

import hardpoint
from hardpoint.core.config.schema import RequestBudgetConfig
from hardpoint.core.config.snapshot import ConfigSnapshot
from hardpoint.core.context import Budget, CacheHandle, Deadline, RunContext, UsageAccumulator
from hardpoint.core.errors import BudgetExceeded
from hardpoint.core.models import Usage


class NullTracer:
    def span(self, name: str, **attrs: object) -> object:
        raise NotImplementedError


class NullMetrics:
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


class DictCache:
    """A minimal CacheBackend, to prove CacheHandle dispatches rather than caches."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.gets = 0
        self.sets = 0

    async def get(self, key: str) -> bytes | None:
        self.gets += 1
        return self.store.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        self.sets += 1
        self.store[key] = value

    async def delete_prefix(self, prefix: str) -> int:
        matched = [k for k in self.store if k.startswith(prefix)]
        for key in matched:
            del self.store[key]
        return len(matched)


def build_context(**overrides: object) -> RunContext:
    fields: dict[str, object] = {
        "run_id": "run-1",
        "tracer": NullTracer(),
        "metrics": NullMetrics(),
        "deadline": Deadline.none(),
        "budget": Budget(),
        "cache": CacheHandle(),
        "config": ConfigSnapshot(env="test", data={}),
        "usage": UsageAccumulator(),
        "extras": {},
    }
    fields.update(overrides)
    return RunContext(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The locked scope                                                            #
# --------------------------------------------------------------------------- #


def test_run_context_scope_is_frozen() -> None:
    """**[LOCKED]** Reproduced verbatim from INSTRUCTIONS.md §5.4.

    RunContext is depended on by every step in the system, so every field added
    to it is a field every future step is coupled to. This test is what turns
    "additions require architectural review" into something that actually stops
    an addition.
    """
    assert set(RunContext.__dataclass_fields__) == {
        "run_id",
        "tracer",
        "metrics",
        "deadline",
        "budget",
        "cache",
        "config",
        "usage",
        "extras",
    }


def test_run_context_is_immutable() -> None:
    """**Do not mutate RunContext** (INSTRUCTIONS.md §13.16)."""
    ctx = build_context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_id = "other"  # type: ignore[misc]


def test_run_context_repr_does_not_render_the_config() -> None:
    """The config can hold secrets; a repr in a log must not carry them."""
    rendered = repr(build_context())
    assert "run-1" in rendered
    assert "data=" not in rendered


def test_the_library_never_reads_extras() -> None:
    """``extras`` is user space, and it stops being so the moment we read a key.

    Scans the library's own source for an attribute access on ``extras``,
    excluding the module that defines it. A string search would trip over the
    word in a docstring, so this parses.
    """
    package_root = Path(hardpoint.__file__).resolve().parent
    offenders: list[str] = []

    for path in package_root.rglob("*.py"):
        if path.name == "context.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{path.relative_to(package_root)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "extras"
        )

    assert not offenders, (
        "the library read RunContext.extras, which makes an untyped mapping part "
        f"of the contract: {offenders}"
    )


# --------------------------------------------------------------------------- #
# Deadline                                                                    #
# --------------------------------------------------------------------------- #


def test_no_deadline_never_expires() -> None:
    deadline = Deadline.none()
    assert deadline.remaining_s() is None
    assert deadline.expired() is False
    assert deadline.check() is None


def test_deadline_counts_down() -> None:
    deadline = Deadline.in_seconds(5.0)
    remaining = deadline.remaining_s()
    assert remaining is not None
    assert 4.0 < remaining <= 5.0
    assert deadline.expired() is False


def test_an_expired_deadline_reports_zero_not_a_negative_number() -> None:
    """A negative timeout passed to a provider client is a bug, not a fast failure."""
    deadline = Deadline(at=time.monotonic() - 10)
    assert deadline.remaining_s() == 0.0
    assert deadline.expired() is True


def test_an_expired_deadline_raises_budget_exceeded_naming_itself() -> None:
    deadline = Deadline(at=time.monotonic() - 1)
    with pytest.raises(BudgetExceeded) as exc_info:
        deadline.check(step="generate", run_id="run-1")

    error = exc_info.value
    assert error.limit == "deadline_s"
    assert error.step == "generate"
    assert error.remedy is not None
    assert "deadline_s" in error.remedy


def test_deadlines_can_only_tighten() -> None:
    """A step may impose its own timeout; it can never extend the run's."""
    tight = Deadline.in_seconds(1.0)
    loose = Deadline.in_seconds(60.0)
    assert tight.earliest(loose) is tight
    assert loose.earliest(tight) is tight


def test_combining_with_no_deadline_yields_the_other() -> None:
    bounded = Deadline.in_seconds(5.0)
    assert Deadline.none().earliest(bounded) is bounded
    assert bounded.earliest(Deadline.none()) is bounded
    assert Deadline.none().earliest(Deadline.none()).at is None


def test_with_deadline_returns_a_copy_and_tightens() -> None:
    ctx = build_context(deadline=Deadline.in_seconds(60.0))
    tightened = ctx.with_deadline(Deadline.in_seconds(1.0))

    assert tightened is not ctx
    original_remaining = ctx.deadline.remaining_s()
    assert original_remaining is not None
    assert original_remaining > 50, "the original context must be unchanged"
    remaining = tightened.deadline.remaining_s()
    assert remaining is not None
    assert remaining <= 1.0


def test_with_deadline_cannot_extend() -> None:
    ctx = build_context(deadline=Deadline.in_seconds(1.0))
    loosened = ctx.with_deadline(Deadline.in_seconds(600.0))
    remaining = loosened.deadline.remaining_s()
    assert remaining is not None
    assert remaining <= 1.0


@pytest.mark.anyio
async def test_as_anyio_deadline_converts_onto_anyios_clock() -> None:
    """anyio's clock is not guaranteed to read the same as time.monotonic().

    Passing a ``time.monotonic()`` value straight into a cancel scope would be
    subtly wrong, so the conversion goes through the remaining duration.
    """
    deadline = Deadline.in_seconds(5.0)
    converted = deadline.as_anyio_deadline()
    now = anyio.current_time()
    assert now < converted <= now + 5.0


@pytest.mark.anyio
async def test_as_anyio_deadline_is_infinite_without_a_deadline() -> None:
    assert Deadline.none().as_anyio_deadline() == float("inf")


@pytest.mark.anyio
async def test_an_anyio_cancel_scope_honours_the_deadline() -> None:
    """The end-to-end property the conversion exists for.

    A flag cannot stop an in-flight await; a cancel scope can, and this proves
    the deadline actually drives one.
    """
    deadline = Deadline.in_seconds(0.05)

    with anyio.CancelScope(deadline=deadline.as_anyio_deadline()) as scope:
        await anyio.sleep(5.0)

    assert scope.cancelled_caught is True, "the sleep should have been cancelled"
    assert deadline.expired() is True


# --------------------------------------------------------------------------- #
# Budget                                                                      #
# --------------------------------------------------------------------------- #


def test_an_unbounded_budget_never_trips() -> None:
    usage = Usage(total_cost_usd=1_000_000.0)
    assert Budget().check(usage) is None


@pytest.mark.parametrize(
    ("budget", "usage", "expected_limit"),
    [
        (Budget(max_cost_usd=0.10), Usage(total_cost_usd=0.11), "max_cost_usd"),
        (Budget(max_tokens=100), Usage(), "max_tokens"),
        (Budget(max_llm_calls=1), Usage(), "max_llm_calls"),
        (Budget(max_steps=1), Usage(), "max_steps"),
    ],
)
def test_each_limit_can_trip_and_names_itself(
    budget: Budget, usage: Usage, expected_limit: str
) -> None:
    """A budget error must say which ceiling was hit, or it is unactionable."""
    if expected_limit in {"max_tokens", "max_llm_calls", "max_steps"}:
        accumulator = UsageAccumulator()
        accumulator.record("a", calls=2, prompt_tokens=200, cost_usd=0.0)
        accumulator.record("b", calls=2, prompt_tokens=200, cost_usd=0.0)
        usage = accumulator.snapshot()

    with pytest.raises(BudgetExceeded) as exc_info:
        budget.check(usage, step="generate", run_id="run-1")

    assert exc_info.value.limit == expected_limit
    assert exc_info.value.step == "generate"
    assert expected_limit in str(exc_info.value)


def test_a_budget_exactly_at_its_limit_does_not_trip() -> None:
    """Ceilings are inclusive; spending exactly the allowance is allowed."""
    assert Budget(max_cost_usd=0.10).check(Usage(total_cost_usd=0.10)) is None


def test_an_unknown_cost_does_not_trip_the_cost_limit() -> None:
    """``cost_usd=None`` means unpriced, not free, and not infinite either.

    Refusing to run because a model is missing from the pricing table would be
    the wrong failure: the run is fine, the table is incomplete.
    """
    usage = Usage(total_cost_usd=None)
    assert Budget(max_cost_usd=0.01).check(usage) is None


def test_budget_from_config() -> None:
    budget = Budget.from_config(
        RequestBudgetConfig(max_cost_usd=0.15, max_llm_calls=6, deadline_s=25)
    )
    assert budget.max_cost_usd == 0.15
    assert budget.max_llm_calls == 6
    assert budget.max_tokens is None


def test_check_limits_reports_the_deadline_before_a_budget() -> None:
    """A run that is out of time should say so.

    Otherwise it reports whichever budget it happened to exhaust while running
    late, which sends the investigation in the wrong direction.
    """
    usage = UsageAccumulator()
    usage.record("a", calls=99, cost_usd=99.0)
    ctx = build_context(
        deadline=Deadline(at=time.monotonic() - 1),
        budget=Budget(max_cost_usd=0.01),
        usage=usage,
    )
    with pytest.raises(BudgetExceeded) as exc_info:
        ctx.check_limits(step="generate")
    assert exc_info.value.limit == "deadline_s"


def test_check_limits_passes_when_within_both() -> None:
    assert build_context().check_limits(step="retrieve") is None


# --------------------------------------------------------------------------- #
# UsageAccumulator                                                            #
# --------------------------------------------------------------------------- #


def test_usage_accumulates_across_calls_to_one_step() -> None:
    """A step making three provider calls must report all three."""
    usage = UsageAccumulator()
    usage.record("generate", calls=1, prompt_tokens=10, cost_usd=0.01)
    usage.record("generate", calls=1, prompt_tokens=5, completion_tokens=7, cost_usd=0.02)

    snapshot = usage.snapshot()
    step = snapshot.by_step["generate"]
    assert step.calls == 2
    assert step.prompt_tokens == 15
    assert step.completion_tokens == 7
    assert step.cost_usd == pytest.approx(0.03)


def test_usage_attributes_per_step() -> None:
    usage = UsageAccumulator()
    usage.record("retrieve", calls=1, embed_tokens=8, cost_usd=0.001)
    usage.record("generate", calls=1, prompt_tokens=100, cost_usd=0.02)

    snapshot = usage.snapshot()
    assert set(snapshot.by_step) == {"retrieve", "generate"}
    assert snapshot.total_tokens == 108
    assert snapshot.total_cost_usd == pytest.approx(0.021)


def test_an_unpriced_call_makes_the_total_unknown_not_understated() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §13.8.

    A partial sum presented as a total is worse than no total: it looks
    authoritative and understates the bill.
    """
    usage = UsageAccumulator()
    usage.record("generate", calls=1, cost_usd=0.02)
    usage.record("rerank", calls=1, cost_usd=None)

    assert usage.snapshot().total_cost_usd is None


def test_a_zero_cost_call_is_not_an_unknown_cost() -> None:
    """A genuinely free call is priced at zero, and must not poison the total."""
    usage = UsageAccumulator()
    usage.record("generate", calls=1, cost_usd=0.02)
    usage.record("local_rerank", calls=1, cost_usd=0.0)

    assert usage.snapshot().total_cost_usd == pytest.approx(0.02)


def test_a_step_with_no_calls_and_no_cost_does_not_poison_the_total() -> None:
    """Recording latency for a pure-computation step must not make cost unknown."""
    usage = UsageAccumulator()
    usage.record("generate", calls=1, cost_usd=0.02)
    usage.record("assemble_context", latency_ms=3.0)

    assert usage.snapshot().total_cost_usd == pytest.approx(0.02)


def test_total_cost_is_none_when_nothing_was_priced() -> None:
    """Not "zero spent"; "no priced call happened"."""
    usage = UsageAccumulator()
    usage.record("assemble_context", latency_ms=3.0)
    assert usage.snapshot().total_cost_usd is None


def test_estimated_is_sticky_across_records() -> None:
    """One estimated count makes the step's figures estimated. It cannot be undone."""
    usage = UsageAccumulator()
    usage.record("generate", calls=1, prompt_tokens=10, estimated=True)
    usage.record("generate", calls=1, prompt_tokens=10, estimated=False)
    assert usage.snapshot().by_step["generate"].estimated is True


def test_latency_sums_into_the_total() -> None:
    usage = UsageAccumulator()
    usage.record("a", latency_ms=1.5)
    usage.record("b", latency_ms=2.5)
    assert usage.snapshot().total_latency_ms == pytest.approx(4.0)


def test_snapshot_is_a_copy_not_a_live_view() -> None:
    """A snapshot handed to a budget check must not change under it."""
    usage = UsageAccumulator()
    usage.record("a", calls=1)
    snapshot = usage.snapshot()
    usage.record("a", calls=1)

    assert snapshot.by_step["a"].calls == 1
    assert usage.snapshot().by_step["a"].calls == 2


def test_empty_accumulator_snapshots_cleanly() -> None:
    snapshot = UsageAccumulator().snapshot()
    assert snapshot.by_step == {}
    assert snapshot.total_cost_usd is None
    assert snapshot.total_latency_ms == 0.0


def test_usage_accumulator_repr_is_a_summary() -> None:
    usage = UsageAccumulator()
    usage.record("a", calls=1, cost_usd=0.1)
    assert "steps=1" in repr(usage)


# --------------------------------------------------------------------------- #
# CacheHandle                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_cache_handle_with_no_backend_is_a_quiet_miss() -> None:
    """A pipeline with caching off must not have to guard every cache call."""
    handle = CacheHandle()
    assert await handle.get("k") is None
    assert await handle.set("k", b"v") is None
    assert handle.enabled("shared") is False
    assert handle.enabled("request") is False


@pytest.mark.anyio
async def test_cache_handle_dispatches_to_the_right_scope() -> None:
    """The two scopes invalidate differently and must not be conflated."""
    request_cache, shared_cache = DictCache(), DictCache()
    handle = CacheHandle(request=request_cache, shared=shared_cache)

    await handle.set("k", b"request-value", scope="request")
    await handle.set("k", b"shared-value", scope="shared")

    assert await handle.get("k", scope="request") == b"request-value"
    assert await handle.get("k", scope="shared") == b"shared-value"
    assert request_cache.sets == 1
    assert shared_cache.sets == 1


@pytest.mark.anyio
async def test_cache_handle_defaults_to_the_shared_scope() -> None:
    shared = DictCache()
    handle = CacheHandle(shared=shared)
    await handle.set("k", b"v")
    assert shared.store == {"k": b"v"}


@pytest.mark.anyio
async def test_one_scope_can_be_enabled_without_the_other() -> None:
    handle = CacheHandle(request=DictCache())
    assert handle.enabled("request") is True
    assert handle.enabled("shared") is False
    assert await handle.get("k", scope="shared") is None
