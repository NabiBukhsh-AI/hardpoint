"""The ``Step`` contract and bounded fan-out. INSTRUCTIONS.md §6.1, ARCHITECTURE.md §9.1.

For steps, the properties that matter are that a five-line class satisfies the
contract and that reporting a degradation is opt-in, so the simple case never
sees ``StepResult``.

For concurrency, the properties are the ones ``ParallelRetrieve`` will depend
on: results in input order regardless of completion order, a real ceiling on
concurrency, siblings cancelled on the first failure, and -- under
``tolerate_partial`` -- every operation accounted for as either a result or a
failure, including the ones a deadline cut short before they started.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from hardpoint.core.context import Deadline
from hardpoint.core.errors import ProviderTimeout, TransientError
from hardpoint.core.models import Degradation
from hardpoint.runtime import (
    FailurePolicy,
    StepResult,
    as_step,
    gather_bounded,
    gather_tolerant,
    step_type_name,
    unwrap,
)
from hardpoint.runtime.step import PipelineInfo, Step, StepInfo
from hardpoint.testing import build_run_context

# --------------------------------------------------------------------------- #
# Step                                                                        #
# --------------------------------------------------------------------------- #


def test_a_five_line_class_is_a_step() -> None:
    """No base class, no registration, no import of hardpoint into its module."""

    class Normalise:
        name = "normalise"

        async def __call__(self, data: str, ctx: Any) -> str:
            return data.strip().lower()

    step: Step[str, str] = Normalise()
    assert step.name == "normalise"


@pytest.mark.anyio
async def test_a_plain_async_function_becomes_a_step() -> None:
    """A function has ``__name__``, not ``name``, so it needs the adapter.

    Worth having, because "can it be a function instead of a class?" should
    usually be answered yes (INSTRUCTIONS.md §17.3).
    """

    async def normalise(data: str, ctx: Any) -> str:
        return data.strip().lower()

    step = as_step(normalise)
    assert step.name == "normalise"
    assert await step("  Hello  ", build_run_context()) == "hello"


@pytest.mark.anyio
async def test_a_function_step_can_be_renamed() -> None:
    async def transform(data: str, ctx: Any) -> str:
        return data

    assert as_step(transform, name="understand").name == "understand"


def test_a_callable_without_a_name_still_gets_one() -> None:
    """A step with no name would break usage attribution and span labelling."""

    class Callable:
        async def __call__(self, data: Any, ctx: Any) -> Any:
            return data

    assert as_step(Callable()).name == "step"  # type: ignore[arg-type]


def test_repr_names_the_wrapped_function() -> None:
    async def normalise(data: str, ctx: Any) -> str:
        return data

    assert "normalise" in repr(as_step(normalise))


def test_unwrapping_a_bare_value_yields_no_degradations() -> None:
    """A step that never degrades never has to know StepResult exists."""
    value, degradations = unwrap(["a", "b"])
    assert value == ["a", "b"]
    assert degradations == ()


def test_unwrapping_a_step_result_splits_it() -> None:
    degradation = Degradation(step="rerank", reason="rerank_skipped")
    value, degradations = unwrap(StepResult(value="x", degradations=(degradation,)))
    assert value == "x"
    assert degradations == (degradation,)


def test_step_result_repr_hides_the_payload() -> None:
    """A repr in a log must not print an entire context bundle."""
    rendered = repr(StepResult(value=["a"] * 1000, degradations=()))
    assert "degradations=0" in rendered
    assert "aaa" not in rendered


def test_step_type_name_reports_the_implementation_class() -> None:
    """Distinct from the step's name: two RerankSteps share a type."""

    class RerankStep:
        name = "rerank_primary"

        async def __call__(self, data: Any, ctx: Any) -> Any:
            return data

    assert step_type_name(RerankStep()) == "RerankStep"


def test_failure_policy_defaults_to_fail() -> None:
    assert StepInfo(name="a", type="A").on_failure is FailurePolicy.FAIL


def test_pipeline_info_renders_an_empty_pipeline() -> None:
    assert PipelineInfo(name="empty").render() == "Pipeline 'empty' (0 steps)"


# --------------------------------------------------------------------------- #
# gather_bounded                                                              #
# --------------------------------------------------------------------------- #


def slow_value(value: int, delay: float) -> Any:
    """Return an operation that resolves to ``value`` after ``delay``."""

    async def operation() -> int:
        await anyio.sleep(delay)
        return value

    return operation


@pytest.mark.anyio
async def test_results_come_back_in_input_order_not_completion_order() -> None:
    """A caller fusing retriever results needs to know which list is which."""
    operations = [slow_value(0, 0.03), slow_value(1, 0.01), slow_value(2, 0.02)]
    assert await gather_bounded(operations, limit=3) == [0, 1, 2]


@pytest.mark.anyio
async def test_an_empty_batch_is_not_an_error() -> None:
    assert await gather_bounded([], limit=4) == []
    assert await gather_tolerant([], limit=4) == ([], [])


@pytest.mark.anyio
async def test_concurrency_is_actually_bounded() -> None:
    """The ceiling exists to keep a batch from being rate-limited wholesale."""
    in_flight = {"now": 0, "peak": 0}

    def tracked() -> Any:
        async def operation() -> int:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            await anyio.sleep(0.01)
            in_flight["now"] -= 1
            return 1

        return operation

    await gather_bounded([tracked() for _ in range(10)], limit=3)
    assert in_flight["peak"] <= 3


@pytest.mark.anyio
async def test_the_first_failure_propagates_and_is_not_an_exception_group() -> None:
    """anyio raises a group; propagating it would defeat ``except RateLimitedError``."""

    async def boom() -> int:
        raise TransientError("upstream 503")

    with pytest.raises(TransientError):
        await gather_bounded([boom, slow_value(1, 0.01)], limit=2)


@pytest.mark.anyio
async def test_a_failure_cancels_its_siblings() -> None:
    """Otherwise a failed batch keeps spending on work whose result is discarded."""
    finished: list[int] = []

    async def boom() -> int:
        await anyio.sleep(0.01)
        raise TransientError("upstream 503")

    def long_running(index: int) -> Any:
        async def operation() -> int:
            await anyio.sleep(5.0)
            finished.append(index)
            return index

        return operation

    with pytest.raises(TransientError):
        await gather_bounded([boom, long_running(1), long_running(2)], limit=3)

    assert finished == [], "siblings must have been cancelled, not left running"


@pytest.mark.anyio
async def test_a_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        await gather_bounded([slow_value(1, 0)], limit=0)


@pytest.mark.anyio
async def test_a_deadline_that_cuts_the_batch_short_raises_rather_than_truncating() -> None:
    """Returning a short list would be silent truncation.

    The caller receives one list per retriever and fuses them. A batch that
    quietly returned two lists where three were asked for would look exactly
    like a retriever that found nothing, which is a very different fact.
    """
    operations = [slow_value(0, 0.001), slow_value(1, 5.0)]

    with pytest.raises(ProviderTimeout) as exc_info:
        await gather_bounded(operations, limit=2, deadline=Deadline.in_seconds(0.05))

    rendered = str(exc_info.value)
    assert "of 2 operations" in rendered
    assert exc_info.value.remedy is not None
    assert "gather_tolerant" in exc_info.value.remedy


@pytest.mark.anyio
async def test_a_batch_that_finishes_inside_its_deadline_is_unaffected() -> None:
    results = await gather_bounded(
        [slow_value(0, 0.001), slow_value(1, 0.001)], limit=2, deadline=Deadline.in_seconds(5)
    )
    assert results == [0, 1]


# --------------------------------------------------------------------------- #
# gather_tolerant                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_partial_failure_yields_results_and_failures() -> None:
    """One retriever failing degrades the result set; it does not fail the request."""

    async def boom() -> int:
        raise TransientError("upstream 503")

    results, failures = await gather_tolerant(
        [slow_value(0, 0.01), boom, slow_value(2, 0.01)], limit=3
    )

    assert results == [0, 2]
    assert [failure.index for failure in failures] == [1]
    assert isinstance(failures[0].error, TransientError)


@pytest.mark.anyio
async def test_a_failure_carries_the_index_that_produced_it() -> None:
    """So a caller can say *which* retriever failed, not only that one did."""

    async def boom() -> int:
        raise TransientError("down")

    _, failures = await gather_tolerant([slow_value(0, 0), boom, boom], limit=3)
    assert [failure.index for failure in failures] == [1, 2]
    assert "TransientError" in repr(failures[0])


@pytest.mark.anyio
async def test_every_operation_is_accounted_for_when_a_deadline_cuts_the_batch_short() -> None:
    """Results plus failures must equal the number of operations.

    Without this, a caller cannot tell "that retriever returned nothing" from
    "that retriever never ran", and the second is a much more serious fact.
    """
    operations = [slow_value(0, 0.001), slow_value(1, 5.0), slow_value(2, 5.0)]
    results, failures = await gather_tolerant(
        operations, limit=3, deadline=Deadline.in_seconds(0.05)
    )

    assert len(results) + len(failures) == len(operations)
    assert [failure.index for failure in failures] == [1, 2]
    assert all(isinstance(failure.error, TimeoutError) for failure in failures)


@pytest.mark.anyio
async def test_a_tolerant_batch_that_all_fails_returns_no_results() -> None:
    async def boom() -> int:
        raise TransientError("down")

    results, failures = await gather_tolerant([boom, boom], limit=2)
    assert results == []
    assert len(failures) == 2
