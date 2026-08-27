"""Bounded fan-out with deadline propagation.

Implements INSTRUCTIONS.md §6.1. What ``ParallelRetrieve`` is built on: run
several operations concurrently, under one deadline, with a ceiling on how many
are in flight, and a choice about whether one failure kills the batch.

## Operations are factories, not coroutine objects

The specification writes ``gather_bounded(coros, limit, deadline)``. This takes
callables that *return* awaitables instead, because a coroutine object that is
created and then never awaited raises ``RuntimeWarning: coroutine was never
awaited`` -- and with ``filterwarnings = ["error"]`` that is a test failure
rather than a warning. Under a deadline, some operations legitimately never
start, so the situation is normal rather than exceptional.

A factory also means a retry can call the operation again, which a spent
coroutine cannot.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import anyio

from hardpoint.core.errors import ProviderTimeout

if TYPE_CHECKING:
    from hardpoint.core.context import Deadline

__all__ = ["Failure", "gather_bounded", "gather_tolerant"]

T = TypeVar("T")

Operation = Callable[[], Awaitable[T]]


@dataclass(frozen=True)
class Failure:
    """One operation that did not succeed.

    Args:
        index: Position in the input sequence, so a caller can say *which*
            retriever failed rather than only that one did.
        error: What it raised.
    """

    index: int
    error: BaseException

    def __repr__(self) -> str:
        """Render the index and the exception type."""
        return f"Failure(index={self.index}, error={type(self.error).__name__})"


def _first_exception(group: BaseException) -> BaseException:
    """Return the first leaf exception of a possibly nested exception group.

    anyio task groups raise a ``BaseExceptionGroup`` even for a single failure.
    Propagating the group would make every caller unwrap it, and would defeat
    ``except RateLimitedError`` in the policy layer, so the first real exception
    is re-raised instead.
    """
    if isinstance(group, BaseExceptionGroup) and group.exceptions:
        return _first_exception(group.exceptions[0])
    return group


async def gather_bounded(
    operations: Sequence[Operation[T]],
    *,
    limit: int = 8,
    deadline: Deadline | None = None,
) -> list[T]:
    """Run operations concurrently, in order, failing on the first error.

    Results come back in input order regardless of completion order, because a
    caller fusing retriever results needs to know which list came from which
    retriever.

    Args:
        operations: Callables returning awaitables.
        limit: Maximum in flight at once. Bounds pressure on a provider that
            would otherwise rate-limit the whole batch.
        deadline: When set, the batch is cancelled at that time.

    Returns:
        One result per operation, in input order.

    Raises:
        BaseException: The first exception any operation raised, with its
            siblings cancelled. Not the ``ExceptionGroup`` anyio produces.
        ProviderTimeout: If the deadline cut the batch short. Returning the
            partial list would be silent truncation, and a missing retriever
            result is indistinguishable from an empty one.
        ValueError: If ``limit`` is below 1.
    """
    results, failures = await _run(operations, limit=limit, deadline=deadline, tolerate=False)
    if failures:  # pragma: no cover - _run re-raises when tolerate is False
        raise failures[0].error
    return [result for _, result in sorted(results.items())]


async def gather_tolerant(
    operations: Sequence[Operation[T]],
    *,
    limit: int = 8,
    deadline: Deadline | None = None,
) -> tuple[list[T], list[Failure]]:
    """Run operations concurrently, tolerating partial failure.

    What ``ParallelRetrieve`` uses: one retriever timing out should degrade the
    result set, not fail the request (ARCHITECTURE.md §18.2). The caller decides
    what the failures mean, and records a ``Degradation`` for each.

    Args:
        operations: Callables returning awaitables.
        limit: Maximum in flight at once.
        deadline: When set, unfinished operations are cancelled at that time and
            reported as failures.

    Returns:
        ``(results, failures)``. Results are in input order and contain only the
        operations that succeeded; each failure carries the index that produced
        it, so the two can be reconciled.

    Raises:
        ValueError: If ``limit`` is below 1.
    """
    results, failures = await _run(operations, limit=limit, deadline=deadline, tolerate=True)
    return [result for _, result in sorted(results.items())], failures


async def _run(
    operations: Sequence[Operation[T]],
    *,
    limit: int,
    deadline: Deadline | None,
    tolerate: bool,
) -> tuple[dict[int, T], list[Failure]]:
    """Run the batch, collecting results by index and failures in order."""
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if not operations:
        return {}, []

    results: dict[int, T] = {}
    failures: list[Failure] = []
    semaphore = anyio.Semaphore(min(limit, len(operations)))

    async def run_one(index: int, operation: Operation[T]) -> None:
        async with semaphore:
            if tolerate:
                try:
                    results[index] = await operation()
                except Exception as exc:  # tolerate_partial: the caller decides what it means
                    failures.append(Failure(index=index, error=exc))
            else:
                results[index] = await operation()

    cancel_at = deadline.as_anyio_deadline() if deadline is not None else float("inf")

    try:
        with anyio.CancelScope(deadline=cancel_at) as scope:
            async with anyio.create_task_group() as group:
                for index, operation in enumerate(operations):
                    group.start_soon(run_one, index, operation)
    except BaseExceptionGroup as group_error:
        raise _first_exception(group_error) from None

    if scope.cancelled_caught and not tolerate:
        # Without this the batch returns a short list and the caller has no way
        # to know it is short: silent truncation, in the one place where a
        # missing retriever result looks exactly like an empty one.
        raise ProviderTimeout(
            f"The batch was cancelled by the run deadline after "
            f"{len(results)} of {len(operations)} operations completed.",
            remedy=(
                "Raise `budgets.request.deadline_s`, reduce the number of parallel "
                "operations, or use gather_tolerant if a partial result set is "
                "acceptable and should be reported as a degradation."
            ),
        )

    if scope.cancelled_caught:
        # Operations the deadline cut short have neither a result nor a recorded
        # failure. Reporting them keeps `len(results) + len(failures)` equal to
        # the number of operations, which is what lets a caller tell "returned
        # nothing" apart from "never ran".
        accounted = set(results) | {failure.index for failure in failures}
        failures.extend(
            Failure(index=index, error=TimeoutError("cancelled by the run deadline"))
            for index in range(len(operations))
            if index not in accounted
        )
        failures.sort(key=lambda failure: failure.index)

    return results, failures
