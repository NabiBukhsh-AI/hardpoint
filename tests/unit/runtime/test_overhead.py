"""Framework overhead stays under 15 ms p95 for a six-step pipeline (INSTRUCTIONS.md §12.6)."""

from __future__ import annotations

import statistics
import time
from typing import Any

import pytest

from hardpoint.observability.metrics import InMemoryMetricSink
from hardpoint.observability.tracing import CollectingTracer
from hardpoint.runtime import Pipeline, as_step
from hardpoint.testing import build_run_context

THRESHOLD_MS = 15.0


@pytest.mark.anyio
async def test_six_step_pipeline_overhead_is_under_15ms_p95() -> None:
    """Steps that do nothing, so every millisecond measured is the runtime's own.

    Measured with a collecting tracer and an in-memory metric sink -- the more
    expensive observability setup -- so the bound holds for the default too.
    """

    async def passthrough(data: Any, ctx: Any) -> Any:
        return data

    pipeline = Pipeline("overhead", [as_step(passthrough, name=f"step_{i}") for i in range(6)])
    timings: list[float] = []
    for _ in range(400):
        ctx = build_run_context(tracer=CollectingTracer(max_traces=1), metrics=InMemoryMetricSink())
        started = time.perf_counter()
        await pipeline("payload", ctx)
        timings.append((time.perf_counter() - started) * 1000.0)

    p95 = statistics.quantiles(timings, n=100)[94]
    assert p95 < THRESHOLD_MS, f"p95 overhead {p95:.2f} ms exceeds {THRESHOLD_MS} ms"
