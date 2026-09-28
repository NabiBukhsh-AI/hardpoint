"""Measure the runtime's own overhead: a six-step pipeline whose steps do nothing.

INSTRUCTIONS.md §12.6 requires framework overhead, excluding provider time, to
stay under 15 ms p95 for a six-step pipeline against fakes. The steps here
return immediately, so what is measured is exactly what the pipeline adds per
step: the deadline and budget checks, the span, latency and usage recording,
metrics, and degradation collection.

    uv run python scripts/benchmark_overhead.py [--runs 2000] [--tracer collecting]

The same measurement is asserted in ``tests/unit/runtime/test_overhead.py``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import anyio

from hardpoint.observability.metrics import InMemoryMetricSink
from hardpoint.observability.tracing import CollectingTracer, NoOpTracer
from hardpoint.runtime import Pipeline, as_step
from hardpoint.testing import build_run_context

THRESHOLD_MS = 15.0


def six_step_pipeline() -> Pipeline[Any, Any]:
    """Six steps that do no work of their own."""

    async def passthrough(data: Any, ctx: Any) -> Any:  # noqa: ARG001 - the Step signature
        return data

    return Pipeline("overhead", [as_step(passthrough, name=f"step_{i}") for i in range(6)])


async def measure(runs: int, *, collecting: bool) -> list[float]:
    """Return per-run wall times in milliseconds."""
    pipeline = six_step_pipeline()
    timings: list[float] = []
    for _ in range(runs):
        tracer = CollectingTracer(max_traces=1) if collecting else NoOpTracer()
        ctx = build_run_context(tracer=tracer, metrics=InMemoryMetricSink())
        started = time.perf_counter()
        await pipeline("payload", ctx)
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def p95(timings: list[float]) -> float:
    """The 95th percentile."""
    return statistics.quantiles(timings, n=100)[94]


def main() -> None:
    """Print the distribution and whether it is under the threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--tracer", choices=["noop", "collecting"], default="collecting")
    args = parser.parse_args()

    timings = anyio.run(lambda: measure(args.runs, collecting=args.tracer == "collecting"))
    value = p95(timings)
    print(  # noqa: T201 - a script's output
        f"six-step pipeline, {args.runs} runs, {args.tracer} tracer: "
        f"median {statistics.median(timings):.3f} ms, p95 {value:.3f} ms, "
        f"max {max(timings):.3f} ms -> {'OK' if value < THRESHOLD_MS else 'OVER'} "
        f"(threshold {THRESHOLD_MS} ms)"
    )


if __name__ == "__main__":
    main()
