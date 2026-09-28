"""Metric sinks and the metric vocabulary.

The ``MetricSink`` port lives in ``core.ports``. :class:`NoOpMetricSink` is the
default for the same reason ``NoOpTracer`` is: a step must be able to increment
a counter without first checking whether anything is listening.

:data:`METRIC_NAMES` is the vocabulary of ARCHITECTURE.md §21. Labels are low
cardinality on purpose -- a model id, a step name, a reason code -- never a
query, a run id or a document id, which would turn every backend's index into a
list of every request ever made.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "METRIC_NAMES",
    "InMemoryMetricSink",
    "InMemoryMetricsConfig",
    "NoOpMetricSink",
    "NoOpMetricsConfig",
    "build_memory",
    "build_noop",
]

METRIC_NAMES: Final = {
    "hardpoint.requests": "counter: pipeline runs, by pipeline and outcome",
    "hardpoint.step.latency_ms": "histogram: wall time per step, by step",
    "hardpoint.tokens": "counter: tokens, by model and direction (input|output|embed)",
    "hardpoint.cost_usd": "counter: priced spend, by model",
    "hardpoint.cost.unpriced_calls": "counter: provider calls with no price, by model",
    "hardpoint.errors": "counter: provider and step errors, by component and error type",
    "hardpoint.degradations": "counter: degradations, by step and reason",
    "hardpoint.retrieval.results": "histogram: results returned per retrieval, by retriever",
    "hardpoint.cache.lookups": "counter: cache lookups, by layer and result (hit|miss)",
}
"""Every metric the library emits, and what it measures."""


class NoOpMetricSink:
    """A ``MetricSink`` that discards every measurement.

    Stateless and safe to share between runs and between threads.
    """

    __slots__ = ()

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Discard a counter increment."""

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Discard a histogram observation."""

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Discard a gauge reading."""

    def __repr__(self) -> str:
        """Render the class name; there is no state to show."""
        return "NoOpMetricSink()"


Series = tuple[str, tuple[tuple[str, str], ...]]


class InMemoryMetricSink:
    """Aggregates measurements in process: totals, histogram samples, last gauges.

    Useful for a service's ``/metrics`` endpoint in development and for tests of
    anything that emits metrics. Thread-safe, because a sync facade or a worker
    thread may emit too. Histogram samples are capped per series.

    Args:
        max_samples: Samples kept per histogram series; older ones are dropped.
    """

    def __init__(self, *, max_samples: int = 10_000) -> None:
        self.max_samples = max_samples
        self.counters: dict[Series, float] = defaultdict(float)
        self.histograms: dict[Series, list[float]] = defaultdict(list)
        self.gauges: dict[Series, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _series(name: str, labels: dict[str, str]) -> Series:
        return name, tuple(sorted(labels.items()))

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        with self._lock:
            self.counters[self._series(name, labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram sample."""
        with self._lock:
            samples = self.histograms[self._series(name, labels)]
            samples.append(value)
            del samples[: max(0, len(samples) - self.max_samples)]

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge."""
        with self._lock:
            self.gauges[self._series(name, labels)] = value

    def total(self, name: str, **labels: str) -> float:
        """Sum a counter across every series whose labels include ``labels``."""
        wanted = set(labels.items())
        return sum(
            value
            for (series, series_labels), value in self.counters.items()
            if series == name and wanted <= set(series_labels)
        )

    def render(self) -> str:
        """Render in the Prometheus text exposition format."""
        lines: list[str] = []
        for (name, labels), value in sorted(self.counters.items()):
            lines.append(f"{_prom(name)}_total{_labels(labels)} {value}")
        for (name, labels), samples in sorted(self.histograms.items()):
            lines.append(f"{_prom(name)}_count{_labels(labels)} {len(samples)}")
            lines.append(f"{_prom(name)}_sum{_labels(labels)} {sum(samples)}")
        for (name, labels), value in sorted(self.gauges.items()):
            lines.append(f"{_prom(name)}{_labels(labels)} {value}")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        """Render how many series are held."""
        return f"InMemoryMetricSink(series={len(self.counters) + len(self.histograms)})"


def _prom(name: str) -> str:
    return name.replace(".", "_")


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    body = ",".join(
        f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in labels
    )
    return "{" + body + "}"


# --------------------------------------------------------------------------- #
# Registry entries                                                            #
# --------------------------------------------------------------------------- #


class NoOpMetricsConfig(BaseModel):
    """``observability.metrics: {type: noop}``. The default."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class InMemoryMetricsConfig(BaseModel):
    """``observability.metrics: {type: memory}``: aggregated in process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_samples: int = Field(default=10_000, ge=1)


def build_noop(config: NoOpMetricsConfig) -> NoOpMetricSink:
    """Registry factory for ``type: noop``."""
    return NoOpMetricSink()


def build_memory(config: InMemoryMetricsConfig) -> InMemoryMetricSink:
    """Registry factory for ``type: memory``."""
    return InMemoryMetricSink(max_samples=config.max_samples)
