"""The default metric sink: one that does nothing, correctly.

The ``MetricSink`` port lives in ``core.ports``. This module supplies the
implementation used when no metrics backend is configured, for the same reason
``NoOpTracer`` exists: a step must be able to increment a counter without first
checking whether anything is listening.

The metric taxonomy and the exporters are M2 (INSTRUCTIONS.md §7).
"""

from __future__ import annotations

__all__ = ["NoOpMetricSink"]


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
