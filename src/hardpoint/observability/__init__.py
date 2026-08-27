"""Tracer and MetricSink implementations, usage aggregation and pricing.

Owns the vocabulary for traces, metrics and cost. Does not own being a backend;
concrete exporters are adapters behind extras.

M1 ships the no-op implementations only, because a ``RunContext`` cannot be
constructed without a tracer and a metric sink and ``hardpoint.testing`` must
never appear on a production path. The span taxonomy, attribute redaction,
pricing and the OpenTelemetry adapter are M2.
"""

from __future__ import annotations

from hardpoint.observability.metrics import NoOpMetricSink
from hardpoint.observability.tracing import NoOpSpan, NoOpTracer

__all__ = ["NoOpMetricSink", "NoOpSpan", "NoOpTracer"]
