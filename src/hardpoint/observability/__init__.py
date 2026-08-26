"""Tracer and MetricSink ports, span helpers, usage aggregation, pricing.

Owns the vocabulary for traces, metrics and cost. Does not own being a backend;
concrete exporters are adapters behind extras.

Populated in milestone M2.
"""
