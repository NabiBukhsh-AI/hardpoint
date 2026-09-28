"""OpenTelemetry tracing and metrics, behind the ``otel`` extra.

ADR-010: hardpoint owns a small ``Tracer`` port and this adapter maps it onto
OpenTelemetry, so OTel's semantic-convention churn is absorbed here rather than
in every step. Span names come from the taxonomy in ``observability.tracing``;
attributes follow the GenAI conventions (``gen_ai.*``) where they exist.

Nesting across async boundaries is OpenTelemetry's own context propagation,
which is built on ``contextvars`` and so follows anyio and asyncio tasks.

The SDK is imported inside the factories, never at module import, so this module
imports in a bare environment (INSTRUCTIONS.md §3 **[LOCKED]**) and the registry
reports a missing extra from its ``requires`` entry before anything is built.

Nothing here touches global OpenTelemetry state unless ``set_global`` asks:
a library that installed a global provider behind an application's back would
be the tracing equivalent of calling ``logging.basicConfig``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hardpoint.core.errors import MissingDependencyError
from hardpoint.core.types import JsonValue

__all__ = [
    "OTelConfig",
    "OTelMetricSink",
    "OTelSpan",
    "OTelTracer",
    "build_metrics",
    "build_tracer",
    "otel_value",
]

_PRIMITIVES = (str, bool, int, float)


def otel_value(value: JsonValue) -> Any:
    """Convert a JSON value into something an OpenTelemetry attribute accepts.

    OTel accepts primitives and homogeneous sequences of primitives. Mappings,
    mixed lists and ``None`` are serialised to JSON text rather than dropped:
    an attribute that silently vanished would be worse than one that reads as
    JSON.
    """
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return list(value)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    ):
        return list(value)
    return json.dumps(value, sort_keys=True, default=str)


class OTelSpan:
    """The ``Span`` port over an OpenTelemetry span."""

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Set an attribute, converting the value to one OTel accepts."""
        self._span.set_attribute(key, otel_value(value))

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Add an event."""
        self._span.add_event(name, {k: otel_value(v) for k, v in (attributes or {}).items()})

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception."""
        self._span.record_exception(exc)

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Set the status."""
        from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415 - the extra

        code = StatusCode.OK if status == "ok" else StatusCode.ERROR
        self._span.set_status(Status(code, description or None))

    @property
    def trace_id(self) -> str | None:
        """The W3C trace id, as 32 hex characters, for log correlation."""
        context = self._span.get_span_context()
        return f"{context.trace_id:032x}" if context.trace_id else None


class OTelTracer:
    """The ``Tracer`` port over an OpenTelemetry tracer.

    Args:
        tracer: An ``opentelemetry.trace.Tracer``.
        provider: The provider that owns it, flushed and shut down by
            :meth:`aclose` so buffered spans are not lost on exit.
    """

    def __init__(self, tracer: Any, provider: Any | None = None) -> None:
        self._tracer = tracer
        self._provider = provider

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[OTelSpan]:
        """Open a span as the current span, nested under whatever is current."""
        attributes = {key: otel_value(value) for key, value in attrs.items()}

        @asynccontextmanager
        async def scope() -> AsyncIterator[OTelSpan]:
            with self._tracer.start_as_current_span(
                name, attributes=attributes, record_exception=True, set_status_on_exception=True
            ) as span:
                yield OTelSpan(span)

        return scope()

    async def aclose(self) -> None:
        """Flush and shut down the provider this tracer owns."""
        if self._provider is not None:
            self._provider.force_flush()
            self._provider.shutdown()

    def __repr__(self) -> str:
        """Render the class name."""
        return "OTelTracer()"


class OTelMetricSink:
    """The ``MetricSink`` port over an OpenTelemetry meter.

    Instruments are created on first use and kept, because creating one per
    measurement is expensive and, in some SDK versions, warns.
    """

    def __init__(self, meter: Any, provider: Any | None = None) -> None:
        self._meter = meter
        self._provider = provider
        self._instruments: dict[tuple[str, str], Any] = {}

    def _instrument(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self._instruments:
            create = {
                "counter": self._meter.create_counter,
                "histogram": self._meter.create_histogram,
                "gauge": self._meter.create_gauge,
            }[kind]
            self._instruments[key] = create(name)
        return self._instruments[key]

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        self._instrument("counter", name).add(value, attributes=labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation."""
        self._instrument("histogram", name).record(value, attributes=labels)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge."""
        self._instrument("gauge", name).set(value, attributes=labels)

    async def aclose(self) -> None:
        """Flush and shut down the provider this sink owns."""
        if self._provider is not None:
            self._provider.force_flush()
            self._provider.shutdown()

    def __repr__(self) -> str:
        """Render how many instruments exist."""
        return f"OTelMetricSink(instruments={len(self._instruments)})"


class OTelConfig(BaseModel):
    """``observability.tracer`` / ``observability.metrics``: ``{type: otel}``.

    Args:
        service_name: The ``service.name`` resource attribute.
        endpoint: OTLP/HTTP collector endpoint, for example
            ``http://localhost:4318``. ``None`` with ``console: false`` builds
            a provider with no exporter, for an application that attaches its
            own.
        console: Export to standard output instead, for local debugging.
        set_global: Install the provider as OpenTelemetry's global. Off by
            default: a library changing global state behind an application's
            back is the tracing equivalent of configuring the root logger.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_name: str = "hardpoint"
    endpoint: str | None = None
    console: bool = False
    set_global: bool = False


def _missing(exc: ModuleNotFoundError) -> MissingDependencyError:
    return MissingDependencyError(
        "OpenTelemetry support requires the 'otel' extra, which is not installed.",
        extra="otel",
        component="otel",
        config_path="observability",
        remedy="pip install 'hardpoint[otel]'",
        cause=exc,
    )


def build_tracer(config: OTelConfig) -> OTelTracer:
    """Registry factory for ``observability.tracer: {type: otel}``.

    Raises:
        MissingDependencyError: If the ``otel`` extra is not installed.
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415 - optional, lazily loaded
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ModuleNotFoundError as exc:
        raise _missing(exc) from exc

    provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
    if config.endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        endpoint = config.endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    elif config.console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if config.set_global:
        trace.set_tracer_provider(provider)
    return OTelTracer(provider.get_tracer("hardpoint"), provider)


def build_metrics(config: OTelConfig) -> OTelMetricSink:
    """Registry factory for ``observability.metrics: {type: otel}``.

    Raises:
        MissingDependencyError: If the ``otel`` extra is not installed.
    """
    try:
        from opentelemetry import metrics  # noqa: PLC0415 - optional, lazily loaded
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
        from opentelemetry.sdk.metrics.export import (  # noqa: PLC0415
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise _missing(exc) from exc

    readers: list[Any] = []
    if config.endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
            OTLPMetricExporter,
        )

        endpoint = config.endpoint.rstrip("/") + "/v1/metrics"
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint)))
    elif config.console:
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    provider = MeterProvider(
        resource=Resource.create({"service.name": config.service_name}), metric_readers=readers
    )
    if config.set_global:
        metrics.set_meter_provider(provider)
    return OTelMetricSink(provider.get_meter("hardpoint"), provider)
