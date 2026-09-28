"""Tracing, redaction, metrics, pricing and the OpenTelemetry adapter (INSTRUCTIONS.md §7)."""

from __future__ import annotations

import io
import warnings
from typing import Any

import anyio
import pytest

from hardpoint.adapters.tracing.otel import OTelConfig, OTelTracer, build_metrics, otel_value
from hardpoint.core.config.schema import ModelPriceConfig
from hardpoint.core.models import StepUsage
from hardpoint.observability.metrics import InMemoryMetricSink
from hardpoint.observability.pricing import PricingTable, UnpricedModelWarning
from hardpoint.observability.tracing import (
    REDACTED_VALUE,
    CollectingTracer,
    ConsoleTracer,
    RedactingTracer,
)

# --------------------------------------------------------------------------- #
# Tracers                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_concurrent_children_nest_under_their_parent() -> None:
    """Two retrievers running at once must both be children of the pipeline span."""
    tracer = CollectingTracer()

    async def retriever(name: str) -> None:
        async with tracer.span("hardpoint.step", **{"step.name": name}):
            await anyio.sleep(0.01)
            async with tracer.span("hardpoint.index.query"):
                await anyio.sleep(0.01)

    async with tracer.span("hardpoint.pipeline"), anyio.create_task_group() as group:
        group.start_soon(retriever, "dense")
        group.start_soon(retriever, "sparse")

    (root,) = tracer.roots
    assert sorted(child.attributes["step.name"] for child in root.children) == ["dense", "sparse"]
    for child in root.children:
        assert [grandchild.name for grandchild in child.children] == ["hardpoint.index.query"]
        assert child.trace == root.trace
    assert all(span.duration_ms is not None for span in tracer.spans())


@pytest.mark.anyio
async def test_a_failing_span_records_the_error_and_reraises() -> None:
    tracer = CollectingTracer()
    with pytest.raises(ValueError, match="boom"):
        async with tracer.span("hardpoint.step"):
            raise ValueError("boom")
    (span,) = tracer.spans()
    assert span.status == ("error", "ValueError")
    assert span.exceptions == ["ValueError: boom"]


@pytest.mark.anyio
async def test_the_collecting_tracer_is_bounded() -> None:
    tracer = CollectingTracer(max_traces=3)
    for _ in range(10):
        async with tracer.span("hardpoint.pipeline"):
            pass
    assert len(tracer.roots) == 3


@pytest.mark.anyio
async def test_the_console_tracer_prints_whole_trees() -> None:
    stream = io.StringIO()
    tracer = ConsoleTracer(stream)
    async with (
        tracer.span("hardpoint.pipeline"),
        tracer.span("hardpoint.step", **{"step.name": "retrieve"}),
    ):
        pass
    output = stream.getvalue()
    assert "hardpoint.pipeline" in output
    assert "  hardpoint.step retrieve" in output, "children are indented under their parent"


@pytest.mark.anyio
async def test_redaction_applies_to_attributes_events_and_everything_under_a_path() -> None:
    """ARCHITECTURE.md §19: traces may hold user content, redacted by config path."""
    inner = CollectingTracer()
    tracer = RedactingTracer(inner, ["messages.content", "chunk.text"])

    async with tracer.span("hardpoint.llm", **{"messages.content": "secret question"}) as span:
        span.set_attribute("messages.content.0", "nested secret")
        span.set_attribute("gen_ai.request.model", "openai/gpt-4o")
        span.add_event("retrieved", {"chunk.text": "private passage", "count": 3})
        assert span.trace_id == inner.roots[0].trace

    (recorded,) = inner.spans()
    assert recorded.attributes["messages.content"] == REDACTED_VALUE
    assert recorded.attributes["messages.content.0"] == REDACTED_VALUE
    assert recorded.attributes["gen_ai.request.model"] == "openai/gpt-4o"
    assert recorded.events == [("retrieved", {"chunk.text": REDACTED_VALUE, "count": 3})]


@pytest.mark.anyio
async def test_redaction_does_not_touch_a_key_that_merely_shares_a_prefix() -> None:
    inner = CollectingTracer()
    async with RedactingTracer(inner, ["chunk.text"]).span("s", **{"chunk.textual": "keep"}):
        pass
    assert inner.spans()[0].attributes["chunk.textual"] == "keep"


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def test_the_in_memory_sink_aggregates_and_renders_prometheus_text() -> None:
    sink = InMemoryMetricSink()
    sink.increment("hardpoint.tokens", 10, model="m", direction="input")
    sink.increment("hardpoint.tokens", 5, model="m", direction="input")
    sink.increment("hardpoint.tokens", 7, model="m", direction="output")
    sink.observe("hardpoint.step.latency_ms", 12.5, step="retrieve")
    sink.gauge("hardpoint.queue", 3)

    assert sink.total("hardpoint.tokens") == 22
    assert sink.total("hardpoint.tokens", direction="input") == 15
    text = sink.render()
    assert 'hardpoint_tokens_total{direction="input",model="m"} 15.0' in text
    assert 'hardpoint_step_latency_ms_count{step="retrieve"} 1' in text
    assert "hardpoint_queue 3" in text


def test_label_values_are_escaped() -> None:
    sink = InMemoryMetricSink()
    sink.increment("x", model='quote"back\\slash')
    assert 'model="quote\\"back\\\\slash"' in sink.render()


# --------------------------------------------------------------------------- #
# Pricing                                                                     #
# --------------------------------------------------------------------------- #


def test_a_known_model_is_priced_per_million_tokens() -> None:
    table = PricingTable({"acme/model": ModelPriceConfig(input=1.0, output=2.0)})
    cost = table.cost("acme/model", StepUsage(calls=1, prompt_tokens=1000, completion_tokens=500))
    assert cost == pytest.approx(0.002)


def test_an_unknown_model_is_unpriced_never_zero_and_warns_once() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §7 and §13.8."""
    table = PricingTable()
    usage = StepUsage(calls=1, prompt_tokens=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert table.cost("acme/unknown", usage) is None
        assert table.cost("acme/unknown", usage) is None
    unpriced = [w for w in caught if issubclass(w.category, UnpricedModelWarning)]
    assert len(unpriced) == 1, "one warning per model, not one per call"
    assert "pricing:" in str(unpriced[0].message)


def test_the_shipped_table_is_loaded_and_overridable() -> None:
    shipped = PricingTable()
    assert shipped.price("openai/text-embedding-3-small") is not None
    assert shipped.embed_price_per_million("openai/text-embedding-3-small") == pytest.approx(0.02)

    overridden = PricingTable({"openai/text-embedding-3-small": ModelPriceConfig(embed=9.0)})
    assert overridden.embed_price_per_million("openai/text-embedding-3-small") == 9.0


def test_fakes_are_free_and_flat_per_call_prices_apply() -> None:
    table = PricingTable({"acme/rerank": ModelPriceConfig(per_call=0.002)})
    assert table.cost("fake/anything", StepUsage(calls=3, prompt_tokens=10)) == 0.0
    assert table.cost("acme/rerank", StepUsage(calls=2)) == pytest.approx(0.004)
    assert table.embed_price_per_million("acme/unknown") is None


# --------------------------------------------------------------------------- #
# OpenTelemetry                                                               #
# --------------------------------------------------------------------------- #


def test_otel_values_are_primitives_or_json() -> None:
    assert otel_value("a") == "a"
    assert otel_value(3) == 3
    assert otel_value(["a", "b"]) == ["a", "b"]
    assert otel_value([1, 2.5]) == [1, 2.5]
    assert otel_value({"k": 1}) == '{"k": 1}'
    assert otel_value(None) == "null"
    assert otel_value([1, "a"]) == '[1, "a"]'


@pytest.mark.anyio
async def test_the_otel_tracer_nests_and_converts_attributes() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider.get_tracer("test"), provider)

    async def child(name: str) -> None:
        async with tracer.span("hardpoint.step", **{"step.name": name}) as span:
            span.set_attribute("hardpoint.meta", {"nested": True})
            span.add_event("retry", {"attempt": 1})

    async with tracer.span("hardpoint.pipeline") as root:
        trace_id = root.trace_id
        async with anyio.create_task_group() as group:
            group.start_soon(child, "a")
            group.start_soon(child, "b")
        root.set_status("ok")
    await tracer.aclose()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    pipeline = spans["hardpoint.pipeline"]
    steps = [s for s in exporter.get_finished_spans() if s.name == "hardpoint.step"]
    assert len(steps) == 2
    for step in steps:
        assert step.parent is not None
        assert step.parent.span_id == pipeline.context.span_id, "nested across tasks"
        assert step.attributes["hardpoint.meta"] == '{"nested": true}'
    assert trace_id == f"{pipeline.context.trace_id:032x}"


@pytest.mark.anyio
async def test_the_otel_tracer_marks_a_failed_span() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider.get_tracer("test"))

    async def fail() -> None:
        async with tracer.span("hardpoint.llm") as span:
            span.set_status("error", "boom")
            span.record_exception(RuntimeError("boom"))
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await fail()
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code is StatusCode.ERROR


def test_the_otel_metric_sink_records_through_a_reader() -> None:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from hardpoint.adapters.tracing.otel import OTelMetricSink

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    sink = OTelMetricSink(provider.get_meter("test"))
    sink.increment("hardpoint.tokens", 5, model="m")
    sink.increment("hardpoint.tokens", 2, model="m")
    sink.observe("hardpoint.step.latency_ms", 3.0, step="s")
    sink.gauge("hardpoint.queue", 4)

    data: Any = reader.get_metrics_data()
    metrics = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert metrics["hardpoint.tokens"].data.data_points[0].value == 7
    assert "hardpoint.step.latency_ms" in metrics
    assert "hardpoint.queue" in metrics
    provider.shutdown()


@pytest.mark.anyio
async def test_the_otel_factories_build_without_exporters() -> None:
    from hardpoint.adapters.tracing.otel import build_tracer

    tracer = build_tracer(OTelConfig(service_name="t"))
    async with tracer.span("hardpoint.pipeline") as span:
        assert span.trace_id is not None
    await tracer.aclose()
    sink = build_metrics(OTelConfig(console=False))
    sink.increment("x")
    await sink.aclose()
