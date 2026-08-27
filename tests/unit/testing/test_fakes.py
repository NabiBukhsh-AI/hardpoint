"""The fakes. INSTRUCTIONS.md §5.8.

Fakes are test infrastructure, so a bug in one is a bug in every test that uses
it, silently. The properties asserted here are the ones the rest of the suite
will lean on:

- **Determinism.** Same input, same output, in a different process. Without it
  an eval baseline is meaningless and a flaky test is indistinguishable from a
  real regression.
- **Call counting.** The six ingestion acceptance tests are all of the form
  "assert zero embeddings happened". They are only checkable because the fake
  counts.
- **Asymmetric embeddings.** Passing the wrong ``EmbedKind`` silently degrades
  recall in production. A symmetric fake would make that bug invisible in tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from hardpoint.core.errors import AuthError, ConfigError, ContractError, TransientError
from hardpoint.core.filters import F
from hardpoint.core.models import CharSpan, Chunk, RetrievedChunk
from hardpoint.core.ports import (
    GenerationRequest,
    IndexRecord,
    IndexSpec,
    Message,
    ToolCall,
    VectorQuery,
)
from hardpoint.testing import (
    FakeCache,
    FakeEmbeddingModel,
    FakeLanguageModel,
    FakeReranker,
    InMemoryVectorIndex,
    RecordingMetricSink,
    RecordingTracer,
    ScriptedResponse,
    build_chunk,
    build_chunks,
    build_document,
    build_retrieved,
    build_run_context,
    deterministic_vector,
)


def ask(text: str = "what is the refund policy?") -> GenerationRequest:
    return GenerationRequest(messages=(Message(role="user", content=text),))


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #


def test_vectors_are_unit_length_and_stable() -> None:
    first = deterministic_vector("hello world", 8)
    second = deterministic_vector("hello world", 8)
    assert first == second
    assert len(first) == 8
    assert sum(v * v for v in first) == pytest.approx(1.0)


def test_vectors_ignore_cosmetic_whitespace() -> None:
    """Matching ``ids.normalise_text``, so a fake and a real id agree on sameness."""
    assert deterministic_vector("hello world", 8) == deterministic_vector("  hello   world ", 8)


def test_different_texts_give_different_vectors() -> None:
    assert deterministic_vector("alpha", 8) != deterministic_vector("beta", 8)


def test_fakes_are_deterministic_across_processes() -> None:
    """A fresh interpreter must produce identical values.

    Hash randomisation differs per process, so anything built on ``hash()`` or
    on an unseeded RNG would drift here. An eval baseline recorded on one
    machine has to replay on another.
    """
    program = textwrap.dedent(
        """
        import json
        from hardpoint.testing import deterministic_vector
        print(json.dumps(deterministic_vector("stable input", 4)))
        """
    )
    outputs = set()
    for seed in ("0", "1", "9999"):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "", "SYSTEMROOT": ""},
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"vectors differed across hash seeds: {outputs}"


# --------------------------------------------------------------------------- #
# FakeLanguageModel                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_deterministic_mode_echoes_a_hash_of_the_prompt() -> None:
    model = FakeLanguageModel()
    ctx = build_run_context()

    first = await model.generate(ask(), ctx)
    second = await model.generate(ask(), ctx)
    other = await model.generate(ask("something else"), ctx)

    assert first.text == second.text
    assert first.text != other.text


@pytest.mark.anyio
async def test_scripted_responses_match_on_the_rendered_prompt() -> None:
    """So a test can make the model say a specific thing about a specific context."""
    model = FakeLanguageModel(
        [
            ScriptedResponse("Refunds take 30 days.", match="refund"),
            ScriptedResponse("I don't know."),
        ]
    )
    ctx = build_run_context()

    assert (await model.generate(ask("refund policy?"), ctx)).text == "Refunds take 30 days."
    assert (await model.generate(ask("shipping?"), ctx)).text == "I don't know."


@pytest.mark.anyio
async def test_the_first_matching_script_wins() -> None:
    model = FakeLanguageModel(
        [ScriptedResponse("first", match="a"), ScriptedResponse("second", match="a")]
    )
    assert (await model.generate(ask("a"), build_run_context())).text == "first"


@pytest.mark.anyio
async def test_generation_always_reports_usage() -> None:
    """**Port invariant** (ARCHITECTURE.md §9.2): never a zero token count."""
    result = await FakeLanguageModel().generate(ask(), build_run_context())
    assert result.usage.calls == 1
    assert result.usage.prompt_tokens > 0
    assert result.usage.estimated is True


@pytest.mark.anyio
async def test_an_unpriced_fake_reports_none_not_zero() -> None:
    """The case a test needs to check that a total goes to ``None`` (§13.8)."""
    model = FakeLanguageModel([ScriptedResponse("hi", cost_usd=None)], cost_per_call=None)
    result = await model.generate(ask(), build_run_context())
    assert result.usage.cost_usd is None


@pytest.mark.anyio
async def test_the_model_records_every_call() -> None:
    model = FakeLanguageModel()
    ctx = build_run_context()
    await model.generate(ask("one"), ctx)
    await model.generate(ask("two"), ctx)

    assert model.call_count == 2
    assert [m.messages[0].content for m in model.calls] == ["one", "two"]


@pytest.mark.anyio
async def test_streaming_yields_deltas_then_a_final_usage_delta() -> None:
    model = FakeLanguageModel([ScriptedResponse("one two three")])
    deltas = [d async for d in model.stream(ask(), build_run_context())]

    assert "".join(d.text for d in deltas) == "one two three"
    assert deltas[-1].finish_reason == "stop"
    assert deltas[-1].usage is not None
    assert model.stream_calls


@pytest.mark.anyio
async def test_streaming_emits_tool_calls_whole() -> None:
    """A half-parsed tool call is not actionable, so it is never split."""
    call = ToolCall(id="1", name="search", arguments='{"q": "x"}')
    model = FakeLanguageModel([ScriptedResponse("", tool_calls=[call], finish_reason="tool_calls")])

    deltas = [d async for d in model.stream(ask(), build_run_context())]
    tool_deltas = [d.tool_call for d in deltas if d.tool_call is not None]
    assert tool_deltas == [call]


@pytest.mark.anyio
async def test_a_failing_model_raises_the_configured_error() -> None:
    """For testing retry, fallback and degradation without a real outage."""
    model = FakeLanguageModel(fail_with=TransientError("upstream 503"))
    with pytest.raises(TransientError):
        await model.generate(ask(), build_run_context())


@pytest.mark.anyio
async def test_a_failing_model_also_fails_when_streaming() -> None:
    model = FakeLanguageModel(fail_with=TransientError("upstream 503"))
    with pytest.raises(TransientError):
        _ = [d async for d in model.stream(ask(), build_run_context())]


@pytest.mark.anyio
async def test_token_counting_is_never_zero() -> None:
    """A zero count would let a context assembler believe anything fits."""
    assert await FakeLanguageModel().count_tokens([Message(role="user", content="")]) >= 1


def test_capabilities_can_be_narrowed_to_test_the_failure_path() -> None:
    from hardpoint.core.capabilities import ModelCapabilities

    model = FakeLanguageModel(capabilities=ModelCapabilities(context_window_tokens=100))
    assert model.capabilities().supports_tools is False
    assert FakeLanguageModel().capabilities().supports_tools is True


# --------------------------------------------------------------------------- #
# FakeEmbeddingModel                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_embedding_is_deterministic_and_ordered() -> None:
    model = FakeEmbeddingModel(dimensions=8)
    ctx = build_run_context()

    first = await model.embed(["alpha", "beta"], "document", ctx)
    second = await model.embed(["alpha", "beta"], "document", ctx)

    assert first.vectors == second.vectors
    assert len(first.vectors) == 2
    assert first.vectors[0] != first.vectors[1]


@pytest.mark.anyio
async def test_query_and_document_embeddings_differ() -> None:
    """Asymmetric on purpose.

    Real asymmetric models exist, and embedding a query as a document degrades
    recall with no error and no symptom other than worse answers. A symmetric
    fake would make that class of bug untestable.
    """
    model = FakeEmbeddingModel()
    ctx = build_run_context()

    as_document = await model.embed(["hello"], "document", ctx)
    as_query = await model.embed(["hello"], "query", ctx)

    assert as_document.vectors != as_query.vectors


@pytest.mark.anyio
async def test_embedding_counts_calls_and_texts() -> None:
    """What "re-running sync performs zero embeddings" asserts against."""
    model = FakeEmbeddingModel()
    ctx = build_run_context()

    await model.embed(["a", "b"], "document", ctx)
    await model.embed(["c"], "document", ctx)

    assert model.call_count == 2
    assert model.embedded_texts == ["a", "b", "c"]


@pytest.mark.anyio
async def test_embedding_reports_its_model_id() -> None:
    """The manifest records it, so a model change forces a re-embed."""
    model = FakeEmbeddingModel(model_id="fake/v2")
    result = await model.embed(["a"], "document", build_run_context())
    assert result.model_id == "fake/v2"


@pytest.mark.anyio
async def test_embedding_dimension_is_configurable() -> None:
    model = FakeEmbeddingModel(dimensions=32)
    result = await model.embed(["a"], "document", build_run_context())
    assert len(result.vectors[0]) == 32
    assert model.dimensions == 32


# --------------------------------------------------------------------------- #
# InMemoryVectorIndex, beyond the contract kit                                #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_ensure_rejects_a_dimension_change_on_a_populated_index() -> None:
    """A mismatch either errors per record or silently truncates. Both are bad."""
    index = InMemoryVectorIndex(dimensions=4)
    ctx = build_run_context()
    await index.upsert([IndexRecord(id="a", vector=(1.0, 0.0, 0.0, 0.0))], ctx)

    with pytest.raises(ConfigError) as exc_info:
        await index.ensure(IndexSpec(name=index.name, dimensions=8))
    assert exc_info.value.remedy is not None
    assert "re-ingest" in exc_info.value.remedy


@pytest.mark.anyio
async def test_ensure_accepts_a_dimension_change_on_an_empty_index() -> None:
    """Configuring a fresh index is not a mismatch."""
    index = InMemoryVectorIndex(dimensions=4)
    await index.ensure(IndexSpec(name=index.name, dimensions=8))
    assert (await index.describe()).dimensions == 8


@pytest.mark.anyio
async def test_upsert_rejects_a_wrong_width_vector() -> None:
    index = InMemoryVectorIndex(dimensions=4)
    with pytest.raises(ContractError, match="dimensions"):
        await index.upsert([IndexRecord(id="a", vector=(1.0, 0.0))], build_run_context())


@pytest.mark.anyio
async def test_document_id_is_filterable_without_being_copied_into_metadata() -> None:
    """Ingestion deletes a document's chunks by filtering on ``document_id``.

    Requiring the caller to duplicate it into metadata would make the most
    important delete path depend on a convention nobody enforces.
    """
    index = InMemoryVectorIndex(dimensions=2)
    ctx = build_run_context()
    await index.upsert(
        [
            IndexRecord(id="a", vector=(1.0, 0.0), document_id="doc_1"),
            IndexRecord(id="b", vector=(0.0, 1.0), document_id="doc_2"),
        ],
        ctx,
    )

    report = await index.delete(ctx, filter=F.field("document_id").eq("doc_1"))
    assert report.deleted == 1
    assert [r.id for r in index.records()] == ["b"]


@pytest.mark.anyio
async def test_equal_scores_order_deterministically() -> None:
    """Otherwise top_k would return different records on different runs."""
    index = InMemoryVectorIndex(dimensions=2)
    ctx = build_run_context()
    await index.upsert([IndexRecord(id=name, vector=(0.0, 0.0)) for name in ("c", "a", "b")], ctx)

    results = await index.query(VectorQuery(vector=(1.0, 0.0), top_k=3), ctx)
    assert [r.id for r in results] == ["a", "b", "c"]


@pytest.mark.anyio
async def test_min_score_filters_server_side() -> None:
    index = InMemoryVectorIndex(dimensions=2)
    ctx = build_run_context()
    await index.upsert(
        [
            IndexRecord(id="near", vector=(1.0, 0.0)),
            IndexRecord(id="far", vector=(0.0, 1.0)),
        ],
        ctx,
    )
    results = await index.query(VectorQuery(vector=(1.0, 0.0), top_k=10, min_score=0.5), ctx)
    assert [r.id for r in results] == ["near"]


@pytest.mark.anyio
async def test_include_text_can_be_turned_off() -> None:
    index = InMemoryVectorIndex(dimensions=2)
    ctx = build_run_context()
    await index.upsert([IndexRecord(id="a", vector=(1.0, 0.0), text="hello")], ctx)

    with_text = await index.query(VectorQuery(vector=(1.0, 0.0), include_text=True), ctx)
    without = await index.query(VectorQuery(vector=(1.0, 0.0), include_text=False), ctx)
    assert with_text[0].text == "hello"
    assert without[0].text is None


@pytest.mark.anyio
async def test_an_unauthorised_index_raises_on_every_operation() -> None:
    """So the contract kit can check error mapping without a real credential."""
    index = InMemoryVectorIndex(unauthorised=True)
    ctx = build_run_context()

    with pytest.raises(AuthError):
        await index.describe()
    with pytest.raises(AuthError):
        await index.upsert([], ctx)
    with pytest.raises(AuthError):
        await index.query(VectorQuery(), ctx)


@pytest.mark.anyio
async def test_epoch_is_reported_and_settable_for_arrangement() -> None:
    index = InMemoryVectorIndex(epoch=3)
    assert (await index.describe()).epoch == 3
    index.set_epoch(4)
    assert (await index.describe()).epoch == 4


@pytest.mark.anyio
async def test_all_three_metrics_rank_an_identical_vector_first() -> None:
    ctx = build_run_context()
    for metric in ("cosine", "dot", "euclidean"):
        index = InMemoryVectorIndex(dimensions=2, metric=metric)  # type: ignore[arg-type]
        await index.upsert(
            [
                IndexRecord(id="same", vector=(1.0, 0.0)),
                IndexRecord(id="other", vector=(0.0, 1.0)),
            ],
            ctx,
        )
        results = await index.query(VectorQuery(vector=(1.0, 0.0), top_k=2), ctx)
        assert next(r.id for r in results) == "same", metric


# --------------------------------------------------------------------------- #
# FakeReranker                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_reranking_prefers_query_word_overlap() -> None:
    """Plausible ordering, so a test's intent reads clearly when it fails."""
    chunks = build_chunks(["nothing relevant here", "the refund policy is thirty days"])
    candidates = build_retrieved(chunks)

    ranked = await FakeReranker().rerank("refund policy", candidates, 2, build_run_context())
    assert ranked[0].chunk.text.startswith("the refund policy")
    assert ranked[0].rank == 0


@pytest.mark.anyio
async def test_reranking_respects_top_k_and_counts_calls() -> None:
    candidates = build_retrieved(build_chunks(["a", "b", "c"]))
    reranker = FakeReranker()
    ranked = await reranker.rerank("a", candidates, 2, build_run_context())

    assert len(ranked) == 2
    assert reranker.call_count == 1


@pytest.mark.anyio
async def test_a_failing_reranker_raises_for_the_skip_path() -> None:
    """Reranking is optional; the default is to skip and record a Degradation."""
    reranker = FakeReranker(fail_with=TransientError("reranker down"))
    with pytest.raises(TransientError):
        await reranker.rerank("q", [], 5, build_run_context())


# --------------------------------------------------------------------------- #
# FakeCache                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_cache_counts_hits_and_misses() -> None:
    """The counters are how "an epoch bump changes the key" gets checked."""
    cache = FakeCache()
    assert await cache.get("k") is None
    await cache.set("k", b"v", None)
    assert await cache.get("k") == b"v"

    assert (cache.misses, cache.hits, cache.sets) == (1, 1, 1)


@pytest.mark.anyio
async def test_cache_records_ttls_without_enforcing_them() -> None:
    """Enforcing expiry on wall-clock time would make tests time-dependent."""
    cache = FakeCache()
    await cache.set("k", b"v", 60)
    assert cache.ttls["k"] == 60
    assert await cache.get("k") == b"v"


@pytest.mark.anyio
async def test_cache_delete_prefix_reports_how_many_it_removed() -> None:
    cache = FakeCache()
    await cache.set("retr:primary:1:a", b"1", None)
    await cache.set("retr:primary:1:b", b"2", None)
    await cache.set("gen:x", b"3", None)

    assert await cache.delete_prefix("retr:primary:1:") == 2
    assert set(cache.store) == {"gen:x"}


# --------------------------------------------------------------------------- #
# Recording tracer and metrics                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_tracer_records_nesting() -> None:
    """ "Traces nest correctly across async boundaries" is a real requirement."""
    tracer = RecordingTracer()

    async with tracer.span("hardpoint.pipeline", pipeline="qa") as outer:
        outer.set_attribute("steps", 2)
        async with tracer.span("hardpoint.step", step="retrieve") as inner:
            inner.add_event("retry", {"attempt": 1})

    assert tracer.names() == ["hardpoint.pipeline", "hardpoint.step"]
    assert len(tracer.roots) == 1
    assert [child.name for child in tracer.roots[0].children] == ["hardpoint.step"]
    assert tracer.find("hardpoint.step")[0].events == [("retry", {"attempt": 1})]
    assert tracer.roots[0].attributes == {"pipeline": "qa", "steps": 2}


@pytest.mark.anyio
async def test_the_tracer_records_exceptions_and_status() -> None:
    tracer = RecordingTracer()
    error = TransientError("boom")

    async with tracer.span("hardpoint.step") as span:
        span.record_exception(error)
        span.set_status("error", "provider failed")

    recorded = tracer.find("hardpoint.step")[0]
    assert recorded.exceptions == [error]
    assert recorded.status == ("error", "provider failed")
    assert recorded.trace_id == "fake-trace"


@pytest.mark.anyio
async def test_sibling_spans_do_not_nest() -> None:
    tracer = RecordingTracer()
    async with tracer.span("a"):
        pass
    async with tracer.span("b"):
        pass
    assert len(tracer.roots) == 2


def test_the_metric_sink_records_and_totals() -> None:
    metrics = RecordingMetricSink()
    metrics.increment("hardpoint.requests", 1.0, pipeline="qa")
    metrics.increment("hardpoint.requests", 2.0, pipeline="qa")
    metrics.observe("hardpoint.latency_ms", 12.5, step="generate")
    metrics.gauge("hardpoint.index.size", 42.0)

    assert metrics.total("hardpoint.requests") == 3.0
    assert metrics.observations[0][1] == 12.5
    assert metrics.gauges[0][0] == "hardpoint.index.size"


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def test_build_run_context_needs_no_arguments() -> None:
    """Every step takes a RunContext; building one must be trivial."""
    ctx = build_run_context()
    assert ctx.run_id == "test-run"
    assert isinstance(ctx.tracer, RecordingTracer)
    assert isinstance(ctx.metrics, RecordingMetricSink)
    assert ctx.deadline.remaining_s() is None
    assert ctx.cache.enabled("shared") is False


def test_build_run_context_accepts_overrides() -> None:
    cache = FakeCache()
    ctx = build_run_context(run_id="r2", shared_cache=cache)
    assert ctx.run_id == "r2"
    assert ctx.cache.enabled("shared") is True


def test_built_chunks_use_the_real_id_derivation() -> None:
    """So a test written against these exercises production identity.

    A fixture that invented ids would let an id bug pass every test and fail
    only on the first real ingestion.
    """
    from hardpoint.core.ids import chunk_id, document_id

    document = build_document("guide.md", source_id="docs")
    chunk = build_chunk("hello", document=document, index=3)

    assert document.id == document_id("docs", "guide.md")
    assert chunk.id == chunk_id(document.id, 3, "hello")


def test_built_chunks_are_untrusted_by_default() -> None:
    assert build_chunk().trust.value == "untrusted"


def test_build_chunks_numbers_them_in_order() -> None:
    chunks = build_chunks(["one", "two", "three"])
    assert [c.index for c in chunks] == [0, 1, 2]
    assert len({c.id for c in chunks}) == 3


def test_build_retrieved_scores_descending() -> None:
    retrieved = build_retrieved(build_chunks(["a", "b", "c"]))
    scores = [r.score for r in retrieved]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in retrieved] == [0, 1, 2]


def test_build_retrieved_accepts_explicit_scores() -> None:
    retrieved = build_retrieved(build_chunks(["a", "b"]), scores=[0.4, 0.9])
    assert [r.score for r in retrieved] == [0.4, 0.9]


def test_fixtures_produce_real_models() -> None:
    """Not dicts pretending to be models: validation must actually have run."""
    assert isinstance(build_chunk(), Chunk)
    assert isinstance(build_chunk().span, CharSpan)
    assert isinstance(build_retrieved(build_chunks(["a"]))[0], RetrievedChunk)
