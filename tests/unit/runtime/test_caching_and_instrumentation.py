"""Cache keys, cache layers, the instrumented ports, and streaming (INSTRUCTIONS.md §7)."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import pytest

from hardpoint.adapters.cache.file import FileCache
from hardpoint.adapters.cache.memory import MemoryCache
from hardpoint.adapters.cache.redis import RedisCache, RedisCacheConfig
from hardpoint.adapters.cache.redis import build as build_redis
from hardpoint.core.cache_keys import (
    embedding_key,
    generation_key,
    params_hash,
    rerank_key,
    retrieval_key,
)
from hardpoint.core.config.schema import CacheLayerConfig, ModelPriceConfig
from hardpoint.core.errors import MissingDependencyError
from hardpoint.core.filters import F
from hardpoint.core.ids import normalise_text
from hardpoint.core.ports import GenerationRequest, IndexRecord, IndexSpec, Message
from hardpoint.generation import Generate, InMemoryPromptStore
from hardpoint.observability.metrics import InMemoryMetricSink
from hardpoint.observability.pricing import PricingTable, UnpricedModelWarning
from hardpoint.observability.tracing import CollectingTracer
from hardpoint.retrieval import ContextAssembler, VectorRetriever
from hardpoint.runtime import Pipeline, PolicyChain
from hardpoint.runtime.wrapped import (
    PolicyEmbeddingModel,
    PolicyLanguageModel,
    PolicyReranker,
    PolicyVectorIndex,
)
from hardpoint.testing import (
    FakeCache,
    FakeEmbeddingModel,
    FakeLanguageModel,
    FakeReranker,
    InMemoryVectorIndex,
    ScriptedResponse,
    build_chunks,
    build_retrieved,
    build_run_context,
)
from hardpoint.testing.contracts import cache_backend_contract

# --------------------------------------------------------------------------- #
# Keys, verbatim  [LOCKED]                                                    #
# --------------------------------------------------------------------------- #


def sha(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def test_the_embedding_key_format_is_verbatim() -> None:
    assert embedding_key("openai/e3", "query", "  Hello   world ") == (
        f"embed:openai/e3:query:{sha(normalise_text('  Hello   world '))}"
    )


def test_the_retrieval_key_format_is_verbatim() -> None:
    key = retrieval_key("primary", 7, "openai/e3", query="q", filter=None, params={"top_k": 5})
    assert key.startswith("retr:primary:7:openai/e3:")
    assert len(key.rsplit(":", 1)[1]) == 64


def test_the_generation_key_format_is_verbatim() -> None:
    digest = params_hash({"temperature": 0.0})
    assert generation_key("openai/gpt", digest, "abc123", "prompt") == (
        f"gen:openai/gpt:{digest}:abc123:{sha('prompt')}"
    )


def test_the_rerank_key_format_is_verbatim() -> None:
    assert rerank_key("cohere/r", "q", ["c1", "c2"]) == f"rr:cohere/r:{sha('q', 'c1', 'c2')}"


def test_bumping_the_epoch_changes_every_retrieval_key() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §7: the test proving an epoch bump invalidates."""
    before = retrieval_key("primary", 1, "m", query="q", filter=None, params={})
    after = retrieval_key("primary", 2, "m", query="q", filter=None, params={})
    assert before != after


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "other"},
        {"query": "different"},
        {"filter": F.field("tenant").eq("b")},
        {"params": {"top_k": 6}},
        {"index": "secondary"},
    ],
)
def test_every_component_of_the_retrieval_key_matters(change: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "index": "primary",
        "epoch": 1,
        "model_id": "m",
        "query": "q",
        "filter": F.field("tenant").eq("a"),
        "params": {"top_k": 5},
    }

    def key(values: dict[str, Any]) -> str:
        return retrieval_key(
            values["index"],
            values["epoch"],
            values["model_id"],
            query=values["query"],
            filter=values["filter"],
            params=values["params"],
        )

    assert key(base) != key({**base, **change})


# --------------------------------------------------------------------------- #
# Retrieval cache, through the step                                           #
# --------------------------------------------------------------------------- #


async def populated_index() -> tuple[InMemoryVectorIndex, FakeEmbeddingModel]:
    embedder = FakeEmbeddingModel(dimensions=16, lexical=True)
    index = InMemoryVectorIndex("primary", dimensions=16)
    await index.ensure(IndexSpec(name="primary", dimensions=16))
    texts = ["refunds take thirty days", "shipping takes five days"]
    vectors = (await embedder.embed(texts, "document", build_run_context())).vectors
    await index.upsert(
        [
            IndexRecord(id=f"c{i}", vector=v, text=t, document_id="d")
            for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
        ],
        build_run_context(),
    )
    embedder.calls.clear()
    return index, embedder


@pytest.mark.anyio
async def test_the_retrieval_cache_serves_repeats_and_misses_after_an_epoch_bump() -> None:
    index, embedder = await populated_index()
    epoch = {"value": 1}

    async def read_epoch() -> int:
        return epoch["value"]

    cache = FakeCache()
    retriever = VectorRetriever(index, embedder, top_k=2, epoch=read_epoch)

    first = await retriever("refund policy", build_run_context(shared_cache=cache))
    second = await retriever("refund policy", build_run_context(shared_cache=cache))
    assert [c.chunk.id for c in first.chunks] == [c.chunk.id for c in second.chunks]
    assert embedder.call_count == 1, "the repeat was served from the cache"
    assert index.query_calls == 1

    epoch["value"] = 2
    await retriever("refund policy", build_run_context(shared_cache=cache))
    assert index.query_calls == 2, "an epoch bump must invalidate the cached retrieval"


@pytest.mark.anyio
async def test_no_epoch_reader_means_no_retrieval_cache() -> None:
    index, embedder = await populated_index()
    cache = FakeCache()
    retriever = VectorRetriever(index, embedder, top_k=2)
    for _ in range(2):
        await retriever("refund policy", build_run_context(shared_cache=cache))
    assert index.query_calls == 2
    assert cache.sets == 0


# --------------------------------------------------------------------------- #
# Embedding and rerank caches, in the wrappers                                #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_embedding_cache_sends_only_misses() -> None:
    inner = FakeEmbeddingModel(dimensions=8)
    model = PolicyEmbeddingModel(inner, PolicyChain(), cache=CacheLayerConfig())
    cache = FakeCache()
    ctx = build_run_context(shared_cache=cache)

    first = await model.embed(["a", "b"], "document", ctx)
    second = await model.embed(["b", "c", "a"], "document", ctx)

    assert inner.calls == [(("a", "b"), "document"), (("c",), "document")]
    assert second.vectors[0] == first.vectors[1], "order is preserved around the hits"
    assert second.vectors[2] == first.vectors[0]
    fully_cached = await model.embed(["a"], "document", ctx)
    assert fully_cached.usage.calls == 0
    assert len(inner.calls) == 2


@pytest.mark.anyio
async def test_query_and_document_embeddings_are_cached_separately() -> None:
    inner = FakeEmbeddingModel(dimensions=8)
    model = PolicyEmbeddingModel(inner, PolicyChain(), cache=CacheLayerConfig())
    ctx = build_run_context(shared_cache=FakeCache())
    await model.embed(["a"], "document", ctx)
    await model.embed(["a"], "query", ctx)
    assert inner.call_count == 2


@pytest.mark.anyio
async def test_the_rerank_cache_serves_a_repeat() -> None:
    inner = FakeReranker()
    reranker = PolicyReranker(inner, PolicyChain(), cache=CacheLayerConfig(ttl_s=60))
    candidates = build_retrieved(build_chunks(["alpha refund", "beta shipping", "gamma"]))
    ctx = build_run_context(shared_cache=FakeCache())

    first = await reranker.rerank("refund", candidates, 2, ctx)
    second = await reranker.rerank("refund", candidates, 2, ctx)
    assert [c.chunk.id for c in first] == [c.chunk.id for c in second]
    assert inner.call_count == 1
    await reranker.rerank("refund", candidates, 3, ctx)
    assert inner.call_count == 2, "a cached ranking shorter than the new top_k is a miss"


# --------------------------------------------------------------------------- #
# Cache backends                                                              #
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _memory() -> MemoryCache:
    return MemoryCache(clock=_Clock())


def _advance(cache: Any, seconds: float) -> None:
    cache._clock.now += seconds


TestMemoryCache = cache_backend_contract(_memory, expire=_advance)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path


_FILE_ROOTS: list[Path] = []


def _file_cache() -> FileCache:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="hardpoint-cache-"))
    _FILE_ROOTS.append(root)
    return FileCache(root, clock=_Clock())


TestFileCache = cache_backend_contract(_file_cache, expire=_advance)


class _FakeRedis:
    """Enough of ``redis.asyncio.Redis`` to run the contract."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self.values[key] = value

    async def scan_iter(self, match: str) -> Any:
        prefix = match.rstrip("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key

    async def delete(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0

    async def aclose(self) -> None:
        return None


TestRedisCache = cache_backend_contract(lambda: RedisCache(_FakeRedis()))


@pytest.mark.anyio
async def test_a_redis_outage_degrades_to_a_miss() -> None:
    class Down(_FakeRedis):
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError("down")

        async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
            raise ConnectionError("down")

    cache = RedisCache(Down())
    await cache.set("k", b"v", None)
    assert await cache.get("k") is None


def test_the_redis_factory_names_the_extra_when_it_is_missing() -> None:
    import importlib.util

    if importlib.util.find_spec("redis") is not None:  # pragma: no cover - dev env lacks it
        pytest.skip("redis is installed here")
    with pytest.raises(MissingDependencyError) as exc_info:
        build_redis(RedisCacheConfig())
    assert exc_info.value.remedy == "pip install 'hardpoint[redis]'"


@pytest.mark.anyio
async def test_a_corrupt_file_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    await cache.set("k", b"value", None)
    (entry,) = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "keys.tsv"]
    entry.write_bytes(b"\x01")
    assert await cache.get("k") is None


def test_the_memory_cache_evicts_least_recently_used() -> None:
    import anyio

    async def scenario() -> None:
        cache = MemoryCache(max_entries=2)
        await cache.set("a", b"1", None)
        await cache.set("b", b"2", None)
        await cache.get("a")
        await cache.set("c", b"3", None)
        assert await cache.get("b") is None
        assert await cache.get("a") == b"1"
        assert len(cache) == 2

    anyio.run(scenario)


# --------------------------------------------------------------------------- #
# Instrumented ports                                                          #
# --------------------------------------------------------------------------- #


def request() -> GenerationRequest:
    return GenerationRequest(messages=(Message(role="user", content="hello"),), temperature=0.0)


@pytest.mark.anyio
async def test_the_llm_span_carries_genai_attributes_and_cost_from_pricing() -> None:
    tracer, metrics = CollectingTracer(), InMemoryMetricSink()
    inner = FakeLanguageModel(
        [ScriptedResponse("hi", cost_usd=None)], model_id="acme/chat", cost_per_call=None
    )
    pricing = PricingTable({"acme/chat": ModelPriceConfig(input=1_000_000.0, output=0.0)})
    model = PolicyLanguageModel(inner, PolicyChain(), pricing=pricing)

    result = await model.generate(request(), build_run_context(tracer=tracer, metrics=metrics))

    (span,) = tracer.find("hardpoint.llm")
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "acme/chat"
    assert span.attributes["gen_ai.system"] == "acme"
    assert span.attributes["gen_ai.usage.input_tokens"] == result.usage.prompt_tokens
    assert span.attributes["messages.content"] == "user: hello"
    assert result.usage.cost_usd == pytest.approx(result.usage.prompt_tokens * 1.0)
    assert metrics.total("hardpoint.cost_usd", model="acme/chat") == result.usage.cost_usd
    assert metrics.total("hardpoint.tokens", direction="input") == result.usage.prompt_tokens


@pytest.mark.anyio
async def test_an_unpriced_model_reports_none_and_counts_the_call() -> None:
    metrics = InMemoryMetricSink()
    inner = FakeLanguageModel(model_id="acme/unknown", cost_per_call=None)
    model = PolicyLanguageModel(inner, PolicyChain(), pricing=PricingTable())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnpricedModelWarning)
        result = await model.generate(request(), build_run_context(metrics=metrics))
    assert result.usage.cost_usd is None
    assert metrics.total("hardpoint.cost.unpriced_calls", model="acme/unknown") == 1


@pytest.mark.anyio
async def test_the_stream_is_traced_and_priced() -> None:
    tracer = CollectingTracer()
    inner = FakeLanguageModel(
        [ScriptedResponse("one two", cost_usd=None)], model_id="acme/chat", cost_per_call=None
    )
    pricing = PricingTable({"acme/chat": ModelPriceConfig(input=0.0, output=1_000_000.0)})
    model = PolicyLanguageModel(inner, PolicyChain(), pricing=pricing)

    deltas = [d async for d in model.stream(request(), build_run_context(tracer=tracer))]
    final = deltas[-1].usage
    assert final is not None
    assert final.cost_usd == pytest.approx(final.completion_tokens * 1.0)
    (span,) = tracer.find("hardpoint.llm")
    assert span.attributes["hardpoint.stream"] is True
    assert span.ended_at is not None


@pytest.mark.anyio
async def test_index_and_embed_spans_are_emitted() -> None:
    tracer, metrics = CollectingTracer(), InMemoryMetricSink()
    ctx = build_run_context(tracer=tracer, metrics=metrics)
    index = PolicyVectorIndex(InMemoryVectorIndex("primary", 4), PolicyChain())
    embedder = PolicyEmbeddingModel(FakeEmbeddingModel(4), PolicyChain())

    await index.upsert([IndexRecord(id="a", vector=(1.0, 0.0, 0.0, 0.0))], ctx)
    vector = (await embedder.embed(["q"], "query", ctx)).vectors[0]
    from hardpoint.core.ports import VectorQuery

    await index.query(VectorQuery(vector=vector, top_k=3), ctx)

    assert [span.name for span in tracer.spans()] == [
        "hardpoint.index.upsert",
        "hardpoint.embed",
        "hardpoint.index.query",
    ]
    assert tracer.find("hardpoint.index.query")[0].attributes["hardpoint.index.results"] == 1
    assert metrics.histograms[("hardpoint.retrieval.results", (("retriever", "primary"),))] == [1]


@pytest.mark.anyio
async def test_a_provider_error_is_counted() -> None:
    from hardpoint.core.errors import AuthError

    metrics = InMemoryMetricSink()
    model = PolicyLanguageModel(
        FakeLanguageModel(fail_with=AuthError("no", remedy="key")), PolicyChain()
    )
    with pytest.raises(AuthError):
        await model.generate(request(), build_run_context(metrics=metrics))
    assert metrics.total("hardpoint.errors", error="AuthError") == 1


# --------------------------------------------------------------------------- #
# Pipeline: metrics, debug mode, streaming                                    #
# --------------------------------------------------------------------------- #


def rag_pipeline(llm: Any, index: Any, embedder: Any) -> Pipeline[str, Any]:
    prompts = InMemoryPromptStore({"answer": "{{ context }}\n\nQ: {{ question }}"})
    return Pipeline(
        "rag",
        [
            VectorRetriever(index, embedder, top_k=2),
            ContextAssembler(token_budget=500),
            Generate(llm, prompts),
        ],
    )


@pytest.mark.anyio
async def test_on_delta_streams_the_final_step_through_the_same_run() -> None:
    index, embedder = await populated_index()
    llm = FakeLanguageModel([ScriptedResponse("Refunds take thirty days.")])
    pipeline = rag_pipeline(llm, index, embedder)
    tracer = CollectingTracer()
    received: list[str] = []

    async def on_delta(text: str) -> None:
        received.append(text)

    run = await pipeline.run_detailed(
        "refunds?", build_run_context(tracer=tracer), on_delta=on_delta
    )

    assert "".join(received) == "Refunds take thirty days."
    assert len(received) > 1
    assert run.value.text == "Refunds take thirty days."
    assert llm.stream_calls, "streamed"
    assert not llm.calls, "not generated"
    steps = [s.attributes["step.name"] for s in tracer.find("hardpoint.step")]
    assert steps == ["retrieve", "assemble_context", "generate"]


@pytest.mark.anyio
async def test_steps_emit_latency_metrics_and_debug_attributes() -> None:
    from hardpoint.core.config.snapshot import ConfigSnapshot

    index, embedder = await populated_index()
    pipeline = rag_pipeline(FakeLanguageModel(), index, embedder)
    tracer, metrics = CollectingTracer(), InMemoryMetricSink()
    config = ConfigSnapshot(env="test", data={"observability": {"debug": True}})

    await pipeline("refunds?", build_run_context(tracer=tracer, metrics=metrics, config=config))

    step = tracer.find("hardpoint.step")[0]
    assert step.attributes["step.input"] == "'refunds?'"
    assert "Retrieved(" in str(step.attributes["step.output"])
    series = {name for (name, _labels) in metrics.histograms}
    assert "hardpoint.step.latency_ms" in series


@pytest.mark.anyio
async def test_debug_attributes_are_off_by_default() -> None:
    index, embedder = await populated_index()
    tracer = CollectingTracer()
    await rag_pipeline(FakeLanguageModel(), index, embedder)("q", build_run_context(tracer=tracer))
    assert all("step.input" not in s.attributes for s in tracer.find("hardpoint.step"))
