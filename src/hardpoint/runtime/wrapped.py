"""Ports wrapped in their configured policies, tracing, metering, pricing and caches.

ADR-004: policies are decorators applied to a port implementation at
construction time, so an adapter stays transport-only. This module is where
that decision is applied: one thin proxy per port, satisfying the same
Protocol, adding in a fixed order:

1. **A cache**, for the content-addressed layers -- embeddings and reranking --
   keyed exactly as ARCHITECTURE.md §22.2 specifies.
2. **A span** from the taxonomy (``hardpoint.llm``, ``hardpoint.embed``,
   ``hardpoint.index.query``, ``hardpoint.index.upsert``, ``hardpoint.rerank``)
   with OpenTelemetry GenAI attributes.
3. **The policy chain**: retry, timeouts, breaker, rate limit, fallback.
4. **Cost**, from the pricing table, when the adapter reported none -- per
   component, so a fallback's answer is priced at the fallback's price.
5. **Metrics**: tokens by model and direction, cost, unpriced calls, errors.

None of this is in an adapter, which is the point: an adapter is transport, and
these concerns are implemented once.

## Streaming is not retried

A stream that failed after delivering tokens cannot be retried without
delivering them twice, so ``stream`` is traced and priced but not retried. The
deadline still applies, because the caller's cancel scope does.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hardpoint.core.cache_keys import embedding_key, rerank_key
from hardpoint.core.models import RetrievedChunk, StepUsage
from hardpoint.core.types import JsonValue
from hardpoint.runtime.policies import Fallback, PolicyChain

if TYPE_CHECKING:
    from hardpoint.core.capabilities import IndexCapabilities, ModelCapabilities
    from hardpoint.core.config.schema import CacheLayerConfig
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import (
        DeleteReport,
        EmbeddingModel,
        EmbedKind,
        EmbedResult,
        GenerationDelta,
        GenerationRequest,
        GenerationResult,
        IndexInfo,
        IndexRecord,
        IndexSpec,
        LanguageModel,
        Message,
        MetadataFilter,
        Reranker,
        ScoredRecord,
        UpsertReport,
        VectorIndex,
        VectorQuery,
    )
    from hardpoint.observability.pricing import PricingTable

__all__ = ["PolicyEmbeddingModel", "PolicyLanguageModel", "PolicyReranker", "PolicyVectorIndex"]


def _provider(model_id: str) -> str:
    """``openai/gpt-4o`` -> ``openai``, the ``gen_ai.system`` attribute."""
    return model_id.split("/", 1)[0] if "/" in model_id else "unknown"


class _Wrapped:
    """Shared plumbing: chain, optional fallback, pricing, metering, closing."""

    def __init__(
        self,
        inner: Any,
        chain: PolicyChain,
        fallback: Any | None,
        pricing: PricingTable | None,
    ) -> None:
        self.inner = inner
        self.chain = chain
        self.fallback = fallback
        self.pricing = pricing

    def _chain_for(self, alternative: Any | None) -> PolicyChain:
        """Return the chain, with a per-call fallback bound to this call's arguments.

        The chain's stateful members -- breaker, rate limiter -- are shared by
        reference, so copying the frozen chain per call keeps their memory.
        """
        if alternative is None:
            return self.chain
        return replace(self.chain, fallback=Fallback(alternative))

    def _priced(self, usage: StepUsage, model_id: str) -> StepUsage:
        """Fill in cost from the table when the adapter reported none."""
        if usage.cost_usd is not None or self.pricing is None or usage.calls == 0:
            return usage
        return usage.model_copy(update={"cost_usd": self.pricing.cost(model_id, usage)})

    @staticmethod
    def _meter(ctx: RunContext, model_id: str, usage: StepUsage) -> None:
        for direction, tokens in (
            ("input", usage.prompt_tokens),
            ("output", usage.completion_tokens),
            ("embed", usage.embed_tokens),
        ):
            if tokens:
                ctx.metrics.increment(
                    "hardpoint.tokens", tokens, model=model_id, direction=direction
                )
        if usage.cost_usd is not None:
            ctx.metrics.increment("hardpoint.cost_usd", usage.cost_usd, model=model_id)
        elif usage.calls:
            ctx.metrics.increment("hardpoint.cost.unpriced_calls", usage.calls, model=model_id)

    @staticmethod
    def _usage_attributes(span: Any, usage: StepUsage) -> None:
        span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens or usage.embed_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
        span.set_attribute("hardpoint.usage.estimated", usage.estimated)
        span.set_attribute("hardpoint.cost_usd", usage.cost_usd)

    async def aclose(self) -> None:
        """Close the inner component and the fallback, when they hold resources."""
        for component in (self.inner, self.fallback):
            closer = getattr(component, "aclose", None)
            if closer is not None:
                await closer()

    def describe_policies(self) -> str:
        """Render the active chain, outermost first."""
        described = self.chain.describe()
        return described if self.fallback is None else f"fallback -> {described}"


# --------------------------------------------------------------------------- #
# Language model                                                              #
# --------------------------------------------------------------------------- #


class PolicyLanguageModel(_Wrapped):
    """A ``LanguageModel`` with policies, a ``hardpoint.llm`` span, cost and metrics."""

    def __init__(
        self,
        inner: LanguageModel,
        chain: PolicyChain,
        fallback: LanguageModel | None = None,
        *,
        pricing: PricingTable | None = None,
    ) -> None:
        super().__init__(inner, chain, fallback, pricing)
        self.id = inner.id

    async def _generate_priced(
        self, model: LanguageModel, req: GenerationRequest, ctx: RunContext
    ) -> GenerationResult:
        result = await model.generate(req, ctx)
        return result.model_copy(update={"usage": self._priced(result.usage, model.id)})

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Generate through the chain, traced and priced."""
        attributes: dict[str, JsonValue] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": _provider(self.id),
            "gen_ai.request.model": self.id,
        }
        if req.max_output_tokens is not None:
            attributes["gen_ai.request.max_tokens"] = req.max_output_tokens
        if req.temperature is not None:
            attributes["gen_ai.request.temperature"] = req.temperature
        async with ctx.tracer.span("hardpoint.llm", **attributes) as span:
            span.set_attribute("messages.content", _render(req.messages))
            fallback = self.fallback
            alternative = (
                (lambda: self._generate_priced(fallback, req, ctx))
                if fallback is not None
                else None
            )
            try:
                result: GenerationResult = await self._chain_for(alternative).run(
                    f"{self.id}.generate", lambda: self._generate_priced(self.inner, req, ctx), ctx
                )
            except Exception as exc:
                ctx.metrics.increment(
                    "hardpoint.errors", component=self.id, error=type(exc).__name__
                )
                raise
            span.set_attribute("gen_ai.response.model", result.model_id)
            span.set_attribute("gen_ai.response.finish_reasons", [result.finish_reason])
            self._usage_attributes(span, result.usage)
            self._meter(ctx, self.id, result.usage)
            return result

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Stream without retry, traced and priced. See the module docstring."""

        async def deltas() -> AsyncIterator[GenerationDelta]:
            attributes: dict[str, JsonValue] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": _provider(self.id),
                "gen_ai.request.model": self.id,
                "hardpoint.stream": True,
            }
            async with ctx.tracer.span("hardpoint.llm", **attributes) as span:
                span.set_attribute("messages.content", _render(req.messages))
                inner: AsyncIterator[GenerationDelta] = self.inner.stream(req, ctx)
                try:
                    async for delta in inner:
                        if delta.usage is not None:
                            usage = self._priced(delta.usage, self.id)
                            self._usage_attributes(span, usage)
                            self._meter(ctx, self.id, usage)
                            delta = delta.model_copy(update={"usage": usage})  # noqa: PLW2901
                        yield delta
                except Exception as exc:
                    ctx.metrics.increment(
                        "hardpoint.errors", component=self.id, error=type(exc).__name__
                    )
                    raise
                finally:
                    closer = getattr(inner, "aclose", None)
                    if closer is not None:
                        await closer()

        return deltas()

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count with the inner model's tokenizer."""
        counted: int = await self.inner.count_tokens(messages)
        return counted

    def capabilities(self) -> ModelCapabilities:
        """Declare the inner model's capabilities."""
        declared: ModelCapabilities = self.inner.capabilities()
        return declared

    def __repr__(self) -> str:
        """Render the inner model and the chain."""
        return f"PolicyLanguageModel({self.inner!r}, policies={self.describe_policies()!r})"


def _render(messages: Sequence[Message]) -> str:
    """Flatten messages for the ``messages.content`` span attribute."""
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #


class PolicyEmbeddingModel(_Wrapped):
    """An ``EmbeddingModel`` with a cache, policies, a ``hardpoint.embed`` span and cost.

    A fallback embedder is only valid when it produces vectors comparable with
    the index's (ARCHITECTURE.md §18.2), so its width is checked here.

    The cache is content-addressed -- ``embed:{model_id}:{kind}:{sha256(text)}`` --
    so it never needs invalidating, and only the misses in a batch are sent.
    """

    def __init__(
        self,
        inner: EmbeddingModel,
        chain: PolicyChain,
        fallback: EmbeddingModel | None = None,
        *,
        pricing: PricingTable | None = None,
        cache: CacheLayerConfig | None = None,
    ) -> None:
        if fallback is not None and fallback.dimensions != inner.dimensions:
            from hardpoint.core.errors import ConfigError  # noqa: PLC0415 - error path only

            raise ConfigError(
                f"The fallback embedder {fallback.id!r} produces {fallback.dimensions} "
                f"dimensions, but the primary {inner.id!r} produces {inner.dimensions}.",
                config_path="providers.embeddings.policies.fallback",
                remedy=(
                    "A fallback embedder must produce vectors comparable with the "
                    "index. Configure one with the same width and model family, or "
                    "remove the fallback."
                ),
            )
        super().__init__(inner, chain, fallback, pricing)
        self.id = inner.id
        self.dimensions = inner.dimensions
        self.cache = cache

    async def _embed_priced(
        self, model: EmbeddingModel, texts: Sequence[str], kind: EmbedKind, ctx: RunContext
    ) -> EmbedResult:
        result = await model.embed(texts, kind, ctx)
        return result.model_copy(update={"usage": self._priced(result.usage, model.id)})

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Embed through the cache and the chain, traced and priced."""
        from hardpoint.core.ports import EmbedResult  # noqa: PLC0415 - avoids a cycle

        caching = self.cache is not None and self.cache.enabled and ctx.cache.enabled("shared")
        found: dict[int, tuple[float, ...]] = {}
        if caching:
            for position, text in enumerate(texts):
                cached = await ctx.cache.get(embedding_key(self.id, kind, text))
                if cached is not None:
                    found[position] = tuple(json.loads(cached))
            ctx.metrics.increment(
                "hardpoint.cache.lookups", len(found), layer="embeddings", result="hit"
            )
            ctx.metrics.increment(
                "hardpoint.cache.lookups",
                len(texts) - len(found),
                layer="embeddings",
                result="miss",
            )

        missing = [position for position in range(len(texts)) if position not in found]
        if not missing:
            return EmbedResult(
                vectors=tuple(found[p] for p in range(len(texts))),
                model_id=self.id,
                usage=StepUsage(),
            )

        batch = [texts[position] for position in missing]
        attributes: dict[str, JsonValue] = {
            "gen_ai.operation.name": "embeddings",
            "gen_ai.system": _provider(self.id),
            "gen_ai.request.model": self.id,
            "hardpoint.embed.kind": kind,
            "hardpoint.embed.batch_size": len(batch),
            "hardpoint.embed.cached": len(found),
        }
        async with ctx.tracer.span("hardpoint.embed", **attributes) as span:
            fallback = self.fallback
            alternative = (
                (lambda: self._embed_priced(fallback, batch, kind, ctx))
                if fallback is not None
                else None
            )
            try:
                result: EmbedResult = await self._chain_for(alternative).run(
                    f"{self.id}.embed",
                    lambda: self._embed_priced(self.inner, batch, kind, ctx),
                    ctx,
                )
            except Exception as exc:
                ctx.metrics.increment(
                    "hardpoint.errors", component=self.id, error=type(exc).__name__
                )
                raise
            self._usage_attributes(span, result.usage)
            self._meter(ctx, self.id, result.usage)

        for position, vector in zip(missing, result.vectors, strict=True):
            found[position] = vector
            if caching and self.cache is not None:
                key = embedding_key(self.id, kind, texts[position])
                await ctx.cache.set(key, json.dumps(list(vector)).encode(), ttl_s=self.cache.ttl_s)

        return result.model_copy(
            update={"vectors": tuple(found[position] for position in range(len(texts)))}
        )

    def __repr__(self) -> str:
        """Render the inner model and the chain."""
        return f"PolicyEmbeddingModel({self.inner!r}, policies={self.describe_policies()!r})"


# --------------------------------------------------------------------------- #
# Vector index                                                                #
# --------------------------------------------------------------------------- #


class PolicyVectorIndex(_Wrapped):
    """A ``VectorIndex`` with policies and ``hardpoint.index.*`` spans.

    No fallback: falling back to a different index would answer from different
    data, which is a correctness failure dressed as availability.
    """

    def __init__(self, inner: VectorIndex, chain: PolicyChain) -> None:
        super().__init__(inner, chain, None, None)
        self.name = inner.name

    async def describe(self) -> IndexInfo:
        """Describe directly: ``doctor`` wants the first failure, not the fourth."""
        info: IndexInfo = await self.inner.describe()
        return info

    async def ensure(self, spec: IndexSpec) -> None:
        """Ensure directly: it runs once, at ingestion start, and a failure is config."""
        await self.inner.ensure(spec)

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Upsert through the chain. Safe to retry: upsert is idempotent by contract."""
        attributes: dict[str, JsonValue] = {
            "hardpoint.index.name": self.name,
            "hardpoint.index.records": len(records),
        }
        async with ctx.tracer.span("hardpoint.index.upsert", **attributes):
            report: UpsertReport = await self._run(
                "upsert", lambda: self.inner.upsert(records, ctx), ctx
            )
            return report

    async def delete(
        self,
        ctx: RunContext,
        *,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,  # noqa: A002 - name fixed by the port
    ) -> DeleteReport:
        """Delete through the chain. Safe to retry: deleting twice deletes once."""
        report: DeleteReport = await self._run(
            "delete", lambda: self.inner.delete(ctx, ids=ids, filter=filter), ctx
        )
        return report

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Query through the chain, traced, with the result count as a metric."""
        attributes: dict[str, JsonValue] = {
            "hardpoint.index.name": self.name,
            "hardpoint.index.top_k": req.top_k,
            "hardpoint.index.filtered": req.filter is not None,
        }
        async with ctx.tracer.span("hardpoint.index.query", **attributes) as span:
            results: list[ScoredRecord] = await self._run(
                "query", lambda: self.inner.query(req, ctx), ctx
            )
            span.set_attribute("hardpoint.index.results", len(results))
            ctx.metrics.observe("hardpoint.retrieval.results", len(results), retriever=self.name)
            return results

    async def _run(self, operation: str, call: Any, ctx: RunContext) -> Any:
        try:
            return await self.chain.run(f"{self.name}.{operation}", call, ctx)
        except Exception as exc:
            ctx.metrics.increment("hardpoint.errors", component=self.name, error=type(exc).__name__)
            raise

    def supports(self) -> IndexCapabilities:
        """Declare the inner index's capabilities."""
        declared: IndexCapabilities = self.inner.supports()
        return declared

    def __repr__(self) -> str:
        """Render the inner index and the chain."""
        return f"PolicyVectorIndex({self.inner!r}, policies={self.describe_policies()!r})"


# --------------------------------------------------------------------------- #
# Reranker                                                                    #
# --------------------------------------------------------------------------- #


class PolicyReranker(_Wrapped):
    """A ``Reranker`` with a cache, policies, a ``hardpoint.rerank`` span and cost.

    The cache key is ``rr:{model_id}:{sha256(query + candidate_ids)}``. A cached
    ranking shorter than the ``top_k`` now asked for is treated as a miss, since
    it cannot say what the next candidates would have been.
    """

    def __init__(
        self,
        inner: Reranker,
        chain: PolicyChain,
        fallback: Reranker | None = None,
        *,
        pricing: PricingTable | None = None,
        cache: CacheLayerConfig | None = None,
    ) -> None:
        super().__init__(inner, chain, fallback, pricing)
        self.id = inner.id
        self.cache = cache

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int, ctx: RunContext
    ) -> list[RetrievedChunk]:
        """Rerank through the cache and the chain, traced and priced."""
        caching = self.cache is not None and self.cache.enabled and ctx.cache.enabled("shared")
        key = rerank_key(self.id, query, [c.chunk.id for c in candidates]) if caching else None
        if key is not None:
            hit = await self._from_cache(key, candidates, top_k, ctx)
            if hit is not None:
                return hit

        attributes: dict[str, JsonValue] = {
            "gen_ai.operation.name": "rerank",
            "gen_ai.system": _provider(self.id),
            "gen_ai.request.model": self.id,
            "hardpoint.rerank.candidates": len(candidates),
            "hardpoint.rerank.top_k": top_k,
        }
        async with ctx.tracer.span("hardpoint.rerank", **attributes):
            fallback = self.fallback
            alternative = (
                (lambda: fallback.rerank(query, candidates, top_k, ctx))
                if fallback is not None
                else None
            )
            try:
                ranked: list[RetrievedChunk] = await self._chain_for(alternative).run(
                    f"{self.id}.rerank",
                    lambda: self.inner.rerank(query, candidates, top_k, ctx),
                    ctx,
                )
            except Exception as exc:
                ctx.metrics.increment(
                    "hardpoint.errors", component=self.id, error=type(exc).__name__
                )
                raise
        # The port returns chunks, not usage, so the call is accounted here, under
        # the component's own name.
        usage = self._priced(StepUsage(calls=1), self.id)
        self._meter(ctx, self.id, usage)
        ctx.usage.record(f"rerank:{self.id}", calls=1, cost_usd=usage.cost_usd)

        if key is not None and self.cache is not None:
            entry = [[item.chunk.id, item.score] for item in ranked]
            await ctx.cache.set(key, json.dumps(entry).encode(), ttl_s=self.cache.ttl_s)
        return ranked

    async def _from_cache(
        self, key: str, candidates: Sequence[RetrievedChunk], top_k: int, ctx: RunContext
    ) -> list[RetrievedChunk] | None:
        cached = await ctx.cache.get(key)
        entries = json.loads(cached) if cached is not None else None
        usable = entries is not None and (len(entries) >= top_k or len(entries) >= len(candidates))
        ctx.metrics.increment(
            "hardpoint.cache.lookups", layer="rerank", result="hit" if usable else "miss"
        )
        if not usable or entries is None:
            return None
        by_id = {candidate.chunk.id: candidate for candidate in candidates}
        return [
            by_id[chunk_id].model_copy(update={"score": score, "rank": rank})
            for rank, (chunk_id, score) in enumerate(entries[:top_k])
            if chunk_id in by_id
        ]

    def __repr__(self) -> str:
        """Render the inner reranker and the chain."""
        return f"PolicyReranker({self.inner!r}, policies={self.describe_policies()!r})"
