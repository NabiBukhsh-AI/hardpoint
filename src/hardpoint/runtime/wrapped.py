"""Ports wrapped in their configured policies.

ADR-004: policies are decorators applied to a port implementation at
construction time, so an adapter stays transport-only and retry behaviour lives
in exactly one place (``runtime/policies.py``). This module is the application
of that decision: one thin proxy per port, satisfying the same Protocol, that
routes each call through the component's :class:`PolicyChain`.

A proxy adds no behaviour of its own beyond the chain and, when configured, a
fallback component of the same kind. It is what ``components list --resolved``
describes, so "is retry on for the embedder" has a printed answer.

## Streaming is not retried

A stream that failed after delivering tokens cannot be retried without
delivering them twice, so ``stream`` passes straight through. The deadline still
applies, because the caller's cancel scope does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hardpoint.runtime.policies import Fallback, PolicyChain

if TYPE_CHECKING:
    from hardpoint.core.capabilities import IndexCapabilities, ModelCapabilities
    from hardpoint.core.context import RunContext
    from hardpoint.core.models import RetrievedChunk
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

__all__ = ["PolicyEmbeddingModel", "PolicyLanguageModel", "PolicyReranker", "PolicyVectorIndex"]


class _Wrapped:
    """Shared plumbing: the chain, the optional fallback, and closing the inner component."""

    def __init__(self, inner: Any, chain: PolicyChain, fallback: Any | None) -> None:
        self.inner = inner
        self.chain = chain
        self.fallback = fallback

    def _chain_for(self, alternative: Any | None) -> PolicyChain:
        """Return the chain, with a per-call fallback bound to this call's arguments.

        The chain's stateful members -- breaker, rate limiter -- are shared by
        reference, so copying the frozen chain per call keeps their memory.
        """
        if alternative is None:
            return self.chain
        return replace(self.chain, fallback=Fallback(alternative))

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


class PolicyLanguageModel(_Wrapped):
    """A ``LanguageModel`` whose calls go through a policy chain."""

    def __init__(
        self, inner: LanguageModel, chain: PolicyChain, fallback: LanguageModel | None = None
    ) -> None:
        super().__init__(inner, chain, fallback)
        self.id = inner.id

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Generate through the chain, falling back when configured."""
        fallback = self.fallback
        alternative = (lambda: fallback.generate(req, ctx)) if fallback is not None else None
        result: GenerationResult = await self._chain_for(alternative).run(
            f"{self.id}.generate", lambda: self.inner.generate(req, ctx), ctx
        )
        return result

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Stream without retry. See the module docstring."""
        stream: AsyncIterator[GenerationDelta] = self.inner.stream(req, ctx)
        return stream

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


class PolicyEmbeddingModel(_Wrapped):
    """An ``EmbeddingModel`` whose calls go through a policy chain.

    A fallback embedder is only valid when it produces vectors comparable with
    the index's (ARCHITECTURE.md §18.2), so its width is checked here.
    """

    def __init__(
        self, inner: EmbeddingModel, chain: PolicyChain, fallback: EmbeddingModel | None = None
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
        super().__init__(inner, chain, fallback)
        self.id = inner.id
        self.dimensions = inner.dimensions

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Embed through the chain."""
        fallback = self.fallback
        alternative = (lambda: fallback.embed(texts, kind, ctx)) if fallback is not None else None
        result: EmbedResult = await self._chain_for(alternative).run(
            f"{self.id}.embed", lambda: self.inner.embed(texts, kind, ctx), ctx
        )
        return result

    def __repr__(self) -> str:
        """Render the inner model and the chain."""
        return f"PolicyEmbeddingModel({self.inner!r}, policies={self.describe_policies()!r})"


class PolicyVectorIndex(_Wrapped):
    """A ``VectorIndex`` whose calls go through a policy chain.

    No fallback: falling back to a different index would answer from different
    data, which is a correctness failure dressed as availability.
    """

    def __init__(self, inner: VectorIndex, chain: PolicyChain) -> None:
        super().__init__(inner, chain, None)
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
        report: UpsertReport = await self.chain.run(
            f"{self.name}.upsert", lambda: self.inner.upsert(records, ctx), ctx
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
        report: DeleteReport = await self.chain.run(
            f"{self.name}.delete", lambda: self.inner.delete(ctx, ids=ids, filter=filter), ctx
        )
        return report

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Query through the chain."""
        results: list[ScoredRecord] = await self.chain.run(
            f"{self.name}.query", lambda: self.inner.query(req, ctx), ctx
        )
        return results

    def supports(self) -> IndexCapabilities:
        """Declare the inner index's capabilities."""
        declared: IndexCapabilities = self.inner.supports()
        return declared

    def __repr__(self) -> str:
        """Render the inner index and the chain."""
        return f"PolicyVectorIndex({self.inner!r}, policies={self.describe_policies()!r})"


class PolicyReranker(_Wrapped):
    """A ``Reranker`` whose calls go through a policy chain."""

    def __init__(
        self, inner: Reranker, chain: PolicyChain, fallback: Reranker | None = None
    ) -> None:
        super().__init__(inner, chain, fallback)
        self.id = inner.id

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int, ctx: RunContext
    ) -> list[RetrievedChunk]:
        """Rerank through the chain."""
        fallback = self.fallback
        alternative = (
            (lambda: fallback.rerank(query, candidates, top_k, ctx))
            if fallback is not None
            else None
        )
        ranked: list[RetrievedChunk] = await self._chain_for(alternative).run(
            f"{self.id}.rerank", lambda: self.inner.rerank(query, candidates, top_k, ctx), ctx
        )
        return ranked

    def __repr__(self) -> str:
        """Render the inner reranker and the chain."""
        return f"PolicyReranker({self.inner!r}, policies={self.describe_policies()!r})"
