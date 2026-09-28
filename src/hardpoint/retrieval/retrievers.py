"""Retrieval steps and the values that flow between them.

Implements the M1 subset of INSTRUCTIONS.md §6.4.

## Retrieval is a composition, not a port

There is deliberately no ``Retriever`` provider port (ARCHITECTURE.md §9.3). A
retriever is embed-then-query-then-filter, and making it a port would invite
adapters that hide the entire retrieval strategy behind a vendor name -- which is
precisely the logic a user must be able to read and change.

So this is an ordinary ``Step``, built from two ports the user can swap
independently.

## Why the query travels with the results

``Generate`` needs the question, and ``ContextAssembler`` needs it to order and
render. The alternative would be stashing it on ``RunContext.extras``, which the
library must never read (INSTRUCTIONS.md §5.4), so it is carried explicitly in
the value each step passes on. Two small frozen types, and the pipeline stays
typed end to end.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hardpoint.core.cache_keys import retrieval_key
from hardpoint.core.models import CharSpan, Chunk, ContextBundle, RetrievedChunk, TrustLevel
from hardpoint.core.ports import MetadataFilter, ScoredRecord, VectorQuery
from hardpoint.core.types import JsonValue

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import EmbeddingModel, VectorIndex

__all__ = ["Assembled", "EpochReader", "FilterFactory", "Retrieved", "VectorRetriever"]

EpochReader = Callable[[], Awaitable[int]]
"""Returns an index's current epoch, which every retrieval cache key includes."""

FilterFactory = Callable[[str, "RunContext"], "MetadataFilter | None"]
"""Builds a metadata filter from the query and the run context.

This is the seam an ACL-aware pipeline uses: the filter is constructed in the
service layer from the *authenticated principal*, never from the query text, and
passed in here. Deriving it from anything the user typed is the RAG
vulnerability that actually gets exploited (ARCHITECTURE.md §27.2).
"""


@dataclass(frozen=True)
class Retrieved:
    """What retrieval produced, with the query that produced it.

    Args:
        query: The query, carried forward for assembly and generation.
        chunks: The results, in the order the retriever ranked them.
    """

    query: str
    chunks: tuple[RetrievedChunk, ...] = ()

    def __repr__(self) -> str:
        """Render the count rather than the chunks."""
        return f"Retrieved(query={self.query[:40]!r}, chunks={len(self.chunks)})"


@dataclass(frozen=True)
class Assembled:
    """Assembled context, with the query it answers.

    Args:
        query: The query.
        context: What will be rendered into the prompt.
    """

    query: str
    context: ContextBundle

    def __repr__(self) -> str:
        """Render the item and drop counts."""
        return (
            f"Assembled(query={self.query[:40]!r}, items={len(self.context.items)}, "
            f"dropped={len(self.context.dropped)})"
        )


class VectorRetriever:
    """Embeds the query and searches one index.

    Args:
        index: Where to search.
        embedder: What embeds the query. Embedded as ``query``, not
            ``document``: asymmetric models exist, and getting the kind wrong
            degrades recall with no error and no symptom other than worse
            answers.
        top_k: How many results to ask for.
        filter: A fixed metadata filter.
        filter_from: Builds a filter per request. Takes precedence over
            ``filter`` when both are given.
        namespace: Index namespace to search.
        min_score: Backend-side score floor.
        epoch: Reads the index's current epoch. When given, and the run has a
            shared cache, results are cached under
            ``retr:{index}:{epoch}:{model_id}:{sha256(query + filter + params)}``
            (ARCHITECTURE.md §22.2) -- so an ingestion run, which bumps the
            epoch, invalidates every cached retrieval built on the old index.
            ``Resources.epoch_reader`` supplies one.
        cache_ttl_s: How long a cached retrieval lives.
        name: The step's name, used for usage attribution and span labelling.
    """

    def __init__(
        self,
        index: VectorIndex,
        embedder: EmbeddingModel,
        *,
        top_k: int = 20,
        filter: MetadataFilter | None = None,  # noqa: A002 - reads as the query parameter it is
        filter_from: FilterFactory | None = None,
        namespace: str | None = None,
        min_score: float | None = None,
        epoch: EpochReader | None = None,
        cache_ttl_s: int | None = 300,
        name: str = "retrieve",
    ) -> None:
        self.name = name
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.filter = filter
        self.filter_from = filter_from
        self.namespace = namespace
        self.min_score = min_score
        self.epoch = epoch
        self.cache_ttl_s = cache_ttl_s

    async def __call__(self, data: str | Retrieved, ctx: RunContext) -> Retrieved:
        """Retrieve chunks for a query.

        Accepts a bare query string, so a pipeline can start with one, or a
        :class:`Retrieved` from an earlier step, whose chunks are replaced.

        Args:
            data: The query, or a previous retrieval whose query is reused.
            ctx: The run context.

        Returns:
            The results, ranked by the index.

        Raises:
            UnsupportedFilterError: If the filter uses an operator this index
                cannot express. Raised at query construction, before any network
                call, so the failure names the operator rather than surfacing as
                an empty result set.
        """
        query = data if isinstance(data, str) else data.query
        active_filter = self.filter_from(query, ctx) if self.filter_from else self.filter

        key = await self._cache_key(query, active_filter, ctx)
        scored = await self._cached(key, ctx) if key is not None else None
        if scored is None:
            scored = await self._search(query, active_filter, ctx)
            if key is not None:
                payload = json.dumps([record.model_dump(mode="json") for record in scored])
                await ctx.cache.set(key, payload.encode("utf-8"), ttl_s=self.cache_ttl_s)

        return Retrieved(
            query=query,
            chunks=tuple(
                RetrievedChunk(
                    chunk=_chunk_from(record),
                    score=record.score,
                    rank=rank,
                    retriever=self.name,
                    raw_scores={self.name: record.score},
                )
                for rank, record in enumerate(scored)
            ),
        )

    async def _search(
        self, query: str, active_filter: MetadataFilter | None, ctx: RunContext
    ) -> list[ScoredRecord]:
        embedded = await self.embedder.embed([query], "query", ctx)
        ctx.usage.record(
            self.name,
            calls=embedded.usage.calls,
            embed_tokens=embedded.usage.embed_tokens,
            cost_usd=embedded.usage.cost_usd,
            estimated=embedded.usage.estimated,
        )
        return await self.index.query(
            VectorQuery(
                vector=embedded.vectors[0],
                top_k=self.top_k,
                filter=active_filter,
                namespace=self.namespace,
                min_score=self.min_score,
                include_text=True,
            ),
            ctx,
        )

    async def _cache_key(
        self, query: str, active_filter: MetadataFilter | None, ctx: RunContext
    ) -> str | None:
        if self.epoch is None or not ctx.cache.enabled("shared"):
            return None
        params: dict[str, JsonValue] = {
            "top_k": self.top_k,
            "namespace": self.namespace,
            "min_score": self.min_score,
        }
        return retrieval_key(
            self.index.name,
            await self.epoch(),
            self.embedder.id,
            query=query,
            filter=active_filter,
            params=params,
        )

    @staticmethod
    async def _cached(key: str, ctx: RunContext) -> list[ScoredRecord] | None:
        cached = await ctx.cache.get(key)
        outcome = "hit" if cached is not None else "miss"
        ctx.metrics.increment("hardpoint.cache.lookups", layer="retrieval", result=outcome)
        if cached is None:
            return None
        return [ScoredRecord.model_validate(item) for item in json.loads(cached)]

    def __repr__(self) -> str:
        """Render the index and top_k."""
        return f"VectorRetriever(index={self.index.name!r}, top_k={self.top_k})"


def _chunk_from(record: object) -> Chunk:
    """Rebuild a ``Chunk`` from what the index returned.

    The index stores text and metadata but not the span or the trust level, so
    those are reconstructed. ``trust`` is ``UNTRUSTED`` unconditionally: content
    that has been through an index came from outside, and a retriever is not the
    place that would know otherwise.
    """
    text = getattr(record, "text", None) or ""
    metadata = dict(getattr(record, "metadata", {}) or {})
    return Chunk(
        id=getattr(record, "id", ""),
        document_id=str(getattr(record, "document_id", None) or metadata.get("document_id") or ""),
        index=int(metadata.get("chunk_index", 0) or 0),
        text=text,
        span=CharSpan(start=0, end=len(text)),
        trust=TrustLevel.UNTRUSTED,
        metadata=metadata,
    )
