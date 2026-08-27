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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hardpoint.core.models import CharSpan, Chunk, ContextBundle, RetrievedChunk, TrustLevel
from hardpoint.core.ports import MetadataFilter, VectorQuery

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import EmbeddingModel, VectorIndex

__all__ = ["Assembled", "FilterFactory", "Retrieved", "VectorRetriever"]

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

        embedded = await self.embedder.embed([query], "query", ctx)
        ctx.usage.record(
            self.name,
            calls=embedded.usage.calls,
            embed_tokens=embedded.usage.embed_tokens,
            cost_usd=embedded.usage.cost_usd,
            estimated=embedded.usage.estimated,
        )

        active_filter = self.filter_from(query, ctx) if self.filter_from else self.filter

        scored = await self.index.query(
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
