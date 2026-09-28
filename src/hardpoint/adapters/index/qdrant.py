"""A Qdrant vector index adapter, over the REST API.

Implements the M1 vector index adapter from INSTRUCTIONS.md §6.3. Qdrant rather
than Chroma because its REST API is small enough to speak directly over
``httpx``, which means this adapter needs **no extra** -- the same argument as
the OpenAI-compatible adapters (``docs/decisions/0004``).

## Filter translation is the whole job

ARCHITECTURE.md §9.2 calls the filter tree "the single most important detail for
vector store portability, because 'filters work differently' is what actually
blocks migrations". This module is where that claim is cashed: the closed
expression tree in ``core.filters`` becomes Qdrant's ``must`` / ``should`` /
``must_not`` shape, and anything it cannot express raises rather than being
dropped.

## What this adapter deliberately does *not* declare

``contains``. hardpoint's ``contains`` means "this list holds the value, or this
string holds the substring". Qdrant can do the list half with ``match``, but the
string half needs a full-text index that has to be created in advance and that
this adapter does not create. Declaring ``contains`` would therefore be *partly*
right, which is worse than not declaring it: a filter that silently matched
nothing on string fields is exactly the failure the capability system exists to
prevent.

Under-declaring costs a clear ``UnsupportedFilterError`` naming the operator.
Over-declaring costs wrong answers with nothing raised.

## Point ids

Qdrant requires an unsigned integer or a UUID. Chunk ids are ``chk_`` followed by
24 hex characters, so they are mapped to a deterministic UUID5 and the original
id is kept in the payload. Deterministic, because a random mapping would break
idempotent re-ingestion just as thoroughly as a random chunk id would.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hardpoint.adapters._http import (
    DEFAULT_TIMEOUT_S,
    as_int,
    as_json,
    build_client,
    map_response_error,
    map_transport_error,
)
from hardpoint.core.capabilities import IndexCapabilities
from hardpoint.core.errors import ConfigError, ContractError, UnsupportedFilterError
from hardpoint.core.filters import And, Comparison, Exists, Not, Or, validate_supported
from hardpoint.core.ports import (
    DeleteReport,
    IndexInfo,
    IndexRecord,
    IndexSpec,
    MetadataFilter,
    ScoredRecord,
    UpsertReport,
    VectorQuery,
)
from hardpoint.core.types import JsonValue

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["QDRANT_FILTER_OPS", "QdrantConfig", "QdrantIndex", "build", "to_qdrant_filter"]

QDRANT_FILTER_OPS: Final = frozenset(
    {"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "exists", "and", "or", "not"}
)
"""Every operator this adapter can translate faithfully.

``contains`` is absent on purpose. See the module docstring.
"""

_ID_NAMESPACE: Final = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")
"""Fixed UUID namespace, so a chunk id maps to the same point id everywhere."""

_ID_FIELD: Final = "_hardpoint_id"
"""Payload key holding the original chunk id."""

_DISTANCES: Final = {"cosine": "Cosine", "dot": "Dot", "euclidean": "Euclid"}

_NOT_FOUND: Final = 404


def point_id(chunk_id: str) -> str:
    """Map a chunk id to a deterministic Qdrant point id.

    UUID5 over a fixed namespace, so the same chunk id always produces the same
    point id -- on any machine, in any process. A random mapping would break
    idempotent re-ingestion exactly as thoroughly as a random chunk id would.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def to_qdrant_filter(node: MetadataFilter) -> dict[str, Any]:
    """Translate a hardpoint filter into Qdrant's filter shape.

    Args:
        node: The filter expression.

    Returns:
        A Qdrant filter object.

    Raises:
        UnsupportedFilterError: For an operator this adapter cannot express.
            Raised rather than dropped: a clause that quietly stops filtering is
            how one tenant sees another tenant's documents.
    """
    if isinstance(node, Comparison):
        return _comparison(node)

    if isinstance(node, Exists):
        # Qdrant has `is_empty`, so "exists" is "not empty". A field holding an
        # explicit null counts as empty in Qdrant and as existing in hardpoint;
        # that difference is documented on IndexCapabilities rather than papered
        # over, because papering over it would need a second query.
        return {"must_not": [{"is_empty": {"key": node.field}}]}

    if isinstance(node, And):
        return {"must": [to_qdrant_filter(clause) for clause in node.clauses]}

    if isinstance(node, Or):
        return {"should": [to_qdrant_filter(clause) for clause in node.clauses]}

    if isinstance(node, Not):
        return {"must_not": [to_qdrant_filter(node.clause)]}

    raise UnsupportedFilterError(  # pragma: no cover - the tree is closed
        f"Unknown filter node {type(node).__name__}.",
        operator=type(node).__name__,
        backend="qdrant",
        remedy=(
            "This is a bug in hardpoint: the filter tree grew a node this adapter "
            "does not know about."
        ),
    )


def _comparison(node: Comparison) -> dict[str, Any]:
    """Translate one leaf comparison."""
    key, value = node.field, node.value

    if node.op == "eq":
        return {"key": key, "match": {"value": value}}
    if node.op == "ne":
        return {"must_not": [{"key": key, "match": {"value": value}}]}
    if node.op == "in":
        return {"key": key, "match": {"any": list(value) if isinstance(value, list) else [value]}}
    if node.op == "nin":
        return {
            "must_not": [
                {
                    "key": key,
                    "match": {"any": list(value) if isinstance(value, list) else [value]},
                }
            ]
        }
    if node.op in {"gt", "gte", "lt", "lte"}:
        bound = {"gt": "gt", "gte": "gte", "lt": "lt", "lte": "lte"}[node.op]
        return {"key": key, "range": {bound: value}}

    raise UnsupportedFilterError(
        f"The Qdrant adapter cannot express the {node.op!r} operator faithfully.",
        operator=node.op,
        backend="qdrant",
        supported=QDRANT_FILTER_OPS,
        component="qdrant",
        remedy=(
            "hardpoint's 'contains' means 'this list holds the value, or this string "
            "holds the substring'. Qdrant can do the list half, but the string half "
            "needs a full-text index created in advance. Declaring it would be partly "
            "right, which returns wrong answers silently. Rewrite the filter using "
            "'in' over a list field, or create a text index and use a custom adapter."
        ),
    )


class QdrantIndex:
    """A ``VectorIndex`` backed by Qdrant's REST API.

    Args:
        collection: The collection name. Also this index's ``name``.
        url: Qdrant's base URL.
        api_key: Sent as ``api-key``, which is Qdrant's own header rather than a
            bearer token.
        timeout_s: Transport timeout.
        client: A pre-built client. When given, this adapter does not close it.
        wait: Whether writes block until they are visible to a search. Defaults
            to ``True``, so ``delete_consistency`` can honestly be declared
            ``consistent`` -- ingestion's correctness depends on a delete being
            visible to the next run.
    """

    def __init__(
        self,
        collection: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        wait: bool = True,
    ) -> None:
        self.name = collection
        self.collection = collection
        self._wait = wait
        self._owns_client = client is None
        self._client = client or build_client(
            base_url=url,
            headers={"api-key": api_key} if api_key else None,
            timeout_s=timeout_s,
        )

    def supports(self) -> IndexCapabilities:
        """Declare what this adapter can express.

        ``contains`` is deliberately absent; see the module docstring. Namespaces
        are absent too: Qdrant's equivalent is a separate collection, and
        pretending a payload field is a namespace would give namespace isolation
        that is only as strong as a filter nobody is obliged to apply.
        """
        return IndexCapabilities(
            filter_ops=QDRANT_FILTER_OPS,
            supports_hybrid=False,
            supports_sparse=False,
            supports_namespaces=False,
            supports_delete_by_filter=True,
            delete_consistency="consistent" if self._wait else "eventual",
        )

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller supplied it."""
        if self._owns_client:
            await self._client.aclose()

    # ----------------------------------------------------------------- #
    # Lifecycle                                                         #
    # ----------------------------------------------------------------- #

    async def describe(self) -> IndexInfo:
        """Report the collection's width, metric and size.

        Raises:
            ConfigError: If the collection does not exist. That is a
                configuration problem -- a name that does not match what was
                created -- rather than a provider failure.
        """
        body = await self._request("GET", f"/collections/{self.collection}", operation="describe")
        result = body.get("result")
        if not isinstance(result, dict):
            raise ContractError(
                f"qdrant: describing {self.collection!r} returned no result object.",
                component=self.name,
                remedy="Check the URL points at a Qdrant instance.",
            )

        config = result.get("config", {})
        params = config.get("params", {}) if isinstance(config, dict) else {}
        vectors = params.get("vectors", {}) if isinstance(params, dict) else {}
        size = as_int(vectors.get("size")) if isinstance(vectors, dict) else 0
        distance = str(vectors.get("distance", "Cosine")) if isinstance(vectors, dict) else "Cosine"

        return IndexInfo(
            name=self.name,
            dimensions=size,
            metric=_metric_from(distance),
            count=as_int(result.get("points_count")),
            # Qdrant has no epoch of its own. The manifest owns it, and reporting
            # zero here rather than inventing one keeps the StateStore the single
            # source of truth (ADR-009).
            epoch=0,
        )

    async def ensure(self, spec: IndexSpec) -> None:
        """Create the collection, or verify an existing one matches.

        Raises:
            ConfigError: If the collection exists with different dimensions.
                Writing vectors of the wrong width either errors per record or,
                worse, is silently accepted and makes retrieval meaningless.
        """
        existing = await self._request(
            "GET", f"/collections/{self.collection}", operation="ensure", allow_404=True
        )

        if existing is not None:
            info = await self.describe()
            if info.dimensions and info.dimensions != spec.dimensions:
                raise ConfigError(
                    f"Qdrant collection {self.collection!r} holds "
                    f"{info.dimensions}-dimension vectors, but the configured "
                    f"embedding model produces {spec.dimensions}.",
                    component=self.name,
                    config_path=f"indexes.{self.name}",
                    remedy=(
                        f"Either configure an embedding model with {info.dimensions} "
                        f"dimensions, or create a new collection and re-ingest. A "
                        f"collection cannot hold vectors of two widths."
                    ),
                )
            return

        await self._request(
            "PUT",
            f"/collections/{self.collection}",
            operation="ensure",
            json={
                "vectors": {
                    "size": spec.dimensions,
                    "distance": _DISTANCES.get(spec.metric, "Cosine"),
                }
            },
        )

    # ----------------------------------------------------------------- #
    # Writes                                                            #
    # ----------------------------------------------------------------- #

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Insert or replace points by id. Idempotent."""
        if not records:
            return UpsertReport(upserted=0)

        points = [
            {
                "id": point_id(record.id),
                "vector": list(record.vector),
                "payload": _payload_for(record),
            }
            for record in records
        ]

        await self._request(
            "PUT",
            f"/collections/{self.collection}/points",
            operation="upsert",
            params={"wait": "true" if self._wait else "false"},
            json={"points": points},
        )
        return UpsertReport(upserted=len(records))

    async def delete(
        self,
        ctx: RunContext,
        *,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,  # noqa: A002 - name fixed by the port
    ) -> DeleteReport:
        """Delete by ids, by filter, or both.

        Raises:
            ValueError: If neither is given.
            UnsupportedFilterError: If the filter uses an operator this adapter
                cannot express.
        """
        if ids is None and filter is None:
            raise ValueError(
                "delete() requires `ids`, `filter`, or both. Refusing to delete every "
                "point in the collection because neither was given."
            )

        selectors: list[dict[str, Any]] = []
        if ids:
            selectors.append({"points": [point_id(chunk_id) for chunk_id in ids]})
        if filter is not None:
            validate_supported(filter, QDRANT_FILTER_OPS, "qdrant")
            selectors.append({"filter": to_qdrant_filter(filter)})

        for selector in selectors:
            await self._request(
                "POST",
                f"/collections/{self.collection}/points/delete",
                operation="delete",
                params={"wait": "true" if self._wait else "false"},
                json=selector,
            )

        # Qdrant reports an operation status, not a count. Reporting `None` says
        # "unknown", which is the truth; a zero would read as "nothing matched",
        # which is a different and much more alarming fact.
        return DeleteReport(deleted=None)

    # ----------------------------------------------------------------- #
    # Reads                                                             #
    # ----------------------------------------------------------------- #

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Search, returning results in descending score order.

        Raises:
            UnsupportedFilterError: Raised at query construction, before the
                request, so the failure names the operator rather than surfacing
                as an empty result set.
        """
        payload: dict[str, Any] = {
            "vector": list(req.vector),
            "limit": req.top_k,
            "with_payload": True,
        }
        if req.filter is not None:
            validate_supported(req.filter, QDRANT_FILTER_OPS, "qdrant")
            payload["filter"] = to_qdrant_filter(req.filter)
        if req.min_score is not None:
            payload["score_threshold"] = req.min_score

        body = await self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            operation="query",
            json=payload,
        )
        results = body.get("result")
        if not isinstance(results, list):
            return []

        return [self._scored_from(item, req) for item in results if isinstance(item, dict)]

    @staticmethod
    def _scored_from(item: Mapping[str, JsonValue], req: VectorQuery) -> ScoredRecord:
        """Rebuild a ``ScoredRecord``, restoring the original chunk id."""
        payload = item.get("payload")
        metadata = dict(payload) if isinstance(payload, dict) else {}
        text = metadata.pop("text", None)
        document_id = metadata.pop("document_id", None)
        original = metadata.pop(_ID_FIELD, None)

        return ScoredRecord(
            id=str(original if original is not None else item.get("id", "")),
            score=float(item.get("score") or 0.0),  # type: ignore[arg-type]
            metadata=metadata,
            text=str(text) if text is not None and req.include_text else None,
            document_id=str(document_id) if document_id is not None else None,
        )

    # ----------------------------------------------------------------- #
    # Transport                                                         #
    # ----------------------------------------------------------------- #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow_404: bool = False,
    ) -> Any:
        """Make one request, mapping every failure into the taxonomy.

        Args:
            method: HTTP method.
            path: Path below the base URL.
            operation: What is being attempted, for the error message.
            json: Request body.
            params: Query parameters.
            allow_404: Return ``None`` for a 404 instead of raising. Used by
                ``ensure``, where "does not exist yet" is the normal case rather
                than an error.

        Returns:
            The decoded body, or ``None`` for a tolerated 404.
        """
        try:
            response = await self._client.request(method, path, json=json, params=params)
        except Exception as exc:
            raise map_transport_error(exc, component=self.name, operation=operation) from exc

        if allow_404 and response.status_code == _NOT_FOUND:
            return None

        if response.is_error:
            raise map_response_error(response, component=self.name, operation=operation)

        return as_json(response, component=self.name, operation=operation)

    def __repr__(self) -> str:
        """Render the collection name."""
        return f"QdrantIndex(collection={self.collection!r})"


class QdrantConfig(BaseModel):
    """Configuration for ``type: qdrant``.

    Args:
        collection: The collection name. Defaults to the key the index is
            configured under.
        url: Qdrant's base URL.
        api_key: Sent as ``api-key``.
        timeout_s: Transport timeout.
        wait: Whether writes block until visible to a search.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = "primary"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    wait: bool = True


def build(config: QdrantConfig) -> QdrantIndex:
    """Registry factory for ``type: qdrant``."""
    return QdrantIndex(
        config.collection,
        url=config.url,
        api_key=config.api_key,
        timeout_s=config.timeout_s,
        wait=config.wait,
    )


def _payload_for(record: IndexRecord) -> dict[str, Any]:
    """Build a point's payload, keeping the original chunk id alongside metadata."""
    payload: dict[str, Any] = dict(record.metadata)
    payload[_ID_FIELD] = record.id
    payload["document_id"] = record.document_id
    payload["text"] = record.text
    return payload


def _metric_from(distance: str) -> Literal["cosine", "dot", "euclidean"]:
    """Map Qdrant's distance name back to the port's vocabulary."""
    for name, qdrant in _DISTANCES.items():
        if qdrant.lower() == distance.lower():
            return name  # type: ignore[return-value]  # keys are exactly the literal
    return "cosine"
