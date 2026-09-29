"""Record and replay provider interactions, so an eval suite runs offline.

ARCHITECTURE.md §20.2 and §23: a recording lets a suite be replayed in a locked-
down CI environment with zero provider calls, deterministically. Wrap the
components, run once in ``record`` mode against the real providers, commit the
cassette, and run in ``replay`` mode from then on.

Each interaction is keyed by a stable hash of the request -- the model, the
messages and every parameter -- so a changed prompt or a changed question misses
rather than replaying an answer to a different request. A miss in ``replay``
mode raises, naming how to re-record, instead of quietly calling the provider.

Three ports are covered: the language model (``generate`` and ``stream``), the
embedding model, and index queries -- the last so a suite whose index is a
remote service replays without it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hardpoint.core.errors import ContractError
from hardpoint.core.ids import stable_hash
from hardpoint.core.ports import (
    EmbedResult,
    GenerationDelta,
    GenerationResult,
    ScoredRecord,
)
from hardpoint.core.types import JsonValue

if TYPE_CHECKING:
    from hardpoint.core.capabilities import IndexCapabilities, ModelCapabilities
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import (
        DeleteReport,
        EmbeddingModel,
        EmbedKind,
        GenerationRequest,
        IndexInfo,
        IndexRecord,
        IndexSpec,
        LanguageModel,
        Message,
        MetadataFilter,
        UpsertReport,
        VectorIndex,
        VectorQuery,
    )

__all__ = [
    "Cassette",
    "CassetteEmbeddingModel",
    "CassetteLanguageModel",
    "CassetteMissError",
    "CassetteMode",
    "CassetteVectorIndex",
    "wrap_resources",
]

CassetteMode = Literal["record", "replay"]


class CassetteMissError(ContractError):
    """A request in replay mode that the cassette has no recording of."""

    default_code = "testing.cassette_miss"


class Cassette:
    """A JSON file of recorded responses, keyed by request hash.

    Args:
        path: The cassette file.
        mode: ``record`` calls through and stores; ``replay`` only reads.
    """

    def __init__(self, path: str | Path, mode: CassetteMode) -> None:
        self.path = Path(path)
        self.mode: CassetteMode = mode
        self.entries: dict[str, JsonValue] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        )
        self.hits = 0
        self.recorded = 0

    def key(self, operation: str, payload: JsonValue) -> str:
        """The stable key for one request."""
        return f"{operation}:{stable_hash(payload)[:32]}"

    def lookup(self, key: str) -> JsonValue:
        """Return a recording, raising in replay mode when there is none.

        Raises:
            CassetteMissError: In replay mode, for an unrecorded request.
        """
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        raise CassetteMissError(
            f"The cassette {self.path} has no recording for {key}: the request changed "
            f"since it was recorded, or it was never recorded.",
            component="cassette",
            remedy=(
                "Re-record against the real providers: "
                "`hardpoint eval run --suite <suite> --cassettes record`, then commit "
                "the cassette."
            ),
        )

    def store(self, key: str, value: JsonValue) -> None:
        """Keep a recording, to be written by :meth:`save`."""
        self.entries[key] = value
        self.recorded += 1

    def save(self) -> Path:
        """Write the cassette, sorted so a re-recording diffs cleanly."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.path

    def __repr__(self) -> str:
        """Render the file, mode and size."""
        return f"Cassette({str(self.path)!r}, mode={self.mode!r}, entries={len(self.entries)})"


class CassetteLanguageModel:
    """A ``LanguageModel`` that records or replays ``generate`` and ``stream``."""

    def __init__(self, inner: LanguageModel, cassette: Cassette) -> None:
        self.inner = inner
        self.cassette = cassette
        self.id = inner.id

    def _key(self, operation: str, req: GenerationRequest) -> str:
        return self.cassette.key(
            operation, {"model": self.id, "request": req.model_dump(mode="json")}
        )

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Replay the recorded result, or call through and record it."""
        key = self._key("generate", req)
        if self.cassette.mode == "replay":
            return GenerationResult.model_validate(self.cassette.lookup(key))
        result = await self.inner.generate(req, ctx)
        self.cassette.store(key, result.model_dump(mode="json"))
        return result

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Replay the recorded deltas, or stream through and record them."""
        key = self._key("stream", req)

        async def deltas() -> AsyncIterator[GenerationDelta]:
            if self.cassette.mode == "replay":
                recorded = self.cassette.lookup(key)
                for item in recorded if isinstance(recorded, list) else []:
                    yield GenerationDelta.model_validate(item)
                return
            captured: list[JsonValue] = []
            async for delta in self.inner.stream(req, ctx):
                captured.append(delta.model_dump(mode="json"))
                yield delta
            self.cassette.store(key, captured)

        return deltas()

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Delegate: tokenizers are local, not provider calls."""
        return await self.inner.count_tokens(messages)

    def capabilities(self) -> ModelCapabilities:
        """Delegate."""
        return self.inner.capabilities()

    async def aclose(self) -> None:
        """Close the inner model."""
        closer = getattr(self.inner, "aclose", None)
        if closer is not None:
            await closer()


class CassetteEmbeddingModel:
    """An ``EmbeddingModel`` that records or replays ``embed``."""

    def __init__(self, inner: EmbeddingModel, cassette: Cassette) -> None:
        self.inner = inner
        self.cassette = cassette
        self.id = inner.id
        self.dimensions = inner.dimensions

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Replay the recorded vectors, or embed through and record them."""
        key = self.cassette.key("embed", {"model": self.id, "kind": kind, "texts": list(texts)})
        if self.cassette.mode == "replay":
            return EmbedResult.model_validate(self.cassette.lookup(key))
        result = await self.inner.embed(texts, kind, ctx)
        self.cassette.store(key, result.model_dump(mode="json"))
        return result

    async def aclose(self) -> None:
        """Close the inner model."""
        closer = getattr(self.inner, "aclose", None)
        if closer is not None:
            await closer()


class CassetteVectorIndex:
    """A ``VectorIndex`` whose queries are recorded or replayed.

    Writes pass straight through: a cassette is for replaying a suite, and a
    suite does not ingest.
    """

    def __init__(self, inner: VectorIndex, cassette: Cassette) -> None:
        self.inner = inner
        self.cassette = cassette
        self.name = inner.name

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Replay the recorded results, or query through and record them."""
        key = self.cassette.key(
            "query", {"index": self.name, "request": req.model_dump(mode="json")}
        )
        if self.cassette.mode == "replay":
            recorded = self.cassette.lookup(key)
            return [
                ScoredRecord.model_validate(item)
                for item in (recorded if isinstance(recorded, list) else [])
            ]
        results = await self.inner.query(req, ctx)
        self.cassette.store(key, [record.model_dump(mode="json") for record in results])
        return results

    async def describe(self) -> IndexInfo:
        """Delegate."""
        return await self.inner.describe()

    async def ensure(self, spec: IndexSpec) -> None:
        """Delegate."""
        await self.inner.ensure(spec)

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Delegate."""
        return await self.inner.upsert(records, ctx)

    async def delete(
        self,
        ctx: RunContext,
        *,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,  # noqa: A002 - name fixed by the port
    ) -> DeleteReport:
        """Delegate."""
        return await self.inner.delete(ctx, ids=ids, filter=filter)

    def supports(self) -> IndexCapabilities:
        """Delegate."""
        return self.inner.supports()

    async def aclose(self) -> None:
        """Close the inner index."""
        closer = getattr(self.inner, "aclose", None)
        if closer is not None:
            await closer()


def wrap_resources(res: Any, cassette: Cassette) -> Any:
    """Put a cassette in front of a ``Resources``' model, embedder and indexes."""
    return res.with_components(
        llm_=CassetteLanguageModel(res.llm_, cassette) if res.llm_ is not None else None,
        embedder_=(
            CassetteEmbeddingModel(res.embedder_, cassette) if res.embedder_ is not None else None
        ),
        indexes={name: CassetteVectorIndex(index, cassette) for name, index in res.indexes.items()},
    )
