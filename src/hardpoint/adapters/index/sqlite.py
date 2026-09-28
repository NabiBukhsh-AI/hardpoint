"""A vector index in one SQLite file, for laptops and CI.

The index the ``rag-minimal`` template starts with (ARCHITECTURE.md §27.1: "runs
on a laptop with no infrastructure"). Standard library only, so it needs no
extra, and it persists between ``hardpoint ingest run`` and ``hardpoint ask``,
which the in-process ``InMemoryVectorIndex`` cannot.

## What it is not

A search engine. Every query scores every record in the namespace in Python.
That is fine for the tens of thousands of chunks a prototype holds and wrong for
millions; the remedy is a real vector database, which is a config change
(``type: qdrant``) because both pass the same contract kit.

## Filters

Evaluated with ``core.filters.matches``, the normative semantics, so this index
declares every operator and honours each one exactly as ``InMemoryVectorIndex``
does. There is no translation step to get wrong.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import anyio
from pydantic import BaseModel, ConfigDict

from hardpoint.core.capabilities import ALL_FILTER_OPS, IndexCapabilities
from hardpoint.core.errors import ConfigError, ContractError
from hardpoint.core.filters import matches, validate_supported
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

__all__ = ["SqliteIndexConfig", "SqliteVectorIndex", "build"]

Metric = Literal["cosine", "dot", "euclidean"]

_NO_NAMESPACE: Final = "\x00"
"""Stored in place of ``None``, so ``""`` stays a namespace distinct from none."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  namespace TEXT NOT NULL, id TEXT NOT NULL, vector TEXT NOT NULL,
  metadata TEXT NOT NULL, document_id TEXT, text TEXT,
  PRIMARY KEY (namespace, id));
CREATE TABLE IF NOT EXISTS index_info (key TEXT PRIMARY KEY, value TEXT);
"""


def _similarity(metric: Metric, left: Sequence[float], right: Sequence[float]) -> float:
    """Score two vectors so that higher is always better."""
    if metric == "dot":
        return sum(a * b for a, b in zip(left, right, strict=True))
    if metric == "euclidean":
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
        return 1.0 / (1.0 + distance)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


class SqliteVectorIndex:
    """A ``VectorIndex`` stored in a single SQLite file.

    Args:
        path: The database file. ``":memory:"`` keeps it in process.
        name: The index name, recorded in the ingestion manifest.

    Raises:
        Nothing on construction; the file is opened on first use.
    """

    def __init__(self, path: str | Path = ":memory:", *, name: str = "primary") -> None:
        self.name = name
        self.path = str(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = anyio.Lock()

    def supports(self) -> IndexCapabilities:
        """Declare every operator: filters are evaluated with the normative semantics."""
        return IndexCapabilities(
            filter_ops=ALL_FILTER_OPS,
            supports_namespaces=True,
            supports_delete_by_filter=True,
            delete_consistency="consistent",
        )

    # ----------------------------------------------------------------- #
    # Plumbing                                                          #
    # ----------------------------------------------------------------- #

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.executescript(_SCHEMA)
            self._connection = connection
        return self._connection

    async def _run(self, work: Any) -> Any:
        """Run blocking database work in a worker thread, one unit at a time."""
        async with self._lock:
            return await anyio.to_thread.run_sync(work)

    async def aclose(self) -> None:
        """Close the file. Safe to call more than once."""
        async with self._lock:
            self.close()

    def close(self) -> None:
        """Close the file from synchronous code, such as a test teardown."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _info(self, connection: sqlite3.Connection) -> dict[str, str]:
        return dict(connection.execute("SELECT key, value FROM index_info").fetchall())

    # ----------------------------------------------------------------- #
    # Lifecycle                                                         #
    # ----------------------------------------------------------------- #

    async def describe(self) -> IndexInfo:
        """Report width, metric, size and the recorded embedding model.

        ``epoch`` is reported as zero: the ingestion manifest owns the epoch
        (ADR-009), and inventing a second one here would give two answers.
        """

        def work() -> IndexInfo:
            connection = self._connect()
            info = self._info(connection)
            (count,) = connection.execute("SELECT COUNT(*) FROM records").fetchone()
            return IndexInfo(
                name=self.name,
                dimensions=int(info.get("dimensions", 0)),
                metric=info.get("metric", "cosine"),  # type: ignore[arg-type]  # written from the Literal
                count=int(count),
                embed_model=info.get("embed_model"),
            )

        result: IndexInfo = await self._run(work)
        return result

    async def ensure(self, spec: IndexSpec) -> None:
        """Record the spec on first use, and refuse a later change of width.

        Raises:
            ConfigError: If the index already holds vectors of another width.
        """

        def work() -> None:
            connection = self._connect()
            info = self._info(connection)
            existing = int(info.get("dimensions", 0))
            if existing and existing != spec.dimensions:
                raise ConfigError(
                    f"Index {self.name!r} ({self.path}) holds {existing}-dimension "
                    f"vectors, but the configured embedding model produces "
                    f"{spec.dimensions}.",
                    component=self.name,
                    config_path=f"indexes.{self.name}",
                    remedy=(
                        f"Configure an embedding model with {existing} dimensions, or "
                        f"delete {self.path} and re-ingest. An index cannot hold "
                        f"vectors of two widths."
                    ),
                )
            with connection:
                updates = {"dimensions": str(spec.dimensions), "metric": spec.metric}
                if spec.embed_model is not None:
                    updates["embed_model"] = spec.embed_model
                connection.executemany(
                    "INSERT OR REPLACE INTO index_info (key, value) VALUES (?, ?)",
                    list(updates.items()),
                )

        await self._run(work)

    # ----------------------------------------------------------------- #
    # Writes                                                            #
    # ----------------------------------------------------------------- #

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Insert or replace records by id. Idempotent.

        Raises:
            ContractError: If a vector's width disagrees with the index.
        """

        def work() -> None:
            connection = self._connect()
            width = int(self._info(connection).get("dimensions", 0))
            for record in records:
                if width and record.vector and len(record.vector) != width:
                    raise ContractError(
                        f"Record {record.id!r} has {len(record.vector)} dimensions, but "
                        f"index {self.name!r} expects {width}.",
                        component=self.name,
                        remedy="`hardpoint doctor` checks embedding and index widths agree.",
                    )
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO records "
                    "(namespace, id, vector, metadata, document_id, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            record.namespace if record.namespace is not None else _NO_NAMESPACE,
                            record.id,
                            json.dumps(list(record.vector)),
                            json.dumps(record.metadata),
                            record.document_id,
                            record.text,
                        )
                        for record in records
                    ],
                )

        await self._run(work)
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
        """
        if ids is None and filter is None:
            raise ValueError(
                "delete() requires `ids`, `filter`, or both. Refusing to delete every "
                "record in the index because neither was given."
            )
        if filter is not None:
            validate_supported(filter, ALL_FILTER_OPS, self.name)

        def work() -> int:
            connection = self._connect()
            doomed: list[tuple[str, str]] = []
            wanted = set(ids or ())
            for namespace, record_id, metadata, document_id in connection.execute(
                "SELECT namespace, id, metadata, document_id FROM records"
            ):
                if record_id in wanted or (
                    filter is not None
                    and matches(filter, _filterable(json.loads(metadata), document_id))
                ):
                    doomed.append((namespace, record_id))
            with connection:
                connection.executemany("DELETE FROM records WHERE namespace = ? AND id = ?", doomed)
            return len(doomed)

        deleted: int = await self._run(work)
        return DeleteReport(deleted=deleted)

    # ----------------------------------------------------------------- #
    # Reads                                                             #
    # ----------------------------------------------------------------- #

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Score every record in the namespace and return the best, highest first."""
        if req.filter is not None:
            validate_supported(req.filter, ALL_FILTER_OPS, self.name)

        def work() -> list[ScoredRecord]:
            connection = self._connect()
            metric: Metric = self._info(connection).get("metric", "cosine")  # type: ignore[assignment]
            namespace = req.namespace if req.namespace is not None else _NO_NAMESPACE
            scored: list[ScoredRecord] = []
            # ponytail: brute-force scan, O(records) per query; switch to qdrant past ~100k chunks.
            for record_id, raw_vector, raw_metadata, document_id, text in connection.execute(
                "SELECT id, vector, metadata, document_id, text FROM records WHERE namespace = ?",
                (namespace,),
            ):
                metadata = json.loads(raw_metadata)
                if req.filter is not None and not matches(
                    req.filter, _filterable(metadata, document_id)
                ):
                    continue
                vector = json.loads(raw_vector)
                value = _similarity(metric, req.vector, vector) if req.vector and vector else 0.0
                if req.min_score is not None and value < req.min_score:
                    continue
                scored.append(
                    ScoredRecord(
                        id=record_id,
                        score=value,
                        metadata=metadata,
                        text=text if req.include_text else None,
                        document_id=document_id,
                    )
                )
            scored.sort(key=lambda item: (-item.score, item.id))
            return scored[: req.top_k]

        results: list[ScoredRecord] = await self._run(work)
        return results

    def __repr__(self) -> str:
        """Render the name and file."""
        return f"SqliteVectorIndex(name={self.name!r}, path={self.path!r})"


def _filterable(metadata: dict[str, JsonValue], document_id: str | None) -> dict[str, JsonValue]:
    """Expose ``document_id`` to filters, which is how ingestion deletes a document."""
    if document_id is not None:
        metadata.setdefault("document_id", document_id)
    return metadata


class SqliteIndexConfig(BaseModel):
    """Configuration for ``type: sqlite`` under ``indexes``.

    Args:
        path: The database file, relative to the project directory.
        name: The index name recorded in the manifest. Defaults to the key the
            index is configured under.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = ".hardpoint/index.db"
    name: str = "primary"


def build(config: SqliteIndexConfig) -> SqliteVectorIndex:
    """Registry factory for ``type: sqlite``."""
    return SqliteVectorIndex(config.path, name=config.name)
