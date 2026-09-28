"""The ingestion manifest, backed by SQLite.

Implements the schema in INSTRUCTIONS.md §6.2 verbatim. The manifest is what
makes ingestion incremental, resumable and correct about deletes, and it is
**part of the public contract**: its schema is versioned, and a change to it
needs a migration rather than a fresh file.

## Why per-document atomicity is the whole design

``record_document`` writes a document row and replaces every one of its chunk
rows in a single transaction. That is what step 5g of the algorithm means by
"commit the manifest rows for this document before moving to the next": a crash
at document 4,000 of 5,000 leaves 3,999 documents committed and the 4,000th
untouched, so the next run picks up where it stopped rather than starting over.

A store that batched writes to the end would be faster and would lose the
property entirely.

## Threading

``sqlite3`` is blocking, so every statement runs in a worker thread through
``anyio.to_thread`` (INSTRUCTIONS.md §12.6). One connection is shared, opened
with ``check_same_thread=False`` and guarded by a lock, because worker threads
vary between calls and SQLite objects are not safe to use from several at once.

The lock also serialises writers, which SQLite would do anyway, but doing it in
process turns a lock timeout into waiting rather than into
``database is locked``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import anyio
from pydantic import BaseModel, ConfigDict

from hardpoint.core.errors import ConfigError, IngestionError
from hardpoint.core.ports import ChunkRecord, DocumentRecord, IndexMeta
from hardpoint.core.types import JsonValue, ModelId

__all__ = ["SCHEMA_VERSION", "SqliteStateConfig", "SqliteStateStore", "build", "utc_now"]

SCHEMA_VERSION: Final = 1
"""The manifest schema this release reads and writes.

Part of the public contract. A release that changes the schema increments this
and ships a migration; opening a newer manifest with an older release is an
error rather than a silent misread.
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  source_id TEXT, document_id TEXT, source_uri TEXT, revision TEXT,
  content_hash TEXT, chunk_count INTEGER, indexed_at TEXT, status TEXT,
  PRIMARY KEY (source_id, document_id));

CREATE TABLE IF NOT EXISTS chunks (
  document_id TEXT, chunk_id TEXT, text_hash TEXT, embedded_with TEXT,
  PRIMARY KEY (document_id, chunk_id));

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, source_id TEXT, started_at TEXT, finished_at TEXT,
  status TEXT, report_json TEXT);

CREATE TABLE IF NOT EXISTS index_meta (
  index_name TEXT PRIMARY KEY, epoch INTEGER, dimensions INTEGER, embed_model TEXT);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_runs_source ON runs (source_id, started_at);
"""


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string.

    Timezone-aware and always UTC: a manifest read on a different machine must
    not have to guess which offset a naive timestamp was written in.
    """
    return datetime.now(UTC).isoformat()


class SqliteStateStore:
    """The default ``StateStore``: a single SQLite file.

    Args:
        path: Where the manifest lives. ``":memory:"`` gives an in-process store
            for tests, which is why it is the default in fixtures rather than a
            second fake implementation -- a fake manifest would not exercise the
            SQL that the real one depends on.

    Raises:
        Nothing on construction. The file is opened on
        :meth:`initialise`, so building the object cannot fail on a path
        problem before the caller is ready to handle it.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = anyio.Lock()

    # ----------------------------------------------------------------- #
    # Lifecycle                                                         #
    # ----------------------------------------------------------------- #

    async def initialise(self) -> None:
        """Create the schema if absent, and refuse a manifest from the future.

        Idempotent: called on every ingest run.

        Raises:
            ConfigError: If the manifest was written by a newer release. Reading
                it with an older schema would silently misinterpret columns.
            IngestionError: If the file cannot be opened.
        """
        await anyio.to_thread.run_sync(self._initialise_sync)

    def _initialise_sync(self) -> None:
        connection = self._connect()
        with connection:
            connection.executescript(_SCHEMA)
            row = connection.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
                return

            found = int(row["version"])
            if found > SCHEMA_VERSION:
                raise ConfigError(
                    f"The ingestion manifest at {self.path} uses schema version "
                    f"{found}, but this release reads version {SCHEMA_VERSION}.",
                    config_path="ingestion.state",
                    remedy=(
                        "Upgrade hardpoint to a release that understands this "
                        "manifest, or point the state store at a new file and "
                        "re-ingest from scratch."
                    ),
                )

    def _connect(self) -> sqlite3.Connection:
        """Open the shared connection, or return the one already open."""
        if self._connection is not None:
            return self._connection

        try:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
        except (OSError, sqlite3.Error) as exc:
            raise IngestionError(
                f"Could not open the ingestion manifest at {self.path}: {exc}.",
                remedy=(
                    "Check the directory exists and is writable. The manifest is "
                    "what makes ingestion incremental; without it every run would "
                    "re-embed the whole corpus."
                ),
                cause=exc,
            ) from exc

        connection.row_factory = sqlite3.Row
        # WAL lets a reader (`ingest status`) run while a writer (`ingest run`)
        # holds the file, which is the normal operational shape.
        if self.path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._connection = connection
        return connection

    async def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        async with self._lock:
            if self._connection is not None:
                await anyio.to_thread.run_sync(self._connection.close)
                self._connection = None

    async def _run(self, work: Any) -> Any:
        """Run a blocking unit of database work in a worker thread, serialised."""
        async with self._lock:
            return await anyio.to_thread.run_sync(work)

    # ----------------------------------------------------------------- #
    # Documents and chunks                                              #
    # ----------------------------------------------------------------- #

    async def documents(self, source_id: str) -> dict[str, DocumentRecord]:
        """Return every known document for a source, keyed by document id.

        Includes tombstoned documents, because a document that reappears must be
        recognised as returning rather than as brand new. Callers that want only
        live documents filter on ``status``.
        """

        def work() -> dict[str, DocumentRecord]:
            rows = self._connect().execute(
                "SELECT * FROM documents WHERE source_id = ? ORDER BY document_id",
                (source_id,),
            )
            return {row["document_id"]: _document_from_row(row) for row in rows}

        result: dict[str, DocumentRecord] = await self._run(work)
        return result

    async def chunks(self, document_id: str) -> dict[str, ChunkRecord]:
        """Return every known chunk for a document, keyed by chunk id.

        What step 5d consults to answer "which of these chunks is already
        embedded, with this same model" -- the question the whole incremental
        embedding saving rests on.
        """

        def work() -> dict[str, ChunkRecord]:
            rows = self._connect().execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_id",
                (document_id,),
            )
            return {row["chunk_id"]: _chunk_from_row(row) for row in rows}

        result: dict[str, ChunkRecord] = await self._run(work)
        return result

    async def record_document(
        self, document: DocumentRecord, chunks: Sequence[ChunkRecord]
    ) -> None:
        """Commit one document and its chunks, atomically.

        The chunk rows are *replaced*, not merged: a document whose chunks
        changed must not leave rows for chunks that no longer exist, or the next
        run would believe they were already embedded and skip re-creating them.

        Atomic per document, and committed before the next document begins. That
        is what makes a crashed run resumable.
        """

        def work() -> None:
            connection = self._connect()
            with connection:
                connection.execute(
                    """
                    INSERT INTO documents
                      (source_id, document_id, source_uri, revision, content_hash,
                       chunk_count, indexed_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (source_id, document_id) DO UPDATE SET
                      source_uri = excluded.source_uri,
                      revision = excluded.revision,
                      content_hash = excluded.content_hash,
                      chunk_count = excluded.chunk_count,
                      indexed_at = excluded.indexed_at,
                      status = excluded.status
                    """,
                    (
                        document.source_id,
                        document.document_id,
                        document.source_uri,
                        document.revision,
                        document.content_hash,
                        document.chunk_count,
                        document.indexed_at or utc_now(),
                        document.status,
                    ),
                )
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?", (document.document_id,)
                )
                connection.executemany(
                    """
                    INSERT INTO chunks (document_id, chunk_id, text_hash, embedded_with)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (chunk.document_id, chunk.chunk_id, chunk.text_hash, chunk.embedded_with)
                        for chunk in chunks
                    ],
                )

        await self._run(work)

    async def tombstone(self, source_id: str, document_id: str) -> None:
        """Mark a document deleted and drop its chunk rows.

        The document row is kept as a tombstone; the chunk rows are not, because
        nothing should ever believe those chunks are still embedded. If the
        document reappears, the tombstone identifies it as returning and its
        chunks are re-created from scratch.
        """

        def work() -> None:
            connection = self._connect()
            with connection:
                connection.execute(
                    """
                    UPDATE documents SET status = 'deleted', indexed_at = ?
                    WHERE source_id = ? AND document_id = ?
                    """,
                    (utc_now(), source_id, document_id),
                )
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        await self._run(work)

    # ----------------------------------------------------------------- #
    # Index metadata                                                    #
    # ----------------------------------------------------------------- #

    async def index_epoch(self, index_name: str) -> int:
        """Return an index's current epoch, or ``0`` if it has none yet."""
        meta = await self.index_meta(index_name)
        return meta.epoch if meta else 0

    async def bump_epoch(self, index_name: str) -> int:
        """Increment and return an index's epoch.

        Called once at the end of a successful run. Every downstream cache key
        includes the epoch, so this single statement is what stops a cache
        serving pre-reindex answers indefinitely (ADR-009).
        """

        def work() -> int:
            connection = self._connect()
            with connection:
                connection.execute(
                    """
                    INSERT INTO index_meta (index_name, epoch) VALUES (?, 1)
                    ON CONFLICT (index_name) DO UPDATE SET epoch = epoch + 1
                    """,
                    (index_name,),
                )
                row = connection.execute(
                    "SELECT epoch FROM index_meta WHERE index_name = ?", (index_name,)
                ).fetchone()
                return int(row["epoch"])

        result: int = await self._run(work)
        return result

    async def index_meta(self, index_name: str) -> IndexMeta | None:
        """Return what is recorded about an index, or ``None`` if it is new."""

        def work() -> IndexMeta | None:
            row = (
                self._connect()
                .execute("SELECT * FROM index_meta WHERE index_name = ?", (index_name,))
                .fetchone()
            )
            if row is None:
                return None
            return IndexMeta(
                index_name=row["index_name"],
                epoch=int(row["epoch"] or 0),
                dimensions=row["dimensions"],
                embed_model=row["embed_model"],
            )

        result: IndexMeta | None = await self._run(work)
        return result

    async def record_index_meta(self, meta: IndexMeta) -> None:
        """Record an index's dimensions and embedding model, preserving its epoch."""

        def work() -> None:
            connection = self._connect()
            with connection:
                connection.execute(
                    """
                    INSERT INTO index_meta (index_name, epoch, dimensions, embed_model)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (index_name) DO UPDATE SET
                      dimensions = excluded.dimensions,
                      embed_model = excluded.embed_model
                    """,
                    (meta.index_name, meta.epoch, meta.dimensions, meta.embed_model),
                )

        await self._run(work)

    # ----------------------------------------------------------------- #
    # Runs                                                              #
    # ----------------------------------------------------------------- #

    async def record_run(self, run_id: str, source_id: str, report: JsonValue) -> None:
        """Persist a run report, so ``ingest status`` can show the last run."""

        def work() -> None:
            connection = self._connect()
            payload = json.dumps(report, sort_keys=True)
            status = report.get("status", "unknown") if isinstance(report, dict) else "unknown"
            started = report.get("started_at") if isinstance(report, dict) else None
            finished = report.get("finished_at") if isinstance(report, dict) else None
            with connection:
                connection.execute(
                    """
                    INSERT INTO runs
                      (run_id, source_id, started_at, finished_at, status, report_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id) DO UPDATE SET
                      finished_at = excluded.finished_at,
                      status = excluded.status,
                      report_json = excluded.report_json
                    """,
                    (run_id, source_id, started, finished, status, payload),
                )

        await self._run(work)

    async def last_run(self, source_id: str) -> dict[str, JsonValue] | None:
        """Return the most recent run report for a source, or ``None``.

        What ``hardpoint ingest status`` prints. Not part of the ``StateStore``
        port: reading history back is a CLI convenience, and adding it to the
        port would oblige every future store to implement it.
        """

        def work() -> dict[str, JsonValue] | None:
            row = (
                self._connect()
                .execute(
                    """
                    SELECT report_json FROM runs WHERE source_id = ?
                    ORDER BY started_at DESC LIMIT 1
                    """,
                    (source_id,),
                )
                .fetchone()
            )
            if row is None or not row["report_json"]:
                return None
            decoded: dict[str, JsonValue] = json.loads(row["report_json"])
            return decoded

        result: dict[str, JsonValue] | None = await self._run(work)
        return result

    def __repr__(self) -> str:
        """Render the manifest path."""
        return f"SqliteStateStore(path={self.path!r})"


class SqliteStateConfig(BaseModel):
    """Configuration for ``type: sqlite`` under ``ingestion.state``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = ".hardpoint/state.db"


def build(config: SqliteStateConfig) -> SqliteStateStore:
    """Registry factory for the ``sqlite`` state store."""
    return SqliteStateStore(config.path)


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        source_id=row["source_id"],
        document_id=row["document_id"],
        source_uri=row["source_uri"],
        revision=row["revision"],
        content_hash=row["content_hash"],
        chunk_count=int(row["chunk_count"] or 0),
        indexed_at=row["indexed_at"],
        status=row["status"] or "indexed",
    )


def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        document_id=row["document_id"],
        chunk_id=row["chunk_id"],
        text_hash=row["text_hash"],
        embedded_with=ModelId(row["embedded_with"]),
    )
