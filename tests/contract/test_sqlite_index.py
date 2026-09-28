"""The SQLite vector index: the contract kit, plus what only a file can prove.

The kit runs against an in-memory database. Persistence -- the reason this
index exists, since ``ingest run`` and ``ask`` are separate processes -- needs a
real file and a second instance, so it is tested separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hardpoint.adapters.index.sqlite import SqliteIndexConfig, SqliteVectorIndex, build
from hardpoint.core.errors import ConfigError, ContractError
from hardpoint.core.filters import F
from hardpoint.core.ports import IndexRecord, IndexSpec, VectorQuery
from hardpoint.testing import build_run_context, vector_index_contract

pytestmark = pytest.mark.contract

TestSqliteVectorIndex = vector_index_contract(
    lambda: SqliteVectorIndex(":memory:"),
    cleanup=lambda index: index.close(),  # type: ignore[attr-defined]  # the factory's type
)


def record(record_id: str, vector: tuple[float, ...], **metadata: object) -> IndexRecord:
    return IndexRecord(
        id=record_id, vector=vector, document_id="doc", text=record_id, metadata=dict(metadata)
    )


@pytest.mark.anyio
async def test_records_survive_reopening_the_file(tmp_path: Path) -> None:
    """``ingest run`` writes and exits; ``ask`` is a different process."""
    path = tmp_path / "index.db"
    writer = SqliteVectorIndex(path)
    await writer.ensure(IndexSpec(name="primary", dimensions=2, embed_model="m"))
    await writer.upsert([record("a", (1.0, 0.0)), record("b", (0.0, 1.0))], build_run_context())
    await writer.aclose()

    reader = SqliteVectorIndex(path)
    results = await reader.query(VectorQuery(vector=(1.0, 0.0), top_k=1), build_run_context())
    info = await reader.describe()
    await reader.aclose()

    assert [r.id for r in results] == ["a"]
    assert (info.dimensions, info.count, info.embed_model) == (2, 2, "m")


@pytest.mark.anyio
async def test_a_width_change_is_refused_with_a_remedy(tmp_path: Path) -> None:
    index = SqliteVectorIndex(tmp_path / "index.db")
    await index.ensure(IndexSpec(name="primary", dimensions=2))
    with pytest.raises(ConfigError) as exc_info:
        await index.ensure(IndexSpec(name="primary", dimensions=3))
    await index.aclose()
    assert "2-dimension" in str(exc_info.value)
    assert exc_info.value.remedy


@pytest.mark.anyio
async def test_a_wrong_width_record_is_refused() -> None:
    index = SqliteVectorIndex()
    await index.ensure(IndexSpec(name="primary", dimensions=2))
    with pytest.raises(ContractError):
        await index.upsert([record("a", (1.0, 0.0, 0.0))], build_run_context())
    await index.aclose()


@pytest.mark.anyio
async def test_delete_by_document_id_works_without_it_in_metadata() -> None:
    """Ingestion deletes a document with a ``document_id`` filter."""
    index = SqliteVectorIndex()
    await index.ensure(IndexSpec(name="primary", dimensions=2))
    await index.upsert([record("a", (1.0, 0.0))], build_run_context())

    report = await index.delete(build_run_context(), filter=F.field("document_id").eq("doc"))
    assert report.deleted == 1
    assert await index.query(VectorQuery(vector=(1.0, 0.0)), build_run_context()) == []
    await index.aclose()


def test_the_registry_factory_uses_the_configured_path() -> None:
    built = build(SqliteIndexConfig(path="somewhere.db", name="docs"))
    assert (built.path, built.name) == ("somewhere.db", "docs")
