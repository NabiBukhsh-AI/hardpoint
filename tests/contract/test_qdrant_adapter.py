"""The Qdrant adapter. INSTRUCTIONS.md §6.3.

Two halves, tested differently.

**Filter translation** is pure logic and gets exhaustive coverage, because
ARCHITECTURE.md §9.2 calls the filter tree the single most important detail for
vector store portability. A mistranslation here does not raise: it returns the
wrong rows, and where the filter is a tenant check that is a data leak.

**Transport** runs against a loopback fake Qdrant, so paths, payload shapes and
status mapping are exercised over real sockets without a container.

A binding against a real Qdrant is included and marked ``integration``, so
``pytest -m integration`` runs the full contract kit against a live instance
when ``HARDPOINT_TEST_QDRANT_URL`` is set.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from hardpoint.adapters.index.qdrant import (
    QDRANT_FILTER_OPS,
    QdrantIndex,
    point_id,
    to_qdrant_filter,
)
from hardpoint.core.errors import ConfigError, UnsupportedFilterError
from hardpoint.core.filters import F
from hardpoint.core.ports import IndexRecord, IndexSpec, VectorQuery
from hardpoint.testing import build_run_context, vector_index_contract
from tests.contract.fake_qdrant_server import fake_qdrant_server

pytestmark = pytest.mark.contract


# --------------------------------------------------------------------------- #
# Filter translation                                                          #
# --------------------------------------------------------------------------- #


def test_equality_becomes_a_match() -> None:
    assert to_qdrant_filter(F.field("tenant").eq("acme")) == {
        "key": "tenant",
        "match": {"value": "acme"},
    }


def test_inequality_becomes_a_negated_match() -> None:
    assert to_qdrant_filter(F.field("tenant").ne("acme")) == {
        "must_not": [{"key": "tenant", "match": {"value": "acme"}}]
    }


def test_membership_becomes_match_any() -> None:
    assert to_qdrant_filter(F.field("tenant").in_(["a", "b"])) == {
        "key": "tenant",
        "match": {"any": ["a", "b"]},
    }


def test_non_membership_negates_match_any() -> None:
    assert to_qdrant_filter(F.field("tenant").nin(["a"])) == {
        "must_not": [{"key": "tenant", "match": {"any": ["a"]}}]
    }


@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
def test_ordering_becomes_a_range(operator: str) -> None:
    translated = to_qdrant_filter(getattr(F.field("year"), operator)(2023))
    assert translated == {"key": "year", "range": {operator: 2023}}


def test_existence_becomes_a_negated_is_empty() -> None:
    assert to_qdrant_filter(F.field("public").exists()) == {
        "must_not": [{"is_empty": {"key": "public"}}]
    }


def test_conjunction_becomes_must() -> None:
    translated = to_qdrant_filter(F.field("a").eq(1) & F.field("b").eq(2))
    assert list(translated) == ["must"]
    assert len(translated["must"]) == 2


def test_disjunction_becomes_should() -> None:
    translated = to_qdrant_filter(F.field("a").eq(1) | F.field("b").eq(2))
    assert list(translated) == ["should"]
    assert len(translated["should"]) == 2


def test_negation_becomes_must_not() -> None:
    assert to_qdrant_filter(~F.field("a").eq(1)) == {
        "must_not": [{"key": "a", "match": {"value": 1}}]
    }


def test_nesting_is_preserved() -> None:
    """A flattened conjunction inside a disjunction changes what matches."""
    translated = to_qdrant_filter((F.field("a").eq(1) & F.field("b").eq(2)) | F.field("c").eq(3))
    assert list(translated) == ["should"]
    assert "must" in translated["should"][0]


def test_contains_is_refused_rather_than_half_translated() -> None:
    """**Under-declaring is safe; over-declaring returns wrong answers silently.**

    hardpoint's ``contains`` covers list membership *and* string substring.
    Qdrant can do the first without a text index and not the second, so
    translating it would be partly right -- and a filter that silently matched
    nothing on string fields is the failure the capability system exists to
    prevent.
    """
    with pytest.raises(UnsupportedFilterError) as exc_info:
        to_qdrant_filter(F.field("tags").contains("urgent"))

    error = exc_info.value
    assert error.operator == "contains"
    assert error.backend == "qdrant"
    assert error.remedy is not None
    assert "text index" in error.remedy


def test_the_declared_operator_set_excludes_contains() -> None:
    """The declaration and the translator must agree, or the kit would skip it."""
    assert "contains" not in QDRANT_FILTER_OPS
    assert {"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "and", "or", "not", "exists"} == set(
        QDRANT_FILTER_OPS
    )


# --------------------------------------------------------------------------- #
# Point ids                                                                   #
# --------------------------------------------------------------------------- #


def test_point_ids_are_deterministic() -> None:
    """A random mapping would break idempotent re-ingestion.

    Qdrant requires an integer or UUID id, so chunk ids have to be mapped -- but
    the mapping must be a function of the chunk id and nothing else.
    """
    assert point_id("chk_abc") == point_id("chk_abc")
    assert point_id("chk_abc") != point_id("chk_abd")


def test_point_ids_are_valid_uuids() -> None:
    import uuid

    assert uuid.UUID(point_id("chk_abc"))


# --------------------------------------------------------------------------- #
# Transport, against a loopback fake                                          #
# --------------------------------------------------------------------------- #


def index_for(base_url: str) -> QdrantIndex:
    return QdrantIndex("docs", url=base_url)


@pytest.mark.anyio
async def test_ensure_creates_a_missing_collection() -> None:
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=8, metric="cosine"))
        await index.aclose()

    created = [
        r for r in state.requests if r["method"] == "PUT" and r["path"] == "/collections/docs"
    ]
    assert created, "the collection must have been created"
    assert created[0]["body"]["vectors"] == {"size": 8, "distance": "Cosine"}


@pytest.mark.anyio
async def test_ensure_is_idempotent_when_the_collection_matches() -> None:
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        spec = IndexSpec(name="docs", dimensions=8)
        await index.ensure(spec)
        await index.ensure(spec)
        await index.aclose()

    creates = [
        r for r in state.requests if r["method"] == "PUT" and r["path"] == "/collections/docs"
    ]
    assert len(creates) == 1, "an existing, matching collection must not be recreated"


@pytest.mark.anyio
async def test_ensure_refuses_a_width_change_with_a_remedy() -> None:
    """Writing the wrong width either errors per record or corrupts silently."""
    with fake_qdrant_server() as (base_url, _state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=8))

        with pytest.raises(ConfigError) as exc_info:
            await index.ensure(IndexSpec(name="docs", dimensions=16))
        await index.aclose()

    assert "8-dimension" in str(exc_info.value)
    assert exc_info.value.remedy is not None
    assert "re-ingest" in exc_info.value.remedy


@pytest.mark.anyio
async def test_upsert_sends_uuid_ids_and_keeps_the_original() -> None:
    """The original chunk id must survive the round trip, or nothing lines up."""
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        await index.upsert(
            [
                IndexRecord(
                    id="chk_original",
                    vector=(1.0, 0.0),
                    document_id="doc_1",
                    text="body",
                    metadata={"folder": "policies"},
                )
            ],
            build_run_context(),
        )

        results = await index.query(VectorQuery(vector=(1.0, 0.0), top_k=5), build_run_context())
        await index.aclose()

    upserts = [r for r in state.requests if r["path"] == "/collections/docs/points"]
    point = upserts[0]["body"]["points"][0]
    assert point["id"] == point_id("chk_original")
    assert point["payload"]["_hardpoint_id"] == "chk_original"

    assert [r.id for r in results] == ["chk_original"]
    assert results[0].document_id == "doc_1"
    assert results[0].text == "body"
    assert results[0].metadata == {"folder": "policies"}


@pytest.mark.anyio
async def test_query_sends_the_translated_filter() -> None:
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        await index.query(
            VectorQuery(vector=(1.0, 0.0), top_k=3, filter=F.field("folder").eq("policies")),
            build_run_context(),
        )
        await index.aclose()

    search = next(r for r in state.requests if r["path"].endswith("/points/search"))
    assert search["body"]["filter"] == {"key": "folder", "match": {"value": "policies"}}
    assert search["body"]["limit"] == 3


@pytest.mark.anyio
async def test_an_unsupported_filter_raises_before_any_request() -> None:
    """Named at query construction, not surfaced as an empty result set."""
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        before = len(state.requests)

        with pytest.raises(UnsupportedFilterError):
            await index.query(
                VectorQuery(vector=(1.0, 0.0), filter=F.field("tags").contains("x")),
                build_run_context(),
            )
        await index.aclose()

    assert len(state.requests) == before, "no request may be sent for an unexpressible filter"


@pytest.mark.anyio
async def test_delete_by_id_maps_to_point_ids() -> None:
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        await index.delete(build_run_context(), ids=["chk_a", "chk_b"])
        await index.aclose()

    deletes = [r for r in state.requests if r["path"].endswith("/points/delete")]
    assert deletes[0]["body"]["points"] == [point_id("chk_a"), point_id("chk_b")]


@pytest.mark.anyio
async def test_delete_by_filter_sends_a_translated_filter() -> None:
    """How ingestion removes a document's chunks."""
    with fake_qdrant_server() as (base_url, state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        await index.delete(build_run_context(), filter=F.field("document_id").eq("doc_1"))
        await index.aclose()

    deletes = [r for r in state.requests if r["path"].endswith("/points/delete")]
    assert deletes[0]["body"]["filter"] == {"key": "document_id", "match": {"value": "doc_1"}}


@pytest.mark.anyio
async def test_delete_with_neither_argument_is_refused() -> None:
    with fake_qdrant_server() as (base_url, _state):
        index = index_for(base_url)
        with pytest.raises(ValueError, match="Refusing to delete"):
            await index.delete(build_run_context())
        await index.aclose()


@pytest.mark.anyio
async def test_delete_reports_unknown_rather_than_zero() -> None:
    """Qdrant reports a status, not a count.

    A zero would read as "nothing matched", which is a different and much more
    alarming fact than "the backend did not say".
    """
    with fake_qdrant_server() as (base_url, _state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        report = await index.delete(build_run_context(), ids=["chk_a"])
        await index.aclose()

    assert report.deleted is None


@pytest.mark.anyio
async def test_describe_reports_the_collections_shape() -> None:
    with fake_qdrant_server() as (base_url, _state):
        index = index_for(base_url)
        await index.ensure(IndexSpec(name="docs", dimensions=8, metric="dot"))
        info = await index.describe()
        await index.aclose()

    assert info.dimensions == 8
    assert info.metric == "dot"
    assert info.epoch == 0, "the manifest owns the epoch, not the backend"


@pytest.mark.anyio
async def test_an_api_key_is_sent_as_qdrants_own_header() -> None:
    """Qdrant uses ``api-key``, not a bearer token."""
    with fake_qdrant_server() as (base_url, state):
        index = QdrantIndex("docs", url=base_url, api_key="secret")
        await index.ensure(IndexSpec(name="docs", dimensions=2))
        await index.aclose()

    assert state.api_keys[-1] == "secret"


@pytest.mark.anyio
async def test_an_error_status_is_mapped_into_the_taxonomy() -> None:
    from hardpoint.core.errors import AuthError

    with fake_qdrant_server() as (base_url, state):
        state.status = 403
        index = index_for(base_url)

        with pytest.raises(AuthError):
            await index.query(VectorQuery(vector=(1.0, 0.0)), build_run_context())
        await index.aclose()


def test_capabilities_are_declared_honestly() -> None:
    index = QdrantIndex("docs")
    capabilities = index.supports()

    assert capabilities.filter_ops == QDRANT_FILTER_OPS
    assert capabilities.supports_delete_by_filter is True
    assert capabilities.supports_namespaces is False, (
        "Qdrant's equivalent is a separate collection; a payload field would give "
        "isolation only as strong as a filter nobody is obliged to apply"
    )
    assert capabilities.delete_consistency == "consistent"


# --------------------------------------------------------------------------- #
# The full contract kit, against a real Qdrant                                #
# --------------------------------------------------------------------------- #

_QDRANT_URL = os.environ.get("HARDPOINT_TEST_QDRANT_URL")

if _QDRANT_URL:  # pragma: no cover - runs only with a live instance

    def _live_index() -> Any:
        import uuid as _uuid

        return QdrantIndex(f"hardpoint_test_{_uuid.uuid4().hex[:8]}", url=_QDRANT_URL)

    TestQdrantIndexLive = pytest.mark.integration(vector_index_contract(_live_index))
    """The whole kit against a live Qdrant.

    Excluded by default (INSTRUCTIONS.md §12.2); run with
    ``HARDPOINT_TEST_QDRANT_URL=http://localhost:6333 pytest -m integration``.
    """
