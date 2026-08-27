"""The ``VectorIndex`` conformance suite. INSTRUCTIONS.md §5.8.

An adapter is not "supported" until it passes this (ARCHITECTURE.md §23). That
rule is what makes "swap the vector database" a real claim rather than an
aspirational one.

## Using it

Bind the kit to a factory in your own test file::

    from hardpoint.testing.contracts import vector_index_contract

    TestMyIndex = vector_index_contract(lambda: MyIndex(url="http://localhost"))

**The name must begin with ``Test``.** pytest collects classes by the name they
are bound to in the module namespace, so the lowercase form in the
specification's snippet would be silently skipped -- and a contract suite that
silently does not run is worse than no contract suite. This is the one place
this kit deviates from the written example, and it deviates because the example
would not execute.

If your backend can be constructed in a state where its credentials are
rejected, pass a second factory and the kit adds error-mapping tests::

    TestMyIndex = vector_index_contract(
        lambda: MyIndex(url=URL, api_key=GOOD),
        unauthorised_factory=lambda: MyIndex(url=URL, api_key="wrong"),
    )

Those tests are *not generated* when no factory is given, rather than generated
and skipped. A skipped test in a report reads as coverage; an absent one reads
as absent.

## Why pytest is imported inside the function

``pytest`` is a development dependency, not one of the five base dependencies.
Importing it at module scope would break the bare-environment guarantee that
every ``hardpoint.*`` module imports with base dependencies only
(INSTRUCTIONS.md §3 **[LOCKED]**). It is imported where it is used, which is the
same rule every optional dependency follows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from hardpoint.core.errors import AuthError, MissingDependencyError, UnsupportedFilterError
from hardpoint.core.filters import F, Filter
from hardpoint.core.ports import IndexRecord, IndexSpec, VectorIndex, VectorQuery

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["vector_index_contract"]

DIMENSIONS = 8
"""Vector width the kit uses. Small enough to read in a failure message."""


def _unit(index: int, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    """Return the ``index``-th basis vector.

    Basis vectors, not random ones, so that expected similarity is obvious: a
    vector scores 1.0 against itself and 0.0 against any other. A contract test
    whose expected ordering needs explaining is a test nobody will trust when it
    fails.
    """
    return tuple(1.0 if position == index else 0.0 for position in range(dimensions))


def _records() -> list[IndexRecord]:
    """The fixture corpus, chosen to exercise every filter operator."""
    return [
        IndexRecord(
            id="chk_a",
            vector=_unit(0),
            document_id="doc_1",
            text="alpha document about billing",
            metadata={
                "tenant": "acme",
                "year": 2024,
                "score": 0.9,
                "tags": ["billing", "urgent"],
                "title": "Invoice handling",
                "public": True,
            },
        ),
        IndexRecord(
            id="chk_b",
            vector=_unit(1),
            document_id="doc_1",
            text="beta document about billing",
            metadata={
                "tenant": "acme",
                "year": 2022,
                "score": 0.5,
                "tags": ["billing"],
                "title": "Payment terms",
                "public": False,
            },
        ),
        IndexRecord(
            id="chk_c",
            vector=_unit(2),
            document_id="doc_2",
            text="gamma document about shipping",
            metadata={
                "tenant": "globex",
                "year": 2024,
                "score": 0.7,
                "tags": ["shipping"],
                "title": "Delivery windows",
            },
        ),
    ]


def vector_index_contract(
    factory: Callable[[], VectorIndex],
    *,
    unauthorised_factory: Callable[[], VectorIndex] | None = None,
    cleanup: Callable[[VectorIndex], None] | None = None,
) -> type:
    """Build a pytest class asserting a ``VectorIndex`` implementation conforms.

    Args:
        factory: Returns a fresh, empty index. Called once per test, so tests
            cannot leak state into each other.
        unauthorised_factory: Returns an index whose credentials will be
            rejected. When given, error-mapping tests are added.
        cleanup: Called with the index after each test, for a backend that needs
            a collection dropped.

    Returns:
        A test class. Bind it to a name beginning with ``Test``.
    """
    try:
        import pytest  # noqa: PLC0415 - see the module docstring
    except ModuleNotFoundError as exc:  # pragma: no cover - pytest is a dev dependency
        raise MissingDependencyError(
            "The contract kits build pytest test classes, and pytest is not installed.",
            component="vector_index_contract",
            remedy="pip install pytest",
            cause=exc,
        ) from exc

    from hardpoint.testing.fixtures import build_run_context  # noqa: PLC0415

    class VectorIndexContract:
        """Behaviour every ``VectorIndex`` implementation must exhibit."""

        @pytest.fixture
        def ctx(self) -> RunContext:
            return build_run_context()

        @pytest.fixture
        def index(self) -> Any:
            built = factory()
            try:
                yield built
            finally:
                if cleanup is not None:
                    cleanup(built)

        @pytest.fixture
        async def populated(self, index: VectorIndex, ctx: RunContext) -> Any:
            await index.ensure(IndexSpec(name=index.name, dimensions=DIMENSIONS))
            await index.upsert(_records(), ctx)
            return index

        # -- shape ---------------------------------------------------- #

        def test_satisfies_the_protocol(self, index: VectorIndex) -> None:
            """Structural conformance, before any behavioural assertion."""
            assert isinstance(index, VectorIndex)

        def test_declares_its_capabilities(self, index: VectorIndex) -> None:
            """Capabilities drive filter validation, so they must be answerable."""
            capabilities = index.supports()
            assert capabilities.filter_ops, (
                "an index declaring no filter operators cannot be filtered, which "
                "makes multi-tenant retrieval unsafe"
            )
            assert capabilities.delete_consistency in {"consistent", "eventual"}

        @pytest.mark.anyio
        async def test_describe_reports_dimensions_and_metric(self, index: VectorIndex) -> None:
            await index.ensure(IndexSpec(name=index.name, dimensions=DIMENSIONS))
            info = await index.describe()
            assert info.dimensions == DIMENSIONS
            assert info.metric in {"cosine", "dot", "euclidean"}

        @pytest.mark.anyio
        async def test_ensure_is_idempotent(self, index: VectorIndex) -> None:
            """Called on every startup, so it cannot be a one-shot operation."""
            spec = IndexSpec(name=index.name, dimensions=DIMENSIONS)
            await index.ensure(spec)
            await index.ensure(spec)
            assert (await index.describe()).dimensions == DIMENSIONS

        # -- upsert --------------------------------------------------- #

        @pytest.mark.anyio
        async def test_upsert_then_query_returns_the_record(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=3), ctx)
            assert next(r.id for r in results) == "chk_a"

        @pytest.mark.anyio
        async def test_upsert_is_idempotent(self, populated: VectorIndex, ctx: RunContext) -> None:
            """Upserting identical records twice leaves one logical record.

            This is what makes re-ingestion safe. An index that appended instead
            would double a corpus on every run, and the symptom -- duplicated
            citations -- would take a long time to trace back here.
            """
            await populated.upsert(_records(), ctx)
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=100), ctx)
            assert len(results) == len(_records())
            assert len({r.id for r in results}) == len(results)

        @pytest.mark.anyio
        async def test_upsert_replaces_rather_than_duplicates(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """The same id with new metadata must overwrite, not coexist."""
            updated = _records()[0].model_copy(update={"metadata": {"tenant": "changed"}})
            await populated.upsert([updated], ctx)

            results = await populated.query(
                VectorQuery(vector=_unit(0), top_k=100, filter=F.field("tenant").eq("changed")),
                ctx,
            )
            assert [r.id for r in results] == ["chk_a"]

        # -- query ---------------------------------------------------- #

        @pytest.mark.anyio
        async def test_query_returns_descending_scores(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """The port's ordering contract. Fusion and top-k both rely on it."""
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=100), ctx)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True), scores

        @pytest.mark.anyio
        async def test_top_k_is_respected(self, populated: VectorIndex, ctx: RunContext) -> None:
            requested = 2
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=requested), ctx)
            assert len(results) <= requested

        @pytest.mark.anyio
        async def test_query_on_an_empty_index_returns_nothing(
            self, index: VectorIndex, ctx: RunContext
        ) -> None:
            """Empty retrieval is a state, not an error (ARCHITECTURE.md §6.2)."""
            await index.ensure(IndexSpec(name=index.name, dimensions=DIMENSIONS))
            assert await index.query(VectorQuery(vector=_unit(0), top_k=5), ctx) == []

        @pytest.mark.anyio
        async def test_query_returns_metadata(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """Without metadata, citations and ACL post-checks are impossible."""
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=1), ctx)
            assert results[0].metadata.get("tenant") == "acme"

        # -- filters -------------------------------------------------- #

        @pytest.mark.anyio
        @pytest.mark.parametrize(
            ("operator", "expression", "expected"),
            [
                ("eq", F.field("tenant").eq("acme"), {"chk_a", "chk_b"}),
                ("ne", F.field("tenant").ne("acme"), {"chk_c"}),
                ("in", F.field("tenant").in_(["acme", "globex"]), {"chk_a", "chk_b", "chk_c"}),
                ("nin", F.field("tenant").nin(["acme"]), {"chk_c"}),
                ("gt", F.field("year").gt(2022), {"chk_a", "chk_c"}),
                ("gte", F.field("year").gte(2022), {"chk_a", "chk_b", "chk_c"}),
                ("lt", F.field("year").lt(2024), {"chk_b"}),
                ("lte", F.field("year").lte(2022), {"chk_b"}),
                ("contains", F.field("tags").contains("urgent"), {"chk_a"}),
                ("exists", F.field("public").exists(), {"chk_a", "chk_b"}),
                (
                    "and",
                    F.field("tenant").eq("acme") & F.field("year").gte(2024),
                    {"chk_a"},
                ),
                (
                    "or",
                    F.field("tenant").eq("globex") | F.field("year").lt(2023),
                    {"chk_b", "chk_c"},
                ),
                ("not", ~F.field("tenant").eq("acme"), {"chk_c"}),
            ],
        )
        async def test_declared_filter_operators_behave_correctly(
            self,
            populated: VectorIndex,
            ctx: RunContext,
            operator: str,
            expression: Filter,
            expected: set[str],
        ) -> None:
            """Every operator the backend *declares* must mean what core says.

            An operator a backend does not declare is skipped, since it will
            raise instead. An operator it declares but translates differently is
            the failure this test exists to catch: it produces wrong results
            silently, with no error anywhere.
            """
            if operator not in populated.supports().filter_ops:
                pytest.skip(f"{operator!r} is not declared in capabilities().filter_ops")

            results = await populated.query(
                VectorQuery(vector=_unit(0), top_k=100, filter=expression), ctx
            )
            assert {r.id for r in results} == expected

        @pytest.mark.anyio
        async def test_an_undeclared_operator_raises_rather_than_being_dropped(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """**[LOCKED]** INSTRUCTIONS.md §5.3.

            A silently dropped clause is how one tenant sees another tenant's
            documents. If the backend declares every operator there is nothing
            to check, and the test says so rather than passing vacuously.
            """
            declared = populated.supports().filter_ops
            candidates: list[tuple[str, Filter]] = [
                ("contains", F.field("tags").contains("urgent")),
                ("or", F.field("a").eq(1) | F.field("b").eq(2)),
                ("not", ~F.field("a").eq(1)),
                ("nin", F.field("a").nin([1])),
            ]
            undeclared = [(op, expr) for op, expr in candidates if op not in declared]
            if not undeclared:
                pytest.skip("this backend declares every operator the kit can withhold")

            _, expression = undeclared[0]
            with pytest.raises(UnsupportedFilterError):
                await populated.query(
                    VectorQuery(vector=_unit(0), top_k=10, filter=expression), ctx
                )

        @pytest.mark.anyio
        async def test_a_filter_matching_nothing_returns_nothing(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            results = await populated.query(
                VectorQuery(vector=_unit(0), top_k=10, filter=F.field("tenant").eq("nobody")),
                ctx,
            )
            assert results == []

        @pytest.mark.anyio
        async def test_an_absent_field_never_satisfies_equality(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """Only ``exists`` observes absence."""
            if "eq" not in populated.supports().filter_ops:
                pytest.skip("'eq' is not declared in capabilities().filter_ops")

            results = await populated.query(
                VectorQuery(vector=_unit(0), top_k=10, filter=F.field("nope").eq("x")),
                ctx,
            )
            assert results == []

        @pytest.mark.anyio
        async def test_an_absent_field_never_satisfies_inequality(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """Absence is not inequality, and conflating them leaks data.

            A backend that treated a missing field as "not equal" would widen
            every ``ne`` filter to match every record lacking that key. Where the
            filter is a tenant check, that is one tenant seeing another's
            documents.

            Separate from the equality case because a backend may legitimately
            not declare ``ne`` at all, and this assertion is important enough to
            be reported as skipped-for-cause rather than folded into a test that
            would then also be skipped.
            """
            if "ne" not in populated.supports().filter_ops:
                pytest.skip("'ne' is not declared in capabilities().filter_ops")

            results = await populated.query(
                VectorQuery(vector=_unit(0), top_k=10, filter=F.field("nope").ne("x")),
                ctx,
            )
            assert results == []

        # -- delete --------------------------------------------------- #

        @pytest.mark.anyio
        async def test_delete_by_id(self, populated: VectorIndex, ctx: RunContext) -> None:
            await populated.delete(ctx, ids=["chk_a"])
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=100), ctx)
            assert "chk_a" not in {r.id for r in results}

        @pytest.mark.anyio
        async def test_delete_by_filter(self, populated: VectorIndex, ctx: RunContext) -> None:
            """How ingestion removes a document's chunks.

            The most commonly skipped requirement in a vector store integration,
            and the reason a system cites a document removed six months ago.
            """
            if not populated.supports().supports_delete_by_filter:
                pytest.skip("this backend does not support delete by filter")

            await populated.delete(ctx, filter=F.field("document_id").eq("doc_1"))
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=100), ctx)
            assert {r.id for r in results} == {"chk_c"}

        @pytest.mark.anyio
        async def test_delete_with_neither_ids_nor_filter_is_refused(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """Deleting the whole index because an argument was forgotten is never intended."""
            with pytest.raises((ValueError, TypeError)):
                await populated.delete(ctx)

        @pytest.mark.anyio
        async def test_deleting_a_missing_id_is_not_an_error(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """Ingestion retries deletes; a second one must not fail the run."""
            await populated.delete(ctx, ids=["chk_does_not_exist"])

        @pytest.mark.anyio
        async def test_delete_then_upsert_restores(
            self, populated: VectorIndex, ctx: RunContext
        ) -> None:
            """A re-appearing document must index cleanly."""
            await populated.delete(ctx, ids=["chk_a"])
            await populated.upsert([_records()[0]], ctx)
            results = await populated.query(VectorQuery(vector=_unit(0), top_k=100), ctx)
            assert "chk_a" in {r.id for r in results}

        # -- epoch ---------------------------------------------------- #

        @pytest.mark.anyio
        async def test_epoch_is_non_negative_and_never_decreases(
            self, populated: VectorIndex
        ) -> None:
            """Every downstream cache key includes the epoch (ADR-009).

            An epoch that went backwards would make a stale cache entry look
            current. Bumping it is the ``StateStore``'s job at the end of an
            ingestion run, so this asserts only what the port itself guarantees.
            """
            first = (await populated.describe()).epoch
            second = (await populated.describe()).epoch
            assert first >= 0
            assert second >= first

        # -- namespaces ----------------------------------------------- #

        @pytest.mark.anyio
        async def test_namespaces_are_isolated(self, index: VectorIndex, ctx: RunContext) -> None:
            """A query in one namespace must not see another's records."""
            if not index.supports().supports_namespaces:
                pytest.skip("this backend does not support namespaces")

            await index.ensure(IndexSpec(name=index.name, dimensions=DIMENSIONS))
            await index.upsert(
                [r.model_copy(update={"namespace": "tenant_a"}) for r in _records()], ctx
            )
            await index.upsert(
                [_records()[0].model_copy(update={"id": "chk_z", "namespace": "tenant_b"})], ctx
            )

            in_a = await index.query(
                VectorQuery(vector=_unit(0), top_k=100, namespace="tenant_a"), ctx
            )
            in_b = await index.query(
                VectorQuery(vector=_unit(0), top_k=100, namespace="tenant_b"), ctx
            )
            assert {r.id for r in in_a} == {"chk_a", "chk_b", "chk_c"}
            assert {r.id for r in in_b} == {"chk_z"}

    if unauthorised_factory is not None:
        build_unauthorised = unauthorised_factory

        class VectorIndexContractWithAuth(VectorIndexContract):
            """The base contract plus error-mapping tests."""

            @pytest.mark.anyio
            async def test_bad_credentials_raise_auth_error(self, ctx: RunContext) -> None:
                """A provider exception must arrive as the taxonomy's ``AuthError``.

                Never a leaked SDK exception type: a caller that had to catch
                the provider's own class would be coupled to the provider,
                which is the coupling this library exists to remove.
                """
                broken = build_unauthorised()
                with pytest.raises(AuthError) as exc_info:
                    await broken.query(VectorQuery(vector=_unit(0), top_k=1), ctx)
                assert exc_info.value.retryable is False, (
                    "an auth failure is not retryable; retrying it burns the "
                    "budget and produces the same 403"
                )

        return VectorIndexContractWithAuth

    return VectorIndexContract


def contract_record_ids() -> Sequence[str]:
    """The ids the kit's fixture corpus uses, for a caller arranging cleanup."""
    return tuple(record.id for record in _records())
