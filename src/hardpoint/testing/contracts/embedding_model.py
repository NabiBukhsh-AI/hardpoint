"""The ``EmbeddingModel`` conformance suite. INSTRUCTIONS.md §6.6.

Bind it in your own tests::

    from hardpoint.testing.contracts import embedding_model_contract

    TestMyEmbeddings = embedding_model_contract(lambda: MyEmbeddings(...))

## The two failures this exists to catch

Both are silent, and both corrupt retrieval while everything keeps working.

**Order.** The caller sends a batch and matches vectors back by position. A
transposition attaches every chunk's vector to a neighbouring chunk. Nothing
errors; retrieval simply returns the wrong passages, and the cause is invisible
from the outside. So the kit sends a batch whose members are distinguishable and
checks that each text's vector is the one that text produces alone.

**Width.** ``dimensions`` is declared on the port so index creation can be
validated against it. An adapter whose declaration and output disagree either
errors per record at the index or, on a backend that pads, produces an index of
meaningless vectors.

Determinism is *not* required: several providers are not bit-identical between
calls. The order check is written to tolerate that by comparing which vector is
closest, not by demanding equality.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from hardpoint.core.errors import AuthError, MissingDependencyError
from hardpoint.core.ports import EmbeddingModel

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["embedding_model_contract"]

# Deliberately unlike each other, so a transposition is detectable by similarity
# even from a model whose outputs are not reproducible between calls.
_DISTINCT = (
    "The refund window is thirty days from the date of purchase.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The bridge was completed in 1937 and spans the strait.",
)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def embedding_model_contract(
    factory: Callable[[], EmbeddingModel],
    *,
    unauthorised_factory: Callable[[], EmbeddingModel] | None = None,
    asymmetric: bool = False,
) -> type:
    """Build a pytest class asserting an ``EmbeddingModel`` implementation conforms.

    Args:
        factory: Returns a ready model.
        unauthorised_factory: Returns one whose credentials will be rejected.
            When given, error-mapping tests are added.
        asymmetric: Whether this model distinguishes query from document
            embeddings. When true, the kit asserts the two differ; when false it
            asserts nothing either way, because a symmetric model returning the
            same vector for both is correct.

    Returns:
        A test class. Bind it to a name beginning with ``Test``.
    """
    try:
        import pytest  # noqa: PLC0415 - see the vector_index kit's docstring
    except ModuleNotFoundError as exc:  # pragma: no cover - pytest is a dev dependency
        raise MissingDependencyError(
            "The contract kits build pytest test classes, and pytest is not installed.",
            component="embedding_model_contract",
            remedy="pip install pytest",
            cause=exc,
        ) from exc

    from hardpoint.testing.fixtures import build_run_context  # noqa: PLC0415

    class EmbeddingModelContract:
        """Behaviour every ``EmbeddingModel`` implementation must exhibit."""

        @pytest.fixture
        def ctx(self) -> RunContext:
            return build_run_context()

        @pytest.fixture
        def model(self) -> Any:
            return factory()

        def test_satisfies_the_protocol(self, model: EmbeddingModel) -> None:
            assert isinstance(model, EmbeddingModel)

        def test_declares_an_id_and_a_positive_width(self, model: EmbeddingModel) -> None:
            """``dimensions`` is on the port so index creation can validate it."""
            assert model.id
            assert model.dimensions > 0

        @pytest.mark.anyio
        async def test_returns_one_vector_per_text(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            result = await model.embed(list(_DISTINCT), "document", ctx)
            assert len(result.vectors) == len(_DISTINCT)

        @pytest.mark.anyio
        async def test_every_vector_has_the_declared_width(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """A declaration that disagrees with the output corrupts the index."""
            result = await model.embed(list(_DISTINCT), "document", ctx)
            assert all(len(vector) == model.dimensions for vector in result.vectors)

        @pytest.mark.anyio
        async def test_vectors_come_back_in_input_order(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """Check the batch comes back in input order.

            **The silent corruption this kit exists for.**

            A transposition attaches every chunk's vector to a neighbouring
            chunk. Nothing errors, retrieval keeps working, and it returns the
            wrong passages.

            Checked by embedding each text alone and asserting the batch's
            vector for that position is closest to it -- which tolerates a
            provider whose output is not reproducible between calls.
            """
            batch = await model.embed(list(_DISTINCT), "document", ctx)

            for position, text in enumerate(_DISTINCT):
                alone = await model.embed([text], "document", ctx)
                similarities = [_cosine(alone.vectors[0], candidate) for candidate in batch.vectors]
                closest = max(range(len(similarities)), key=similarities.__getitem__)
                assert closest == position, (
                    f"text {position} matched batch vector {closest}: the batch was "
                    f"returned out of order"
                )

        @pytest.mark.anyio
        async def test_different_texts_give_different_vectors(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """A model returning one vector for everything would pass the width test."""
            result = await model.embed(list(_DISTINCT), "document", ctx)
            assert result.vectors[0] != result.vectors[1]

        @pytest.mark.anyio
        async def test_usage_is_reported_and_never_zero(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """Usage is reported and never zero.

            Embedding spend during a large ingestion is exactly the number
            nobody wants understated.
            """
            result = await model.embed(list(_DISTINCT), "document", ctx)
            assert result.usage.calls >= 1
            assert result.usage.embed_tokens > 0

        @pytest.mark.anyio
        async def test_the_reported_model_id_is_populated(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """The ingestion manifest records it to decide what needs re-embedding."""
            result = await model.embed(["one text"], "document", ctx)
            assert result.model_id

        @pytest.mark.anyio
        async def test_an_empty_batch_is_handled(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            """The sync engine reaches this whenever every chunk was reused."""
            result = await model.embed([], "document", ctx)
            assert result.vectors == ()

        @pytest.mark.anyio
        async def test_query_and_document_kinds_are_both_accepted(
            self, model: EmbeddingModel, ctx: RunContext
        ) -> None:
            as_query = await model.embed(["a question"], "query", ctx)
            as_document = await model.embed(["a question"], "document", ctx)

            assert len(as_query.vectors) == 1
            assert len(as_document.vectors) == 1
            if asymmetric:
                assert as_query.vectors != as_document.vectors, (
                    "an asymmetric model must embed a query differently from a document"
                )

    if unauthorised_factory is not None:
        build_unauthorised = unauthorised_factory

        class EmbeddingModelContractWithAuth(EmbeddingModelContract):
            """The base contract plus error mapping."""

            @pytest.mark.anyio
            async def test_bad_credentials_raise_auth_error(self, ctx: RunContext) -> None:
                with pytest.raises(AuthError) as exc_info:
                    await build_unauthorised().embed(["text"], "document", ctx)
                assert exc_info.value.retryable is False

        return EmbeddingModelContractWithAuth

    return EmbeddingModelContract
