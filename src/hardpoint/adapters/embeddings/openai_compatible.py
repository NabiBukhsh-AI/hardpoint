"""An OpenAI-compatible embeddings adapter.

Implements the M1 embedding adapter from INSTRUCTIONS.md §6.3. Speaks
``POST /embeddings``, which OpenAI, Ollama, vLLM, LM Studio and every gateway
in between serve identically, so a new provider costs a ``base_url``.

Over ``httpx``, so it needs no extra. See the chat adapter's docstring for the
full argument.

## Two invariants this adapter exists to hold

**Order is preserved.** The caller sends texts and matches vectors back by
position. OpenAI-compatible responses carry an ``index`` per item and are *not*
guaranteed to arrive in order, so they are sorted before being returned. A
transposition here would attach every chunk's vector to a neighbouring chunk --
retrieval would keep working and quietly return the wrong passages.

**Dimensions are what the model actually produced.** Declared up front for index
validation, and checked against the first response. An adapter that declared 1536
and returned 3072 would fail per record at the index, or worse, be padded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from hardpoint.adapters._http import (
    DEFAULT_TIMEOUT_S,
    as_int,
    as_json,
    build_client,
    map_response_error,
    map_transport_error,
)
from hardpoint.core.errors import ContractError
from hardpoint.core.models import StepUsage
from hardpoint.core.ports import EmbedKind, EmbedResult
from hardpoint.core.tokens import estimate_tokens
from hardpoint.core.types import JsonValue, ModelId

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["OpenAICompatibleEmbeddings"]


class OpenAICompatibleEmbeddings:
    """An ``EmbeddingModel`` speaking the OpenAI embeddings protocol.

    Args:
        model: The model name the endpoint expects.
        dimensions: The vector width this model produces. Carried on the port so
            index creation can be validated against it, which turns the classic
            dimension-mismatch failure into a startup error rather than a
            first-upsert error.
        base_url: The API root.
        api_key: Bearer token, omitted entirely when ``None``.
        request_dimensions: Ask the endpoint to truncate to ``dimensions``.
            Supported by ``text-embedding-3-*`` and ignored elsewhere, so it is
            off by default rather than sent to endpoints that would reject it.
        timeout_s: Transport timeout.
        client: A pre-built client. When given, this adapter does not close it.
        model_id: What to report. Defaults to ``openai/<model>``, which is what
            the ingestion manifest records to decide whether a chunk needs
            re-embedding.
        extra_headers: Additional headers.
    """

    def __init__(
        self,
        model: str,
        *,
        dimensions: int,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        request_dimensions: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        model_id: ModelId | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.id = model_id or f"openai/{model}"
        self._request_dimensions = request_dimensions
        self._owns_client = client is None
        self._client = client or build_client(
            base_url=base_url, api_key=api_key, headers=extra_headers, timeout_s=timeout_s
        )

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller supplied it."""
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Call ``POST /embeddings`` for a batch of texts.

        ``kind`` is accepted and not sent: the OpenAI protocol has no
        query/document distinction, and the models it serves are symmetric. An
        adapter for an asymmetric model -- and they exist -- would prefix the
        text or switch endpoint here. Accepting the argument keeps that adapter
        a drop-in replacement.

        Args:
            texts: What to embed. An empty batch returns immediately without a
                request, because a provider asked to embed nothing bills for it
                on some plans and errors on others.
            kind: Query or document. See above.
            ctx: The run context.

        Returns:
            Vectors in input order, one per text.

        Raises:
            AuthError, RateLimitedError, TransientError, InvalidRequestError,
            ProviderTimeout: Mapped from the failure.
            ContractError: If the response returns the wrong number of vectors
                or vectors of the wrong width.
        """
        if not texts:
            return EmbedResult(
                vectors=(), model_id=self.id, usage=StepUsage(calls=0, estimated=False)
            )

        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self._request_dimensions:
            payload["dimensions"] = self.dimensions

        try:
            response = await self._client.post("/embeddings", json=payload)
        except Exception as exc:
            raise map_transport_error(exc, component=self.id, operation="embed") from exc

        if response.is_error:
            raise map_response_error(response, component=self.id, operation="embed")

        body = as_json(response, component=self.id, operation="embed")
        vectors = self._vectors_from(body, expected=len(texts))
        return EmbedResult(
            vectors=vectors,
            model_id=str(body.get("model") or self.id),
            usage=self._usage_from(body, texts),
        )

    def _vectors_from(
        self, body: Mapping[str, JsonValue], *, expected: int
    ) -> tuple[tuple[float, ...], ...]:
        """Extract vectors, in input order, checking count and width."""
        data = body.get("data")
        if not isinstance(data, list):
            raise ContractError(
                f"{self.id}: the embeddings response contained no data array.",
                component=self.id,
                remedy="Check the base URL points at an OpenAI-compatible API root.",
            )

        # Sorted by the response's own index. The protocol does not guarantee
        # arrival order, and a transposition here would attach every chunk's
        # vector to a neighbouring chunk -- retrieval would keep working and
        # quietly return the wrong passages.
        ordered = sorted(
            (item for item in data if isinstance(item, dict)),
            key=lambda item: as_int(item.get("index")),
        )

        if len(ordered) != expected:
            raise ContractError(
                f"{self.id}: asked for {expected} embeddings and received {len(ordered)}.",
                component=self.id,
                remedy=(
                    "The endpoint returned a different number of vectors than texts "
                    "sent. Reduce `ingestion.embed_batch_size` if the endpoint caps "
                    "batch size silently."
                ),
            )

        vectors: list[tuple[float, ...]] = []
        for item in ordered:
            raw = item.get("embedding")
            if not isinstance(raw, list):
                raise ContractError(
                    f"{self.id}: an embedding was not an array of numbers.",
                    component=self.id,
                    remedy="Check the model name is an embedding model.",
                )
            vector = tuple(float(value) for value in raw)  # type: ignore[arg-type]
            if len(vector) != self.dimensions:
                raise ContractError(
                    f"{self.id}: declared {self.dimensions} dimensions but produced {len(vector)}.",
                    component=self.id,
                    remedy=(
                        f"Set the adapter's `dimensions` to {len(vector)}, or configure "
                        f"a model that produces {self.dimensions}. An index cannot hold "
                        f"vectors of two widths."
                    ),
                )
            vectors.append(vector)

        return tuple(vectors)

    def _usage_from(self, body: Mapping[str, JsonValue], texts: Sequence[str]) -> StepUsage:
        """Build usage, estimating when the provider omitted it.

        **Never a zero.** Embedding spend during a large ingestion is precisely
        the number nobody wants understated.
        """
        usage = body.get("usage")
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            return StepUsage(
                calls=1, embed_tokens=as_int(usage.get("prompt_tokens")), estimated=False
            )
        return StepUsage(
            calls=1,
            embed_tokens=max(1, sum(estimate_tokens(text) for text in texts)),
            estimated=True,
        )

    def __repr__(self) -> str:
        """Render the model id and width."""
        return f"OpenAICompatibleEmbeddings(model={self.model!r}, dimensions={self.dimensions})"
