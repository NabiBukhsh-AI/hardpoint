"""Deterministic fakes for every port. INSTRUCTIONS.md §5.8.

These are the reason a user's pipeline test runs in milliseconds with no network
access. They ship in M0 because everything after M0 depends on them.

## Deterministic, not random

Every fake derives its output from a hash of its input. The same query returns
the same vector, the same prompt returns the same text, on every machine and in
every process. A fake that returned random values would make an eval suite
unreproducible and a flaky test indistinguishable from a real regression.

## They record what happened

Each fake counts its calls and keeps its arguments. This is what the six
ingestion acceptance tests assert against: "re-running sync performs zero
embeddings" is only checkable because ``FakeEmbeddingModel`` counts.

## InMemoryVectorIndex is a conformance target, not a toy **[LOCKED]**

It implements the full ``MetadataFilter`` semantics, all three distance metrics,
namespaces, delete-by-filter and epoch reporting, and it passes the same
``VectorIndex`` contract kit a real adapter must pass. If it were a simplified
stand-in, every test written against it would prove nothing about production.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Literal

from hardpoint.core.capabilities import ALL_FILTER_OPS, IndexCapabilities, ModelCapabilities
from hardpoint.core.errors import AuthError, ConfigError, ContractError
from hardpoint.core.filters import matches, validate_supported
from hardpoint.core.ids import normalise_text
from hardpoint.core.models import RetrievedChunk, StepUsage
from hardpoint.core.ports import (
    DeleteReport,
    EmbedKind,
    EmbedResult,
    GenerationDelta,
    GenerationRequest,
    GenerationResult,
    IndexInfo,
    IndexRecord,
    IndexSpec,
    Message,
    MetadataFilter,
    ScoredRecord,
    ToolCall,
    UpsertReport,
    VectorQuery,
)
from hardpoint.core.types import JsonValue, ModelId

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = [
    "FakeCache",
    "FakeEmbeddingModel",
    "FakeLanguageModel",
    "FakeReranker",
    "InMemoryVectorIndex",
    "RecordedSpan",
    "RecordingMetricSink",
    "RecordingTracer",
    "ScriptedResponse",
    "deterministic_vector",
]

Metric = Literal["cosine", "dot", "euclidean"]


def _digest_floats(seed: str, count: int) -> tuple[float, ...]:
    """Derive ``count`` floats in ``[-1, 1)`` deterministically from a string.

    Uses sha256 over a counter so the sequence is reproducible across processes
    and Python versions. ``random.Random(seed)`` would also be reproducible but
    couples the values to CPython's Mersenne Twister implementation.
    """
    values: list[float] = []
    block = 0
    while len(values) < count:
        digest = hashlib.sha256(f"{seed}|{block}".encode()).digest()
        for offset in range(0, len(digest), 4):
            if len(values) >= count:
                break
            (raw,) = struct.unpack(">I", digest[offset : offset + 4])
            values.append((raw / 2**31) - 1.0)
        block += 1
    return tuple(values)


def deterministic_vector(text: str, dimensions: int, *, salt: str = "") -> tuple[float, ...]:
    """Return a stable unit vector for a text.

    Normalised to unit length so cosine similarity behaves like a real embedding
    space: identical texts score 1.0, and a dot product equals the cosine.

    Args:
        text: The text to embed. Normalised first, so whitespace differences do
            not produce different vectors.
        dimensions: Vector width.
        salt: Distinguishes otherwise identical inputs, used to make query and
            document embeddings differ.

    Returns:
        A unit vector of the requested width.
    """
    raw = _digest_floats(f"{salt}|{normalise_text(text)}", dimensions)
    norm = math.sqrt(sum(value * value for value in raw))
    if norm == 0.0:  # pragma: no cover - a zero digest is not reachable in practice
        return tuple(1.0 / math.sqrt(dimensions) for _ in range(dimensions))
    return tuple(value / norm for value in raw)


# --------------------------------------------------------------------------- #
# Language model                                                              #
# --------------------------------------------------------------------------- #


class ScriptedResponse:
    """One scripted reply, matched against the rendered prompt.

    Args:
        match: A substring that must appear in the rendered prompt for this
            response to fire. ``None`` matches anything.
        text: The text to return.
        tool_calls: Tool calls to return instead of, or alongside, text.
        finish_reason: What to report as the reason generation stopped.
        prompt_tokens: Reported prompt tokens. ``None`` uses the estimate.
        completion_tokens: Reported completion tokens. ``None`` uses the estimate.
        cost_usd: Reported cost. ``None`` models an unpriced model, which is the
            case a test needs in order to check that a total goes to ``None``
            rather than to zero.
    """

    __slots__ = (
        "completion_tokens",
        "cost_usd",
        "finish_reason",
        "match",
        "prompt_tokens",
        "text",
        "tool_calls",
    )

    def __init__(
        self,
        text: str = "",
        *,
        match: str | None = None,
        tool_calls: Sequence[ToolCall] = (),
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"] = "stop",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = 0.0,
    ) -> None:
        self.text = text
        self.match = match
        self.tool_calls = tuple(tool_calls)
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd


class FakeLanguageModel:
    """A ``LanguageModel`` that never leaves the process.

    Two modes. **Scripted**: responses are matched against the rendered prompt,
    so a test can make the model say a specific thing when it sees a specific
    context. **Deterministic** (the default): the reply echoes a hash of the
    prompt, which is meaningless to read but stable, so a cassette or an eval
    baseline recorded against it stays valid.

    Args:
        responses: Scripted replies, tried in order. The first whose ``match``
            appears in the rendered prompt wins.
        model_id: What to report as the model.
        capabilities: What to declare. Defaults to supporting everything, since
            most tests want the capability check to pass; pass a narrower value
            to test the failure path.
        fail_with: An exception to raise instead of generating, for testing
            retry, fallback and degradation paths.
        cost_per_call: Reported cost per call, or ``None`` to model an unpriced
            model.
    """

    def __init__(
        self,
        responses: Sequence[ScriptedResponse] = (),
        *,
        model_id: ModelId = "fake/language-model",
        capabilities: ModelCapabilities | None = None,
        fail_with: Exception | None = None,
        cost_per_call: float | None = 0.0,
    ) -> None:
        self.id = model_id
        self._responses = tuple(responses)
        self._capabilities = capabilities or ModelCapabilities(
            context_window_tokens=32_000,
            max_output_tokens=4_096,
            supports_tools=True,
            supports_structured_output=True,
            supports_streaming=True,
            supports_vision=False,
        )
        self._fail_with = fail_with
        self._cost_per_call = cost_per_call

        self.calls: list[GenerationRequest] = []
        self.stream_calls: list[GenerationRequest] = []
        self.token_count_calls: list[tuple[Message, ...]] = []

    @property
    def call_count(self) -> int:
        """How many times the model was invoked, generating or streaming."""
        return len(self.calls) + len(self.stream_calls)

    def capabilities(self) -> ModelCapabilities:
        """Declare what this fake supports."""
        return self._capabilities

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate tokens as roughly four characters each.

        A crude heuristic, but a *stated* one, and never zero: a token count of
        zero would let a context assembler believe anything fits.
        """
        self.token_count_calls.append(tuple(messages))
        return self._estimate(messages)

    @staticmethod
    def _estimate(messages: Sequence[Message]) -> int:
        characters = sum(len(message.content) for message in messages)
        return max(1, characters // 4)

    @staticmethod
    def render(request: GenerationRequest) -> str:
        """Flatten a request's messages into the text a script is matched against."""
        return "\n".join(f"{m.role}: {m.content}" for m in request.messages)

    def _select(self, request: GenerationRequest) -> ScriptedResponse:
        rendered = self.render(request)
        for response in self._responses:
            if response.match is None or response.match in rendered:
                return response
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        return ScriptedResponse(text=f"fake answer {digest}", cost_usd=self._cost_per_call)

    def _usage(self, request: GenerationRequest, response: ScriptedResponse) -> StepUsage:
        prompt_tokens = (
            response.prompt_tokens
            if response.prompt_tokens is not None
            else self._estimate(request.messages)
        )
        completion_tokens = (
            response.completion_tokens
            if response.completion_tokens is not None
            else max(1, len(response.text) // 4)
        )
        cost = response.cost_usd if response.cost_usd is not None else self._cost_per_call
        return StepUsage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            estimated=True,
        )

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Return the scripted or deterministic reply for this prompt."""
        self.calls.append(req)
        if self._fail_with is not None:
            raise self._fail_with

        response = self._select(req)
        return GenerationResult(
            text=response.text,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            model_id=self.id,
            usage=self._usage(req, response),
            raw_response_id=f"fake-{len(self.calls)}",
        )

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Stream the reply one word at a time, with usage on the final delta."""
        self.stream_calls.append(req)

        async def iterator() -> AsyncIterator[GenerationDelta]:
            if self._fail_with is not None:
                raise self._fail_with

            response = self._select(req)
            words = response.text.split(" ")
            for index, word in enumerate(words):
                if not word and len(words) == 1:
                    break
                suffix = "" if index == len(words) - 1 else " "
                yield GenerationDelta(text=f"{word}{suffix}")

            for call in response.tool_calls:
                yield GenerationDelta(tool_call=call)

            yield GenerationDelta(
                finish_reason=response.finish_reason, usage=self._usage(req, response)
            )

        return iterator()


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #


class FakeEmbeddingModel:
    """An ``EmbeddingModel`` producing stable vectors from a hash of the text.

    Deliberately asymmetric: the same text embeds differently as a query and as
    a document. Real asymmetric models exist, and getting ``EmbedKind`` wrong
    silently degrades recall with no error and no symptom other than worse
    answers. A symmetric fake would make that bug invisible in tests.

    Args:
        dimensions: Vector width.
        model_id: What to report as the model, and what the ingestion manifest
            records. Change it in a test to simulate switching models.
        fail_with: An exception to raise instead of embedding.
    """

    def __init__(
        self,
        dimensions: int = 8,
        *,
        model_id: ModelId = "fake/embedding-model",
        fail_with: Exception | None = None,
    ) -> None:
        self.id = model_id
        self.dimensions = dimensions
        self._fail_with = fail_with

        self.calls: list[tuple[tuple[str, ...], EmbedKind]] = []

    @property
    def call_count(self) -> int:
        """How many batches were embedded."""
        return len(self.calls)

    @property
    def embedded_texts(self) -> list[str]:
        """Every text embedded, in order. What an idempotency test asserts on."""
        return [text for texts, _ in self.calls for text in texts]

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Return one deterministic unit vector per text, in input order."""
        self.calls.append((tuple(texts), kind))
        if self._fail_with is not None:
            raise self._fail_with

        vectors = tuple(deterministic_vector(text, self.dimensions, salt=kind) for text in texts)
        return EmbedResult(
            vectors=vectors,
            model_id=self.id,
            usage=StepUsage(
                calls=1,
                embed_tokens=sum(max(1, len(text) // 4) for text in texts),
                cost_usd=0.0,
                estimated=True,
            ),
        )


# --------------------------------------------------------------------------- #
# Vector index  [LOCKED]                                                      #
# --------------------------------------------------------------------------- #


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _euclidean_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Convert euclidean distance to a similarity, so higher is always better.

    A raw distance would sort backwards against the port's contract that results
    come back in descending score order.
    """
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    return 1.0 / (1.0 + distance)


_METRICS: dict[Metric, Callable[[Sequence[float], Sequence[float]], float]] = {
    "cosine": _cosine,
    "dot": _dot,
    "euclidean": _euclidean_similarity,
}


class InMemoryVectorIndex:
    """A complete, in-process ``VectorIndex`` (INSTRUCTIONS.md §5.8, **[LOCKED]**).

    The reference implementation the ``VectorIndex`` contract kit is written
    against, and it passes that kit unmodified. Full ``MetadataFilter``
    semantics including ``and``/``or``/``not``/``exists``, three distance
    metrics, namespaces, delete by id and by filter, and epoch reporting.

    Args:
        name: The index name.
        dimensions: Vector width. Enforced on upsert and on ``ensure``.
        metric: Distance function.
        capabilities: What to declare. Defaults to supporting every filter
            operator. Narrow it to simulate a limited backend, which is how a
            test checks that ``UnsupportedFilterError`` is raised rather than
            the clause being dropped.
        epoch: Starting epoch.
        unauthorised: When true, every operation raises ``AuthError``. Lets the
            contract kit check error mapping without a real credential.
    """

    def __init__(
        self,
        name: str = "memory",
        dimensions: int = 8,
        metric: Metric = "cosine",
        *,
        capabilities: IndexCapabilities | None = None,
        epoch: int = 0,
        unauthorised: bool = False,
    ) -> None:
        self.name = name
        self.dimensions = dimensions
        self.metric: Metric = metric
        self._records: dict[tuple[str | None, str], IndexRecord] = {}
        self._capabilities = capabilities or IndexCapabilities(
            filter_ops=ALL_FILTER_OPS,
            supports_namespaces=True,
            supports_delete_by_filter=True,
            delete_consistency="consistent",
        )
        self._epoch = epoch
        self._unauthorised = unauthorised
        self._embed_model: ModelId | None = None

        self.upsert_calls = 0
        self.query_calls = 0
        self.delete_calls = 0

    # -- helpers ------------------------------------------------------- #

    def _authorise(self) -> None:
        if self._unauthorised:
            raise AuthError(
                f"The index {self.name!r} rejected the credentials.",
                component=self.name,
                remedy="Check the index API key in the environment and retry.",
            )

    @property
    def record_count(self) -> int:
        """How many records are stored, across every namespace."""
        return len(self._records)

    def records(self, namespace: str | None = None) -> list[IndexRecord]:
        """Return stored records, for a test to assert against directly."""
        return [
            record
            for (record_namespace, _), record in sorted(self._records.items())
            if record_namespace == namespace
        ]

    def set_epoch(self, epoch: int) -> None:
        """Set the reported epoch.

        Not part of the ``VectorIndex`` port: bumping an epoch is the
        ``StateStore``'s job at the end of an ingestion run
        (INSTRUCTIONS.md §6.2 step 7). This exists so a test can arrange an
        epoch without reaching through the sync engine.
        """
        self._epoch = epoch

    # -- the port ------------------------------------------------------ #

    async def describe(self) -> IndexInfo:
        """Report the index's dimensions, metric, size and epoch."""
        self._authorise()
        return IndexInfo(
            name=self.name,
            dimensions=self.dimensions,
            metric=self.metric,
            count=len(self._records),
            epoch=self._epoch,
            embed_model=self._embed_model,
        )

    async def ensure(self, spec: IndexSpec) -> None:
        """Create or verify the index against a spec.

        A dimension mismatch raises rather than being tolerated. Writing 1536
        vectors into a 768-dimension index either errors per record or, worse,
        silently truncates them, and the truncated version is a quality
        regression nobody can trace.
        """
        self._authorise()
        if self._records and spec.dimensions != self.dimensions:
            raise ConfigError(
                f"Index {self.name!r} holds {self.dimensions}-dimension vectors, but "
                f"the configured embedding model produces {spec.dimensions}.",
                component=self.name,
                config_path=f"indexes.{self.name}",
                remedy=(
                    "Either configure an embedding model with "
                    f"{self.dimensions} dimensions, or create a new index and "
                    "re-ingest. An index cannot hold vectors of two widths."
                ),
            )
        self.dimensions = spec.dimensions
        self.metric = spec.metric
        if spec.embed_model is not None:
            self._embed_model = spec.embed_model

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Insert or replace records by id. Idempotent."""
        self._authorise()
        self.upsert_calls += 1

        for record in records:
            if record.vector and len(record.vector) != self.dimensions:
                raise ContractError(
                    f"Record {record.id!r} has {len(record.vector)} dimensions, but "
                    f"index {self.name!r} expects {self.dimensions}.",
                    component=self.name,
                    remedy=(
                        "This is a bug in the caller: the embedding model and the "
                        "index disagree on dimensions. `hardpoint doctor` checks "
                        "for this at startup."
                    ),
                )
            self._records[(record.namespace, record.id)] = record

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
            ValueError: If neither is given. Deleting the whole index because a
                caller forgot an argument is never what anyone meant.
        """
        self._authorise()
        if ids is None and filter is None:
            raise ValueError(
                "delete() requires `ids`, `filter`, or both. Refusing to delete "
                "every record in the index because neither was given."
            )
        self.delete_calls += 1

        if filter is not None:
            validate_supported(filter, self._capabilities.filter_ops, self.name)

        wanted = set(ids or ())
        doomed = [
            key
            for key, record in self._records.items()
            if (ids is not None and record.id in wanted)
            or (filter is not None and matches(filter, self._metadata_of(record)))
        ]
        for key in doomed:
            del self._records[key]
        return DeleteReport(deleted=len(doomed))

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Search, returning results in descending score order."""
        self._authorise()
        self.query_calls += 1

        if req.filter is not None:
            validate_supported(req.filter, self._capabilities.filter_ops, self.name)

        score = _METRICS[self.metric]
        scored: list[ScoredRecord] = []
        for (namespace, _), record in self._records.items():
            if namespace != req.namespace:
                continue
            if req.filter is not None and not matches(req.filter, self._metadata_of(record)):
                continue
            value = score(req.vector, record.vector) if req.vector and record.vector else 0.0
            if req.min_score is not None and value < req.min_score:
                continue
            scored.append(
                ScoredRecord(
                    id=record.id,
                    score=value,
                    metadata=dict(record.metadata),
                    text=record.text if req.include_text else None,
                    document_id=record.document_id,
                )
            )

        # Sorted by descending score, then by id, so equal scores order the same
        # way on every run rather than by insertion order.
        scored.sort(key=lambda item: (-item.score, item.id))
        return scored[: req.top_k]

    def supports(self) -> IndexCapabilities:
        """Declare filter operators, namespaces and delete consistency."""
        return self._capabilities

    @staticmethod
    def _metadata_of(record: IndexRecord) -> dict[str, JsonValue]:
        """Return the record's metadata with ``document_id`` visible to filters.

        Ingestion deletes a document's chunks with a filter on ``document_id``,
        so that field has to be filterable whether or not the caller also copied
        it into metadata.
        """
        metadata: dict[str, JsonValue] = dict(record.metadata)
        if record.document_id is not None:
            metadata.setdefault("document_id", record.document_id)
        return metadata


# --------------------------------------------------------------------------- #
# Reranker                                                                    #
# --------------------------------------------------------------------------- #


class FakeReranker:
    """A ``Reranker`` scoring by word overlap with the query.

    Overlap rather than a hash, so the ordering it produces is *plausible*: a
    test asserting "the relevant chunk moved up" reads as intended instead of
    depending on an arbitrary permutation.

    Args:
        model_id: What to report as the model.
        fail_with: An exception to raise, for testing the ``on_failure: skip``
            path that records a ``Degradation`` rather than failing the request.
    """

    def __init__(
        self,
        *,
        model_id: ModelId = "fake/reranker",
        fail_with: Exception | None = None,
    ) -> None:
        self.id = model_id
        self._fail_with = fail_with
        self.calls: list[tuple[str, int]] = []

    @property
    def call_count(self) -> int:
        """How many times reranking was requested."""
        return len(self.calls)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
        ctx: RunContext,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` candidates with the most query-word overlap."""
        self.calls.append((query, len(candidates)))
        if self._fail_with is not None:
            raise self._fail_with

        terms = set(normalise_text(query).lower().split())

        def overlap(candidate: RetrievedChunk) -> float:
            words = set(normalise_text(candidate.chunk.text).lower().split())
            return len(terms & words) / len(terms) if terms else 0.0

        ranked = sorted(candidates, key=lambda c: (-overlap(c), c.chunk.id))
        return [
            candidate.model_copy(update={"score": overlap(candidate), "rank": index})
            for index, candidate in enumerate(ranked[:top_k])
        ]


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #


class FakeCache:
    """An in-memory ``CacheBackend`` that counts hits and misses.

    The counters are the point: "does bumping the epoch change the retrieval
    cache key" is checked by observing a miss where there would otherwise have
    been a hit.

    TTLs are recorded but not enforced. A fake that expired entries on wall-clock
    time would make tests time-dependent, which is a worse failure than not
    covering expiry.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int | None] = {}
        self.hits = 0
        self.misses = 0
        self.sets = 0

    async def get(self, key: str) -> bytes | None:
        """Return the cached value, counting the hit or miss."""
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        """Store a value and remember the TTL it was given."""
        self.sets += 1
        self.store[key] = value
        self.ttls[key] = ttl_s

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix, returning how many were removed."""
        doomed = [key for key in self.store if key.startswith(prefix)]
        for key in doomed:
            del self.store[key]
            self.ttls.pop(key, None)
        return len(doomed)


# --------------------------------------------------------------------------- #
# Observability                                                               #
# --------------------------------------------------------------------------- #


class RecordedSpan:
    """A span captured by :class:`RecordingTracer`."""

    def __init__(self, name: str, attributes: dict[str, JsonValue]) -> None:
        self.name = name
        self.attributes = attributes
        self.events: list[tuple[str, dict[str, JsonValue]]] = []
        self.exceptions: list[BaseException] = []
        self.status: tuple[str, str] | None = None
        self.children: list[RecordedSpan] = []

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Attach an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Record a point-in-time event, such as a retry attempt."""
        self.events.append((name, dict(attributes or {})))

    def record_exception(self, exc: BaseException) -> None:
        """Attach an exception."""
        self.exceptions.append(exc)

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Mark the span's outcome."""
        self.status = (status, description)

    @property
    def trace_id(self) -> str | None:
        """A fixed trace id, so log-correlation assertions have something stable."""
        return "fake-trace"

    def __repr__(self) -> str:
        """Render the span name and how many children it has."""
        return f"RecordedSpan({self.name!r}, children={len(self.children)})"


class RecordingTracer:
    """A ``Tracer`` that keeps every span, including its nesting.

    Nesting is tracked because "traces nest correctly across async boundaries"
    is a real M2 requirement and ad hoc logging is exactly what fails it.
    """

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self.roots: list[RecordedSpan] = []
        self._stack: list[RecordedSpan] = []

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[RecordedSpan]:
        """Open a span as an async context manager."""

        @asynccontextmanager
        async def scope() -> AsyncIterator[RecordedSpan]:
            recorded = RecordedSpan(name, dict(attrs))
            self.spans.append(recorded)
            if self._stack:
                self._stack[-1].children.append(recorded)
            else:
                self.roots.append(recorded)
            self._stack.append(recorded)
            try:
                yield recorded
            finally:
                self._stack.pop()

        return scope()

    def names(self) -> list[str]:
        """Every span name in the order the spans were opened."""
        return [span.name for span in self.spans]

    def find(self, name: str) -> list[RecordedSpan]:
        """Every recorded span with a given name."""
        return [span for span in self.spans if span.name == name]


class RecordingMetricSink:
    """A ``MetricSink`` that keeps every measurement."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, float, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []
        self.gauges: list[tuple[str, float, dict[str, str]]] = []

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        self.counters.append((name, value, dict(labels)))

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation."""
        self.observations.append((name, value, dict(labels)))

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge."""
        self.gauges.append((name, value, dict(labels)))

    def total(self, name: str) -> float:
        """Sum every increment recorded under a counter name."""
        return sum(value for recorded, value, _ in self.counters if recorded == name)
