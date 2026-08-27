"""Every provider port, as a ``typing.Protocol``.

Implements ARCHITECTURE.md §9.2 and §9.3.

## Why Protocols and not abstract base classes (ADR-002)

Structural typing lets a third-party object be adapted without wrapping it in an
inheritance chain, lets a test fake be a five-line class, and keeps this library
out of the user's class hierarchy. An ABC-required design forces the library to
be imported into places that should not depend on it.

There is deliberately **no** ``Component`` base class that everything inherits
from (INSTRUCTIONS.md §13.4). It looks tidy on a whiteboard and destroys IDE
support in practice.

## What is *not* here, and why

- ``Step`` lives in ``runtime/step.py``. It is the composition primitive, not a
  boundary to an external system.
- ``Source`` lives in ``ingestion/sources.py``, next to the implementations and
  the sync engine that drives it.
- ``Retriever`` is not a port at all. It is a composition of embed, query and
  filter, and making it a provider port would invite adapters that hide the
  whole retrieval strategy -- precisely the logic a user must be able to read
  (ARCHITECTURE.md §9.3).

## Ports carry their own request and result types

The DTOs are in this module rather than in ``models``, because they exist only
to give a port a typed surface and reading them beside the Protocol is how the
port becomes comprehensible. ``models`` holds the data that flows *through* a
pipeline; this holds the data that crosses a *boundary*.

## The invariants adapters must honour

These are contract, not convention, and the contract kits check what they can:

- Every ``generate`` returns ``Usage``. A provider that does not report tokens
  gets an estimate with ``estimated=True``, never a zero.
- Every provider exception is mapped into the ``core.errors`` taxonomy.
- No provider SDK type appears in any signature here or in any implementation's
  public surface.
- ``capabilities()`` is honest. Under-declaring is safe; over-declaring is a bug.
- An unsupported filter operator raises, never silently drops the clause.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.filters import Filter
from hardpoint.core.models import Chunk, ParsedDocument, RetrievedChunk, StepUsage
from hardpoint.core.types import JsonValue, ModelId

if TYPE_CHECKING:
    from hardpoint.core.capabilities import IndexCapabilities, ModelCapabilities
    from hardpoint.core.context import RunContext

__all__ = [
    "BlobStore",
    "CacheBackend",
    "CacheHint",
    "ChunkRecord",
    "Chunker",
    "DeleteReport",
    "DocumentParser",
    "DocumentRecord",
    "EmbedKind",
    "EmbedResult",
    "EmbeddingModel",
    "GenerationDelta",
    "GenerationRequest",
    "GenerationResult",
    "IndexInfo",
    "IndexMeta",
    "IndexRecord",
    "IndexSpec",
    "LanguageModel",
    "Message",
    "MetadataFilter",
    "MetricSink",
    "PromptStore",
    "RenderedPrompt",
    "Reranker",
    "Role",
    "ScoredRecord",
    "SourceBlob",
    "SourceEntry",
    "Span",
    "StateStore",
    "Tool",
    "ToolCall",
    "ToolPermissions",
    "ToolResult",
    "ToolSpec",
    "Tracer",
    "UpsertReport",
    "VectorQuery",
]

MetadataFilter = Filter
"""ARCHITECTURE.md's name for the filter expression tree defined in ``filters``.

An alias rather than a second type, so there is exactly one filter language.
"""

_FROZEN = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Language model                                                              #
# --------------------------------------------------------------------------- #

Role = Literal["system", "user", "assistant", "tool"]
"""Message roles. Closed, because retrieved content must never reach ``system``."""


class CacheHint(BaseModel):
    """A request to the provider to cache this message's prefix.

    Expressed as a hint on a message rather than through each provider's own
    prompt-caching API, so that turning it on does not couple a pipeline to one
    vendor. An adapter whose provider has no such feature ignores it.
    """

    model_config = _FROZEN

    cache: bool = False
    ttl_s: int | None = Field(default=None, gt=0)


class Message(BaseModel):
    """One message in a conversation.

    Args:
        role: Who is speaking.
        content: The text.
        name: Optional speaker name, for providers that support it.
        tool_call_id: Set on a ``tool`` message, linking it to the call it
            answers.
        tool_calls: Set on an ``assistant`` message that requested tool calls.
        cache_hint: Provider-side prompt caching hint.
    """

    model_config = _FROZEN

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    cache_hint: CacheHint | None = None


class ToolCall(BaseModel):
    """A model's request to invoke a tool.

    ``arguments`` is raw JSON text rather than a parsed object because models
    emit malformed JSON often enough that repair is a normal path, and parsing
    here would throw away the text a repair attempt needs.
    """

    model_config = _FROZEN

    id: str
    name: str
    arguments: str


class GenerationRequest(BaseModel):
    """What to ask a language model for.

    Args:
        messages: The conversation.
        max_output_tokens: Cap on generated tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        stop: Stop sequences.
        seed: Determinism hint, honoured by some providers.
        response_schema: JSON Schema the output must satisfy. Requires
            ``ModelCapabilities.supports_structured_output``.
        tools: Tool definitions the model may call. Requires
            ``ModelCapabilities.supports_tools``.
        tool_choice: ``auto`` lets the model decide, ``none`` forbids calls,
            ``required`` demands one.
        provider_options: Provider-specific knobs an adapter interprets. An
            adapter that does not recognise a key **warns**; it does not fail
            and does not silently ignore it (ARCHITECTURE.md §9.2). Kept opaque
            so that a vendor-specific setting never becomes part of this port.
    """

    model_config = _FROZEN

    messages: tuple[Message, ...]
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    response_schema: dict[str, JsonValue] | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: Literal["auto", "none", "required"] = "auto"
    provider_options: dict[str, JsonValue] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """What a language model returned.

    Args:
        text: The generated text. Empty when the model returned only tool calls.
        tool_calls: Tool invocations the model requested.
        finish_reason: Why generation stopped. ``length`` matters: it means the
            answer was truncated, which a caller must be able to detect.
        model_id: The model that actually served the request, which may differ
            from the one configured when a fallback fired.
        usage: Tokens, cost and latency. Always present.
        raw_response_id: The provider's own identifier, for correlating with
            their dashboard during an incident.
    """

    model_config = _FROZEN

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"] = "stop"
    model_id: ModelId
    usage: StepUsage
    raw_response_id: str | None = None


class GenerationDelta(BaseModel):
    """One increment of a streaming response.

    Args:
        text: Text produced since the previous delta.
        tool_call: A tool call, delivered once complete rather than in
            fragments, because a half-parsed call is not actionable.
        finish_reason: Set on the final delta.
        usage: Set on the final delta, when the provider reports it there.
    """

    model_config = _FROZEN

    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"] | None = None
    usage: StepUsage | None = None


@runtime_checkable
class LanguageModel(Protocol):
    """Generate text, optionally streaming, optionally with tools.

    Owns transport to one provider. Does not own retries, timeouts, caching,
    tracing decisions or prompt content -- all of those live elsewhere, which is
    what keeps an adapter to transport only.

    Attributes:
        id: The model identifier, by convention ``"<provider>/<model>"``.
    """

    id: ModelId

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Generate a complete response.

        Must always populate ``usage``. If the provider does not report token
        counts, estimate them and set ``estimated=True`` rather than reporting
        zero, because a zero is silently wrong.

        Raises:
            AuthError, RateLimitedError, TransientError, InvalidRequestError,
            ProviderTimeout: Mapped from the provider's own exceptions.
        """
        ...

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Stream a response as it is produced.

        Not ``async def``: it returns the iterator directly so a caller can
        ``async for`` over it without an extra await.
        """
        ...

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count tokens under this model's tokenizer.

        Used by ``ContextAssembler`` to fill a token budget accurately. An
        adapter with no tokenizer available uses a documented heuristic; it must
        not return zero.
        """
        ...

    def capabilities(self) -> ModelCapabilities:
        """Declare what this model supports.

        Queried rather than assumed, which is how provider switching stays safe.
        Over-declaring is a bug the contract kit exists to catch.
        """
        ...


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #

EmbedKind = Literal["document", "query"]
"""Which side of an asymmetric embedding model a text belongs to.

Part of the port because asymmetric models exist, and embedding a query as a
document silently degrades recall -- a failure with no error and no symptom
except worse answers.
"""


class EmbedResult(BaseModel):
    """Vectors for a batch of texts, in the order they were given.

    Args:
        vectors: One vector per input text, same order.
        model_id: The model that produced them. Recorded in the ingestion
            manifest so a model change forces re-embedding.
        usage: Tokens, cost and latency.
    """

    model_config = _FROZEN

    vectors: tuple[tuple[float, ...], ...]
    model_id: ModelId
    usage: StepUsage


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turn text into vectors.

    Attributes:
        id: The model identifier.
        dimensions: Vector width. Carried on the port so index creation can be
            validated against the configured model, turning the classic
            dimension-mismatch failure into a startup error rather than a
            first-upsert error.
    """

    id: ModelId
    dimensions: int

    async def embed(self, texts: Sequence[str], kind: EmbedKind, ctx: RunContext) -> EmbedResult:
        """Embed a batch of texts.

        Returns vectors in input order, one per text. Batching is the caller's
        decision; the adapter honours the batch it is given.
        """
        ...


# --------------------------------------------------------------------------- #
# Vector index                                                                #
# --------------------------------------------------------------------------- #


class IndexSpec(BaseModel):
    """What an index must look like for ``ensure`` to accept it.

    Args:
        name: Index or collection name.
        dimensions: Vector width, checked against the embedding model.
        metric: Distance function.
        namespace: Logical partition, where the backend supports one.
        embed_model: The embedding model the index was built with. Recorded so
            that changing models is detected rather than producing an index of
            mixed, incomparable vectors.
    """

    model_config = _FROZEN

    name: str
    dimensions: int = Field(gt=0)
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    namespace: str | None = None
    embed_model: ModelId | None = None


class IndexInfo(BaseModel):
    """The current state of an index.

    Args:
        name: Index name.
        dimensions: Vector width the index actually has.
        metric: Distance function in use.
        count: Number of records, when the backend can report it cheaply.
        epoch: Monotonically increasing version, bumped once per successful
            ingestion run. Every downstream cache key includes it, which is what
            stops a cache serving pre-reindex answers indefinitely (ADR-009).
        embed_model: The embedding model recorded at creation.
    """

    model_config = _FROZEN

    name: str
    dimensions: int
    metric: Literal["cosine", "dot", "euclidean"]
    count: int | None = None
    epoch: int = 0
    embed_model: ModelId | None = None


class IndexRecord(BaseModel):
    """One record to upsert.

    Args:
        id: Deterministic chunk id. Never random: a random id makes
            re-ingestion write duplicates instead of updating (ADR-009).
        vector: The dense vector.
        sparse_vector: Term index to weight, for hybrid backends.
        metadata: Filterable metadata.
        document_id: The owning document, so a delete-by-document filter works.
        text: The chunk text, when the backend stores it.
        namespace: Logical partition.
    """

    model_config = _FROZEN

    id: str
    vector: tuple[float, ...] = ()
    sparse_vector: dict[str, float] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    document_id: str | None = None
    text: str | None = None
    namespace: str | None = None


class UpsertReport(BaseModel):
    """What an upsert did.

    ``upserted`` counts records accepted. It is not a claim about how many were
    new: most backends cannot distinguish an insert from an update, and
    pretending otherwise would make an idempotency test assert a lie.
    """

    model_config = _FROZEN

    upserted: int = Field(default=0, ge=0)
    failed: tuple[str, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)


class DeleteReport(BaseModel):
    """What a delete did.

    ``deleted`` is ``None`` when the backend does not report a count, rather
    than zero. Zero would read as "nothing matched", which is a different and
    much more alarming fact.
    """

    model_config = _FROZEN

    deleted: int | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class VectorQuery(BaseModel):
    """A search against an index.

    Args:
        vector: Dense query vector.
        sparse_vector: Sparse query vector, for hybrid search.
        text: The query text, for backends that embed server-side.
        top_k: Maximum results.
        filter: Metadata filter. An operator the backend cannot express raises
            ``UnsupportedFilterError`` at query construction.
        namespace: Logical partition to search.
        include_text: Whether to return stored chunk text.
        min_score: Backend-side score floor, where supported.
    """

    model_config = _FROZEN

    vector: tuple[float, ...] = ()
    sparse_vector: dict[str, float] | None = None
    text: str | None = None
    top_k: int = Field(default=10, ge=1)
    filter: MetadataFilter | None = None
    namespace: str | None = None
    include_text: bool = True
    min_score: float | None = None


class ScoredRecord(BaseModel):
    """One search result.

    Args:
        id: The record's id.
        score: Backend-reported similarity. Raw, not normalised: normalisation
            across retrievers is a fusion concern, and normalising here would
            destroy the information fusion needs.
        metadata: Stored metadata.
        text: Stored text, when requested and available.
        document_id: The owning document.
        vector: The stored vector, when the backend returns it.
    """

    model_config = _FROZEN

    id: str
    score: float
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    text: str | None = None
    document_id: str | None = None
    vector: tuple[float, ...] | None = None


@runtime_checkable
class VectorIndex(Protocol):
    """Store and search vectors with filterable metadata.

    The port the whole portability claim rests on. Its contract kit is the
    largest in the library, and an adapter is not "supported" until it passes.

    Owns transport and filter translation. Does not own schema migration,
    backup, replication, or index lifecycle beyond ``ensure``.

    Attributes:
        name: The index or collection name.
    """

    name: str

    async def describe(self) -> IndexInfo:
        """Report the index's current state, including its epoch."""
        ...

    async def ensure(self, spec: IndexSpec) -> None:
        """Create the index if absent, and verify it matches the spec if present.

        A dimension mismatch must raise rather than be tolerated: writing 1536
        vectors into a 768-dimension index either errors per record or, worse,
        silently truncates.
        """
        ...

    async def upsert(self, records: Sequence[IndexRecord], ctx: RunContext) -> UpsertReport:
        """Insert or replace records by id.

        Must be idempotent: upserting identical records twice leaves one logical
        record. This is what makes re-ingestion safe, and the contract kit
        checks it directly.
        """
        ...

    async def delete(
        self,
        ctx: RunContext,
        *,
        ids: Sequence[str] | None = None,
        filter: MetadataFilter | None = None,  # noqa: A002 - name fixed by ARCHITECTURE.md §9.2
    ) -> DeleteReport:
        """Delete by explicit ids, by filter, or both.

        Deleting by filter is how ingestion removes a document's chunks, and it
        is the most commonly skipped requirement in a vector store integration.
        Without it, the system cites a document that was removed months ago.

        Raises:
            ValueError: If neither ``ids`` nor ``filter`` is given. Deleting
                everything by omission is never what anyone meant.
        """
        ...

    async def query(self, req: VectorQuery, ctx: RunContext) -> list[ScoredRecord]:
        """Search, returning results in descending score order."""
        ...

    def supports(self) -> IndexCapabilities:
        """Declare filter operators, hybrid support, and delete consistency."""
        ...


# --------------------------------------------------------------------------- #
# Reranker                                                                    #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Reranker(Protocol):
    """Reorder retrieved chunks by relevance to a query.

    An optional quality stage. When it fails the default is to skip it and
    record a ``Degradation``, not to fail the request.
    """

    id: ModelId

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
        ctx: RunContext,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant candidates, most relevant first."""
        ...


# --------------------------------------------------------------------------- #
# Ingestion inputs                                                            #
# --------------------------------------------------------------------------- #


class SourceEntry(BaseModel):
    """What a source knows about a document *before* fetching it.

    Listing must be cheap, because the sync engine lists everything on every run
    to compute a diff. ``revision`` is what makes that diff possible without
    downloading: an etag, an mtime, a commit sha.

    Args:
        id: Identifier within the source, normally a path or key.
        uri: Where the document lives.
        revision: The source's own version marker.
        size_bytes: Size, when the source reports it. Used for cost estimation
            in ``--plan``.
        media_type: IANA media type, used to select a parser.
        metadata: Source metadata carried through to the document.
    """

    model_config = _FROZEN

    id: str
    uri: str
    revision: str
    size_bytes: int | None = Field(default=None, ge=0)
    media_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SourceBlob(BaseModel):
    """A fetched document's bytes and what the source knew about it.

    Bytes rather than text: the parser decides the encoding, and
    ``Document.content_hash`` must be over the raw bytes so that a re-encoding
    is detected as a change.
    """

    model_config = _FROZEN

    entry: SourceEntry
    data: bytes
    media_type: str


@runtime_checkable
class DocumentParser(Protocol):
    """Turn a fetched blob into text and structure.

    Attributes:
        media_types: Which media types this parser handles. Used to select a
            parser, so it must be accurate.
    """

    media_types: frozenset[str]

    async def parse(self, blob: SourceBlob, ctx: RunContext) -> ParsedDocument:
        """Extract text and recoverable structure.

        Recoverable problems go in ``ParsedDocument.parse_warnings``. Only an
        unusable document raises ``IngestionError``, which quarantines it and
        lets the run continue.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Split a parsed document into indexable chunks.

    Chunk ids must come from ``ids.chunk_id`` so that re-chunking unchanged text
    produces unchanged ids. A chunker that invents ids breaks incremental
    ingestion completely and silently.
    """

    async def chunk(self, doc: ParsedDocument, ctx: RunContext) -> list[Chunk]:
        """Return the document's chunks, in document order."""
        ...


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #


@runtime_checkable
class CacheBackend(Protocol):
    """A byte-oriented key-value cache.

    Bytes, not objects: serialisation is the caller's decision, and a cache that
    pickles would be both a compatibility hazard and a deserialisation risk.

    A cache miss is ``None``. A backend that is unreachable should return
    ``None`` rather than raise: a cache outage must degrade latency, not
    availability.
    """

    async def get(self, key: str) -> bytes | None:
        """Return the cached value, or ``None`` on a miss."""
        ...

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        """Store a value, with an optional time to live in seconds."""
        ...

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix, returning how many were removed.

        Prefix deletion is how an epoch bump invalidates a whole cache layer.
        """
        ...


# --------------------------------------------------------------------------- #
# Observability                                                               #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Span(Protocol):
    """A unit of traced work.

    Deliberately tiny. The default adapter maps this onto OpenTelemetry GenAI
    semantic conventions; inventing a richer vocabulary here would guarantee
    that no backend understands it (ADR-010).
    """

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Attach an attribute, subject to the configured redaction paths."""
        ...

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Record a point-in-time event, such as a retry attempt."""
        ...

    def record_exception(self, exc: BaseException) -> None:
        """Attach an exception to the span."""
        ...

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Mark the span's outcome."""
        ...

    @property
    def trace_id(self) -> str | None:
        """The trace this span belongs to, for log correlation."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Where spans go.

    The default is a no-op tracer, never ``None``, so that no step has to guard
    a tracing call.
    """

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[Span]:
        """Open a span as an async context manager.

        Async so that a span survives an ``await`` inside it and nests correctly
        across task boundaries, which is exactly where ad hoc logging stops
        composing.
        """
        ...


@runtime_checkable
class MetricSink(Protocol):
    """Where counters, histograms and gauges go.

    Three methods, because a metrics abstraction that grows richer than this
    starts constraining which backends can implement it.
    """

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        ...

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record an observation in a histogram."""
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge."""
        ...


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #


class ToolPermissions(BaseModel):
    """What a tool is allowed to do, and on whose say-so.

    Present from day one because tool authorisation retrofitted later is a
    security incident, not a feature (ARCHITECTURE.md §9.2).

    This is a gate, not a sandbox, and the documentation says so. It cannot stop
    a tool that decides to misbehave; it decides whether the tool is called.

    Args:
        side_effect: ``read`` observes, ``write`` mutates state the operator
            owns, ``external`` reaches a third party.
        allow_untrusted_input: Whether this tool may be invoked when the model's
            decision was influenced by ``UNTRUSTED`` context. Defaults to
            ``False``, which is what contains indirect prompt injection: a
            document cannot talk the model into calling a write tool.
        allowlist: Optional restriction on argument values, interpreted by the
            tool. An empty allowlist means no restriction.
    """

    model_config = _FROZEN

    side_effect: Literal["read", "write", "external"] = "read"
    allow_untrusted_input: bool = False
    allowlist: tuple[str, ...] = ()


class ToolSpec(BaseModel):
    """A tool as described to a model.

    Provider-agnostic: adapters translate this into their own tool format, so
    a tool definition is written once and works across providers.
    """

    model_config = _FROZEN

    name: str
    description: str
    parameters: dict[str, JsonValue]


class ToolResult(BaseModel):
    """What a tool returned.

    Args:
        content: The result as text, which is what goes back to the model.
        structured: The result as data, for a caller that wants it typed.
        is_error: Whether the tool failed. A failed tool returns this rather
            than raising, so the loop can show the model what went wrong and let
            it recover.
        trust: How far the result may be trusted. Defaults to ``UNTRUSTED``,
            because a tool that fetches a web page returns whatever that page
            says.
    """

    model_config = _FROZEN

    content: str
    structured: JsonValue | None = None
    is_error: bool = False
    trust: Literal["trusted", "untrusted", "user_supplied"] = "untrusted"


@runtime_checkable
class Tool(Protocol):
    """Something an agent may call.

    Attributes:
        name: The name the model uses to call it.
        description: What it does, read by the model. This is prompt content and
            it matters.
        parameters: A Pydantic model defining the arguments. Typed, so a
            malformed call is caught before the tool runs.
        permissions: What this tool is allowed to do.
    """

    name: str
    description: str
    parameters: type[BaseModel]
    permissions: ToolPermissions

    async def __call__(self, args: BaseModel, ctx: RunContext) -> ToolResult:
        """Run the tool with validated arguments."""
        ...


# --------------------------------------------------------------------------- #
# Prompts                                                                     #
# --------------------------------------------------------------------------- #


class RenderedPrompt(BaseModel):
    """A prompt after rendering, with the version that produced it.

    ``version`` is carried because it belongs in the run manifest and in every
    generation cache key. A prompt edit that did not change the cache key would
    serve answers from the previous prompt indefinitely.
    """

    model_config = _FROZEN

    name: str
    version: str
    messages: tuple[Message, ...]


@runtime_checkable
class PromptStore(Protocol):
    """Where versioned prompt templates live.

    A port so that prompts can come from a file, a database or a prompt-
    management service. **Not** a template engine abstraction: one restricted,
    sandbox-safe renderer ships, and a user who needs Jinja passes a callable.

    The library ships no prompt *content*. Prompts are product assets and live in
    the generated project (INSTRUCTIONS.md §13.11).
    """

    async def render(
        self, name: str, variables: Mapping[str, JsonValue], *, version: str | None = None
    ) -> RenderedPrompt:
        """Render a named prompt, defaulting to its current version."""
        ...

    async def version_of(self, name: str) -> str:
        """Return a prompt's current version, without rendering it.

        Cheap, because it is called to build a cache key on every request.
        """
        ...


# --------------------------------------------------------------------------- #
# Ingestion state                                                             #
# --------------------------------------------------------------------------- #


class DocumentRecord(BaseModel):
    """A document's row in the ingestion manifest.

    Args:
        source_id: The configured source.
        document_id: The document.
        source_uri: Where it came from.
        revision: The revision last ingested.
        content_hash: The bytes last ingested. Compared before parsing, so a
            revision bump that did not change the content costs nothing.
        chunk_count: How many chunks it produced.
        indexed_at: ISO-8601 UTC timestamp of the last successful index.
        status: ``indexed``, ``quarantined`` or ``deleted``. Tombstones are kept
            rather than rows removed, so a re-appearing document is recognised.
    """

    model_config = _FROZEN

    source_id: str
    document_id: str
    source_uri: str
    revision: str
    content_hash: str
    chunk_count: int = Field(default=0, ge=0)
    indexed_at: str | None = None
    status: Literal["indexed", "quarantined", "deleted"] = "indexed"


class ChunkRecord(BaseModel):
    """A chunk's row in the ingestion manifest.

    ``embedded_with`` is the load-bearing field. It records which embedding
    model produced this chunk's vector, so that changing models re-embeds
    everything instead of leaving an index of mixed, incomparable vectors.
    """

    model_config = _FROZEN

    document_id: str
    chunk_id: str
    text_hash: str
    embedded_with: ModelId


class IndexMeta(BaseModel):
    """What the manifest remembers about an index.

    Mirrors the ``index_meta`` table in INSTRUCTIONS.md §6.2. The three fields
    beyond the name each prevent a specific silent corruption:

    - ``epoch`` is in every downstream cache key, so a re-index invalidates
      them rather than letting a cache serve pre-reindex answers indefinitely.
    - ``dimensions`` is checked before writing, so switching to a model of a
      different width fails loudly instead of erroring per record or, worse,
      silently truncating.
    - ``embed_model`` is what makes "has this chunk already been embedded"
      answerable. Without it, changing models would leave an index holding a
      mixture of incomparable vectors and no way to tell which was which.
    """

    model_config = _FROZEN

    index_name: str
    epoch: int = Field(default=0, ge=0)
    dimensions: int | None = Field(default=None, gt=0)
    embed_model: ModelId | None = None


@runtime_checkable
class StateStore(Protocol):
    """The ingestion manifest: what has been ingested, and with what.

    Deliberately narrow. This is not a general database port; it covers the
    manifest and nothing else (ARCHITECTURE.md §9.3).

    The manifest is committed incrementally, document by document, which is what
    makes a crashed run resumable rather than a restart from zero.
    """

    async def initialise(self) -> None:
        """Create or migrate the manifest schema. Idempotent."""
        ...

    async def documents(self, source_id: str) -> dict[str, DocumentRecord]:
        """Return every known document for a source, keyed by document id."""
        ...

    async def chunks(self, document_id: str) -> dict[str, ChunkRecord]:
        """Return every known chunk for a document, keyed by chunk id."""
        ...

    async def record_document(
        self, document: DocumentRecord, chunks: Sequence[ChunkRecord]
    ) -> None:
        """Commit one document and its chunks atomically.

        Atomic per document, and committed before the next document starts. That
        is what makes a crash at document 4,000 of 5,000 resumable.
        """
        ...

    async def tombstone(self, source_id: str, document_id: str) -> None:
        """Mark a document deleted, keeping the row.

        A tombstone rather than a deletion, so that a document which reappears
        is recognised as returning rather than as brand new.
        """
        ...

    async def index_epoch(self, index_name: str) -> int:
        """Return an index's current epoch, or ``0`` if it has none yet."""
        ...

    async def bump_epoch(self, index_name: str) -> int:
        """Increment and return an index's epoch.

        Called once at the end of a successful run. Every downstream cache key
        includes the epoch, so this is what invalidates them.
        """
        ...

    async def index_meta(self, index_name: str) -> IndexMeta | None:
        """Return what is recorded about an index, or ``None`` if it is new."""
        ...

    async def record_index_meta(self, meta: IndexMeta) -> None:
        """Record an index's dimensions and embedding model."""
        ...

    async def record_run(self, run_id: str, source_id: str, report: JsonValue) -> None:
        """Persist a run report, so ``ingest status`` can show the last run."""
        ...


# --------------------------------------------------------------------------- #
# Blob storage                                                                #
# --------------------------------------------------------------------------- #


@runtime_checkable
class BlobStore(Protocol):
    """Minimal object storage, for ingestion sources and eval artefacts.

    Three methods. Anything richer belongs to whichever storage service the user
    already has, not to this library.
    """

    async def get(self, key: str) -> bytes:
        """Read an object.

        Raises:
            KeyError: If the object does not exist.
        """
        ...

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        """Write an object, replacing any existing one."""
        ...

    async def list(self, prefix: str) -> list[str]:
        """List keys under a prefix."""
        ...


def _rebuild_models() -> None:
    """Resolve the forward references between the message and tool models."""
    for model in (Message, GenerationRequest, GenerationResult, GenerationDelta):
        model.model_rebuild()


_rebuild_models()
