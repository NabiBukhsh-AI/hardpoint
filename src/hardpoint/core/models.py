"""The data models that cross every boundary.

Implements INSTRUCTIONS.md §5.1 and ARCHITECTURE.md §10. Every model is Pydantic
v2, frozen, and forbids unknown fields.

**Models are data.** None of them performs I/O, and none of them acquires a
resource, opens a connection or reads a file (INSTRUCTIONS.md §5.1
**[LOCKED]**). A test asserts this: it walks every model class here and fails on
a public method that is not a validator or a pure derived property.

Three field-level decisions are worth reading before the code, because each one
exists to prevent a specific production failure:

- ``Chunk.trust`` defaults to ``UNTRUSTED``. Everything ingested from outside is
  untrusted until something says otherwise. Guards and tool permissioning
  consult it, and it is the only structural mitigation against indirect prompt
  injection an SDK can honestly offer (ARCHITECTURE.md §10).
- ``ContextBundle.dropped`` records what did not fit and why. Silent truncation
  is the most common invisible quality bug in RAG.
- ``Answer.manifest`` captures the config hash, prompt versions, model ids and
  index epochs. It is what makes an answer reproducible and a regression
  bisectable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hardpoint.core.types import JsonValue, ModelId

__all__ = [
    "Answer",
    "Block",
    "CharSpan",
    "Chunk",
    "Citation",
    "ContextBundle",
    "ContextItem",
    "Degradation",
    "Document",
    "DropRecord",
    "ParsedDocument",
    "RetrievedChunk",
    "RunManifest",
    "StepUsage",
    "TrustLevel",
    "Usage",
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


class TrustLevel(StrEnum):
    """How far the content of a chunk may be trusted.

    Attributes:
        TRUSTED: Authored or vetted by the operator. Safe to act on.
        UNTRUSTED: Ingested from an external source. The default for everything
            the ingestion pipeline produces.
        USER_SUPPLIED: Provided by the end user in this request.
    """

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    USER_SUPPLIED = "user_supplied"


class CharSpan(BaseModel):
    """A half-open character range ``[start, end)`` into a document's text.

    Character offsets rather than byte offsets, so a span is meaningful after
    decoding and can be highlighted in a UI without re-deriving an encoding.
    """

    model_config = _MODEL_CONFIG

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        """Reject a span that ends before it starts."""
        if self.end < self.start:
            raise ValueError(f"CharSpan end ({self.end}) precedes start ({self.start})")
        return self

    @property
    def length(self) -> int:
        """Number of characters the span covers."""
        return self.end - self.start


class Document(BaseModel):
    """A unit of source content, before parsing.

    Args:
        id: Stable identifier, normally from ``ids.document_id``.
        source_uri: Where the document lives within its source.
        source_id: Which configured source produced it. Part of identity, so the
            same file reachable through two sources is two documents.
        media_type: IANA media type, used to select a parser.
        content_hash: sha256 of the raw bytes. Lets ingestion skip a document
            whose revision changed but whose bytes did not.
        revision: The source's own version marker: an etag, an mtime, a commit.
        metadata: Arbitrary source metadata, carried through to chunks.
    """

    model_config = _MODEL_CONFIG

    id: str
    source_uri: str
    source_id: str
    media_type: str
    content_hash: str
    revision: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Block(BaseModel):
    """A structural unit recovered by a parser.

    Blocks are what a structure-aware chunker uses to avoid splitting mid-table
    or mid-heading. A parser that recovers no structure returns no blocks; that
    is a quality signal, not an error.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["heading", "paragraph", "list", "table", "code", "caption", "other"]
    text: str
    level: int | None = Field(default=None, ge=1)
    span: CharSpan


class ParsedDocument(BaseModel):
    """A document's extracted text and recovered structure.

    Args:
        document: The source document.
        text: Full extracted text. Spans on blocks and chunks index into this.
        blocks: Recovered structure, empty when the parser found none.
        parse_warnings: Recoverable problems, for example an unreadable embedded
            image. Warnings do not quarantine a document; failures do.
    """

    model_config = _MODEL_CONFIG

    document: Document
    text: str
    blocks: list[Block] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """An indexable unit of text.

    Args:
        id: Deterministic, from ``ids.chunk_id``. Never random: a random id
            destroys idempotent re-ingestion (ADR-009).
        document_id: The owning document.
        index: Zero-based position within the document.
        text: The chunk's text.
        span: Where the text sits in the parsed document.
        parent_id: The larger chunk this one was split from, when parent-child
            retrieval is enabled. A data property, not a separate architecture.
        trust: Defaults to ``UNTRUSTED``.
        metadata: Filterable metadata, normally inherited from the document.
        token_count: Tokens under the embedding model's tokenizer, when known.
            ``None`` means not counted, never zero.
    """

    model_config = _MODEL_CONFIG

    id: str
    document_id: str
    index: int = Field(ge=0)
    text: str
    span: CharSpan
    parent_id: str | None = None
    trust: TrustLevel = TrustLevel.UNTRUSTED
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    token_count: int | None = Field(default=None, ge=0)


class RetrievedChunk(BaseModel):
    """A chunk together with why it was retrieved.

    Args:
        chunk: The chunk itself.
        score: Relevance. Raw from a single retriever; normalised to 0..1 after
            fusion, because raw scores across retrievers are not comparable.
        rank: Zero-based position in the result list that produced it.
        retriever: Name of the retriever that produced it, for attribution when
            several ran in parallel.
        raw_scores: Per-retriever scores kept through fusion, so a debugging
            session can see what each contributed.
    """

    model_config = _MODEL_CONFIG

    chunk: Chunk
    score: float
    rank: int = Field(ge=0)
    retriever: str
    raw_scores: dict[str, float] = Field(default_factory=dict)


class ContextItem(BaseModel):
    """One chunk as it will actually appear in the prompt.

    Args:
        chunk: The source chunk.
        citation_key: What appears in the rendered prompt, for example ``"1"``.
        included_text: The text actually included, which may be compressed or
            truncated relative to ``chunk.text``.
    """

    model_config = _MODEL_CONFIG

    chunk: Chunk
    citation_key: str
    included_text: str


class DropRecord(BaseModel):
    """Why a candidate chunk did not make it into the context.

    Every drop is recorded. Assembling context is where quality quietly dies,
    and an unexplained absence is unreviewable.
    """

    model_config = _MODEL_CONFIG

    chunk_id: str
    reason: Literal["token_budget", "duplicate", "below_threshold", "filtered", "compressed_out"]
    detail: str | None = None


class ContextBundle(BaseModel):
    """The assembled context handed to generation.

    Args:
        items: The included chunks in prompt order.
        rendered: The exact text placed in the prompt, with the delimiters that
            mark it as untrusted data rather than instructions.
        token_count: Tokens in ``rendered``.
        dropped: Everything that did not make it, with a reason.
    """

    model_config = _MODEL_CONFIG

    items: list[ContextItem] = Field(default_factory=list)
    rendered: str = ""
    token_count: int = Field(default=0, ge=0)
    dropped: list[DropRecord] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether no context was assembled.

        Callers use this to apply the configured no-context policy. Empty
        retrieval is a policy decision, not an error (ARCHITECTURE.md §6.2).
        """
        return not self.items


class Citation(BaseModel):
    """A mapping from a citation key in the answer back to its source.

    Carries the document and the character span, so a claim can be traced to the
    exact text that supported it rather than to a document name.
    """

    model_config = _MODEL_CONFIG

    citation_key: str
    chunk_id: str
    document_id: str
    source_uri: str
    span: CharSpan | None = None


class Degradation(BaseModel):
    """A machine-readable record that something was skipped or reduced.

    Degradation is data, not a log line (ADR-012). A pipeline that skipped
    reranking because the reranker timed out returns an answer *and* says so, so
    that SLOs, eval exclusion and user-facing disclaimers all have something to
    read.

    Args:
        step: The step that degraded.
        reason: A stable machine-readable code, for example ``"rerank_skipped"``.
        detail: Human-readable explanation.
        severity: ``info`` for an expected reduction, ``warn`` for an unplanned
            one.
    """

    model_config = _MODEL_CONFIG

    step: str
    reason: str
    detail: str = ""
    severity: Literal["info", "warn"] = "warn"


class StepUsage(BaseModel):
    """Resource consumption attributed to one step.

    ``cost_usd`` is ``None`` for an unpriced model, never ``0``
    (INSTRUCTIONS.md §13.8 **[LOCKED]**). A zero would be silently wrong and
    would quietly understate a bill; ``None`` is visibly unknown.

    ``estimated`` is set when token counts came from an estimator rather than
    from the provider, so a cost figure can be read with the right confidence.
    """

    model_config = _MODEL_CONFIG

    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embed_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt, completion and embedding tokens combined."""
        return self.prompt_tokens + self.completion_tokens + self.embed_tokens


class Usage(BaseModel):
    """Resource consumption for a whole run, attributed per step.

    ``total_cost_usd`` is ``None`` when any contributing step's cost is unknown,
    rather than a partial sum presented as a total.
    """

    model_config = _MODEL_CONFIG

    by_step: dict[str, StepUsage] = Field(default_factory=dict)
    total_cost_usd: float | None = Field(default=None, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        """Every token counted across every step."""
        return sum(step.total_tokens for step in self.by_step.values())

    @property
    def total_calls(self) -> int:
        """Every provider call counted across every step."""
        return sum(step.calls for step in self.by_step.values())


class RunManifest(BaseModel):
    """Everything needed to reproduce or bisect a run.

    Recorded on every answer. Without the index epoch a re-index makes past
    answers unexplainable; without the prompt version a prompt edit makes a
    regression untraceable.

    Args:
        hardpoint_version: The library version that produced the answer.
        config_hash: Hash of the resolved, redacted configuration.
        pipeline_name: Which pipeline ran.
        model_ids: Role to model id, for example ``{"llm": "openai/gpt-4o"}``.
        prompt_versions: Prompt name to version.
        index_epochs: Index name to the epoch that was queried.
        route: The route a dispatcher chose, when one did.
    """

    model_config = _MODEL_CONFIG

    hardpoint_version: str
    config_hash: str
    pipeline_name: str
    model_ids: dict[str, ModelId] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    index_epochs: dict[str, int] = Field(default_factory=dict)
    route: str | None = None


class Answer(BaseModel):
    """The result of a pipeline run.

    ``text`` is never ``None`` (INSTRUCTIONS.md §5.1 **[LOCKED]**). An abstention
    is a real message plus ``abstained=True``, not an empty string and not a
    null, because every caller would otherwise have to invent its own copy for
    the "I don't know" case.

    Args:
        text: The answer. Always present.
        structured: Structured output, when the pipeline asked for it.
        citations: Claims mapped back to chunks and spans.
        context: The context bundle that produced the answer, when retained.
        usage: Cost, tokens and latency, attributed per step.
        degradations: What was skipped or reduced, and why.
        blocked: Whether a guard blocked the output.
        abstained: Whether the pipeline declined to answer.
        run_id: Identifier of this run.
        trace_id: Trace identifier, when tracing was enabled.
        manifest: Everything needed to reproduce the run.
    """

    model_config = _MODEL_CONFIG

    text: str
    structured: JsonValue | None = None
    citations: list[Citation] = Field(default_factory=list)
    context: ContextBundle | None = None
    usage: Usage
    degradations: list[Degradation] = Field(default_factory=list)
    blocked: bool = False
    abstained: bool = False
    run_id: str
    trace_id: str | None = None
    manifest: RunManifest

    @model_validator(mode="after")
    def _check_abstention_carries_a_message(self) -> Self:
        """An abstention must say something. An empty abstention is a bug."""
        if self.abstained and not self.text.strip():
            raise ValueError(
                "An abstaining Answer must carry a real message in `text`. "
                "Set the abstention template in the generated project's prompts."
            )
        return self

    @property
    def is_healthy(self) -> bool:
        """Whether the run completed with nothing skipped, blocked or abstained.

        What a caller checks to decide whether a response is fit to show
        without a disclaimer, or fit to include in an eval sample.
        """
        return not self.degradations and not self.blocked and not self.abstained
