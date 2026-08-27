"""The provider ports. ARCHITECTURE.md §9.2, §9.3, ADR-002.

Protocols have no behaviour to test, so these tests assert the things that
actually go wrong with a port layer:

- **A five-line class satisfies a port.** That is the whole argument for
  Protocols over ABCs, and if it stops being true the design has quietly
  changed.
- **No ``Component`` base class.** Forbidden shortcut §13.4, and easy to
  introduce by accident when two ports start sharing a helper.
- **The method sets match the specification**, so a port cannot grow a method
  without someone noticing.
- **No provider SDK type appears in a signature.** Checked structurally, since
  this is the property that lets an adapter be rewritten against a new major SDK
  version without a hardpoint breaking change.
"""

from __future__ import annotations

import inspect
import typing
from typing import Protocol, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

from hardpoint.core import ports
from hardpoint.core.filters import Filter
from hardpoint.core.ports import (
    BlobStore,
    CacheBackend,
    Chunker,
    DocumentParser,
    EmbeddingModel,
    GenerationRequest,
    GenerationResult,
    IndexRecord,
    LanguageModel,
    Message,
    MetadataFilter,
    MetricSink,
    PromptStore,
    Reranker,
    SourceBlob,
    SourceEntry,
    StateStore,
    Tool,
    ToolCall,
    ToolPermissions,
    ToolResult,
    Tracer,
    VectorIndex,
    VectorQuery,
)
from hardpoint.core.types import ModelId


def protocol_classes() -> list[type]:
    """Every Protocol defined in ``core.ports``."""
    return [
        obj
        for obj in vars(ports).values()
        if inspect.isclass(obj) and issubclass(obj, Protocol) and obj is not Protocol  # type: ignore[arg-type]
    ]


def port_methods(protocol: type) -> set[str]:
    """The public method names a Protocol declares."""
    return {
        name
        for name, member in vars(protocol).items()
        if not name.startswith("_") and (inspect.isfunction(member) or isinstance(member, property))
    }


# --------------------------------------------------------------------------- #
# The Protocol design (ADR-002)                                               #
# --------------------------------------------------------------------------- #


def test_a_five_line_class_satisfies_a_port() -> None:
    """The entire argument for Protocols over ABCs, asserted.

    No base class, no registration, no import of hardpoint into the class's own
    module. If this stops holding, adapters and fakes both get harder to write
    and the design has changed without anyone deciding to change it.
    """

    class TinyCache:
        async def get(self, key: str) -> bytes | None:
            return None

        async def set(self, key: str, value: bytes, ttl_s: int | None) -> None: ...

        async def delete_prefix(self, prefix: str) -> int:
            return 0

    assert isinstance(TinyCache(), CacheBackend)


def test_a_class_missing_a_method_does_not_satisfy_the_port() -> None:
    """Otherwise the runtime check would be decoration rather than a check."""

    class Incomplete:
        async def get(self, key: str) -> bytes | None:
            return None

    assert not isinstance(Incomplete(), CacheBackend)


def test_there_is_no_component_base_class() -> None:
    """**Forbidden shortcut §13.4.**

    Every port must derive from ``Protocol`` and nothing else. A shared base
    class is the tidy-looking change that destroys IDE support and couples every
    implementation to the library's hierarchy.
    """
    offenders: list[str] = []
    for protocol in protocol_classes():
        bases = [base.__name__ for base in protocol.__bases__ if base is not Protocol]
        if bases and bases != ["Generic"]:
            offenders.append(f"{protocol.__name__} inherits from {bases}")
    assert not offenders, offenders


def test_every_port_is_runtime_checkable() -> None:
    """``isinstance`` against a port is how ``doctor`` and the contract kits check."""
    for protocol in protocol_classes():
        assert getattr(protocol, "_is_runtime_protocol", False), (
            f"{protocol.__name__} is not @runtime_checkable"
        )


# --------------------------------------------------------------------------- #
# The port surface                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        (LanguageModel, {"generate", "stream", "count_tokens", "capabilities"}),
        (EmbeddingModel, {"embed"}),
        (
            VectorIndex,
            {"describe", "ensure", "upsert", "delete", "query", "supports"},
        ),
        (Reranker, {"rerank"}),
        (DocumentParser, {"parse"}),
        (Chunker, {"chunk"}),
        (CacheBackend, {"get", "set", "delete_prefix"}),
        (Tracer, {"span"}),
        (MetricSink, {"increment", "observe", "gauge"}),
        (PromptStore, {"render", "version_of"}),
        (BlobStore, {"get", "put", "list"}),
        (
            StateStore,
            {
                "initialise",
                "documents",
                "chunks",
                "record_document",
                "tombstone",
                "index_epoch",
                "bump_epoch",
                "index_meta",
                "record_index_meta",
                "record_run",
            },
        ),
    ],
)
def test_port_method_sets_match_the_specification(protocol: type, expected: set[str]) -> None:
    """Ports are narrow on purpose; a new method is an architectural change.

    Pinning the set means growing a port requires editing this list, which is a
    small speed bump in exactly the right place.
    """
    assert port_methods(protocol) == expected


def test_tool_is_callable_rather_than_having_a_run_method() -> None:
    """A tool is a function with metadata, not an object with a lifecycle."""
    assert "__call__" in vars(Tool)


def test_step_is_not_a_provider_port() -> None:
    """``Step`` belongs to ``runtime``: it is composition, not a boundary."""
    assert not hasattr(ports, "Step")


def test_retriever_is_not_a_port() -> None:
    """ARCHITECTURE.md §9.3 rejects it explicitly.

    A ``Retriever`` provider port would let an adapter hide the whole retrieval
    strategy, which is precisely the logic the user must be able to read.
    """
    assert not hasattr(ports, "Retriever")


def test_source_lives_with_the_sync_engine_not_here() -> None:
    """INSTRUCTIONS.md §6.2 places it in ``ingestion/sources.py``."""
    assert not hasattr(ports, "Source")


def test_metadata_filter_is_the_same_type_as_filter() -> None:
    """Two filter languages would be one too many."""
    assert MetadataFilter is Filter


# --------------------------------------------------------------------------- #
# No provider SDK types in signatures                                         #
# --------------------------------------------------------------------------- #

ALLOWED_MODULE_PREFIXES = (
    "hardpoint.",
    "builtins",
    "typing",
    "collections.abc",
    "contextlib",
    "pydantic",
    "abc",
)


def test_no_provider_sdk_type_appears_in_a_port_signature() -> None:
    """**Adapter rule [LOCKED]** (INSTRUCTIONS.md §6.3).

    This is what lets an adapter be rewritten against a new major SDK version
    without a hardpoint breaking change. Checked structurally rather than by
    reading, because it is exactly the kind of leak that arrives via a
    convenience parameter nobody reviewed closely.
    """
    # ``ports`` imports these only under TYPE_CHECKING, to avoid a runtime cycle
    # with ``context``. Supplying them here lets get_type_hints resolve, which is
    # the point: an unresolvable annotation would hide a leak rather than reveal
    # one, so a resolution failure is reported as a finding too.
    from hardpoint.core.capabilities import IndexCapabilities, ModelCapabilities
    from hardpoint.core.context import RunContext

    namespace = {
        **vars(ports),
        **vars(typing),
        "RunContext": RunContext,
        "ModelCapabilities": ModelCapabilities,
        "IndexCapabilities": IndexCapabilities,
    }
    offenders: list[str] = []

    for protocol in protocol_classes():
        for name, member in vars(protocol).items():
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            try:
                hints = get_type_hints(member, globalns=namespace)
            except Exception as exc:  # pragma: no cover - a resolution failure is a finding
                offenders.append(f"{protocol.__name__}.{name}: unresolvable hints ({exc})")
                continue

            for parameter, annotation in hints.items():
                for referenced in _referenced_types(annotation):
                    module = getattr(referenced, "__module__", "")
                    if module and not module.startswith(ALLOWED_MODULE_PREFIXES):
                        offenders.append(
                            f"{protocol.__name__}.{name}({parameter}) exposes "
                            f"{module}.{getattr(referenced, '__qualname__', referenced)}"
                        )

    assert not offenders, offenders


def _referenced_types(annotation: object) -> list[object]:
    """Flatten a type annotation into the concrete types it mentions."""
    args = typing.get_args(annotation)
    if not args:
        return [annotation]
    found: list[object] = []
    for arg in args:
        found.extend(_referenced_types(arg))
    return found


# --------------------------------------------------------------------------- #
# Port DTOs                                                                   #
# --------------------------------------------------------------------------- #


def test_generation_request_defaults_are_conservative() -> None:
    """Nothing is sampled, capped or tool-enabled unless asked for."""
    request = GenerationRequest(messages=(Message(role="user", content="hi"),))
    assert request.temperature is None
    assert request.max_output_tokens is None
    assert request.tools == ()
    assert request.tool_choice == "auto"
    assert request.provider_options == {}


def test_provider_options_stay_opaque() -> None:
    """A vendor-specific knob must never become part of this port.

    Keeping it an untyped mapping is what stops one provider's parameter from
    appearing in the shared request model.
    """
    request = GenerationRequest(
        messages=(), provider_options={"vendor_specific_thing": {"nested": [1, 2]}}
    )
    assert request.provider_options["vendor_specific_thing"] == {"nested": [1, 2]}


def test_message_roles_are_a_closed_set() -> None:
    """Retrieved content must never be renderable into a system role."""
    assert Message(role="system").role == "system"
    with pytest.raises(ValidationError):
        Message(role="retrieved_context")  # type: ignore[arg-type]


def test_generation_result_always_carries_usage() -> None:
    """**Invariant:** ``generate`` always returns ``Usage`` (ARCHITECTURE.md §9.2)."""
    with pytest.raises(ValidationError):
        GenerationResult(model_id="openai/gpt-4o")  # type: ignore[call-arg]


def test_finish_reason_can_report_truncation() -> None:
    """A caller must be able to tell a complete answer from a truncated one."""
    from hardpoint.core.models import StepUsage

    result = GenerationResult(
        text="partial", model_id="m", usage=StepUsage(calls=1), finish_reason="length"
    )
    assert result.finish_reason == "length"


def test_tool_call_arguments_stay_as_text() -> None:
    """Models emit malformed JSON often enough that repair is a normal path.

    Parsing here would discard the text a repair attempt needs.
    """
    call = ToolCall(id="1", name="search", arguments='{"q": "unterminated')
    assert isinstance(call.arguments, str)


def test_tool_permissions_deny_untrusted_input_by_default() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §10.

    This default is the practical containment for indirect prompt injection: an
    ingested document cannot talk the model into calling a write tool.
    """
    permissions = ToolPermissions()
    assert permissions.side_effect == "read"
    assert permissions.allow_untrusted_input is False
    assert permissions.allowlist == ()


def test_tool_result_defaults_to_untrusted() -> None:
    """A tool that fetches a web page returns whatever that page says."""
    assert ToolResult(content="x").trust == "untrusted"


def test_tool_failure_is_data_not_an_exception() -> None:
    """So the loop can show the model what went wrong and let it recover."""
    result = ToolResult(content="connection refused", is_error=True)
    assert result.is_error is True


def test_index_record_requires_a_deterministic_id() -> None:
    with pytest.raises(ValidationError):
        IndexRecord()  # type: ignore[call-arg]


def test_vector_query_carries_the_closed_filter_tree() -> None:
    from hardpoint.core.filters import F

    query = VectorQuery(vector=(0.1, 0.2), top_k=5, filter=F.field("tenant").eq("acme"))
    assert query.filter is not None
    assert query.filter.required_ops() == {"eq"}


def test_vector_query_rejects_a_non_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        VectorQuery(top_k=0)


def test_delete_report_distinguishes_unknown_from_zero() -> None:
    """Zero reads as "nothing matched", which is a much more alarming fact."""
    from hardpoint.core.ports import DeleteReport

    assert DeleteReport().deleted is None
    assert DeleteReport(deleted=0).deleted == 0


def test_index_info_starts_at_epoch_zero() -> None:
    from hardpoint.core.ports import IndexInfo

    info = IndexInfo(name="primary", dimensions=768, metric="cosine")
    assert info.epoch == 0


def test_index_spec_rejects_a_non_positive_dimension() -> None:
    from hardpoint.core.ports import IndexSpec

    with pytest.raises(ValidationError):
        IndexSpec(name="x", dimensions=0)


def test_source_blob_carries_bytes_not_text() -> None:
    """The parser decides the encoding, and content_hash is over raw bytes.

    Decoding here would make a re-encoding invisible to change detection.
    """
    entry = SourceEntry(id="a.md", uri="file://a.md", revision="1")
    blob = SourceBlob(entry=entry, data=b"# heading", media_type="text/markdown")
    assert isinstance(blob.data, bytes)


def test_chunk_record_records_the_embedding_model() -> None:
    """The field that makes an embedding-model change force a re-embed."""
    from hardpoint.core.ports import ChunkRecord

    record = ChunkRecord(
        document_id="doc_1", chunk_id="chk_1", text_hash="abc", embedded_with="openai/small"
    )
    assert record.embedded_with == "openai/small"


def test_document_record_defaults_to_indexed_and_supports_tombstones() -> None:
    from hardpoint.core.ports import DocumentRecord

    record = DocumentRecord(
        source_id="docs",
        document_id="doc_1",
        source_uri="a.md",
        revision="1",
        content_hash="abc",
    )
    assert record.status == "indexed"
    assert record.model_copy(update={"status": "deleted"}).status == "deleted"


def test_rendered_prompt_carries_its_version() -> None:
    """A prompt edit that did not change the cache key serves the old prompt forever."""
    from hardpoint.core.ports import RenderedPrompt

    rendered = RenderedPrompt(name="answer", version="v3", messages=())
    assert rendered.version == "v3"


def test_every_port_dto_is_frozen_and_forbids_unknown_fields() -> None:
    dto_classes = [
        obj
        for obj in vars(ports).values()
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]
    assert dto_classes, "no DTOs were discovered"
    for cls in dto_classes:
        assert cls.model_config.get("frozen") is True, f"{cls.__name__} is not frozen"
        assert cls.model_config.get("extra") == "forbid", f"{cls.__name__} allows extras"


def test_model_id_is_a_plain_string_alias() -> None:
    """A class here would earn nothing (INSTRUCTIONS.md §17.3)."""
    assert ModelId is str
