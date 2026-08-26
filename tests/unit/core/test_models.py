"""The boundary data models. INSTRUCTIONS.md §5.1, ARCHITECTURE.md §10.

Two locked rules are asserted here reflectively rather than by inspection:

- ``Answer.text`` is never ``None``, and an abstention carries a real message.
- No model performs I/O. Models are data.

The rest checks the field-level decisions that exist to prevent specific
production failures: untrusted-by-default chunks, recorded drops, ``None``
rather than ``0`` for an unpriced model.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hardpoint.core import models
from hardpoint.core.models import (
    Answer,
    Block,
    CharSpan,
    Chunk,
    Citation,
    ContextBundle,
    ContextItem,
    Degradation,
    Document,
    DropRecord,
    ParsedDocument,
    RetrievedChunk,
    RunManifest,
    StepUsage,
    TrustLevel,
    Usage,
)

# Names that perform I/O or acquire a resource. INSTRUCTIONS.md §5.1 [LOCKED]
# forbids a model method from doing any of this.
IO_INDICATORS = frozenset(
    {
        "open",
        "read",
        "write",
        "save",
        "load",
        "fetch",
        "download",
        "upload",
        "connect",
        "query",
        "send",
        "post",
        "get_from",
        "persist",
        "commit",
        "flush",
    }
)


def model_classes() -> list[type[BaseModel]]:
    """Every Pydantic model defined in ``core.models``."""
    return [
        obj
        for obj in vars(models).values()
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


def manifest() -> RunManifest:
    return RunManifest(hardpoint_version="0.0.0", config_hash="abc", pipeline_name="qa")


def span(start: int = 0, end: int = 5) -> CharSpan:
    return CharSpan(start=start, end=end)


def chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "id": "chk_1",
        "document_id": "doc_1",
        "index": 0,
        "text": "hello",
        "span": span(),
    }
    fields.update(overrides)
    return Chunk(**fields)


# --------------------------------------------------------------------------- #
# Locked rules                                                                #
# --------------------------------------------------------------------------- #


def test_answer_text_is_required_and_not_nullable() -> None:
    """**[LOCKED]** ``Answer.text`` is never ``None``."""
    with pytest.raises(ValidationError):
        Answer(usage=Usage(), run_id="r", manifest=manifest())  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Answer(text=None, usage=Usage(), run_id="r", manifest=manifest())  # type: ignore[arg-type]


def test_abstention_must_carry_a_real_message() -> None:
    """**[LOCKED]** Abstention is a message plus a flag, not an empty string.

    Otherwise every caller invents its own copy for the "I don't know" case, and
    the abstention rate stops being measurable.
    """
    with pytest.raises(ValidationError, match="real message"):
        Answer(text="   ", abstained=True, usage=Usage(), run_id="r", manifest=manifest())

    answer = Answer(
        text="I could not find anything about that in the indexed documents.",
        abstained=True,
        usage=Usage(),
        run_id="r",
        manifest=manifest(),
    )
    assert answer.abstained is True
    assert answer.text


def test_no_model_performs_io() -> None:
    """**[LOCKED]** Models are data.

    Walks every model class and fails on a public method whose name suggests it
    touches the outside world. Catches the convenience method someone adds in a
    hurry, which is how a data model becomes a service client.
    """
    offenders: list[str] = []
    for cls in model_classes():
        for name, member in vars(cls).items():
            if name.startswith("_") or not callable(member):
                continue
            lowered = name.lower()
            if any(indicator in lowered for indicator in IO_INDICATORS):
                offenders.append(f"{cls.__name__}.{name}")
    assert not offenders, f"models must not perform I/O: {offenders}"


def test_every_model_is_frozen_and_forbids_unknown_fields() -> None:
    for cls in model_classes():
        config = cls.model_config
        assert config.get("frozen") is True, f"{cls.__name__} is not frozen"
        assert config.get("extra") == "forbid", f"{cls.__name__} allows extra fields"


def test_frozen_models_reject_mutation() -> None:
    item = chunk()
    with pytest.raises(ValidationError):
        item.text = "changed"  # type: ignore[misc]


def test_unknown_field_is_an_error_not_a_shrug() -> None:
    with pytest.raises(ValidationError):
        Degradation(step="rerank", reason="skipped", sevrity="warn")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Field-level decisions                                                       #
# --------------------------------------------------------------------------- #


def test_chunks_are_untrusted_by_default() -> None:
    """Everything ingested from outside is untrusted until something says otherwise.

    This default is the only structural mitigation against indirect prompt
    injection the library can honestly offer, and it only works if it is the
    default rather than an opt-in.
    """
    assert chunk().trust is TrustLevel.UNTRUSTED
    assert chunk(trust=TrustLevel.TRUSTED).trust is TrustLevel.TRUSTED


def test_trust_level_values_are_stable_strings() -> None:
    """The values are serialised into manifests and traces; they cannot drift."""
    assert [level.value for level in TrustLevel] == [
        "trusted",
        "untrusted",
        "user_supplied",
    ]


def test_unpriced_step_costs_none_not_zero() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §13.8. A zero would silently understate a bill."""
    assert StepUsage().cost_usd is None
    assert Usage().total_cost_usd is None


def test_token_count_defaults_to_none_not_zero() -> None:
    """Not counted and counted-as-zero are different facts."""
    assert chunk().token_count is None


def test_step_usage_totals() -> None:
    usage = StepUsage(prompt_tokens=10, completion_tokens=5, embed_tokens=2, calls=1)
    assert usage.total_tokens == 17


def test_usage_aggregates_across_steps() -> None:
    usage = Usage(
        by_step={
            "retrieve": StepUsage(embed_tokens=8, calls=1),
            "generate": StepUsage(prompt_tokens=100, completion_tokens=20, calls=1),
        }
    )
    assert usage.total_tokens == 128
    assert usage.total_calls == 2


def test_char_span_rejects_a_reversed_range() -> None:
    assert span(2, 5).length == 3
    with pytest.raises(ValidationError, match="precedes start"):
        CharSpan(start=5, end=2)
    with pytest.raises(ValidationError):
        CharSpan(start=-1, end=2)


def test_context_bundle_records_what_it_dropped() -> None:
    """Silent truncation is the most common invisible quality bug in RAG."""
    bundle = ContextBundle(
        items=[ContextItem(chunk=chunk(), citation_key="1", included_text="hello")],
        rendered="[1] hello",
        token_count=3,
        dropped=[DropRecord(chunk_id="chk_2", reason="token_budget", detail="over by 40")],
    )
    assert bundle.is_empty is False
    assert bundle.dropped[0].reason == "token_budget"


def test_empty_context_is_a_state_not_an_error() -> None:
    """Empty retrieval is a policy decision, not an exception (ARCHITECTURE.md §6.2)."""
    assert ContextBundle().is_empty is True


def test_drop_reason_is_a_closed_set() -> None:
    with pytest.raises(ValidationError):
        DropRecord(chunk_id="c", reason="felt_like_it")  # type: ignore[arg-type]


def test_answer_health_reflects_degradation_blocking_and_abstention() -> None:
    """What a caller checks before showing a response without a disclaimer."""
    healthy = Answer(text="yes", usage=Usage(), run_id="r", manifest=manifest())
    assert healthy.is_healthy is True

    degraded = Answer(
        text="yes",
        usage=Usage(),
        run_id="r",
        manifest=manifest(),
        degradations=[Degradation(step="rerank", reason="rerank_skipped")],
    )
    assert degraded.is_healthy is False

    blocked = Answer(text="refused", blocked=True, usage=Usage(), run_id="r", manifest=manifest())
    assert blocked.is_healthy is False


def test_manifest_carries_what_reproduces_a_run() -> None:
    """Without the epoch a re-index makes past answers unexplainable."""
    run_manifest = RunManifest(
        hardpoint_version="0.1.0",
        config_hash="deadbeef",
        pipeline_name="support_qa",
        model_ids={"llm": "openai/gpt-4o-mini"},
        prompt_versions={"answer": "v3"},
        index_epochs={"primary": 7},
        route="billing",
    )
    assert run_manifest.index_epochs["primary"] == 7
    assert run_manifest.prompt_versions["answer"] == "v3"
    assert run_manifest.route == "billing"


def test_parent_child_is_a_data_property() -> None:
    """Parent-child retrieval needs no separate architecture, only a field."""
    assert chunk().parent_id is None
    assert chunk(parent_id="chk_parent").parent_id == "chk_parent"


def test_document_identity_includes_its_source() -> None:
    document = Document(
        id="doc_1",
        source_uri="guide.md",
        source_id="docs",
        media_type="text/markdown",
        content_hash="abc",
        revision="etag-1",
    )
    assert document.source_id == "docs"
    assert document.metadata == {}


def test_parsed_document_tolerates_a_parser_that_found_no_structure() -> None:
    parsed = ParsedDocument(
        document=Document(
            id="doc_1",
            source_uri="a.txt",
            source_id="docs",
            media_type="text/plain",
            content_hash="abc",
            revision="1",
        ),
        text="hello",
    )
    assert parsed.blocks == []
    assert parsed.parse_warnings == []


def test_block_requires_a_known_kind_and_a_positive_level() -> None:
    assert Block(kind="heading", text="Title", level=1, span=span()).level == 1
    with pytest.raises(ValidationError):
        Block(kind="sidebar", text="x", span=span())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Block(kind="heading", text="x", level=0, span=span())


def test_retrieved_chunk_keeps_per_retriever_scores() -> None:
    """Fusion must not destroy the evidence of what each retriever contributed."""
    retrieved = RetrievedChunk(
        chunk=chunk(),
        score=0.8,
        rank=0,
        retriever="fused",
        raw_scores={"dense": 0.9, "sparse": 0.4},
    )
    assert retrieved.raw_scores == {"dense": 0.9, "sparse": 0.4}


def test_citation_maps_back_to_a_span() -> None:
    """A claim traces to the exact text that supported it, not to a file name."""
    citation = Citation(
        citation_key="1",
        chunk_id="chk_1",
        document_id="doc_1",
        source_uri="guide.md",
        span=span(10, 40),
    )
    assert citation.span is not None
    assert citation.span.length == 30


def test_degradation_defaults_to_warn() -> None:
    assert Degradation(step="rerank", reason="rerank_skipped").severity == "warn"
    assert Degradation(step="x", reason="y", severity="info").severity == "info"


def test_models_round_trip_through_json() -> None:
    """Every model ends up in a trace, a report or a cassette at some point."""
    answer = Answer(
        text="hello",
        usage=Usage(by_step={"generate": StepUsage(calls=1)}),
        run_id="run-1",
        manifest=manifest(),
        citations=[
            Citation(
                citation_key="1",
                chunk_id="chk_1",
                document_id="doc_1",
                source_uri="guide.md",
            )
        ],
        context=ContextBundle(
            items=[ContextItem(chunk=chunk(), citation_key="1", included_text="hello")],
            rendered="[1] hello",
            token_count=3,
        ),
    )
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
