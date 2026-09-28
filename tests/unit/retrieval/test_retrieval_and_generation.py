"""Retrieval, assembly and generation. INSTRUCTIONS.md §6.4.

The locked rules are the two about untrusted content:

- Retrieved content is **never** rendered into a system message.
- The context block is delimited and labelled as data.

Both are asserted against the request the model actually receives, not against
the code that builds it, because these are exactly the rules a later refactor
breaks while making a prompt tidier.

The rest covers what ``ContextAssembler`` owns and every team rebuilds slightly
wrong: the token budget, the ordering, the citation keys, and -- most
importantly -- a record of everything that was dropped and why. Silent
truncation is the most common invisible quality bug in RAG.
"""

from __future__ import annotations

from typing import Any

import pytest

from hardpoint.core.errors import ConfigError, RetrievalError
from hardpoint.core.filters import F
from hardpoint.core.models import Answer, CharSpan, Chunk, RetrievedChunk, TrustLevel
from hardpoint.core.ports import GenerationDelta, IndexRecord
from hardpoint.generation import Generate, InMemoryPromptStore, PromptTemplate, render_template
from hardpoint.retrieval import (
    DEFAULT_PREAMBLE,
    Assembled,
    ContextAssembler,
    Retrieved,
    VectorRetriever,
    citations_for,
)
from hardpoint.testing import (
    FakeCache,
    FakeEmbeddingModel,
    FakeLanguageModel,
    InMemoryVectorIndex,
    ScriptedResponse,
    build_run_context,
    deterministic_vector,
)

ANSWER_PROMPT = """\
--- system ---
You answer questions about the handbook. Cite your sources.

--- user ---
Question: {{ question }}

{{ context }}
"""


def chunk(text: str, *, index: int = 0, document: str = "doc_1", **metadata: Any) -> Chunk:
    return Chunk(
        id=f"chk_{document}_{index}",
        document_id=document,
        index=index,
        text=text,
        span=CharSpan(start=0, end=len(text)),
        trust=TrustLevel.UNTRUSTED,
        metadata={"source_uri": f"{document}.md", **metadata},
    )


def retrieved(*texts: str, scores: list[float] | None = None) -> Retrieved:
    values = scores or [1.0 - index * 0.1 for index in range(len(texts))]
    return Retrieved(
        query="what is the refund window?",
        chunks=tuple(
            RetrievedChunk(
                chunk=chunk(text, index=index),
                score=score,
                rank=index,
                retriever="retrieve",
            )
            for index, (text, score) in enumerate(zip(texts, values, strict=True))
        ),
    )


async def assemble(source: Retrieved, **kwargs: Any) -> Assembled:
    return await ContextAssembler(**kwargs)(source, build_run_context())


# --------------------------------------------------------------------------- #
# VectorRetriever                                                             #
# --------------------------------------------------------------------------- #


async def populated_index(dimensions: int = 8) -> InMemoryVectorIndex:
    index = InMemoryVectorIndex(dimensions=dimensions)
    ctx = build_run_context()
    await index.upsert(
        [
            IndexRecord(
                id="chk_a",
                vector=deterministic_vector("refunds within 30 days", dimensions, salt="document"),
                text="Refunds are processed within 30 days.",
                document_id="doc_1",
                metadata={"source_uri": "billing.md", "folder": "policies"},
            ),
            IndexRecord(
                id="chk_b",
                vector=deterministic_vector("shipping takes 5 days", dimensions, salt="document"),
                text="Shipping takes five working days.",
                document_id="doc_2",
                metadata={"source_uri": "shipping.md", "folder": "logistics"},
            ),
        ],
        ctx,
    )
    return index


@pytest.mark.anyio
async def test_the_retriever_embeds_the_query_as_a_query_not_a_document() -> None:
    """Asymmetric models exist, and the wrong kind degrades recall silently.

    There is no error and no symptom other than worse answers, so this is
    asserted directly against what the embedder was asked for.
    """
    embedder = FakeEmbeddingModel(dimensions=8)
    step = VectorRetriever(await populated_index(), embedder)

    await step("refund window", build_run_context())

    assert [kind for _, kind in embedder.calls] == ["query"]


@pytest.mark.anyio
async def test_the_retriever_returns_ranked_chunks_with_the_query() -> None:
    step = VectorRetriever(await populated_index(), FakeEmbeddingModel(dimensions=8), top_k=2)
    result = await step("refund window", build_run_context())

    assert result.query == "refund window"
    assert len(result.chunks) == 2
    assert [c.rank for c in result.chunks] == [0, 1]
    assert result.chunks[0].score >= result.chunks[1].score


@pytest.mark.anyio
async def test_retrieved_chunks_are_untrusted() -> None:
    """Content that came through an index came from outside."""
    step = VectorRetriever(await populated_index(), FakeEmbeddingModel(dimensions=8))
    result = await step("anything", build_run_context())

    assert all(c.chunk.trust is TrustLevel.UNTRUSTED for c in result.chunks)


@pytest.mark.anyio
async def test_the_retriever_applies_a_fixed_filter() -> None:
    step = VectorRetriever(
        await populated_index(),
        FakeEmbeddingModel(dimensions=8),
        filter=F.field("folder").eq("policies"),
    )
    result = await step("anything", build_run_context())

    assert {c.chunk.id for c in result.chunks} == {"chk_a"}


@pytest.mark.anyio
async def test_a_per_request_filter_takes_precedence() -> None:
    """The seam an ACL-aware pipeline uses.

    The filter comes from the authenticated principal in the service layer, and
    is never derived from the query text -- that is the RAG vulnerability that
    actually gets exploited.
    """
    seen: list[str] = []

    def by_tenant(query: str, ctx: Any) -> Any:
        seen.append(query)
        return F.field("folder").eq("logistics")

    step = VectorRetriever(
        await populated_index(),
        FakeEmbeddingModel(dimensions=8),
        filter=F.field("folder").eq("policies"),
        filter_from=by_tenant,
    )
    result = await step("anything", build_run_context())

    assert seen == ["anything"]
    assert {c.chunk.id for c in result.chunks} == {"chk_b"}


@pytest.mark.anyio
async def test_the_retriever_records_its_embedding_spend() -> None:
    ctx = build_run_context()
    await VectorRetriever(await populated_index(), FakeEmbeddingModel(dimensions=8))(
        "anything", ctx
    )

    usage = ctx.usage.snapshot()
    assert usage.by_step["retrieve"].embed_tokens > 0


@pytest.mark.anyio
async def test_the_retriever_accepts_a_previous_retrieval() -> None:
    """So a pipeline can put an UNDERSTAND step before it without losing the query."""
    step = VectorRetriever(await populated_index(), FakeEmbeddingModel(dimensions=8))
    result = await step(Retrieved(query="reused query"), build_run_context())
    assert result.query == "reused query"


# --------------------------------------------------------------------------- #
# ContextAssembler: budget and drops                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_everything_dropped_is_recorded_with_a_reason() -> None:
    """**Silent truncation is the most common invisible quality bug in RAG.**"""
    long = "word " * 200
    result = await assemble(retrieved(long, long + "x", long + "y"), token_budget=80)

    assert len(result.context.items) < 3, "the budget must have excluded something"
    assert result.context.dropped, "exclusions must be recorded, not silent"
    assert all(drop.reason == "token_budget" for drop in result.context.dropped)
    assert all("budget" in (drop.detail or "") for drop in result.context.dropped)


@pytest.mark.anyio
async def test_the_budget_is_filled_in_score_order_not_input_order() -> None:
    """Ordering decides arrangement, not survival.

    Filling in document order would let a weak early chunk consume the budget a
    strong later one needed.
    """
    # Sized so exactly one passage fits alongside the preamble, which counts
    # against the budget because it is real tokens in the real prompt.
    result = await assemble(
        retrieved("weak " * 40, "strong " * 40, scores=[0.1, 0.9]),
        token_budget=130,
        ordering="document_order",
    )

    kept = [item.chunk.text for item in result.context.items]
    assert len(kept) == 1, kept
    assert "strong" in kept[0], "the higher-scoring chunk must be the survivor"


@pytest.mark.anyio
async def test_the_preamble_counts_against_the_budget() -> None:
    """It is real tokens in the real prompt, so it cannot be free.

    A budget too small to fit the preamble and one passage produces an empty
    context -- which is confusing until you read the drop records, so the drop
    records have to say it.
    """
    result = await assemble(retrieved("word " * 40), token_budget=60)

    assert result.context.items == []
    assert result.context.dropped, "the exclusion must be explained, not silent"
    assert "remaining" in (result.context.dropped[0].detail or "")


@pytest.mark.anyio
async def test_duplicate_passages_are_dropped() -> None:
    """Each near-duplicate spends budget on something already said."""
    result = await assemble(retrieved("identical text here", "identical text here"))

    assert len(result.context.items) == 1
    assert [d.reason for d in result.context.dropped] == ["duplicate"]


@pytest.mark.anyio
async def test_deduplication_can_be_switched_off() -> None:
    result = await assemble(
        retrieved("identical text here", "identical text here"), deduplicate=False
    )
    assert len(result.context.items) == 2


@pytest.mark.anyio
async def test_a_budget_that_fits_everything_drops_nothing() -> None:
    result = await assemble(retrieved("one", "two", "three"), token_budget=4000)
    assert len(result.context.items) == 3
    assert result.context.dropped == []


def test_a_non_positive_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ContextAssembler(token_budget=0)


# --------------------------------------------------------------------------- #
# ContextAssembler: ordering and citations                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_relevance_ordering_is_descending_score() -> None:
    result = await assemble(retrieved("a", "b", "c", scores=[0.2, 0.9, 0.5]))
    assert [item.chunk.text for item in result.context.items] == ["b", "c", "a"]


@pytest.mark.anyio
async def test_document_order_restores_source_order() -> None:
    result = await assemble(
        retrieved("a", "b", "c", scores=[0.2, 0.9, 0.5]), ordering="document_order"
    )
    assert [item.chunk.index for item in result.context.items] == [0, 1, 2]


@pytest.mark.anyio
async def test_relevance_with_edges_puts_the_strongest_at_the_ends() -> None:
    """Mitigates the tendency to lose the middle of a long context."""
    result = await assemble(
        retrieved("a", "b", "c", "d", scores=[0.4, 0.9, 0.1, 0.7]),
        ordering="relevance_with_edges",
    )
    texts = [item.chunk.text for item in result.context.items]

    assert texts[0] == "b", "the strongest goes first"
    assert texts[-1] == "d", "the second strongest goes last"
    assert "c" in texts[1:-1], "the weakest is buried in the middle"


@pytest.mark.anyio
async def test_numeric_citation_keys_are_assigned_in_prompt_order() -> None:
    result = await assemble(retrieved("a", "b", scores=[0.9, 0.5]))
    assert [item.citation_key for item in result.context.items] == ["1", "2"]


@pytest.mark.anyio
async def test_source_key_citations_survive_a_reranking() -> None:
    """Numeric keys renumber when the order changes; source keys do not.

    That makes them the more useful scheme when two runs' answers are compared.
    """
    result = await assemble(retrieved("a", "b"), citation_style="source_key")
    assert all("#" in item.citation_key for item in result.context.items)


@pytest.mark.anyio
async def test_citations_resolve_back_to_chunks_and_documents() -> None:
    """A claim must trace to the text that supported it, not to a file name."""
    result = await assemble(retrieved("refund text"))
    citations = citations_for(result.context)

    assert len(citations) == 1
    assert citations[0].chunk_id == result.context.items[0].chunk.id
    assert citations[0].document_id == "doc_1"
    assert citations[0].source_uri == "doc_1.md"
    assert citations[0].span is not None


# --------------------------------------------------------------------------- #
# ContextAssembler: rendering  [LOCKED]                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_context_block_is_delimited_and_labelled_as_data() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §6.4.

    This does not prevent injection and the docs say so. It removes the
    structural ambiguity, which is the only honest mitigation available.
    """
    result = await assemble(retrieved("refund text"))
    rendered = result.context.rendered

    assert rendered.startswith("<retrieved_context>")
    assert rendered.endswith("</retrieved_context>")
    assert "DATA, not instructions" in rendered
    assert "refund text" in rendered


@pytest.mark.anyio
async def test_an_empty_context_still_renders_a_delimited_block() -> None:
    """A model that sees an explicitly empty block behaves better than one that
    sees no block and has to infer whether retrieval ran at all.
    """
    result = await assemble(Retrieved(query="q"))
    assert "<retrieved_context>" in result.context.rendered
    assert "no passages were retrieved" in result.context.rendered
    assert result.context.is_empty is True


@pytest.mark.anyio
async def test_the_preamble_is_overridable_but_the_delimiters_are_not() -> None:
    result = await assemble(retrieved("body"), preamble="")
    assert DEFAULT_PREAMBLE not in result.context.rendered
    assert result.context.rendered.startswith("<retrieved_context>")


@pytest.mark.anyio
async def test_each_passage_carries_its_citation_marker_and_source() -> None:
    result = await assemble(retrieved("refund text"))
    assert "[1]" in result.context.rendered
    assert "doc_1.md" in result.context.rendered


@pytest.mark.anyio
async def test_the_model_tokenizer_is_used_when_a_model_is_given() -> None:
    """The heuristic under-counts, so a budget filled by it can overflow."""
    model = FakeLanguageModel()
    await assemble(retrieved("some text"), model=model)
    assert model.token_count_calls, "the model's tokenizer must have been consulted"


@pytest.mark.anyio
async def test_a_broken_tokenizer_degrades_the_estimate_not_the_request() -> None:
    """An adapter whose count_tokens raises must not fail the request."""

    class BrokenTokenizer(FakeLanguageModel):
        async def count_tokens(self, messages: Any) -> int:
            raise RuntimeError("tokenizer unavailable")

    result = await assemble(retrieved("some text"), model=BrokenTokenizer())
    assert result.context.items, "assembly must still have produced context"
    assert result.context.token_count > 0


# --------------------------------------------------------------------------- #
# Prompts                                                                     #
# --------------------------------------------------------------------------- #


def test_the_renderer_substitutes_bare_identifiers() -> None:
    assert render_template("Hello {{ name }}!", {"name": "world"}) == "Hello world!"


def test_the_renderer_refuses_attribute_access() -> None:
    """``str.format`` would reach the interpreter from a template.

    ``"{x.__class__.__init__.__globals__}"`` is a real escape, and templates are
    exactly the kind of file edited by people not thinking about sandboxing.
    """
    template = "{{ x.__class__ }} and {x.__class__.__init__.__globals__}"
    rendered = render_template(template, {"x": "safe"})

    assert "__globals__" in rendered, "left as literal text, not evaluated"
    assert "safe" not in rendered, "the malformed placeholder was not substituted"


def test_an_unsupplied_variable_is_an_error_not_a_hole() -> None:
    """A hole in a prompt produces a confidently wrong answer, not an error."""
    with pytest.raises(ConfigError) as exc_info:
        render_template("Answer {{ question }} using {{ context }}", {"question": "q"})
    assert "context" in str(exc_info.value)
    assert exc_info.value.remedy is not None


def test_a_prompt_version_is_a_content_hash() -> None:
    """Every generation cache key includes it.

    If it did not move when the text moved, editing a prompt would keep serving
    answers from the previous one indefinitely.
    """
    first = PromptTemplate("answer", "Say hello")
    same = PromptTemplate("answer", "Say hello")
    edited = PromptTemplate("answer", "Say hello politely")

    assert first.version == same.version
    assert first.version != edited.version
    assert len(first.version) == 12


def test_a_template_splits_into_role_sections() -> None:
    template = PromptTemplate("answer", ANSWER_PROMPT)
    rendered = template.render({"question": "why?", "context": "CTX"})

    assert [m.role for m in rendered.messages] == ["system", "user"]
    assert "handbook" in rendered.messages[0].content
    assert "why?" in rendered.messages[1].content


def test_a_template_without_markers_is_a_user_message() -> None:
    """The safe default: retrieved content must never reach a system message."""
    rendered = PromptTemplate("answer", "Just a body").render({})
    assert [m.role for m in rendered.messages] == ["user"]


def test_a_template_reports_the_variables_it_needs() -> None:
    """What ``doctor`` checks against, so a missing input is caught at startup."""
    assert PromptTemplate("answer", ANSWER_PROMPT).variables() == {"question", "context"}


@pytest.mark.anyio
async def test_an_unknown_prompt_names_the_ones_that_exist() -> None:
    store = InMemoryPromptStore({"answer": "hi"})
    with pytest.raises(ConfigError) as exc_info:
        await store.render("summarise", {})
    assert "answer" in (exc_info.value.remedy or "")


@pytest.mark.anyio
async def test_requesting_a_version_the_store_does_not_hold_is_refused() -> None:
    """Silently rendering a different version makes a reproduction meaningless."""
    store = InMemoryPromptStore({"answer": "hi"})
    with pytest.raises(ConfigError, match="version"):
        await store.render("answer", {}, version="000000000000")


# --------------------------------------------------------------------------- #
# Generate  [LOCKED]                                                          #
# --------------------------------------------------------------------------- #


def prompts() -> InMemoryPromptStore:
    return InMemoryPromptStore({"answer": ANSWER_PROMPT})


@pytest.mark.anyio
async def test_retrieved_content_never_reaches_a_system_message() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §6.4 and §13.12.

    Asserted against the request the model actually received, because this is
    the rule a later refactor breaks while tidying a prompt.
    """
    model = FakeLanguageModel([ScriptedResponse("Within 30 days.")])
    assembled = await assemble(retrieved("Refunds are processed within 30 days."))

    await Generate(model, prompts())(assembled, build_run_context())

    request = model.calls[0]
    system = [m for m in request.messages if m.role == "system"]
    assert system, "the prompt's own instructions must still be in a system message"

    for message in system:
        assert "<retrieved_context>" not in message.content
        assert "Refunds are processed" not in message.content

    users = " ".join(m.content for m in request.messages if m.role == "user")
    assert "<retrieved_context>" in users
    assert "Refunds are processed" in users


@pytest.mark.anyio
async def test_the_context_block_is_appended_when_the_prompt_omits_it() -> None:
    """A prompt that forgot the context still gets it -- in the right role."""
    model = FakeLanguageModel()
    store = InMemoryPromptStore({"answer": "--- user ---\nQuestion: {{ question }}"})
    assembled = await assemble(retrieved("body text"))

    await Generate(model, store)(assembled, build_run_context())

    users = [m for m in model.calls[0].messages if m.role == "user"]
    assert any("<retrieved_context>" in m.content for m in users)


@pytest.mark.anyio
async def test_generate_attaches_citations_from_the_bundle() -> None:
    model = FakeLanguageModel([ScriptedResponse("Within 30 days [1].")])
    assembled = await assemble(retrieved("Refunds within 30 days."))

    answer = await Generate(model, prompts())(assembled, build_run_context())

    assert answer.text == "Within 30 days [1]."
    assert len(answer.citations) == 1
    assert answer.citations[0].citation_key == "1"
    assert answer.citations[0].source_uri == "doc_1.md"


@pytest.mark.anyio
async def test_generate_populates_the_run_manifest() -> None:
    """What makes a run reproducible and a regression bisectable."""
    model = FakeLanguageModel(model_id="fake/reported-model")
    assembled = await assemble(retrieved("body"))

    answer = await Generate(model, prompts(), pipeline_name="support_qa")(
        assembled, build_run_context(run_id="run-7")
    )

    manifest = answer.manifest
    assert manifest.pipeline_name == "support_qa"
    assert manifest.model_ids["llm"] == "fake/reported-model"
    assert manifest.prompt_versions["answer"]
    assert manifest.hardpoint_version
    assert answer.run_id == "run-7"


@pytest.mark.anyio
async def test_the_manifest_records_the_model_the_adapter_reported() -> None:
    """Configured and reported differ exactly when a fallback fired."""
    model = FakeLanguageModel(model_id="fallback/secondary")
    answer = await Generate(model, prompts())(
        await assemble(retrieved("body")), build_run_context()
    )
    assert answer.manifest.model_ids["llm"] == "fallback/secondary"


@pytest.mark.anyio
async def test_generate_records_its_usage() -> None:
    ctx = build_run_context()
    await Generate(FakeLanguageModel(), prompts())(await assemble(retrieved("body")), ctx)

    usage = ctx.usage.snapshot()
    assert usage.by_step["generate"].calls == 1
    assert usage.by_step["generate"].prompt_tokens > 0


@pytest.mark.anyio
async def test_streaming_produces_the_same_answer() -> None:
    """Streaming changes when the first token arrives, not what is said."""
    assembled = await assemble(retrieved("body"))
    script = [ScriptedResponse("The refund window is thirty days.")]

    plain = await Generate(FakeLanguageModel(script), prompts())(assembled, build_run_context())
    streamed = await Generate(FakeLanguageModel(script), prompts(), stream=True)(
        assembled, build_run_context()
    )

    assert streamed.text == plain.text
    assert streamed.citations == plain.citations


@pytest.mark.anyio
async def test_stream_events_yields_text_as_it_arrives_then_the_answer() -> None:
    """What a service's SSE endpoint consumes, through ``Pipeline(on_delta=...)``."""
    step = Generate(FakeLanguageModel([ScriptedResponse("one two three")]), prompts())
    assembled = await assemble(retrieved("body"))

    events = [event async for event in step.stream_events(assembled, build_run_context())]
    pieces, final = events[:-1], events[-1]
    assert all(isinstance(piece, str) for piece in pieces)
    assert "".join(str(piece) for piece in pieces) == "one two three"
    assert len(pieces) > 1, "the point of streaming is more than one delta"
    assert isinstance(final, Answer)
    assert final.text == "one two three"
    assert final.citations, "the terminal event carries the citations"


@pytest.mark.anyio
async def test_a_streamed_answer_without_provider_usage_is_estimated_not_zero() -> None:
    class Silent(FakeLanguageModel):
        def stream(self, req: Any, ctx: Any) -> Any:
            async def deltas() -> Any:
                yield GenerationDelta(text="hello there")

            return deltas()

    answer = await Generate(Silent(), prompts(), stream=True)(
        await assemble(retrieved("body")), build_run_context()
    )
    usage = answer.usage.by_step["generate"]
    assert usage.estimated
    assert usage.completion_tokens > 0
    assert usage.prompt_tokens > 0


@pytest.mark.anyio
async def test_generation_cache_serves_a_repeat_and_misses_on_a_new_prompt_version() -> None:
    """``gen:`` keys hold the prompt version: an edited prompt never serves a stale answer."""
    cache = FakeCache()
    model = FakeLanguageModel([ScriptedResponse("cached answer")])
    assembled = await assemble(retrieved("body"))

    store = prompts()
    step = Generate(model, store, cache=True)
    first = await step(assembled, build_run_context(shared_cache=cache))
    second = await step(assembled, build_run_context(shared_cache=cache))
    assert (first.text, second.text) == ("cached answer", "cached answer")
    assert model.call_count == 1, "the repeat was served from the cache"
    assert second.usage.by_step["generate"].calls == 0

    store.add("answer", "Edited. {{ context }} {{ question }}")
    await Generate(model, store, cache=True)(assembled, build_run_context(shared_cache=cache))
    assert model.call_count == 2, "a new prompt version misses"


@pytest.mark.anyio
async def test_generation_cache_is_off_by_default() -> None:
    cache = FakeCache()
    model = FakeLanguageModel([ScriptedResponse("x")])
    assembled = await assemble(retrieved("body"))
    for _ in range(2):
        await Generate(model, prompts())(assembled, build_run_context(shared_cache=cache))
    assert model.call_count == 2
    assert cache.sets == 0


@pytest.mark.anyio
async def test_a_truncated_answer_is_reported_as_a_degradation() -> None:
    """``finish_reason == "length"`` means the answer is incomplete."""
    model = FakeLanguageModel([ScriptedResponse("partial", finish_reason="length")])
    answer = await Generate(model, prompts())(
        await assemble(retrieved("body")), build_run_context()
    )

    assert [d.reason for d in answer.degradations] == ["output_truncated"]
    assert answer.is_healthy is False


# --------------------------------------------------------------------------- #
# No-context policy                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_empty_retrieval_abstains_by_default() -> None:
    """Empty retrieval is a policy decision, not an error.

    Modelling it as an exception forces try/except into every application
    (ARCHITECTURE.md §6.2).
    """
    model = FakeLanguageModel()
    assembled = await assemble(Retrieved(query="q"))

    answer = await Generate(model, prompts())(assembled, build_run_context())

    assert answer.abstained is True
    assert answer.text, "an abstention carries a real message"
    assert answer.citations == []
    assert model.call_count == 0, "abstaining must not cost a model call"
    assert [d.reason for d in answer.degradations] == ["abstained"]


@pytest.mark.anyio
async def test_the_abstention_wording_comes_from_the_caller() -> None:
    """Product voice belongs in the generated project, not the library."""
    answer = await Generate(
        FakeLanguageModel(), prompts(), abstention_text="Sorry, I have nothing on that."
    )(await assemble(Retrieved(query="q")), build_run_context())

    assert answer.text == "Sorry, I have nothing on that."


@pytest.mark.anyio
async def test_answer_without_context_calls_the_model_and_degrades() -> None:
    model = FakeLanguageModel([ScriptedResponse("Answering unaided.")])
    answer = await Generate(model, prompts(), no_context_policy="answer_without_context")(
        await assemble(Retrieved(query="q")), build_run_context()
    )

    assert answer.abstained is False
    assert model.call_count == 1
    assert [d.reason for d in answer.degradations] == ["no_context"]


@pytest.mark.anyio
async def test_the_raise_policy_raises_with_a_remedy() -> None:
    with pytest.raises(RetrievalError) as exc_info:
        await Generate(FakeLanguageModel(), prompts(), no_context_policy="raise")(
            await assemble(Retrieved(query="q")), build_run_context()
        )

    remedy = exc_info.value.remedy or ""
    assert "top_k" in remedy
    assert "abstain" in remedy


@pytest.mark.anyio
async def test_escalate_abstains_but_says_so_differently() -> None:
    """``escalate`` and ``abstain`` differ only in what the caller does next."""
    answer = await Generate(FakeLanguageModel(), prompts(), no_context_policy="escalate")(
        await assemble(Retrieved(query="q")), build_run_context()
    )
    assert answer.abstained is True
    assert [d.reason for d in answer.degradations] == ["escalated"]


# --------------------------------------------------------------------------- #
# The M1 slice, composed                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_three_steps_compose_into_a_pipeline() -> None:
    """The M1 vertical slice: retrieve, assemble, generate, with citations.

    Runs through the real ``Pipeline``, so the steps are exercised the way a
    generated project composes them rather than by being called directly.
    """
    from hardpoint.runtime import Pipeline

    index = await populated_index()
    embedder = FakeEmbeddingModel(dimensions=8)
    model = FakeLanguageModel([ScriptedResponse("Refunds take 30 days [1].")])

    pipeline: Pipeline[str, Any] = Pipeline(
        "support_qa",
        [
            VectorRetriever(index, embedder, top_k=4),
            ContextAssembler(token_budget=500, model=model),
            Generate(model, prompts(), pipeline_name="support_qa"),
        ],
    )

    ctx = build_run_context(run_id="run-1")
    answer = await pipeline("what is the refund window?", ctx)

    assert answer.text == "Refunds take 30 days [1]."
    assert answer.citations, "the M1 DoD requires non-empty citations"
    assert answer.manifest.pipeline_name == "support_qa"

    usage = ctx.usage.snapshot()
    assert set(usage.by_step) == {"retrieve", "assemble_context", "generate"}
    assert usage.total_calls >= 2
