"""Sources, parsers, the chunker and validation. INSTRUCTIONS.md §6.2.

These are the inputs the sync engine drives, and the properties that matter are
the ones incremental ingestion depends on:

- **Listing is cheap and stable.** No content read, sorted order, so two runs
  over an unchanged corpus diff cleanly.
- **Chunk ids come from the real derivation.** Re-chunking unchanged text must
  produce unchanged ids, or every run re-embeds the whole corpus while
  reporting that it changed nothing.
- **Structure is respected.** Headings start new chunks and code fences are not
  cut in half, because both damage retrieval invisibly.
- **Rejections carry reasons.** A silently dropped chunk is one nobody will ever
  know was missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hardpoint.core.errors import IngestionError
from hardpoint.core.ids import chunk_id
from hardpoint.core.models import CharSpan, Chunk
from hardpoint.core.ports import SourceBlob, SourceEntry
from hardpoint.core.tokens import estimate_tokens
from hardpoint.ingestion.chunkers import HEADING_SEPARATOR, RecursiveChunker
from hardpoint.ingestion.parsers import MarkdownParser, TextParser, decode
from hardpoint.ingestion.sources import DEFAULT_MEDIA_TYPE, LocalFileSource, Source, media_type_for
from hardpoint.ingestion.validation import (
    BoilerplateRule,
    ChunkValidator,
    DuplicateRule,
    EmptyRule,
    TooLongRule,
    TooShortRule,
    default_rules,
)
from hardpoint.testing import build_run_context

MARKDOWN = """\
# Billing

Invoices are issued monthly.

## Refunds

Refunds are processed within 30 days of purchase.

```python
def refund(order):
    return order.total
```

## Disputes

Raise a dispute inside 60 days.
"""


def corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def blob(text: str, *, name: str = "guide.md", media_type: str = "text/markdown") -> SourceBlob:
    entry = SourceEntry(
        id=name,
        uri=f"file:///corpus/{name}",
        revision="1",
        media_type=media_type,
        metadata={"source_id": "docs"},
    )
    return SourceBlob(entry=entry, data=text.encode("utf-8"), media_type=media_type)


# --------------------------------------------------------------------------- #
# Media types                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.md", "text/markdown"),
        ("a.markdown", "text/markdown"),
        ("a.MD", "text/markdown"),
        ("a.txt", "text/plain"),
        ("a.unknownext", DEFAULT_MEDIA_TYPE),
    ],
)
def test_media_type_detection(name: str, expected: str) -> None:
    """Getting Markdown wrong sends a structured document to the text parser.

    Which silently loses every heading the chunker would have split on, with no
    error and no symptom other than worse retrieval.
    """
    assert media_type_for(name) == expected


# --------------------------------------------------------------------------- #
# LocalFileSource                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_listing_yields_entries_in_sorted_order(tmp_path: Path) -> None:
    """Filesystem walk order is not stable; an unstable order breaks report diffs."""
    root = corpus(tmp_path, {"c.md": "c", "a.md": "a", "b/inner.md": "inner"})
    source = LocalFileSource(root, source_id="docs")

    entries = [entry async for entry in source.list()]
    assert [entry.id for entry in entries] == ["a.md", "b/inner.md", "c.md"]


@pytest.mark.anyio
async def test_a_relative_root_yields_absolute_file_uris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration says `root: docs`, relative to the project directory.

    `Path.as_uri` refuses a relative path, so without resolving the root every
    `hardpoint ingest run` in a generated project crashed on the first file.
    """
    corpus(tmp_path, {"docs/a.md": "hello"})
    monkeypatch.chdir(tmp_path)
    entries = [entry async for entry in LocalFileSource("docs").list()]

    assert entries[0].id == "a.md"
    assert entries[0].uri.startswith("file:")


@pytest.mark.anyio
async def test_listing_reports_a_revision_without_reading_content(tmp_path: Path) -> None:
    """Listing must be cheap: the whole source is listed on every run."""
    root = corpus(tmp_path, {"a.md": "hello"})
    entries = [entry async for entry in LocalFileSource(root).list()]

    assert entries[0].revision.count(":") == 1, "mtime_ns:size"
    assert entries[0].size_bytes == 5
    assert entries[0].media_type == "text/markdown"


@pytest.mark.anyio
async def test_the_revision_changes_when_the_file_changes(tmp_path: Path) -> None:
    """What the whole diff step depends on."""
    root = corpus(tmp_path, {"a.md": "hello"})
    source = LocalFileSource(root)

    before = (await anext(aiter(source.list()))).revision
    (root / "a.md").write_text("hello world", encoding="utf-8")
    after = (await anext(aiter(source.list()))).revision

    assert before != after


@pytest.mark.anyio
async def test_patterns_and_excludes_are_applied(tmp_path: Path) -> None:
    root = corpus(tmp_path, {"keep.md": "k", "skip.md": "s", "notes.txt": "n", "image.png": "x"})
    source = LocalFileSource(root, patterns=("**/*.md",), exclude=("skip.md",))

    assert [entry.id async for entry in source.list()] == ["keep.md"]


@pytest.mark.anyio
async def test_folder_metadata_is_carried_for_filtering(tmp_path: Path) -> None:
    """The most commonly used filter in a small corpus, free of configuration."""
    root = corpus(tmp_path, {"policies/refunds.md": "r", "top.md": "t"})
    entries = {entry.id: entry async for entry in LocalFileSource(root).list()}

    assert entries["policies/refunds.md"].metadata["folder"] == "policies"
    assert entries["top.md"].metadata["folder"] == ""


@pytest.mark.anyio
async def test_fetch_reads_the_bytes(tmp_path: Path) -> None:
    root = corpus(tmp_path, {"a.md": "hello"})
    source = LocalFileSource(root)
    entry = await anext(aiter(source.list()))

    fetched = await source.fetch(entry)
    assert fetched.data == b"hello"
    assert isinstance(fetched.data, bytes), "bytes, so content_hash is over raw bytes"


@pytest.mark.anyio
async def test_a_file_removed_between_listing_and_fetching_is_an_ingestion_error(
    tmp_path: Path,
) -> None:
    """Quarantined and survivable, not a crashed run."""
    root = corpus(tmp_path, {"a.md": "hello"})
    source = LocalFileSource(root)
    entry = await anext(aiter(source.list()))
    (root / "a.md").unlink()

    with pytest.raises(IngestionError) as exc_info:
        await source.fetch(entry)
    assert exc_info.value.document_id == "a.md"
    assert exc_info.value.remedy is not None
    assert "quarantined" in exc_info.value.remedy


def test_a_missing_root_fails_at_construction(tmp_path: Path) -> None:
    """Better than listing an empty corpus and reporting that nothing changed."""
    with pytest.raises(IngestionError) as exc_info:
        LocalFileSource(tmp_path / "nope", source_id="docs")
    assert exc_info.value.remedy is not None


def test_the_local_source_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(LocalFileSource(tmp_path), Source)


# --------------------------------------------------------------------------- #
# Parsers                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_markdown_parser_recovers_headings_with_levels() -> None:
    """The structure the chunker splits on."""
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())

    headings = [(b.text, b.level) for b in parsed.blocks if b.kind == "heading"]
    assert headings == [("Billing", 1), ("Refunds", 2), ("Disputes", 2)]


@pytest.mark.anyio
async def test_the_markdown_parser_keeps_a_code_fence_whole() -> None:
    """A fence cut in half is syntactically broken and embeds as noise."""
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())

    code = [b for b in parsed.blocks if b.kind == "code"]
    assert len(code) == 1
    assert "def refund(order):" in code[0].text
    assert "return order.total" in code[0].text


@pytest.mark.anyio
async def test_the_markdown_parser_recognises_setext_headings() -> None:
    """Plenty of real corpora use them, and the underline would become a chunk."""
    parsed = await MarkdownParser().parse(
        blob("Title\n=====\n\nBody text here.\n"), build_run_context()
    )
    headings = [(b.text, b.level) for b in parsed.blocks if b.kind == "heading"]
    assert headings == [("Title", 1)]


@pytest.mark.anyio
async def test_the_markdown_parser_recognises_lists() -> None:
    parsed = await MarkdownParser().parse(blob("# T\n\n- one\n- two\n"), build_run_context())
    assert any(b.kind == "list" for b in parsed.blocks)


@pytest.mark.anyio
async def test_an_unterminated_code_fence_keeps_its_text() -> None:
    """Discarding it would lose content; a malformed document is still a document."""
    parsed = await MarkdownParser().parse(
        blob("# T\n\n```python\ndef f():\n    pass\n"), build_run_context()
    )
    assert any("def f():" in b.text for b in parsed.blocks)


@pytest.mark.anyio
async def test_markdown_without_headings_warns() -> None:
    """A quality signal: this document will chunk worse than the others."""
    parsed = await MarkdownParser().parse(blob("Just a paragraph.\n"), build_run_context())
    assert any("No headings" in warning for warning in parsed.parse_warnings)


@pytest.mark.anyio
async def test_the_text_parser_splits_on_blank_lines() -> None:
    parsed = await TextParser().parse(
        blob("First para.\n\nSecond para.\n", name="a.txt", media_type="text/plain"),
        build_run_context(),
    )
    assert [b.text for b in parsed.blocks] == ["First para.", "Second para."]


@pytest.mark.anyio
async def test_an_empty_document_is_quarantined_with_a_reason() -> None:
    """Indexing nothing and reporting success is the failure to avoid."""
    for parser in (TextParser(), MarkdownParser()):
        with pytest.raises(IngestionError) as exc_info:
            await parser.parse(blob("   \n\n  \n"), build_run_context())
        assert exc_info.value.remedy is not None


@pytest.mark.anyio
async def test_the_document_id_is_deterministic_and_source_scoped() -> None:
    """The same file under two sources is two documents."""
    from hardpoint.core.ids import document_id

    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    assert parsed.document.id == document_id("docs", "guide.md")
    assert parsed.document.content_hash == __import__(
        "hardpoint.core.ids", fromlist=["content_hash"]
    ).content_hash(MARKDOWN.encode("utf-8"))


def test_decoding_falls_back_and_warns_rather_than_failing() -> None:
    """One malformed byte in a 200-page document must not quarantine it."""
    text, warnings = decode(
        SourceBlob(
            entry=SourceEntry(id="a.txt", uri="file:///a.txt", revision="1"),
            data=b"caf\xe9 latte",
            media_type="text/plain",
        )
    )
    assert "latte" in text
    assert any("latin-1" in warning for warning in warnings)


def test_decoding_utf8_produces_no_warning() -> None:
    text, warnings = decode(
        SourceBlob(
            entry=SourceEntry(id="a.txt", uri="file:///a.txt", revision="1"),
            data="café".encode(),
            media_type="text/plain",
        )
    )
    assert text == "café"
    assert warnings == []


# --------------------------------------------------------------------------- #
# Chunker                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_chunk_ids_come_from_the_real_derivation() -> None:
    """A chunker that invents ids breaks incremental ingestion silently.

    Every run would re-embed every chunk while reporting that it had only
    embedded what changed.
    """
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=40, overlap_tokens=8).chunk(
        parsed, build_run_context()
    )

    for chunk in chunks:
        assert chunk.id == chunk_id(parsed.document.id, chunk.index, chunk.text)


@pytest.mark.anyio
async def test_re_chunking_unchanged_text_yields_identical_ids() -> None:
    """The property the whole idempotency guarantee rests on."""
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunker = RecursiveChunker(target_tokens=40, overlap_tokens=8)

    first = await chunker.chunk(parsed, build_run_context())
    second = await chunker.chunk(parsed, build_run_context())

    assert [c.id for c in first] == [c.id for c in second]


@pytest.mark.anyio
async def test_a_heading_starts_a_new_chunk() -> None:
    """The best available split point, because a heading marks a topic change."""
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=200, overlap_tokens=0).chunk(
        parsed, build_run_context()
    )

    paths = [chunk.metadata["heading_path"] for chunk in chunks]
    assert any("Refunds" in str(path) for path in paths)
    assert any("Disputes" in str(path) for path in paths)


@pytest.mark.anyio
async def test_overlap_never_carries_across_a_heading() -> None:
    """With overlap on, a new section must still open with its own heading.

    Carrying the previous section's tail across a heading put that text *above*
    the new heading, so the chunk took the previous section's heading path: a
    chunk about disputes labelled as refunds. Every other heading test here ran
    with ``overlap_tokens=0``, which is how it went unnoticed.
    """
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=200, overlap_tokens=60).chunk(
        parsed, build_run_context()
    )

    owners = {"issued monthly": "Billing", "30 days": "Refunds", "Raise a dispute": "Disputes"}
    for chunk in chunks:
        label = str(chunk.metadata["heading_path"]).split(HEADING_SEPARATOR)[-1]
        for text, section in owners.items():
            if text in chunk.text:
                assert section == label, f"{section} text sits in a chunk labelled {label!r}"


@pytest.mark.anyio
async def test_chunks_carry_their_heading_path_as_context() -> None:
    """ "within 30 days" retrieves for nothing; "Billing > Refunds" makes it findable."""
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=60, overlap_tokens=0).chunk(
        parsed, build_run_context()
    )

    refunds = [c for c in chunks if "30 days" in c.text]
    assert refunds, "the refunds text must be in some chunk"
    assert refunds[0].metadata["heading_path"] == f"Billing{HEADING_SEPARATOR}Refunds"
    assert "Billing" in refunds[0].text
    assert "Refunds" in refunds[0].text


@pytest.mark.anyio
async def test_the_heading_prefix_does_not_repeat_the_heading_it_opens_with() -> None:
    """A chunk beginning at a heading already contains it.

    Prefixing the full trail would print the heading immediately below itself
    and charge the token budget twice for the same words.
    """
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=60, overlap_tokens=0).chunk(
        parsed, build_run_context()
    )

    refunds = next(c for c in chunks if "30 days" in c.text)
    assert refunds.text.count("Refunds\n") <= 1, refunds.text
    assert refunds.text.startswith("Billing")


@pytest.mark.anyio
async def test_the_heading_path_can_be_switched_off() -> None:
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(
        include_heading_path=False, target_tokens=60, overlap_tokens=8
    ).chunk(parsed, build_run_context())
    assert not any(HEADING_SEPARATOR in chunk.text for chunk in chunks)


@pytest.mark.anyio
async def test_chunks_stay_within_the_target_where_the_text_allows() -> None:
    parsed = await MarkdownParser().parse(blob(MARKDOWN * 4), build_run_context())
    target = 80
    chunks = await RecursiveChunker(target_tokens=target, overlap_tokens=0).chunk(
        parsed, build_run_context()
    )

    oversized = [c for c in chunks if (c.token_count or 0) > target * 2]
    assert not oversized, f"{len(oversized)} chunks far exceeded the target"


@pytest.mark.anyio
async def test_a_single_oversized_paragraph_is_split_rather_than_emitted_whole() -> None:
    """The last-resort word split. It loses meaning, so it only happens here."""
    giant = "word " * 2000
    parsed = await TextParser().parse(
        blob(giant, name="a.txt", media_type="text/plain"), build_run_context()
    )
    chunks = await RecursiveChunker(target_tokens=100, overlap_tokens=0).chunk(
        parsed, build_run_context()
    )

    assert len(chunks) > 1
    assert all((c.token_count or 0) <= 200 for c in chunks)


@pytest.mark.anyio
async def test_chunks_are_untrusted_and_numbered_in_order() -> None:
    parsed = await MarkdownParser().parse(blob(MARKDOWN), build_run_context())
    chunks = await RecursiveChunker(target_tokens=40, overlap_tokens=8).chunk(
        parsed, build_run_context()
    )

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.trust.value == "untrusted" for c in chunks)
    assert all(c.document_id == parsed.document.id for c in chunks)


@pytest.mark.anyio
async def test_a_document_with_no_structure_still_chunks() -> None:
    """A parser that recovered nothing degrades to paragraph splitting, not to nothing."""
    from hardpoint.core.models import Document, ParsedDocument

    document = Document(
        id="doc_x",
        source_uri="a.txt",
        source_id="docs",
        media_type="text/plain",
        content_hash="0" * 64,
        revision="1",
    )
    parsed = ParsedDocument(document=document, text="One para.\n\nTwo para.\n", blocks=[])

    chunks = await RecursiveChunker(target_tokens=5, overlap_tokens=0, min_tokens=0).chunk(
        parsed, build_run_context()
    )
    assert len(chunks) >= 1


def test_overlap_must_be_smaller_than_the_target() -> None:
    """Otherwise every chunk contains the whole previous one and nothing advances."""
    with pytest.raises(ValueError, match="never advance"):
        RecursiveChunker(target_tokens=100, overlap_tokens=100)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def make_chunk(text: str, *, index: int = 0, tokens: int | None = None) -> Chunk:
    return Chunk(
        id=chunk_id("doc_1", index, text),
        document_id="doc_1",
        index=index,
        text=text,
        span=CharSpan(start=0, end=len(text)),
        token_count=tokens if tokens is not None else estimate_tokens(text),
    )


def test_an_empty_chunk_is_dropped_with_a_reason() -> None:
    report = ChunkValidator([EmptyRule()]).validate([make_chunk("   ")])
    assert report.kept == ()
    assert report.rejections[0].rule == "empty"
    assert report.rejections[0].disposition == "drop"


def test_a_short_chunk_is_dropped_and_the_reason_names_the_floor() -> None:
    """A three-token chunk matches on one term and contributes nothing."""
    report = ChunkValidator([TooShortRule(min_tokens=10)]).validate([make_chunk("tiny")])
    assert "below the 10-token floor" in report.rejections[0].reason
    assert report.rejections[0].excerpt == "tiny"


def test_an_over_long_chunk_quarantines_the_whole_document() -> None:
    """Not a drop: it means the chunker's target and the model's limit disagree.

    Dropping would index most of a document and hide the misconfiguration.
    """
    report = ChunkValidator([TooLongRule(max_tokens=5)]).validate([make_chunk("word " * 100)])
    assert report.rejections[0].disposition == "quarantine"
    assert report.quarantines_document is True
    assert "target_tokens" in report.rejections[0].reason


def test_duplicate_text_is_dropped_on_the_second_occurrence() -> None:
    """Each duplicate costs an embedding and then competes with the original."""
    report = ChunkValidator([DuplicateRule()]).validate(
        [make_chunk("Same body text here.", index=0), make_chunk("Same body text here.", index=1)]
    )
    assert len(report.kept) == 1
    assert report.rejections[0].rule == "duplicate"


def test_duplicate_detection_normalises_before_comparing() -> None:
    """A repeated header formatted two ways is still one duplicate."""
    report = ChunkValidator([DuplicateRule()]).validate(
        [make_chunk("Same body text.", index=0), make_chunk("Same   body\ttext.", index=1)]
    )
    assert len(report.kept) == 1


def test_boilerplate_is_judged_by_ratio_not_by_a_pattern_list() -> None:
    """The shapes vary by format, and a pattern list would never be complete."""
    validator = ChunkValidator([BoilerplateRule()])
    assert validator.validate([make_chunk("--- === --- === ---")]).kept == ()
    assert len(validator.validate([make_chunk("Real sentence with words.")]).kept) == 1


def test_the_first_matching_rule_wins() -> None:
    """A chunk reported under two rules is twice the artefact for one problem."""
    report = ChunkValidator(default_rules(min_tokens=100)).validate([make_chunk("   ")])
    assert [r.rule for r in report.rejections] == ["empty"]


def test_the_default_rule_order_reports_the_most_specific_cause() -> None:
    validator = ChunkValidator(default_rules())
    assert [rule.name for rule in validator.rules] == [
        "empty",
        "too_long",
        "too_short",
        "boilerplate",
        "duplicate",
    ]


def test_a_clean_document_passes_everything() -> None:
    chunks = [
        make_chunk(f"Paragraph number {n} with enough words to survive.", index=n) for n in range(5)
    ]
    report = ChunkValidator().validate(chunks)
    assert len(report.kept) == 5
    assert report.rejections == ()
    assert report.quarantines_document is False


def test_the_validator_reports_its_rules() -> None:
    assert "empty" in repr(ChunkValidator())
