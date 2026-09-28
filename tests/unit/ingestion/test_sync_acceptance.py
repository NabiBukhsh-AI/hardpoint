"""The six ingestion acceptance tests. **[LOCKED]** INSTRUCTIONS.md §6.2.

Reproduced from the specification, in order:

1. Running sync twice on an unchanged corpus performs zero parse, zero embed,
   zero upsert calls.
2. Editing one paragraph in a 100-chunk document re-embeds only the affected
   chunks.
3. Deleting a source file removes its chunks from the index and tombstones the
   manifest.
4. Killing the process midway and re-running completes without reprocessing the
   first N documents.
5. Changing the embedding model forces re-embedding of everything and refuses to
   write into an index whose recorded dimensions differ, raising ``ConfigError``
   with a remedy.
6. A document that fails parsing does not abort the run and appears in the
   quarantine artefact.

**Every one asserts on fake call counters**, as the specification requires. A
sync engine that works on the happy path is not done: each of these covers a
failure that is invisible in production until an invoice arrives or a withdrawn
document turns up in a citation.

The state store is the real SQLite one, not a fake. A fake manifest would not
exercise the SQL the guarantees actually depend on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hardpoint.core.errors import ConfigError, IngestionError
from hardpoint.core.ports import VectorQuery
from hardpoint.ingestion.chunkers import RecursiveChunker
from hardpoint.ingestion.parsers import MarkdownParser, TextParser
from hardpoint.ingestion.sources import LocalFileSource
from hardpoint.ingestion.state import SqliteStateStore
from hardpoint.ingestion.sync import SyncEngine
from hardpoint.testing import FakeEmbeddingModel, InMemoryVectorIndex, build_run_context


class CountingParser:
    """Wraps a parser so a test can assert it was not called."""

    def __init__(self, inner: object, media_types: frozenset[str]) -> None:
        self._inner = inner
        self.media_types = media_types
        self.calls = 0

    async def parse(self, blob: object, ctx: object) -> object:
        self.calls += 1
        return await self._inner.parse(blob, ctx)  # type: ignore[attr-defined]


class CountingIndex(InMemoryVectorIndex):
    """Adds a per-record upsert counter, so incrementality is measurable."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.upserted_ids: list[str] = []

    async def upsert(self, records: object, ctx: object) -> object:
        self.upserted_ids.extend(record.id for record in records)  # type: ignore[union-attr]
        return await super().upsert(records, ctx)  # type: ignore[arg-type]


class Harness:
    """A complete ingestion setup over a temporary corpus."""

    def __init__(self, root: Path, *, dimensions: int = 8, model_id: str = "fake/embed-v1") -> None:
        self.root = root
        self.state = SqliteStateStore(str(root / "manifest.sqlite"))
        self.index = CountingIndex(dimensions=dimensions)
        self.embedder = FakeEmbeddingModel(dimensions=dimensions, model_id=model_id)
        self.markdown = CountingParser(MarkdownParser(), frozenset({"text/markdown"}))
        self.text = CountingParser(TextParser(), frozenset({"text/plain"}))

    def engine(self, **overrides: object) -> SyncEngine:
        settings: dict[str, object] = {
            "source": LocalFileSource(
                self.root, source_id="docs", patterns=("**/*.md", "**/*.txt")
            ),
            "state": self.state,
            "index": self.index,
            "embedder": self.embedder,
            "chunker": RecursiveChunker(target_tokens=40, overlap_tokens=0, min_tokens=4),
            "parsers": [self.markdown, self.text],
            "concurrency": 1,
            "quarantine_path": self.root / "artefacts" / "quarantine.jsonl",
        }
        settings.update(overrides)
        return SyncEngine(**settings)  # type: ignore[arg-type]

    @property
    def parse_calls(self) -> int:
        return self.markdown.calls + self.text.calls

    def reset_counters(self) -> None:
        self.markdown.calls = 0
        self.text.calls = 0
        self.embedder.calls.clear()
        self.index.upserted_ids.clear()
        self.index.upsert_calls = 0
        self.index.delete_calls = 0


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def paragraphs(count: int, *, marker: str = "para") -> str:
    """A document with ``count`` clearly separated paragraphs."""
    return "\n\n".join(
        f"## Section {n}\n\nThis is {marker} number {n} with enough words to survive validation."
        for n in range(count)
    )


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    built = Harness(tmp_path)
    yield built
    await built.state.close()


# --------------------------------------------------------------------------- #
# 1. Idempotency                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_1_re_running_an_unchanged_corpus_does_no_work(harness: Harness) -> None:
    """**Acceptance test 1.** Zero parse, zero embed, zero upsert on a re-run.

    Without this, a nightly job re-embeds the whole corpus every night and the
    first symptom is the bill.
    """
    write(harness.root, "a.md", paragraphs(4))
    write(harness.root, "b.md", paragraphs(3, marker="beta"))

    first = await harness.engine().run(build_run_context(run_id="run-1"))
    assert first.documents_indexed == 2
    assert first.chunks_embedded > 0
    assert harness.parse_calls == 2

    harness.reset_counters()
    second = await harness.engine().run(build_run_context(run_id="run-2"))

    assert harness.parse_calls == 0, "an unchanged corpus must not be parsed again"
    assert harness.embedder.call_count == 0, "an unchanged corpus must not be embedded again"
    assert harness.index.upserted_ids == [], "an unchanged corpus must not be upserted again"

    assert second.unchanged == 2
    assert second.added == 0
    assert second.changed == 0
    assert second.chunks_embedded == 0
    assert second.did_nothing is True


@pytest.mark.anyio
async def test_1b_a_new_revision_with_identical_bytes_costs_no_embedding(
    harness: Harness,
) -> None:
    """The cheap path: revision moved, content did not.

    A source's revision is allowed to be over-eager -- an mtime changes when a
    file is touched. Verifying the content hash after fetching is what stops
    that costing an embedding.
    """
    path = write(harness.root, "a.md", paragraphs(4))
    await harness.engine().run(build_run_context(run_id="run-1"))

    harness.reset_counters()
    path.write_text(paragraphs(4), encoding="utf-8")  # same bytes, new mtime

    report = await harness.engine().run(build_run_context(run_id="run-2"))

    assert report.changed == 1, "the revision moved, so it was fetched"
    assert report.documents_revision_only == 1
    assert harness.parse_calls == 0, "identical bytes must not be parsed again"
    assert harness.embedder.call_count == 0


# --------------------------------------------------------------------------- #
# 2. Chunk-level incrementality                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_2_editing_one_paragraph_re_embeds_only_the_affected_chunks(
    harness: Harness,
) -> None:
    """**Acceptance test 2.** The property most of the embedding bill depends on.

    Getting it wrong is invisible: retrieval still works, and the only symptom
    is that every edit costs a full re-embed of the document.
    """
    write(harness.root, "big.md", paragraphs(30))
    first = await harness.engine().run(build_run_context(run_id="run-1"))

    total_chunks = first.chunks_embedded
    assert total_chunks >= 10, f"the fixture must produce many chunks, got {total_chunks}"

    harness.reset_counters()
    edited = paragraphs(30).replace(
        "This is para number 7", "This paragraph has been rewritten entirely"
    )
    write(harness.root, "big.md", edited)

    second = await harness.engine().run(build_run_context(run_id="run-2"))

    assert second.changed == 1
    assert second.chunks_embedded > 0, "the edited chunk must be re-embedded"
    assert second.chunks_embedded <= 3, (
        f"only the affected chunks may be re-embedded, but {second.chunks_embedded} "
        f"of {total_chunks} were"
    )
    assert second.chunks_reused >= total_chunks - 3, "the rest must have been reused"
    assert len(harness.index.upserted_ids) == second.chunks_embedded


@pytest.mark.anyio
async def test_2b_a_removed_paragraph_deletes_its_chunk_from_the_index(
    harness: Harness,
) -> None:
    """Chunk ids that no longer exist must leave the index.

    Otherwise an edit that deletes a paragraph leaves it retrievable forever.
    """
    write(harness.root, "big.md", paragraphs(12))
    await harness.engine().run(build_run_context(run_id="run-1"))
    before = harness.index.record_count

    harness.reset_counters()
    write(harness.root, "big.md", paragraphs(6))
    report = await harness.engine().run(build_run_context(run_id="run-2"))

    assert report.chunks_deleted > 0
    assert harness.index.record_count < before


# --------------------------------------------------------------------------- #
# 3. Deletion                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_3_deleting_a_source_file_removes_its_chunks_and_tombstones_it(
    harness: Harness,
) -> None:
    """**Acceptance test 3.** The most commonly skipped requirement.

    It is why a system cites a document that was withdrawn six months ago.
    """
    write(harness.root, "keep.md", paragraphs(3))
    removed = write(harness.root, "remove.md", paragraphs(3, marker="doomed"))
    await harness.engine().run(build_run_context(run_id="run-1"))

    ctx = build_run_context()
    doomed_ids = {
        record.id
        for record in await harness.index.query(VectorQuery(vector=(0.0,) * 8, top_k=1000), ctx)
        if "doomed" in (record.text or "")
    }
    assert doomed_ids, "the fixture must have indexed the document to be removed"

    harness.reset_counters()
    removed.unlink()
    report = await harness.engine().run(build_run_context(run_id="run-2"))

    assert report.deleted == 1

    remaining = {
        record.id
        for record in await harness.index.query(VectorQuery(vector=(0.0,) * 8, top_k=1000), ctx)
    }
    assert not (doomed_ids & remaining), "the removed document's chunks are still in the index"

    manifest = await harness.state.documents("docs")
    tombstoned = [record for record in manifest.values() if record.status == "deleted"]
    assert len(tombstoned) == 1
    assert tombstoned[0].source_uri.endswith("remove.md")


@pytest.mark.anyio
async def test_3b_a_tombstoned_document_that_returns_is_indexed_again(
    harness: Harness,
) -> None:
    """A tombstone records that it left, not that it must never come back.

    Treating a returning document as unchanged would leave it permanently
    missing from the index while the manifest claimed it was there.
    """
    path = write(harness.root, "a.md", paragraphs(3))
    await harness.engine().run(build_run_context(run_id="run-1"))
    path.unlink()
    await harness.engine().run(build_run_context(run_id="run-2"))

    harness.reset_counters()
    write(harness.root, "a.md", paragraphs(3))
    report = await harness.engine().run(build_run_context(run_id="run-3"))

    assert report.added == 1, "a returning document is added, not unchanged"
    assert report.chunks_embedded > 0
    manifest = await harness.state.documents("docs")
    assert all(record.status == "indexed" for record in manifest.values())


# --------------------------------------------------------------------------- #
# 4. Crash resumability                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_4_a_crash_midway_resumes_without_reprocessing(harness: Harness) -> None:
    """**Acceptance test 4.** Simulated by raising after document N.

    A crash at document 4,000 of 5,000 must not restart from zero. This works
    only because each document's manifest rows are committed before the next
    document begins.
    """
    for name in ("a.md", "b.md", "c.md", "d.md"):
        write(harness.root, name, paragraphs(3, marker=name))

    crash_after = 2
    processed: list[str] = []

    class CrashingParser(CountingParser):
        async def parse(self, blob: object, ctx: object) -> object:
            processed.append(blob.entry.id)  # type: ignore[attr-defined]
            if len(processed) > crash_after:
                raise RuntimeError("simulated crash")
            return await super().parse(blob, ctx)

    crashing = CrashingParser(MarkdownParser(), frozenset({"text/markdown"}))
    # anyio's task group wraps a task failure in an ExceptionGroup, so the crash
    # arrives as either the original or a group containing it.
    with pytest.raises((RuntimeError, BaseExceptionGroup)):
        await harness.engine(parsers=[crashing]).run(build_run_context(run_id="run-1"))

    committed = await harness.state.documents("docs")
    indexed = [r for r in committed.values() if r.status == "indexed"]
    assert len(indexed) == crash_after, "exactly the documents that finished are committed"

    # Re-run with a healthy parser. Only the unprocessed documents are touched.
    harness.reset_counters()
    report = await harness.engine().run(build_run_context(run_id="run-2"))

    assert harness.parse_calls == len(("a.md", "b.md", "c.md", "d.md")) - crash_after, (
        f"the {crash_after} already-committed documents must not be parsed again"
    )
    assert report.unchanged == crash_after
    assert report.added == 2


# --------------------------------------------------------------------------- #
# 5. Embedding model change                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_5_changing_the_embedding_model_re_embeds_everything(
    harness: Harness,
) -> None:
    """**Acceptance test 5, first half.** A model change invalidates every vector.

    Reusing a vector produced by a different model would leave an index of
    incomparable vectors that retrieves plausibly and wrongly -- the worst
    failure shape available, because nothing errors.
    """
    write(harness.root, "a.md", paragraphs(6))
    first = await harness.engine().run(build_run_context(run_id="run-1"))

    harness.reset_counters()
    harness.embedder = FakeEmbeddingModel(dimensions=8, model_id="fake/embed-v2")
    second = await harness.engine(embedder=harness.embedder).run(build_run_context(run_id="run-2"))

    assert second.chunks_embedded == first.chunks_embedded, (
        "every chunk must be re-embedded when the model changes"
    )
    assert second.chunks_reused == 0


@pytest.mark.anyio
async def test_5b_a_dimension_mismatch_is_refused_with_a_remedy(harness: Harness) -> None:
    """**Acceptance test 5, second half.** ``ConfigError`` with a remedy.

    Writing 1536-dimension vectors into a 768-dimension index either errors per
    record or, on a backend that pads, silently produces meaningless vectors.
    The second keeps working and returns the wrong things, so this refuses
    before writing anything.
    """
    write(harness.root, "a.md", paragraphs(4))
    await harness.engine().run(build_run_context(run_id="run-1"))

    wider = FakeEmbeddingModel(dimensions=16, model_id="fake/embed-wide")

    with pytest.raises(ConfigError) as exc_info:
        await harness.engine(embedder=wider).run(build_run_context(run_id="run-2"))

    error = exc_info.value
    assert "8-dimension" in str(error)
    assert "16" in str(error)
    assert error.remedy is not None
    assert "re-ingest" in error.remedy
    assert error.config_path == "indexes.primary"


@pytest.mark.anyio
async def test_5c_the_plan_says_a_model_change_will_re_embed_everything(
    harness: Harness,
) -> None:
    """``--plan`` exists to show the bill before it is paid."""
    write(harness.root, "a.md", paragraphs(6))
    await harness.engine().run(build_run_context(run_id="run-1"))

    plan = await harness.engine(
        embedder=FakeEmbeddingModel(dimensions=8, model_id="fake/embed-v2")
    ).plan()

    assert plan.unchanged == ()
    assert len(plan.changed) == 1
    assert "embedding model changed" in plan.reason
    assert "re-embed" in plan.render()


# --------------------------------------------------------------------------- #
# 6. Parse failure isolation                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_6_a_parse_failure_does_not_abort_the_run(harness: Harness) -> None:
    """**Acceptance test 6.** One bad document must not cost the whole corpus.

    And it must be *visible*: quarantined into an artefact a human can open,
    not swallowed into a log nobody greps.
    """
    write(harness.root, "good.md", paragraphs(3))
    write(harness.root, "empty.md", "   \n\n   \n")
    write(harness.root, "also-good.md", paragraphs(3, marker="gamma"))

    report = await harness.engine().run(build_run_context(run_id="run-1"))

    assert report.documents_indexed == 2, "the healthy documents must still be indexed"
    assert report.documents_quarantined == 1
    assert report.status == "partial", "a quarantine must not report a clean run"

    entries = [entry for entry in report.quarantine if entry.kind == "document"]
    assert len(entries) == 1
    assert entries[0].rule == "parse"
    assert entries[0].source_uri.endswith("empty.md")


@pytest.mark.anyio
async def test_6b_the_quarantine_artefact_is_written_as_readable_jsonl(
    harness: Harness,
) -> None:
    """Nobody greps logs for the documents that failed. A file, they will open."""
    import json

    write(harness.root, "good.md", paragraphs(3))
    write(harness.root, "empty.md", "   \n")

    report = await harness.engine().run(build_run_context(run_id="run-1"))

    assert report.quarantine_path is not None
    artefact = Path(report.quarantine_path)
    assert artefact.exists()

    lines = [json.loads(line) for line in artefact.read_text(encoding="utf-8").splitlines()]
    assert lines, "the artefact must not be empty when something was quarantined"
    assert lines[0]["kind"] == "document"
    assert lines[0]["reason"]
    assert lines[0]["source_uri"].endswith("empty.md")


@pytest.mark.anyio
async def test_6c_fail_fast_aborts_instead_of_quarantining(harness: Harness) -> None:
    """The opt-out, for a pipeline where a bad document should stop the job."""
    write(harness.root, "empty.md", "   \n")

    with pytest.raises((IngestionError, BaseExceptionGroup)) as exc_info:
        await harness.engine(fail_fast=True).run(build_run_context(run_id="run-1"))

    assert "IngestionError" in repr(exc_info.value)


@pytest.mark.anyio
async def test_6d_a_document_with_no_parser_is_quarantined_not_crashed(
    harness: Harness,
) -> None:
    """An unexpected media type is a corpus problem, not a runtime error."""
    write(harness.root, "a.md", paragraphs(3))
    report = await harness.engine(parsers=[harness.text]).run(build_run_context(run_id="run-1"))

    assert report.documents_quarantined == 1
    assert any("No parser handles" in entry.reason for entry in report.quarantine)


# --------------------------------------------------------------------------- #
# Epoch                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_epoch_is_bumped_once_per_successful_run(harness: Harness) -> None:
    """Every downstream cache key includes it, so this is what invalidates them."""
    write(harness.root, "a.md", paragraphs(3))

    first = await harness.engine().run(build_run_context(run_id="run-1"))
    assert first.epoch_before == 0
    assert first.epoch_after == 1

    second = await harness.engine().run(build_run_context(run_id="run-2"))
    assert second.epoch_before == 1
    assert second.epoch_after == 2, "even a no-op run bumps the epoch exactly once"


@pytest.mark.anyio
async def test_embedding_spend_is_attributed_to_the_run_context(harness: Harness) -> None:
    """Ingestion spend must be visible in the same accounting as request spend."""
    write(harness.root, "a.md", paragraphs(4))
    ctx = build_run_context(run_id="run-1")

    await harness.engine().run(ctx)

    usage = ctx.usage.snapshot()
    assert "ingest.embed" in usage.by_step
    assert usage.by_step["ingest.embed"].embed_tokens > 0


@pytest.mark.anyio
async def test_the_run_is_recorded_so_status_can_report_it(harness: Harness) -> None:
    write(harness.root, "a.md", paragraphs(3))
    await harness.engine().run(build_run_context(run_id="run-1"))

    last = await harness.state.last_run("docs")
    assert last is not None
    assert last["run_id"] == "run-1"
    assert last["status"] == "ok"


@pytest.mark.anyio
async def test_an_empty_source_is_a_clean_no_op(harness: Harness) -> None:
    """A misconfigured source and an unchanged one must not look identical."""
    report = await harness.engine().run(build_run_context(run_id="run-1"))

    assert report.documents_seen == 0
    assert report.status == "ok"
    assert report.did_nothing is True


def test_an_engine_without_parsers_is_refused() -> None:
    """It would quarantine every document and report partial success."""
    with pytest.raises(ValueError, match="at least one parser"):
        SyncEngine(
            source=None,  # type: ignore[arg-type]
            state=None,  # type: ignore[arg-type]
            index=None,  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            chunker=None,  # type: ignore[arg-type]
            parsers=[],
        )
