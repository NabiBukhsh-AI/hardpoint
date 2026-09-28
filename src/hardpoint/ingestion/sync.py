"""The ingestion sync engine.

Implements the algorithm in INSTRUCTIONS.md §6.2, step by step. This is the
single hardest thing in the system to get right, and the one ARCHITECTURE.md §31
names as the most likely to be under-implemented, so the structure below follows
the specified steps literally and each one says what it is.

## The four properties, and what breaks without each

**Idempotent.** Re-running over an unchanged corpus performs zero parses, zero
embeddings and zero upserts. Without it, a nightly job re-embeds the whole
corpus every night and the bill is the first thing anyone notices.

**Incremental at chunk granularity.** A one-paragraph edit in a 200-page
document re-embeds the affected chunks only. This is where most of the embedding
spend lives, and getting it wrong is invisible except on an invoice.

**Correct about deletes.** A removed document's chunks leave the index. This is
the most commonly skipped requirement in a vector store integration, and it is
why a system cites a document that was withdrawn six months ago.

**Resumable.** Each document's manifest rows are committed before the next
document begins, so a crash at 4,000 of 5,000 resumes rather than restarting.

## Where the incrementality actually comes from

Two comparisons, and they do different jobs:

- ``revision`` decides whether to *fetch*, and comes from the source's listing.
  It is cheap and slightly unreliable.
- ``content_hash`` decides whether to *parse*, and is computed from the fetched
  bytes. A revision that moved without the content changing costs one fetch and
  a manifest update, not a re-embed.
- ``chunk_id`` plus ``embedded_with`` decides whether to *embed*. A chunk whose
  normalised text is unchanged has the same deterministic id, and if the
  manifest says it was embedded with this same model, it is skipped.

Remove any one of the three and the layer above it starts doing work it does not
need to do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from hardpoint.core.errors import ConfigError, IngestionError
from hardpoint.core.filters import F
from hardpoint.core.ids import content_hash as hash_bytes
from hardpoint.core.ids import document_id as derive_document_id
from hardpoint.core.ids import text_hash
from hardpoint.core.models import Chunk
from hardpoint.core.ports import (
    ChunkRecord,
    DocumentRecord,
    IndexMeta,
    IndexRecord,
    IndexSpec,
    SourceBlob,
    SourceEntry,
)
from hardpoint.core.tokens import estimate_tokens
from hardpoint.ingestion.report import IngestReport, QuarantineEntry, write_quarantine
from hardpoint.ingestion.state import utc_now
from hardpoint.ingestion.validation import ChunkValidator

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import (
        Chunker,
        DocumentParser,
        EmbeddingModel,
        StateStore,
        VectorIndex,
    )
    from hardpoint.ingestion.sources import Source

__all__ = ["SyncEngine", "SyncPlan"]


@dataclass(frozen=True)
class SyncPlan:
    """What a run would do, computed without side effects.

    What ``ingest run --plan`` prints. Its whole purpose is to let somebody see
    the embedding bill before paying it, which prevents a recurring class of
    expensive mistake (ARCHITECTURE.md §14).

    Args:
        source_id: The source.
        added: Documents in the source and not in the manifest.
        changed: Documents whose revision differs from the manifest.
        deleted: Documents in the manifest and no longer in the source.
        unchanged: Documents whose revision matches.
        estimated_embed_tokens: Upper bound on tokens to be embedded, derived
            from listed sizes without fetching anything.
        estimated_cost_usd: Spend, or ``None`` when no price was supplied.
            Never a guess presented as a number.
        reason: Why everything is being re-embedded, when it is.
    """

    source_id: str
    added: tuple[SourceEntry, ...] = ()
    changed: tuple[SourceEntry, ...] = ()
    deleted: tuple[DocumentRecord, ...] = ()
    unchanged: tuple[SourceEntry, ...] = ()
    estimated_embed_tokens: int = 0
    estimated_cost_usd: float | None = None
    reason: str = ""

    @property
    def has_work(self) -> bool:
        """Whether the run would do anything at all."""
        return bool(self.added or self.changed or self.deleted)

    def render(self, *, limit: int = 20) -> str:
        """Render the plan, truncating the document lists.

        Truncated because a first run over a large corpus would otherwise print
        every path and bury the number that matters.
        """
        cost = "unpriced" if self.estimated_cost_usd is None else f"${self.estimated_cost_usd:.4f}"
        lines = [
            f"plan for {self.source_id}:",
            f"  added      {len(self.added)}",
            f"  changed    {len(self.changed)}",
            f"  deleted    {len(self.deleted)}",
            f"  unchanged  {len(self.unchanged)}",
            f"  estimated  {self.estimated_embed_tokens} embedding tokens ({cost})",
        ]
        if self.reason:
            lines.append(f"  note       {self.reason}")

        for label, entries in (("added", self.added), ("changed", self.changed)):
            lines.extend(f"    + {label:<8} {entry.id}" for entry in entries[:limit])
            if len(entries) > limit:
                lines.append(f"    ... and {len(entries) - limit} more {label}")

        lines.extend(f"    - deleted  {record.source_uri}" for record in self.deleted[:limit])
        if len(self.deleted) > limit:
            lines.append(f"    ... and {len(self.deleted) - limit} more deleted")

        return "\n".join(lines)


@dataclass
class _DocumentOutcome:
    """What processing one document produced, before it is folded into the report."""

    indexed: bool = False
    quarantined: bool = False
    revision_only: bool = False
    chunks_indexed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0
    chunks_rejected: int = 0
    embed_calls: int = 0
    embed_tokens: int = 0
    cost_usd: float | None = 0.0
    quarantine: list[QuarantineEntry] = field(default_factory=list)


class SyncEngine:
    """Brings an index into line with a source, doing as little work as possible.

    Args:
        source: Where documents come from.
        state: The manifest.
        index: Where vectors go.
        embedder: What produces them.
        chunker: How documents are split.
        parsers: Parsers to choose from, by media type. The first whose
            ``media_types`` covers a document's type wins.
        validator: Chunk validation. Defaults to the standard rule set.
        index_name: The index's name in the manifest, which is what the epoch
            and the recorded dimensions are keyed on.
        concurrency: How many documents to process at once.
        fail_fast: Abort on the first document failure instead of quarantining
            it and continuing.
        quarantine_path: Where the rejection artefact is written.
        price_per_million_embed_tokens: Used for the ``--plan`` estimate. ``None``
            means unpriced, and the plan says so rather than inventing a figure.
        embed_batch_size: Texts per embedding call. A 200-chunk document is
            several calls, not one request the provider rejects as too large.

    Raises:
        ValueError: If no parsers were given. A sync engine that cannot parse
            anything would quarantine every document and report partial success.
    """

    def __init__(
        self,
        *,
        source: Source,
        state: StateStore,
        index: VectorIndex,
        embedder: EmbeddingModel,
        chunker: Chunker,
        parsers: Sequence[DocumentParser],
        validator: ChunkValidator | None = None,
        index_name: str = "primary",
        concurrency: int = 4,
        fail_fast: bool = False,
        quarantine_path: str | Path = "artefacts/quarantine.jsonl",
        price_per_million_embed_tokens: float | None = None,
        embed_batch_size: int = 128,
    ) -> None:
        if not parsers:
            raise ValueError(
                "SyncEngine needs at least one parser; with none, every document "
                "would be quarantined as unparseable."
            )
        self.source = source
        self.state = state
        self.index = index
        self.embedder = embedder
        self.chunker = chunker
        self.parsers = tuple(parsers)
        self.validator = validator or ChunkValidator()
        self.index_name = index_name
        self.concurrency = max(1, concurrency)
        self.fail_fast = fail_fast
        self.quarantine_path = Path(quarantine_path)
        self.price_per_million_embed_tokens = price_per_million_embed_tokens
        self.embed_batch_size = max(1, embed_batch_size)

    # ----------------------------------------------------------------- #
    # Steps 1-3: list, load, diff                                       #
    # ----------------------------------------------------------------- #

    async def _diff(
        self,
    ) -> tuple[list[SourceEntry], list[SourceEntry], list[DocumentRecord], list[SourceEntry], int]:
        """Steps 1-3. List the source, load the manifest, and partition.

        Returns ``(added, changed, deleted, unchanged, seen)``.
        """
        # Step 1: list the source.
        entries: dict[str, SourceEntry] = {}
        async for entry in self.source.list():
            entries[derive_document_id(self.source.id, entry.id)] = entry

        # Step 2: load the manifest for this source.
        manifest = await self.state.documents(self.source.id)

        # Step 3: partition. A tombstoned document that reappears counts as
        # added, not unchanged, because its chunks were removed from the index
        # and nothing would put them back otherwise.
        added: list[SourceEntry] = []
        changed: list[SourceEntry] = []
        unchanged: list[SourceEntry] = []

        for document_id, entry in entries.items():
            record = manifest.get(document_id)
            if record is None or record.status == "deleted":
                added.append(entry)
            elif record.revision != entry.revision:
                changed.append(entry)
            else:
                unchanged.append(entry)

        deleted = [
            record
            for document_id, record in manifest.items()
            if document_id not in entries and record.status != "deleted"
        ]

        return added, changed, deleted, unchanged, len(entries)

    async def plan(self) -> SyncPlan:
        """Step 4. Compute what a run would do, without side effects.

        The estimate is deliberately an upper bound derived from listed sizes:
        it does not fetch, because a plan that downloaded the corpus to tell you
        what it would cost to embed it has already spent most of the time.
        """
        added, changed, deleted, unchanged, _ = await self._diff()
        reason = ""

        meta = await self.state.index_meta(self.index_name)
        if meta is not None and meta.embed_model and meta.embed_model != self.embedder.id:
            # Every chunk was embedded with a different model, so none can be
            # reused, whatever the manifest says about individual chunks.
            reason = (
                f"the embedding model changed from {meta.embed_model!r} to "
                f"{self.embedder.id!r}, so every chunk will be re-embedded"
            )
            changed = [*changed, *unchanged]
            unchanged = []

        tokens = sum(estimate_tokens("x" * (entry.size_bytes or 0)) for entry in (*added, *changed))
        cost: float | None = None
        if self.price_per_million_embed_tokens is not None:
            cost = tokens / 1_000_000 * self.price_per_million_embed_tokens

        return SyncPlan(
            source_id=self.source.id,
            added=tuple(added),
            changed=tuple(changed),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            estimated_embed_tokens=tokens,
            estimated_cost_usd=cost,
            reason=reason,
        )

    # ----------------------------------------------------------------- #
    # The run                                                           #
    # ----------------------------------------------------------------- #

    async def run(self, ctx: RunContext) -> IngestReport:
        """Execute the full algorithm.

        Args:
            ctx: The run context. Its usage accumulator receives embedding spend.

        Returns:
            The report, also persisted to the manifest.

        Raises:
            ConfigError: If the index's recorded dimensions differ from the
                configured embedding model's.
            IngestionError: If ``fail_fast`` is set and a document fails.
        """
        await self.state.initialise()
        report = IngestReport(run_id=ctx.run_id, source_id=self.source.id, started_at=utc_now())

        model_changed = await self._check_index_compatibility()
        report.epoch_before = await self.state.index_epoch(self.index_name)

        added, changed, deleted, unchanged, seen = await self._diff()
        if model_changed:
            # The diff is revision-based, and changing the embedding model moves
            # no revisions. Without this, a model change would record the new
            # model, mark every document unchanged, and leave the index full of
            # vectors from the old one -- incomparable, and retrieving plausibly
            # and wrongly with nothing raising.
            changed, unchanged = [*changed, *unchanged], []
        report.documents_seen = seen
        report.added, report.changed = len(added), len(changed)
        report.deleted, report.unchanged = len(deleted), len(unchanged)

        await self.index.ensure(
            IndexSpec(
                name=self.index.name,
                dimensions=self.embedder.dimensions,
                embed_model=self.embedder.id,
            )
        )

        # Step 5: process added and changed documents, with bounded concurrency.
        outcomes: list[_DocumentOutcome] = []
        limiter = anyio.Semaphore(self.concurrency)

        async def process(entry: SourceEntry) -> None:
            async with limiter:
                outcomes.append(await self._process_document(entry, ctx, force_embed=model_changed))

        async with anyio.create_task_group() as group:
            for entry in (*added, *changed):
                group.start_soon(process, entry)

        for outcome in outcomes:
            _fold(report, outcome)

        # Step 6: deletions.
        for record in deleted:
            report.chunks_deleted += await self._delete_document(record, ctx)

        # Step 7: bump the epoch once, at the end of a successful run.
        report.epoch_after = await self.state.bump_epoch(self.index_name)

        # Step 8: write the artefacts.
        report.finished_at = utc_now()
        report.status = "partial" if report.documents_quarantined else "ok"
        self._write_quarantine(report)
        await self.state.record_run(ctx.run_id, self.source.id, report.as_json())
        return report

    async def _check_index_compatibility(self) -> bool:
        """Refuse to write vectors of the wrong width into an existing index.

        Writing 1536-dimension vectors into a 768-dimension index either errors
        per record or, on a backend that pads or truncates, silently produces an
        index whose vectors are meaningless. The second is far worse, because
        retrieval keeps working and simply returns the wrong things.

        Returns:
            Whether the embedding model changed since the last run. The caller
            uses it to force every document through, because the diff is
            revision-based and a model change moves no revisions.
        """
        meta = await self.state.index_meta(self.index_name)
        if meta is None:
            await self.state.record_index_meta(
                IndexMeta(
                    index_name=self.index_name,
                    epoch=0,
                    dimensions=self.embedder.dimensions,
                    embed_model=self.embedder.id,
                )
            )
            return False

        if meta.dimensions is not None and meta.dimensions != self.embedder.dimensions:
            raise ConfigError(
                f"Index {self.index_name!r} holds {meta.dimensions}-dimension vectors, "
                f"but the configured embedding model {self.embedder.id!r} produces "
                f"{self.embedder.dimensions}.",
                component=self.index_name,
                config_path=f"indexes.{self.index_name}",
                remedy=(
                    f"Either configure an embedding model with {meta.dimensions} "
                    f"dimensions, or create a new index and re-ingest from scratch. "
                    f"An index cannot hold vectors of two widths, and writing the "
                    f"wrong width would corrupt retrieval silently."
                ),
            )

        if meta.embed_model != self.embedder.id:
            # Not an error: switching models is legitimate. Recording the new one
            # is what makes every chunk's `embedded_with` mismatch, which is what
            # forces the re-embed.
            await self.state.record_index_meta(
                IndexMeta(
                    index_name=self.index_name,
                    epoch=meta.epoch,
                    dimensions=self.embedder.dimensions,
                    embed_model=self.embedder.id,
                )
            )
            return True

        return False

    # ----------------------------------------------------------------- #
    # Step 5: one document                                              #
    # ----------------------------------------------------------------- #

    async def _process_document(
        self, entry: SourceEntry, ctx: RunContext, *, force_embed: bool = False
    ) -> _DocumentOutcome:
        """Steps 5a-5g for a single document.

        Args:
            entry: The document to process.
            ctx: The run context.
            force_embed: Bypass the content-hash shortcut. Set when the embedding
                model changed: the bytes are unchanged, so the shortcut would
                skip the document entirely, but every vector still has to be
                recomputed with the new model.
        """
        outcome = _DocumentOutcome()
        document_id = derive_document_id(self.source.id, entry.id)

        # 5a: fetch and verify the content hash.
        try:
            blob = await self.source.fetch(entry)
        except IngestionError as exc:
            return self._quarantine(
                outcome, document_id, entry, rule="fetch", reason=exc.message, cause=exc
            )

        # The parser derives the document id from the entry's `source_id`, and
        # only the engine knows which source it asked. Injecting it here keeps
        # the id the parser computes identical to the one the manifest uses --
        # without it the two disagreed, and deleting a document by
        # `document_id` filter silently matched nothing.
        blob = blob.model_copy(
            update={
                "entry": blob.entry.model_copy(
                    update={"metadata": {**blob.entry.metadata, "source_id": self.source.id}}
                )
            }
        )

        digest = hash_bytes(blob.data)
        previous = (await self.state.documents(self.source.id)).get(document_id)
        if (
            not force_embed
            and previous is not None
            and previous.status == "indexed"
            and previous.content_hash == digest
        ):
            # The revision moved but the bytes did not. Update the revision so
            # the next run does not fetch again, and skip everything expensive.
            #
            # `status == "indexed"` is load-bearing: a tombstoned document whose
            # bytes are unchanged still needs full re-indexing, because its
            # chunks were deleted from the index when it was removed.
            await self.state.record_document(
                previous.model_copy(update={"revision": entry.revision, "indexed_at": utc_now()}),
                list((await self.state.chunks(document_id)).values()),
            )
            outcome.revision_only = True
            return outcome

        # 5b: parse. A parse failure quarantines the document and continues.
        parser = self._parser_for(blob)
        if parser is None:
            return self._quarantine(
                outcome,
                document_id,
                entry,
                rule="parse",
                reason=f"No parser handles media type {blob.media_type!r}.",
            )

        try:
            parsed = await parser.parse(blob, ctx)
        except IngestionError as exc:
            return self._quarantine(
                outcome, document_id, entry, rule="parse", reason=exc.message, cause=exc
            )

        # 5c: chunk, then validate.
        chunks = await self.chunker.chunk(parsed, ctx)
        validation = self.validator.validate(chunks)
        outcome.chunks_rejected = len(validation.rejections)
        outcome.quarantine.extend(
            QuarantineEntry(
                kind="chunk",
                document_id=document_id,
                source_uri=entry.uri,
                chunk_id=rejection.chunk_id,
                rule=rejection.rule,
                reason=rejection.reason,
                excerpt=rejection.excerpt,
            )
            for rejection in validation.rejections
        )

        if validation.quarantines_document:
            outcome.quarantined = True
            if self.fail_fast:
                raise IngestionError(
                    f"{entry.uri} produced a chunk that failed validation fatally.",
                    document_id=document_id,
                    source_id=self.source.id,
                    remedy="Inspect the quarantine artefact, or drop --fail-fast to continue.",
                )
            return outcome

        kept = list(validation.kept)

        # 5d: embed only what is not already embedded with this same model.
        known = await self.state.chunks(document_id)
        to_embed = [
            chunk for chunk in kept if not _already_embedded(chunk, known, self.embedder.id)
        ]
        outcome.chunks_reused = len(kept) - len(to_embed)

        vectors: dict[str, tuple[float, ...]] = {}
        for start in range(0, len(to_embed), self.embed_batch_size):
            batch = to_embed[start : start + self.embed_batch_size]
            result = await self.embedder.embed([chunk.text for chunk in batch], "document", ctx)
            vectors.update(
                (chunk.id, vector) for chunk, vector in zip(batch, result.vectors, strict=True)
            )
            outcome.chunks_embedded += len(batch)
            outcome.embed_calls += result.usage.calls
            outcome.embed_tokens += result.usage.embed_tokens
            if result.usage.cost_usd is None or outcome.cost_usd is None:
                outcome.cost_usd = None
            else:
                outcome.cost_usd += result.usage.cost_usd
            ctx.usage.record(
                "ingest.embed",
                calls=result.usage.calls,
                embed_tokens=result.usage.embed_tokens,
                cost_usd=result.usage.cost_usd,
                estimated=result.usage.estimated,
            )

        # 5e: upsert what was embedded, and remove chunk ids that no longer exist.
        #
        # Only the newly embedded chunks are upserted. A chunk whose text is
        # unchanged has the same deterministic id and is already in the index
        # with the same vector, so re-upserting it would mean re-embedding it
        # first -- which is exactly the cost this step exists to avoid.
        if vectors:
            await self.index.upsert(
                [
                    IndexRecord(
                        id=chunk.id,
                        vector=vectors[chunk.id],
                        metadata=dict(chunk.metadata),
                        document_id=chunk.document_id,
                        text=chunk.text,
                    )
                    for chunk in to_embed
                ],
                ctx,
            )
            outcome.chunks_indexed = len(to_embed)

        stale = sorted(set(known) - {chunk.id for chunk in kept})
        if stale:
            await self.index.delete(ctx, ids=stale)
            outcome.chunks_deleted = len(stale)

        # 5f: the DocumentStore write for parent-child retrieval is M4; parent
        # expansion is not part of the M1 pipeline, so there is nothing to store.

        # 5g: commit this document's manifest rows before the next one starts.
        await self.state.record_document(
            DocumentRecord(
                source_id=self.source.id,
                document_id=document_id,
                source_uri=entry.uri,
                revision=entry.revision,
                content_hash=digest,
                chunk_count=len(kept),
                indexed_at=utc_now(),
                status="indexed",
            ),
            [
                ChunkRecord(
                    document_id=document_id,
                    chunk_id=chunk.id,
                    text_hash=text_hash(chunk.text),
                    embedded_with=self.embedder.id,
                )
                for chunk in kept
            ],
        )
        outcome.indexed = True
        return outcome

    def _parser_for(self, blob: SourceBlob) -> DocumentParser | None:
        """Return the first parser whose declared media types cover this blob."""
        for parser in self.parsers:
            if blob.media_type in parser.media_types:
                return parser
        return None

    def _quarantine(
        self,
        outcome: _DocumentOutcome,
        document_id: str,
        entry: SourceEntry,
        *,
        rule: str,
        reason: str,
        cause: Exception | None = None,
    ) -> _DocumentOutcome:
        """Record a document-level rejection, honouring ``--fail-fast``."""
        outcome.quarantined = True
        outcome.quarantine.append(
            QuarantineEntry(
                kind="document",
                document_id=document_id,
                source_uri=entry.uri,
                rule=rule,
                reason=reason,
            )
        )
        if self.fail_fast:
            raise IngestionError(
                f"{entry.uri} could not be ingested: {reason}",
                document_id=document_id,
                source_id=self.source.id,
                remedy=(
                    "Fix or exclude the document, or drop --fail-fast to quarantine "
                    "it and continue with the rest of the corpus."
                ),
                cause=cause,
            )
        return outcome

    # ----------------------------------------------------------------- #
    # Step 6: deletions                                                 #
    # ----------------------------------------------------------------- #

    async def _delete_document(self, record: DocumentRecord, ctx: RunContext) -> int:
        """Remove a document's chunks from the index, then tombstone the manifest.

        By ``document_id`` filter rather than by collecting ids, because the
        manifest's chunk rows could be incomplete after a crash and the filter
        catches whatever is actually there. The tombstone comes second, so a
        crash between the two leaves the document still marked live and the next
        run deletes again -- which is harmless, where the reverse would leave
        orphaned vectors nothing would ever remove.
        """
        deleted = await self.index.delete(ctx, filter=F.field("document_id").eq(record.document_id))
        await self.state.tombstone(self.source.id, record.document_id)
        return deleted.deleted if deleted.deleted is not None else record.chunk_count

    # ----------------------------------------------------------------- #
    # Step 8: artefacts                                                 #
    # ----------------------------------------------------------------- #

    def _write_quarantine(self, report: IngestReport) -> None:
        """Write the quarantine artefact, degrading to a warning if it fails.

        Losing the artefact is bad; losing a completed ingestion because the
        artefact could not be written would be worse.
        """
        if not report.quarantine:
            return
        try:
            report.quarantine_path = str(write_quarantine(self.quarantine_path, report.quarantine))
        except OSError as exc:
            report.warnings.append(
                f"Could not write the quarantine artefact to {self.quarantine_path}: {exc}. "
                f"The rejections are in this report's `quarantine` field."
            )

    def __repr__(self) -> str:
        """Render the source and index this engine connects."""
        return f"SyncEngine(source={self.source.id!r}, index={self.index_name!r})"


def _already_embedded(chunk: Chunk, known: Mapping[str, ChunkRecord], model_id: str) -> bool:
    """Whether a chunk is already embedded, with this same model.

    Both halves matter. The id alone would reuse a vector produced by a
    different model, leaving an index of incomparable vectors that retrieves
    plausibly and wrongly.
    """
    record = known.get(chunk.id)
    return record is not None and record.embedded_with == model_id


def _fold(report: IngestReport, outcome: _DocumentOutcome) -> None:
    """Accumulate one document's outcome into the run report."""
    report.documents_indexed += int(outcome.indexed)
    report.documents_quarantined += int(outcome.quarantined)
    report.documents_revision_only += int(outcome.revision_only)
    report.chunks_indexed += outcome.chunks_indexed
    report.chunks_embedded += outcome.chunks_embedded
    report.chunks_reused += outcome.chunks_reused
    report.chunks_deleted += outcome.chunks_deleted
    report.chunks_rejected += outcome.chunks_rejected
    report.embed_calls += outcome.embed_calls
    report.embed_tokens += outcome.embed_tokens
    report.quarantine.extend(outcome.quarantine)

    # An unpriced embedding makes the run's total unknown rather than
    # understated (INSTRUCTIONS.md §13.8), and unknown stays unknown.
    if outcome.chunks_embedded and outcome.cost_usd is None:
        report.cost_usd = None
    elif report.cost_usd is not None:
        report.cost_usd += outcome.cost_usd or 0.0
