"""What an ingestion run did, and what it refused.

Implements step 8 of INSTRUCTIONS.md §6.2. Two artefacts come out of every run:
a machine-readable :class:`IngestReport` and a quarantine file listing every
document and chunk that was rejected, with the reason.

## Why the quarantine artefact is a file and not a log line

Nobody greps logs for the documents that failed to parse. A file that
``ingest run`` names at the end, that a human can open and scan, is the
difference between a corpus with forty silently missing documents and one where
somebody notices. ARCHITECTURE.md §14 makes it a requirement for exactly that
reason.

The most useful signal in it is not any single entry: it is a *document* that
produced a hundred chunk rejections, which means the parser chose wrongly or the
chunker's target is mis-set, and neither would otherwise surface at all.

## Counters distinguish "did nothing" from "did nothing because nothing changed"

``chunks_embedded == 0`` on a re-run is the headline success of incremental
ingestion. ``documents_seen == 0`` is a misconfigured source. A report that
collapsed both into "0 chunks indexed" would make the good outcome and the bad
one look identical.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.types import JsonValue

__all__ = ["IngestReport", "QuarantineEntry", "write_quarantine"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class QuarantineEntry(BaseModel):
    """One document or chunk that was rejected, and why.

    Args:
        kind: Whether a whole document was refused or a single chunk dropped.
        document_id: The document, always present -- a chunk rejection is only
            actionable if you know which document to open.
        source_uri: Where the document came from, so a human can open it.
        chunk_id: The chunk, when the rejection was chunk-level.
        rule: What refused it: a validation rule name, or ``parse`` / ``fetch``.
        reason: A human-readable explanation.
        excerpt: The start of the offending text.
    """

    model_config = _FROZEN

    kind: Literal["document", "chunk"]
    document_id: str
    source_uri: str = ""
    chunk_id: str | None = None
    rule: str = ""
    reason: str = ""
    excerpt: str = ""


class IngestReport(BaseModel):
    """What one ingestion run did.

    Written to the manifest's ``runs`` table and printed by ``ingest run``.

    Args:
        run_id: Identifier for this run.
        source_id: The source that was ingested.
        started_at: ISO-8601 UTC.
        finished_at: ISO-8601 UTC, set when the run ends.
        status: ``ok`` when nothing was quarantined, ``partial`` when something
            was, ``failed`` when the run aborted. ``partial`` exits non-zero, so
            a scheduled job surfaces quarantined documents rather than reporting
            success (ARCHITECTURE.md §18.2).
        documents_seen: How many the source listed.
        added, changed, deleted, unchanged: The diff.
        documents_indexed: How many were parsed, chunked and written.
        documents_quarantined: How many were refused.
        documents_revision_only: Documents whose revision moved but whose bytes
            did not. Counted separately because it is the cheap path and seeing
            it dominate tells you the source's revision is too eager.
        chunks_indexed: Chunks written to the index this run.
        chunks_embedded: Chunks actually sent to the embedding model. **The
            number incremental ingestion exists to keep small.**
        chunks_reused: Chunks already embedded with this model, and skipped.
        chunks_deleted: Chunk ids removed because they no longer exist.
        chunks_rejected: Chunks refused by validation.
        embed_calls: Calls made to the embedding model.
        embed_tokens: Tokens embedded, as reported or estimated.
        cost_usd: Spend, or ``None`` when the model is not priced. Never ``0``
            for an unpriced model (INSTRUCTIONS.md §13.8).
        epoch_before, epoch_after: The index epoch either side of the run.
        quarantine: Every rejection.
        quarantine_path: Where the artefact was written.
        warnings: Run-level problems that were not fatal.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    run_id: str
    source_id: str
    started_at: str
    finished_at: str | None = None
    status: Literal["ok", "partial", "failed"] = "ok"

    documents_seen: int = 0
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0

    documents_indexed: int = 0
    documents_quarantined: int = 0
    documents_revision_only: int = 0

    chunks_indexed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0
    chunks_rejected: int = 0

    embed_calls: int = 0
    embed_tokens: int = 0
    cost_usd: float | None = 0.0

    epoch_before: int = 0
    epoch_after: int = 0

    quarantine: list[QuarantineEntry] = Field(default_factory=list)
    quarantine_path: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def did_nothing(self) -> bool:
        """Whether the run embedded and wrote nothing.

        The expected outcome of re-ingesting an unchanged corpus, and the
        assertion the idempotency acceptance test makes.
        """
        return self.chunks_embedded == 0 and self.chunks_indexed == 0

    def as_json(self) -> dict[str, JsonValue]:
        """Return the report as a JSON-compatible mapping."""
        payload: dict[str, JsonValue] = self.model_dump(mode="json")
        return payload

    def render(self) -> str:
        """Render a short summary for the terminal.

        Ordered so the two numbers that matter come first: what changed, and how
        much was embedded.
        """
        cost = "unpriced" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        lines = [
            f"ingest {self.source_id}: {self.status}",
            f"  diff       +{self.added} ~{self.changed} -{self.deleted} "
            f"={self.unchanged} of {self.documents_seen}",
            f"  embedded   {self.chunks_embedded} chunks "
            f"({self.chunks_reused} reused, {self.embed_calls} calls, {cost})",
            f"  indexed    {self.chunks_indexed} chunks, "
            f"{self.chunks_deleted} deleted, {self.documents_indexed} documents",
            f"  epoch      {self.epoch_before} -> {self.epoch_after}",
        ]
        if self.documents_revision_only:
            lines.append(
                f"  unchanged bytes: {self.documents_revision_only} documents had a new "
                f"revision but identical content"
            )
        if self.quarantine:
            lines.append(
                f"  quarantined {self.documents_quarantined} documents, "
                f"{self.chunks_rejected} chunks -> {self.quarantine_path}"
            )
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def write_quarantine(path: str | Path, entries: Iterable[QuarantineEntry]) -> Path:
    """Write the quarantine artefact as JSONL.

    One entry per line, so the file is greppable, diffable between runs, and
    streamable when a corpus produces thousands of rejections.

    Args:
        path: Where to write.
        entries: The rejections.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written. The caller degrades to a warning
            rather than failing the run: losing the artefact is bad, losing the
            ingestion because the artefact could not be written is worse.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
    return destination
