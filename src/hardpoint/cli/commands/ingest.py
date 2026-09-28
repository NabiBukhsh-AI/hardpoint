"""``hardpoint ingest run|plan|status``.

Ingestion is a job, never part of the request path (INSTRUCTIONS.md §13.18). It
is safe to run while the service is serving, because the epoch bump at the end
of a run is what invalidates the caches that depend on the old index.
"""

from collections.abc import Sequence

from hardpoint.core.errors import ConfigError
from hardpoint.ingestion.chunkers.recursive import RecursiveChunker
from hardpoint.ingestion.parsers.text import MarkdownParser, TextParser
from hardpoint.ingestion.sync import SyncEngine
from hardpoint.runtime.resources import Resources, build_source

__all__ = ["build_engine", "ingest", "ingest_status", "source_names"]


def source_names(res: Resources, requested: Sequence[str]) -> list[str]:
    """Return the sources to ingest: the ones asked for, or every configured one.

    Raises:
        ConfigError: If nothing is configured to ingest.
    """
    names = list(requested) or sorted(res.config.sources)
    if not names:
        raise ConfigError(
            "No ingestion sources are configured.",
            config_path="sources",
            remedy="Add `sources.docs: {type: local_files, root: docs}` to config/base.yaml.",
        )
    return names


async def build_engine(res: Resources, source: str, *, fail_fast: bool = False) -> SyncEngine:
    """Build the sync engine for one configured source."""
    ingestion = res.config.ingestion
    chunking = ingestion.chunking
    return SyncEngine(
        source=await build_source(res, source),
        state=res.state,
        index=res.index(ingestion.index),
        embedder=res.embedder,
        chunker=RecursiveChunker(
            target_tokens=chunking.target_tokens,
            overlap_tokens=chunking.overlap_tokens,
            min_tokens=chunking.min_tokens,
        ),
        parsers=[MarkdownParser(), TextParser()],
        index_name=ingestion.index,
        concurrency=ingestion.concurrency,
        fail_fast=fail_fast or ingestion.fail_fast,
        quarantine_path=ingestion.quarantine_path,
        embed_batch_size=ingestion.embed_batch_size,
        price_per_million_embed_tokens=res.pricing.embed_price_per_million(res.embedder.id),
    )


async def ingest(
    res: Resources, sources: Sequence[str], *, plan: bool = False, fail_fast: bool = False
) -> tuple[str, int]:
    """Plan or run ingestion for each source.

    Returns:
        ``(report text, exit code)``. A run that quarantined anything exits 1,
        so a scheduled job surfaces it instead of reporting success
        (ARCHITECTURE.md §18.2).
    """
    outputs: list[str] = []
    exit_code = 0
    await res.state.initialise()
    for name in source_names(res, sources):
        engine = await build_engine(res, name, fail_fast=fail_fast)
        if plan:
            outputs.append((await engine.plan()).render())
            continue
        report = await engine.run(res.run_context())
        outputs.append(report.render())
        if report.status != "ok":
            exit_code = 1
    return "\n\n".join(outputs), exit_code


async def ingest_status(res: Resources, sources: Sequence[str]) -> str:
    """Render the last run for each source and each index's epoch."""
    await res.state.initialise()
    lines: list[str] = []
    # `last_run` is a SQLite-store convenience, not part of the StateStore port.
    read_last = getattr(res.state, "last_run", None)
    for name in source_names(res, sources):
        last = await read_last(name) if read_last is not None else None
        if last is None:
            lines.append(f"{name}: never ingested")
            continue
        lines.append(
            f"{name}: {last.get('status')} at {last.get('finished_at')} "
            f"(+{last.get('added')} ~{last.get('changed')} -{last.get('deleted')}, "
            f"{last.get('chunks_embedded')} chunks embedded)"
        )
    for index, epoch in (await res.epochs()).items():
        lines.append(f"index {index}: epoch {epoch}")
    return "\n".join(lines)
