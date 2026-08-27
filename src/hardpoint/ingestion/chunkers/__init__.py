"""Chunkers mapping a ParsedDocument to a list of Chunk.

A chunker must derive ids through ``core.ids.chunk_id``. One that invents ids
breaks incremental ingestion completely and silently: every run would re-embed
every chunk while reporting that it had only embedded what changed.
"""

from __future__ import annotations

from hardpoint.ingestion.chunkers.recursive import HEADING_SEPARATOR, RecursiveChunker

__all__ = ["HEADING_SEPARATOR", "RecursiveChunker"]
