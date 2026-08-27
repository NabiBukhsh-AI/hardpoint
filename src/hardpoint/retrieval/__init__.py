"""Retriever composition, fusion, refinement and context assembly.

Owns the composable steps from which a retrieval strategy is built. Does not own
which vector store you use, and never imports an adapter.

There is deliberately no ``Retriever`` provider port (ARCHITECTURE.md §9.3):
retrieval is a composition, and a port would let an adapter hide the whole
strategy behind a vendor name.

M1 ships the vertical slice -- ``VectorRetriever`` and ``ContextAssembler``.
Fusion, reranking, expansion, compression and routing are M4, and adding them
must require no change to ``core`` or ``runtime``.
"""

from __future__ import annotations

from hardpoint.retrieval.assemble import (
    DEFAULT_PREAMBLE,
    ContextAssembler,
    Ordering,
    citations_for,
)
from hardpoint.retrieval.retrievers import Assembled, FilterFactory, Retrieved, VectorRetriever

__all__ = [
    "DEFAULT_PREAMBLE",
    "Assembled",
    "ContextAssembler",
    "FilterFactory",
    "Ordering",
    "Retrieved",
    "VectorRetriever",
    "citations_for",
]
