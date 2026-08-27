"""Fakes, contract test kits and deterministic fixtures.

Everything here is public API: third parties import the contract kits to prove
their own adapters conform. Nothing here belongs on a production path.

Imports ``core`` and ``runtime`` only, so that a test kit never drags an
optional dependency into a consumer's environment (ARCHITECTURE.md §8.1 R6).
``pytest`` is not imported at module scope anywhere in this package, so the whole
of ``hardpoint.testing`` imports cleanly in a bare environment.
"""

from __future__ import annotations

from hardpoint.testing.contracts import vector_index_contract
from hardpoint.testing.fakes import (
    FakeCache,
    FakeEmbeddingModel,
    FakeLanguageModel,
    FakeReranker,
    InMemoryVectorIndex,
    RecordedSpan,
    RecordingMetricSink,
    RecordingTracer,
    ScriptedResponse,
    deterministic_vector,
)
from hardpoint.testing.fixtures import (
    build_chunk,
    build_chunks,
    build_document,
    build_retrieved,
    build_run_context,
)

__all__ = [
    "FakeCache",
    "FakeEmbeddingModel",
    "FakeLanguageModel",
    "FakeReranker",
    "InMemoryVectorIndex",
    "RecordedSpan",
    "RecordingMetricSink",
    "RecordingTracer",
    "ScriptedResponse",
    "build_chunk",
    "build_chunks",
    "build_document",
    "build_retrieved",
    "build_run_context",
    "deterministic_vector",
    "vector_index_contract",
]
