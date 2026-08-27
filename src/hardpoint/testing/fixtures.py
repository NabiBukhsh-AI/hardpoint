"""Deterministic builders for test inputs. INSTRUCTIONS.md §5.8.

``build_run_context`` is the important one: every step and every port method
takes a ``RunContext``, so a test that cannot cheaply construct one cannot test
anything. Defaults are wired to the recording fakes, so a test gets a working
tracer, metric sink and usage accumulator without asking.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hardpoint.core.config.snapshot import ConfigSnapshot
from hardpoint.core.context import Budget, CacheHandle, Deadline, RunContext, UsageAccumulator
from hardpoint.core.ids import chunk_id, document_id
from hardpoint.core.models import CharSpan, Chunk, Document, RetrievedChunk, TrustLevel
from hardpoint.core.ports import CacheBackend, MetricSink, Tracer
from hardpoint.core.types import JsonValue
from hardpoint.testing.fakes import RecordingMetricSink, RecordingTracer

__all__ = [
    "build_chunk",
    "build_chunks",
    "build_document",
    "build_retrieved",
    "build_run_context",
]


def build_run_context(
    *,
    run_id: str = "test-run",
    tracer: Tracer | None = None,
    metrics: MetricSink | None = None,
    deadline: Deadline | None = None,
    budget: Budget | None = None,
    request_cache: CacheBackend | None = None,
    shared_cache: CacheBackend | None = None,
    config: ConfigSnapshot | None = None,
    usage: UsageAccumulator | None = None,
    extras: Mapping[str, Any] | None = None,
) -> RunContext:
    """Build a ``RunContext`` wired to recording fakes.

    Every argument has a default, so the common case is ``build_run_context()``.
    Override only what a test is actually about.

    Args:
        run_id: Identifier for the run.
        tracer: Defaults to a ``RecordingTracer``, so span assertions are
            available without arranging one.
        metrics: Defaults to a ``RecordingMetricSink``.
        deadline: Defaults to no deadline. A test about deadlines passes one.
        budget: Defaults to unbounded.
        request_cache: Backend for the request-scoped cache. Absent by default.
        shared_cache: Backend for the shared cache. Absent by default.
        config: Defaults to an empty snapshot in the ``test`` environment.
        usage: Defaults to a fresh accumulator.
        extras: User space. The library never reads it.

    Returns:
        A ready ``RunContext``.
    """
    return RunContext(
        run_id=run_id,
        tracer=tracer or RecordingTracer(),
        metrics=metrics or RecordingMetricSink(),
        deadline=deadline or Deadline.none(),
        budget=budget or Budget(),
        cache=CacheHandle(request=request_cache, shared=shared_cache),
        config=config or ConfigSnapshot(env="test", data={}),
        usage=usage or UsageAccumulator(),
        extras=dict(extras or {}),
    )


def build_document(
    uri: str = "guide.md",
    *,
    source_id: str = "docs",
    media_type: str = "text/markdown",
    revision: str = "rev-1",
    content_hash: str = "0" * 64,
    metadata: Mapping[str, JsonValue] | None = None,
) -> Document:
    """Build a ``Document`` with a deterministic id derived from its source."""
    return Document(
        id=document_id(source_id, uri),
        source_uri=uri,
        source_id=source_id,
        media_type=media_type,
        content_hash=content_hash,
        revision=revision,
        metadata=dict(metadata or {}),
    )


def build_chunk(
    text: str = "hello world",
    *,
    document: Document | None = None,
    index: int = 0,
    parent_id: str | None = None,
    trust: TrustLevel = TrustLevel.UNTRUSTED,
    metadata: Mapping[str, JsonValue] | None = None,
    token_count: int | None = None,
) -> Chunk:
    """Build a ``Chunk`` whose id comes from ``ids.chunk_id``.

    Uses the real derivation rather than a made-up id, so a test written against
    these chunks exercises the same identity the sync engine will compute.
    """
    owner = document or build_document()
    return Chunk(
        id=chunk_id(owner.id, index, text),
        document_id=owner.id,
        index=index,
        text=text,
        span=CharSpan(start=0, end=len(text)),
        parent_id=parent_id,
        trust=trust,
        metadata=dict(metadata or {}),
        token_count=token_count,
    )


def build_chunks(
    texts: list[str],
    *,
    document: Document | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> list[Chunk]:
    """Build a document's chunks, numbered in order."""
    owner = document or build_document()
    return [
        build_chunk(text, document=owner, index=index, metadata=metadata)
        for index, text in enumerate(texts)
    ]


def build_retrieved(
    chunks: list[Chunk],
    *,
    retriever: str = "fake",
    scores: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Wrap chunks as retrieval results, scored in descending order by default."""
    values = scores or [1.0 - (index * 0.1) for index in range(len(chunks))]
    return [
        RetrievedChunk(chunk=chunk, score=score, rank=index, retriever=retriever)
        for index, (chunk, score) in enumerate(zip(chunks, values, strict=True))
    ]
