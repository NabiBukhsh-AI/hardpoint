"""The four cache key formats of ARCHITECTURE.md §22.2, verbatim (**[LOCKED]**).

======================  ==============================================================
Layer                   Key
======================  ==============================================================
Embedding               ``embed:{model_id}:{kind}:{sha256(normalised_text)}``
Retrieval               ``retr:{index}:{epoch}:{model_id}:{sha256(query + filter + params)}``
Generation              ``gen:{model_id}:{params_hash}:{prompt_version}:{sha256(rendered_prompt)}``
Rerank                  ``rr:{model_id}:{sha256(query + candidate_ids)}``
======================  ==============================================================

What makes caching *safe* rather than a source of stale-answer incidents is
what the keys contain: the embedding model, so a model change misses; the index
**epoch**, so an ingestion run invalidates every retrieval built on the old
index; and the **prompt version**, a content hash, so editing a prompt misses
instead of serving answers written under the old one. A test asserts each.

Serves the ``CacheBackend`` port: these are the only keys the library writes.
Pure functions over ``ids.stable_hash`` and ``hashlib``, so they are identical
on every machine and in every process.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from hardpoint.core.filters import Filter
from hardpoint.core.ids import normalise_text, stable_hash
from hardpoint.core.types import JsonValue

__all__ = ["embedding_key", "generation_key", "params_hash", "rerank_key", "retrieval_key"]

_SEPARATOR = "\x00"
"""Joins hashed components, so ``("ab", "c")`` and ``("a", "bc")`` differ."""


def _sha256(*parts: str) -> str:
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


def embedding_key(model_id: str, kind: str, text: str) -> str:
    """``embed:{model_id}:{kind}:{sha256(normalised_text)}``.

    Content-addressed, so it never needs invalidating: the same text embedded
    by the same model as the same kind is the same vector.
    """
    return f"embed:{model_id}:{kind}:{_sha256(normalise_text(text))}"


def retrieval_key(
    index: str,
    epoch: int,
    model_id: str,
    *,
    query: str,
    filter: Filter | None,  # noqa: A002 - reads as what it is
    params: Mapping[str, JsonValue],
) -> str:
    """``retr:{index}:{epoch}:{model_id}:{sha256(query + filter + params)}``.

    ``epoch`` is what an ingestion run bumps; ``model_id`` is the embedding model
    that produced the query vector.
    """
    filter_part = stable_hash(filter.model_dump(mode="json")) if filter is not None else ""
    return (
        f"retr:{index}:{epoch}:{model_id}:"
        f"{_sha256(normalise_text(query), filter_part, stable_hash(dict(params)))}"
    )


def params_hash(params: Mapping[str, JsonValue]) -> str:
    """Hash of the generation parameters: temperature, token caps, schema, tools."""
    return stable_hash(dict(params))[:16]


def generation_key(
    model_id: str, params_digest: str, prompt_version: str, rendered_prompt: str
) -> str:
    """``gen:{model_id}:{params_hash}:{prompt_version}:{sha256(rendered_prompt)}``."""
    return f"gen:{model_id}:{params_digest}:{prompt_version}:{_sha256(rendered_prompt)}"


def rerank_key(model_id: str, query: str, candidate_ids: Sequence[str]) -> str:
    """``rr:{model_id}:{sha256(query + candidate_ids)}``.

    Candidate ids in the order given: a reranker's output depends on its input,
    and the ids already change whenever the corpus does.
    """
    return f"rr:{model_id}:{_sha256(normalise_text(query), *candidate_ids)}"
