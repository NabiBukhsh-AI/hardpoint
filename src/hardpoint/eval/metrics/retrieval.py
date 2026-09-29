"""Deterministic retrieval metrics (INSTRUCTIONS.md §8: in base, no LLM).

Each takes the retrieved items in rank order as relevance flags -- ``True`` for a
relevant item -- so the functions are pure arithmetic and trivially testable.
They catch the majority of regressions, cheaply enough for every pull request
(ARCHITECTURE.md §20.2).

Definitions, so a number in a report means one thing:

- ``hit_rate@k``: 1 if any of the top ``k`` is relevant, else 0.
- ``precision@k``: relevant among the top ``k``, divided by ``k``. Retrieving
  fewer than ``k`` does not flatter it.
- ``recall@k``: expected items found in the top ``k``, divided by how many were
  expected.
- ``mrr``: the reciprocal rank of the first relevant item, 0 when none.
- ``ndcg@k``: discounted cumulative gain of the top ``k``, normalised by the
  best achievable ordering of the same relevance flags plus any expected items
  that were not retrieved.
- ``context_precision``: relevant among the items actually placed in the prompt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "context_precision",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def hit_rate_at_k(relevant: Sequence[bool], k: int) -> float:
    """1.0 if any of the top ``k`` is relevant."""
    return 1.0 if any(relevant[:k]) else 0.0


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    """Relevant among the top ``k``, over ``k``."""
    return sum(relevant[:k]) / k


def recall_at_k(found: Sequence[str], expected: set[str] | frozenset[str], k: int) -> float:
    """Distinct expected items among the top ``k`` found, over the number expected.

    Args:
        found: What was retrieved, in rank order -- document ids for
            document-level recall, chunk ids for chunk-level.
        expected: What should have been.
        k: The cut-off.
    """
    if not expected:
        return 0.0
    return len(set(found[:k]) & expected) / len(expected)


def mrr(relevant: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item."""
    for rank, flag in enumerate(relevant, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Sequence[bool], k: int, *, missing: int = 0) -> float:
    """Normalised discounted cumulative gain at ``k``, with binary relevance.

    Args:
        relevant: Relevance flags in rank order.
        k: The cut-off.
        missing: Expected items that were not retrieved at all. They belong in
            the ideal ordering; leaving them out would score a retriever that
            found one of five relevant items as perfect.
    """
    gains = [1.0 if flag else 0.0 for flag in relevant[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_count = min(k, sum(relevant) + missing)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def context_precision(included: Sequence[bool]) -> float:
    """Relevant among the items placed in the prompt."""
    return sum(included) / len(included) if included else 0.0
