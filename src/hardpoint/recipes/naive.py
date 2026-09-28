"""Naive RAG: retrieve, assemble, generate (ARCHITECTURE.md §12.2, "6 lines").

The pipeline the ``rag-minimal`` template generates, and the one every other
recipe extends by adding steps to the list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hardpoint.generation import Generate
from hardpoint.retrieval import ContextAssembler, VectorRetriever
from hardpoint.runtime.pipeline import Pipeline

if TYPE_CHECKING:
    from hardpoint.runtime.resources import Resources

__all__ = ["build"]


def build(res: Resources, *, abstention_text: str | None = None) -> Pipeline[str, Any]:
    """Build the naive pipeline from configured resources.

    Args:
        res: The configured components.
        abstention_text: What to say when nothing was retrieved. The wording is
            product voice and belongs in the project (INSTRUCTIONS.md §7).

    Returns:
        ``str -> Answer``.
    """
    retrieval, context = res.config.retrieval, res.config.retrieval.context
    return Pipeline(
        name="rag",
        steps=[
            VectorRetriever(
                res.index(),
                res.embedder,
                top_k=retrieval.top_k,
                min_score=retrieval.score_threshold,
            ),
            ContextAssembler(
                token_budget=context.token_budget,
                ordering=context.ordering,
                citation_style=context.citation_style,
                model=res.llm,
            ),
            Generate(
                res.llm,
                res.prompts,
                no_context_policy=retrieval.no_context_policy,
                abstention_text=abstention_text,
            ),
        ],
    )
