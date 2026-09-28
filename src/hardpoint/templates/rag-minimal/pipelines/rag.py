"""The question-answering pipeline. Read it top to bottom: this is what runs.

`hardpoint ask`, the tests, and (later) the service and the eval suite all build
through `build`, so there is exactly one definition of what answering means.

To change the strategy, change the list: add a reranker, a second retriever and
a fusion step, a guard. Each is one line.
"""

from pathlib import Path

from hardpoint.core.models import Answer
from hardpoint.generation import Generate
from hardpoint.retrieval import ContextAssembler, VectorRetriever
from hardpoint.runtime import Pipeline, Resources

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def build(res: Resources) -> Pipeline[str, Answer]:
    """Build the pipeline from the configured components."""
    retrieval = res.config.retrieval
    return Pipeline(
        name="rag",
        steps=[
            VectorRetriever(res.index(), res.embedder, top_k=retrieval.top_k),
            ContextAssembler(token_budget=retrieval.context.token_budget, model=res.llm),
            Generate(
                res.llm,
                res.prompts,
                prompt="answer",
                no_context_policy=retrieval.no_context_policy,
                abstention_text=(PROMPTS / "abstain.md").read_text(encoding="utf-8").strip(),
            ),
        ],
    )
