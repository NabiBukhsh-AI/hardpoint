"""The question-answering pipeline. Read it top to bottom: this is what runs.

The service, `hardpoint ask` and the eval suite all build through `build`, so
there is exactly one definition of what answering means.

To change the strategy, change the list. Guards come from `guards:` in config;
caching from `cache:`. Each step is one line.
"""

from pathlib import Path

from hardpoint.core.models import Answer
from hardpoint.generation import Generate
from hardpoint.retrieval import ContextAssembler, VectorRetriever
from hardpoint.runtime import Pipeline, Resources

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def voice(name: str) -> str:
    """Product wording kept next to the prompts: abstention and refusal text."""
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def build(res: Resources) -> Pipeline[str, Answer]:
    """Build the pipeline from the configured components."""
    retrieval, cache = res.config.retrieval, res.config.cache
    return Pipeline(
        name="rag",
        steps=[
            *res.input_guards(),
            VectorRetriever(
                res.index(),
                res.embedder,
                top_k=retrieval.top_k,
                min_score=retrieval.score_threshold,
                epoch=res.epoch_reader() if cache.retrieval.enabled else None,
                cache_ttl_s=cache.retrieval.ttl_s,
            ),
            ContextAssembler(
                token_budget=retrieval.context.token_budget,
                ordering=retrieval.context.ordering,
                model=res.llm,
            ),
            res.guarded(
                Generate(
                    res.llm,
                    res.prompts,
                    prompt="answer",
                    no_context_policy=retrieval.no_context_policy,
                    abstention_text=voice("abstain"),
                    cache=cache.generation.enabled,
                    cache_ttl_s=cache.generation.ttl_s,
                ),
                refusal=voice("refusal"),
            ),
        ],
    )
