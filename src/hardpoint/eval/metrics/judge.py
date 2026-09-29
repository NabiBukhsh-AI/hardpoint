"""LLM-judged metrics: faithfulness and answer relevance (INSTRUCTIONS.md §8 **[LOCKED]**).

Every score records the judge's model id, the judge prompt's version and the
temperature. A judge change -- a new model, an edited prompt -- makes scores
incomparable, and a report that let two such runs be compared as if they were
the same would be worse than no judge at all (ARCHITECTURE.md §20.2).

The judge prompts are the project's, not the library's (INSTRUCTIONS.md §13.11):
``prompts/judge_faithfulness.md`` and ``prompts/judge_relevance.md`` in a
generated project. Each receives ``question``, ``answer``, ``context`` and
``reference`` and must ask for ``{"score": <0..1>, "reason": "..."}``.

These go through the ``LanguageModel`` port like everything else, so they need
no dependency beyond the base install; the ``eval-judge`` extra exists to name
the capability, not to install anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.errors import ConfigError
from hardpoint.core.models import Answer
from hardpoint.core.ports import GenerationRequest
from hardpoint.generation.structured import generate_structured

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import LanguageModel, PromptStore

__all__ = ["Judge", "JudgeScore"]


class JudgeScore(BaseModel):
    """One judged score, with everything needed to know whether it is comparable.

    Args:
        metric: ``faithfulness`` or ``answer_relevance``.
        score: Between 0 and 1.
        reason: The judge's explanation.
        judge_model: The model that judged.
        prompt_version: The judge prompt's content hash.
        temperature: The sampling temperature used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    score: float = Field(ge=0, le=1)
    reason: str = ""
    judge_model: str
    prompt_version: str
    temperature: float


class _Verdict(BaseModel):
    """What a judge prompt must return."""

    score: float = Field(ge=0, le=1)
    reason: str = ""


class Judge:
    """Scores answers with an LLM, through the project's judge prompts.

    Args:
        model: The judge model. A different, stronger model than the one being
            judged is better practice; the same one works.
        prompts: Where the judge prompts live.
        temperature: Zero by default, for repeatability.
        faithfulness_prompt: Name of the faithfulness prompt.
        relevance_prompt: Name of the relevance prompt.
    """

    def __init__(
        self,
        model: LanguageModel,
        prompts: PromptStore,
        *,
        temperature: float = 0.0,
        faithfulness_prompt: str = "judge_faithfulness",
        relevance_prompt: str = "judge_relevance",
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.temperature = temperature
        self.faithfulness_prompt = faithfulness_prompt
        self.relevance_prompt = relevance_prompt

    async def faithfulness(
        self, question: str, answer: Answer, ctx: RunContext, *, reference: str | None = None
    ) -> JudgeScore:
        """Is every claim in the answer supported by the retrieved context?"""
        return await self._score(
            "faithfulness",
            prompt=self.faithfulness_prompt,
            question=question,
            answer=answer,
            ctx=ctx,
            reference=reference,
        )

    async def answer_relevance(
        self, question: str, answer: Answer, ctx: RunContext, *, reference: str | None = None
    ) -> JudgeScore:
        """Does the answer address the question that was asked?"""
        return await self._score(
            "answer_relevance",
            prompt=self.relevance_prompt,
            question=question,
            answer=answer,
            ctx=ctx,
            reference=reference,
        )

    async def _score(
        self,
        metric: str,
        *,
        prompt: str,
        question: str,
        answer: Answer,
        ctx: RunContext,
        reference: str | None,
    ) -> JudgeScore:
        variables = {
            "question": question,
            "answer": answer.text,
            "context": answer.context.rendered if answer.context else "",
            "reference": reference or "",
        }
        try:
            rendered = await self.prompts.render(prompt, variables)
        except ConfigError as exc:
            raise ConfigError(
                f"The judge prompt {prompt!r} is missing: {exc.message}",
                config_path="eval.judge",
                remedy=(
                    f"Add prompts/{prompt}.md asking for "
                    '{"score": <0 to 1>, "reason": "..."} given {{ question }}, '
                    "{{ answer }}, {{ context }} and {{ reference }}, or disable "
                    "`eval.judge.enabled`."
                ),
                cause=exc,
            ) from exc

        request = GenerationRequest(messages=rendered.messages, temperature=self.temperature)
        verdict = await generate_structured(self.model, request, _Verdict, ctx)
        usage = verdict.usage
        ctx.usage.record(
            f"judge:{metric}",
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            estimated=usage.estimated,
        )
        return JudgeScore(
            metric=metric,
            score=verdict.value.score,
            reason=verdict.value.reason,
            judge_model=self.model.id,
            prompt_version=rendered.version,
            temperature=self.temperature,
        )
