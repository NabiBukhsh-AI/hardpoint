"""Structured output: coerce a model's reply into a Pydantic model, with repair.

ARCHITECTURE.md §18.2: when a model returns invalid JSON for a structured
output, make **one** repair attempt with the validation error appended, then
fail. Models produce malformed JSON often enough that repair is a normal path,
and rarely enough that more than one attempt mostly buys a larger bill.

When the model declares ``supports_structured_output``, the schema is also sent
as ``response_schema`` so the provider can constrain decoding; the validation
here still runs, because a declaration is not a guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from hardpoint.core.errors import ContractError
from hardpoint.core.models import StepUsage
from hardpoint.core.ports import GenerationRequest, GenerationResult, Message

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import LanguageModel

__all__ = ["Structured", "generate_structured", "parse_json"]

M = TypeVar("M", bound=BaseModel)


def parse_json(text: str) -> Any:
    """Parse JSON from model output, tolerating a Markdown code fence around it.

    Raises:
        ValueError: If no JSON value can be parsed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return json.loads(stripped)


@dataclass(frozen=True)
class Structured(Generic[M]):
    """A validated structured reply.

    Args:
        value: The validated model instance.
        result: The final generation result, whose text parsed.
        attempts: How many generations it took: 1, or 2 after a repair.
        usage: Consumption across every attempt, repairs included.
    """

    value: M
    result: GenerationResult
    attempts: int
    usage: StepUsage


def _add(total: StepUsage, usage: StepUsage) -> StepUsage:
    cost = (
        None
        if total.cost_usd is None or usage.cost_usd is None
        else total.cost_usd + usage.cost_usd
    )
    return StepUsage(
        calls=total.calls + usage.calls,
        prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
        completion_tokens=total.completion_tokens + usage.completion_tokens,
        latency_ms=total.latency_ms + usage.latency_ms,
        cost_usd=cost,
        estimated=total.estimated or usage.estimated,
    )


async def generate_structured(
    model: LanguageModel,
    request: GenerationRequest,
    schema: type[M],
    ctx: RunContext,
    *,
    repair_attempts: int = 1,
) -> Structured[M]:
    """Generate and validate against ``schema``, repairing at most ``repair_attempts`` times.

    Args:
        model: The language model.
        request: The request. Its messages should ask for JSON.
        schema: The Pydantic model the reply must validate against.
        ctx: The run context.
        repair_attempts: Extra generations allowed after a failed validation.

    Returns:
        The validated value, the final result, the attempt count and usage.

    Raises:
        ContractError: If the reply still does not validate after the repairs,
            carrying the last validation error.
    """
    if model.capabilities().supports_structured_output:
        request = request.model_copy(update={"response_schema": schema.model_json_schema()})

    usage = StepUsage(cost_usd=0.0)
    messages = list(request.messages)
    problem = ""
    for attempt in range(1, repair_attempts + 2):
        result = await model.generate(request.model_copy(update={"messages": tuple(messages)}), ctx)
        usage = _add(usage, result.usage)
        try:
            value = schema.model_validate(parse_json(result.text))
        except (ValueError, ValidationError) as exc:
            problem = str(exc)
            messages.extend(
                [
                    Message(role="assistant", content=result.text),
                    Message(
                        role="user",
                        content=(
                            "That reply is not valid for the required format. "
                            f"The error was:\n{problem}\n"
                            "Reply again with only the corrected JSON."
                        ),
                    ),
                ]
            )
            continue
        return Structured(value=value, result=result, attempts=attempt, usage=usage)

    raise ContractError(
        f"{model.id} did not produce a valid {schema.__name__} after "
        f"{repair_attempts + 1} attempt(s): {problem}",
        code="generation.structured_output_invalid",
        component=model.id,
        remedy=(
            "Make the prompt state the required JSON shape explicitly, use a model "
            "that declares supports_structured_output, or raise repair_attempts."
        ),
    )
