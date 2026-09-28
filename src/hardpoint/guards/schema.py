"""The schema guard: the answer must be the structure the caller asked for.

Correctness-critical, so it defaults to ``block`` (ARCHITECTURE.md §18.2:
"schema validation is not" optional). It validates ``Answer.structured`` when the
pipeline produced one, and otherwise parses ``Answer.text`` as JSON -- so it
works for a model that returned JSON as text as well as for structured output.
On success the parsed value is placed in ``Answer.structured``; the text is never
touched.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from hardpoint.core.errors import ConfigError
from hardpoint.core.models import Answer
from hardpoint.generation.structured import parse_json
from hardpoint.guards.base import GuardAction, GuardResult

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["SchemaGuard", "SchemaGuardConfig", "build", "parse_json"]


class SchemaGuard:
    """Checks the answer against a Pydantic model, or just for valid JSON.

    Args:
        model: The structure required. ``None`` requires only a JSON object.
        action: What to do on a violation. Defaults to ``block``.
        name: The guard's name.
    """

    def __init__(
        self,
        model: type[BaseModel] | None = None,
        *,
        action: GuardAction = "block",
        name: str = "schema",
    ) -> None:
        self.model = model
        self.action: GuardAction = action
        self.name = name

    async def check(self, value: Answer, ctx: RunContext) -> GuardResult:
        """Validate the answer's structure."""
        raw: Any = value.structured
        if raw is None:
            try:
                raw = parse_json(value.text)
            except ValueError as exc:
                return self._violation("invalid_json", f"The answer is not valid JSON: {exc}.")

        if self.model is None:
            if isinstance(raw, dict):
                return GuardResult(guard=self.name)
            return self._violation("not_an_object", "The answer is JSON but not an object.")

        try:
            self.model.model_validate(raw)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return self._violation("schema_mismatch", f"The answer does not match: {errors}.")
        return GuardResult(guard=self.name)

    def _violation(self, reason: str, detail: str) -> GuardResult:
        return GuardResult(guard=self.name, action=self.action, reason=reason, detail=detail)


class SchemaGuardConfig(BaseModel):
    """``guards.output: [{type: schema, model: "myproject.schemas:Reply"}]``.

    Args:
        action: What to do on a violation.
        model: ``module:ClassName`` of a Pydantic model in the project. When
            absent, any JSON object passes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: GuardAction = "block"
    model: str | None = None


def build(config: SchemaGuardConfig) -> SchemaGuard:
    """Registry factory for ``type: schema``.

    Raises:
        ConfigError: If ``model`` cannot be imported or is not a Pydantic model.
    """
    if config.model is None:
        return SchemaGuard(action=config.action)
    module_name, _, class_name = config.model.partition(":")
    try:
        found = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ConfigError(
            f"The schema guard's model {config.model!r} could not be imported.",
            config_path="guards.output",
            remedy="Give `model` as `module:ClassName`, importable from the project directory.",
            cause=exc,
        ) from exc
    if not (isinstance(found, type) and issubclass(found, BaseModel)):
        raise ConfigError(
            f"{config.model!r} is not a Pydantic model.",
            config_path="guards.output",
            remedy="Point `model` at a class that subclasses pydantic.BaseModel.",
        )
    return SchemaGuard(found, action=config.action)
