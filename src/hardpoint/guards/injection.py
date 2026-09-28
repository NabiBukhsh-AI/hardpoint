"""Input guards: a prompt-injection heuristic and an input shape check.

## The injection heuristic flags; it does not block

ARCHITECTURE.md §19: "a heuristic injection detector that flags rather than
blocks by default (false positives on legitimate content are common)". A
support question about "how do I ignore previous invoices" matches half the
patterns anybody writes. A detector that blocked by default would be switched
off within a week, and a flag in the trace is worth more than a disabled guard.

What this is: a pattern list over the text, recorded as evidence. What it is
not: prevention. The structural mitigations -- trust levels on chunks, retrieved
content never in the system role, tool permissions checked against trust -- are
the ones that hold; this is visibility.

## The input shape check does block

An empty question or a 200,000-character one is not a phrasing problem. The
shape check defaults to ``block``, because there is nothing useful to do with
either.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.types import JsonValue
from hardpoint.guards.base import GuardAction, GuardResult

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = [
    "DEFAULT_PATTERNS",
    "InjectionGuardConfig",
    "InjectionHeuristic",
    "InputShapeConfig",
    "InputShapeGuard",
    "build",
    "build_input_shape",
]

DEFAULT_PATTERNS: Final = (
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|system)\b",
    r"\bforget\s+(everything|all)\s+(you|that)\b",
    r"\byou\s+are\s+now\s+(a|an|the|in)\b",
    r"\b(reveal|print|show|repeat|output)\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"\bact\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(unrestricted|jailbroken|developer\s+mode)",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"\bBEGIN\s+(SYSTEM|ADMIN)\s+(PROMPT|INSTRUCTIONS)\b",
)
"""Phrasings common in published injection attempts. A list, deliberately short."""


class InjectionHeuristic:
    """Flags text matching known injection phrasings.

    Args:
        patterns: Regular expressions, matched case-insensitively.
        action: What to do on a match. Defaults to ``flag``; see the module
            docstring for why not ``block``.
        name: The guard's name.
    """

    def __init__(
        self,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        *,
        action: GuardAction = "flag",
        name: str = "injection_heuristic",
    ) -> None:
        self.patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        self.action: GuardAction = action
        self.name = name

    async def check(self, value: str, ctx: RunContext) -> GuardResult:
        """Match the text against every pattern."""
        matched: list[JsonValue] = [
            match.group(0) for p in self.patterns if (match := p.search(value))
        ]
        if not matched:
            return GuardResult(guard=self.name)
        replacement = None
        if self.action == "redact":
            replacement = value
            for pattern in self.patterns:
                replacement = pattern.sub("[removed]", replacement)
        return GuardResult(
            guard=self.name,
            action=self.action,
            reason="possible_injection",
            detail=f"The text matched {len(matched)} known injection phrasing(s).",
            evidence={"matched": matched},
            replacement=replacement,
        )


class InputShapeGuard:
    """Rejects an input that is empty or unreasonably long.

    Args:
        min_chars: Shortest acceptable input, after stripping whitespace.
        max_chars: Longest acceptable input.
        action: What to do on a violation. Defaults to ``block``.
        name: The guard's name.
    """

    def __init__(
        self,
        *,
        min_chars: int = 1,
        max_chars: int = 4000,
        action: GuardAction = "block",
        name: str = "input_shape",
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.action: GuardAction = action
        self.name = name

    async def check(self, value: str, ctx: RunContext) -> GuardResult:
        """Check the input's length."""
        length = len(value.strip())
        if self.min_chars <= length <= self.max_chars:
            return GuardResult(guard=self.name)
        reason = "too_short" if length < self.min_chars else "too_long"
        replacement = value.strip()[: self.max_chars] if self.action == "redact" else None
        return GuardResult(
            guard=self.name,
            action=self.action,
            reason=reason,
            detail=(
                f"The input is {length} characters; between {self.min_chars} and "
                f"{self.max_chars} are accepted."
            ),
            evidence={"length": length},
            replacement=replacement,
        )


class InjectionGuardConfig(BaseModel):
    """``guards.input: [{type: injection_heuristic, action: flag}]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: GuardAction = "flag"
    patterns: tuple[str, ...] = DEFAULT_PATTERNS


class InputShapeConfig(BaseModel):
    """``guards.input: [{type: input_shape, max_chars: 4000}]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: GuardAction = "block"
    min_chars: int = Field(default=1, ge=0)
    max_chars: int = Field(default=4000, ge=1)


def build(config: InjectionGuardConfig) -> InjectionHeuristic:
    """Registry factory for ``type: injection_heuristic``."""
    return InjectionHeuristic(config.patterns, action=config.action)


def build_input_shape(config: InputShapeConfig) -> InputShapeGuard:
    """Registry factory for ``type: input_shape``."""
    return InputShapeGuard(
        min_chars=config.min_chars, max_chars=config.max_chars, action=config.action
    )
