"""The groundedness guard: does each claim map to a passage it cites?

ARCHITECTURE.md §19: a citation-grounding check. Deterministic and cheap, so it
can run on every request -- which also means it is a heuristic, and the
documentation says so. It measures; it does not prove.

## How a sentence counts as supported

A sentence of at least ``min_words`` words is a claim. A claim is **supported**
when it cites at least one passage that was actually in the context -- by its
``[key]`` -- and enough of its content words appear in the passages it cites
(``min_overlap``). A claim citing nothing, or citing a key that was never in the
context, is unsupported; a claim whose words appear nowhere in what it cites is
unsupported too.

The guard fires when the supported share of claims falls below
``min_supported_ratio``. Its default action is ``flag``: an LLM judge (M3's
faithfulness metric) is the better instrument for deciding what to *block*.

``redact`` removes the unsupported sentences and says so in a degradation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.models import Answer
from hardpoint.core.types import JsonValue
from hardpoint.guards.base import GuardAction, GuardResult

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["GroundednessGuard", "GroundednessGuardConfig", "build", "claims", "grounding"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[([^\[\]\n]{1,40})\]")
_CITATION_ONLY = re.compile(r"^(\[[^\[\]\n]{1,40}\]\s*[.!?]?\s*)+$")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
    ]
)


def claims(text: str, *, min_words: int = 4) -> list[str]:
    """Split an answer into claim sentences, ignoring fragments.

    A citation written after the full stop -- ``"... the old key. [1]"`` -- is
    attached to the sentence before it; split off on its own, it would leave
    that sentence looking uncited.
    """
    merged: list[str] = []
    for part in (part.strip() for part in _SENTENCE.split(text)):
        if merged and part and _CITATION_ONLY.match(part):
            merged[-1] = f"{merged[-1]} {part}"
        elif part:
            merged.append(part)
    return [part for part in merged if len(_WORD.findall(part.lower())) >= min_words]


def _content_words(text: str) -> set[str]:
    return {word[:6] for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def grounding(
    answer: Answer, *, min_words: int = 4, min_overlap: float = 0.5
) -> tuple[list[str], list[str]]:
    """Return ``(supported, unsupported)`` claims for an answer."""
    passages = (
        {item.citation_key: item.included_text for item in answer.context.items}
        if answer.context
        else {}
    )
    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims(answer.text, min_words=min_words):
        cited = [passages[key] for key in _CITATION.findall(claim) if key in passages]
        words = _content_words(_CITATION.sub(" ", claim))
        if cited and words:
            available = set().union(*(_content_words(passage) for passage in cited))
            overlap = len(words & available) / len(words)
        else:
            overlap = 0.0
        (supported if cited and overlap >= min_overlap else unsupported).append(claim)
    return supported, unsupported


class GroundednessGuard:
    """Flags an answer whose claims are not supported by the passages they cite.

    Args:
        min_supported_ratio: The share of claims that must be supported.
        min_overlap: The share of a claim's content words that must appear in
            the passages it cites.
        min_words: Sentences shorter than this are not claims.
        action: What to do below the ratio. Defaults to ``flag``.
        name: The guard's name.
    """

    def __init__(
        self,
        min_supported_ratio: float = 0.7,
        *,
        min_overlap: float = 0.5,
        min_words: int = 4,
        action: GuardAction = "flag",
        name: str = "groundedness",
    ) -> None:
        self.min_supported_ratio = min_supported_ratio
        self.min_overlap = min_overlap
        self.min_words = min_words
        self.action: GuardAction = action
        self.name = name

    async def check(self, value: Answer, ctx: RunContext) -> GuardResult:
        """Measure the supported share of the answer's claims."""
        if value.abstained or value.blocked:
            return GuardResult(guard=self.name)
        supported, unsupported = grounding(
            value, min_words=self.min_words, min_overlap=self.min_overlap
        )
        total = len(supported) + len(unsupported)
        ratio = len(supported) / total if total else 1.0
        evidence: dict[str, JsonValue] = {
            "supported_ratio": round(ratio, 4),
            "unsupported": list(unsupported),
        }
        if ratio >= self.min_supported_ratio:
            return GuardResult(guard=self.name, evidence=evidence)

        replacement = None
        if self.action == "redact":
            kept = " ".join(supported)
            replacement = kept or "The answer could not be supported by the retrieved passages."
        return GuardResult(
            guard=self.name,
            action=self.action,
            reason="ungrounded",
            detail=(
                f"{len(unsupported)} of {total} claims are not supported by the passages "
                f"they cite ({ratio:.0%} supported, {self.min_supported_ratio:.0%} required)."
            ),
            evidence=evidence,
            replacement=replacement,
        )


class GroundednessGuardConfig(BaseModel):
    """``guards.output: [{type: groundedness, min_supported_ratio: 0.7, action: flag}]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: GuardAction = "flag"
    min_supported_ratio: float = Field(default=0.7, ge=0, le=1)
    min_overlap: float = Field(default=0.5, ge=0, le=1)
    min_words: int = Field(default=4, ge=1)


def build(config: GroundednessGuardConfig) -> GroundednessGuard:
    """Registry factory for ``type: groundedness``."""
    return GroundednessGuard(
        config.min_supported_ratio,
        min_overlap=config.min_overlap,
        min_words=config.min_words,
        action=config.action,
    )
