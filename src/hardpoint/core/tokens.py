"""Token counting, and the documented fallback when no tokenizer is available.

Two modules need to count tokens: the chunker, deciding where to split, and
``ContextAssembler``, deciding what fits in the budget. Both may import ``core``
and neither may import the other, so the heuristic lives here rather than being
written twice and drifting -- and a drift between "how big the chunker thought
this was" and "how big the assembler thinks it is" surfaces as context that
mysteriously does not fit.

## The heuristic, stated so it can be judged

``ceil(len(text) / 4)``, with a floor of 1 for any non-empty string.

Four characters per token is the widely quoted average for English prose under
byte-pair encodings. It is *wrong* for code, for non-Latin scripts, and for text
dense in punctuation, and it is wrong in the dangerous direction for those: it
under-counts, so a budget filled by this estimate can overflow the real one.

That is why it is a fallback rather than the mechanism. ``LanguageModel`` carries
``count_tokens``, and any caller holding a model should use it. This exists for
the ingestion path, which chunks documents long before a model is chosen, and
for adapters whose provider exposes no tokenizer.

**It is never zero for non-empty text.** A zero would let a context assembler
believe anything fits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

__all__ = ["CHARACTERS_PER_TOKEN", "TokenCounter", "estimate_tokens"]

CHARACTERS_PER_TOKEN = 4
"""Characters per token in the fallback estimate. See the module docstring."""

TokenCounter: TypeAlias = Callable[[str], int]
"""Anything that can count tokens in a string.

:func:`estimate_tokens` satisfies it, and so does a closure over a real
tokenizer -- which is how a caller holding a model upgrades from the heuristic
without either side knowing about the other.
"""


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string.

    Args:
        text: The text to measure.

    Returns:
        An estimate. Zero for the empty string, and never zero for anything else.

    Raises:
        Nothing.
    """
    if not text:
        return 0
    return max(1, -(-len(text) // CHARACTERS_PER_TOKEN))
