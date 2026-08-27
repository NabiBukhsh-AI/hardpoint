"""Prompt rendering, LLM invocation steps and structured output handling.

Owns the mechanics of turning a context bundle into an answer. Does not own
prompt content, which is a user asset living in the generated project
(INSTRUCTIONS.md §13.11).
"""

from __future__ import annotations

from hardpoint.generation.generate import Generate, NoContextPolicy
from hardpoint.generation.prompts import (
    FilePromptStore,
    InMemoryPromptStore,
    PromptTemplate,
    render_template,
)

__all__ = [
    "FilePromptStore",
    "Generate",
    "InMemoryPromptStore",
    "NoContextPolicy",
    "PromptTemplate",
    "render_template",
]
