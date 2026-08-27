"""LanguageModel adapters.

M1 ships one, speaking the OpenAI chat-completions wire protocol over ``httpx``.
That covers OpenAI, Ollama, vLLM, LM Studio, Together, Groq and Azure OpenAI
with a different ``base_url``, and needs no extra.
"""

from __future__ import annotations

from hardpoint.adapters.llm.openai_compatible import OpenAICompatibleChat

__all__ = ["OpenAICompatibleChat"]
