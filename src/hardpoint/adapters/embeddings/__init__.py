"""EmbeddingModel adapters.

M1 ships one, speaking the OpenAI embeddings wire protocol over ``httpx``.
Needs no extra.
"""

from __future__ import annotations

from hardpoint.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddings

__all__ = ["OpenAICompatibleEmbeddings"]
