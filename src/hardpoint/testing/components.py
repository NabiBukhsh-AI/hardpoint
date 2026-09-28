"""Registry entries for the fakes, so ``type: fake_llm`` works from YAML.

What lets a generated project run its whole path -- ingest, ask, eval -- in CI
with zero network access and no credentials (INSTRUCTIONS.md §6.6), selected by
configuration rather than by editing code. The ``offline`` overlay of every
template switches to these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.testing.fakes import (
    FakeEmbeddingModel,
    FakeLanguageModel,
    FakeReranker,
    InMemoryVectorIndex,
)

__all__ = [
    "FakeEmbeddingsConfig",
    "FakeLLMConfig",
    "FakeRerankerConfig",
    "MemoryIndexConfig",
    "build_embeddings",
    "build_index",
    "build_llm",
    "build_reranker",
]

_STRICT = ConfigDict(frozen=True, extra="forbid")


class FakeLLMConfig(BaseModel):
    """``type: fake_llm``. Deterministic, offline, and extractive by default."""

    model_config = _STRICT

    model_id: str = "fake/language-model"
    extractive: bool = True
    cost_per_call: float | None = Field(default=0.0, ge=0.0)


class FakeEmbeddingsConfig(BaseModel):
    """``type: fake_embeddings``. Hashed word stems, so shared wording retrieves."""

    model_config = _STRICT

    model_id: str = "fake/embedding-model"
    dimensions: int = Field(default=256, gt=0)
    lexical: bool = True


class MemoryIndexConfig(BaseModel):
    """``type: memory``. In process, so nothing survives the process."""

    model_config = _STRICT

    name: str = "primary"
    dimensions: int = Field(default=64, gt=0)
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class FakeRerankerConfig(BaseModel):
    """``type: fake_reranker``. Word overlap with the query."""

    model_config = _STRICT

    model_id: str = "fake/reranker"


def build_llm(config: FakeLLMConfig) -> FakeLanguageModel:
    """Registry factory for ``type: fake_llm``."""
    return FakeLanguageModel(
        model_id=config.model_id,
        extractive=config.extractive,
        cost_per_call=config.cost_per_call,
    )


def build_embeddings(config: FakeEmbeddingsConfig) -> FakeEmbeddingModel:
    """Registry factory for ``type: fake_embeddings``."""
    return FakeEmbeddingModel(config.dimensions, model_id=config.model_id, lexical=config.lexical)


def build_index(config: MemoryIndexConfig) -> InMemoryVectorIndex:
    """Registry factory for ``type: memory``."""
    return InMemoryVectorIndex(config.name, config.dimensions, config.metric)


def build_reranker(config: FakeRerankerConfig) -> FakeReranker:
    """Registry factory for ``type: fake_reranker``."""
    return FakeReranker(model_id=config.model_id)
