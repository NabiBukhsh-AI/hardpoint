"""Configuration to live components, and the one path from a question to an Answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hardpoint.core.config import resolve
from hardpoint.core.errors import ConfigError, ContractError, TransientError
from hardpoint.core.models import Answer
from hardpoint.core.ports import GenerationRequest, Message
from hardpoint.core.registry import ComponentRegistry, Kind
from hardpoint.runtime import (
    Pipeline,
    Resources,
    answer_query,
    as_step,
    build_resources,
    build_source,
    load_pipeline,
)
from hardpoint.runtime.wrapped import PolicyLanguageModel, PolicyVectorIndex
from hardpoint.testing import FakeLanguageModel, build_run_context
from hardpoint.testing.components import FakeLLMConfig

OFFLINE: dict[str, Any] = {
    "project": {"pipeline": "hardpoint.recipes.naive:build"},
    "providers": {
        "llm": {"type": "fake_llm"},
        "embeddings": {"type": "fake_embeddings", "dimensions": 32},
    },
    "indexes": {"primary": {"type": "memory", "dimensions": 32}},
    "ingestion": {"state": {"type": "sqlite", "path": ":memory:"}},
}


async def resources(overrides: dict[str, Any] | None = None, **kwargs: Any) -> Resources:
    return await build_resources(
        resolve(base={**OFFLINE, **(overrides or {})}, environ={}), **kwargs
    )


@pytest.mark.anyio
async def test_configured_components_are_built_and_wrapped_in_policies() -> None:
    res = await resources()
    assert isinstance(res.llm, PolicyLanguageModel)
    assert isinstance(res.index(), PolicyVectorIndex)
    assert res.embedder.dimensions == 32
    assert res.index().name == "primary", "the index takes its configured key as its name"
    await res.aclose()


@pytest.mark.anyio
async def test_an_unconfigured_component_fails_with_its_config_path() -> None:
    res = await resources({"providers": {}})
    with pytest.raises(ConfigError) as exc_info:
        _ = res.llm
    assert exc_info.value.config_path == "providers.llm"
    with pytest.raises(ConfigError) as index_error:
        res.index("secondary")
    assert "primary" in (index_error.value.remedy or "")
    await res.aclose()


@pytest.mark.anyio
async def test_retry_policy_from_config_is_applied() -> None:
    """Retry lives in the runtime, configured per component (ADR-004)."""
    attempts: list[int] = []

    class Flaky(FakeLanguageModel):
        async def generate(self, req: GenerationRequest, ctx: Any) -> Any:
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientError("blip", remedy="retry")
            return await super().generate(req, ctx)

    registry = ComponentRegistry()
    registry.register("flaky", kind=Kind.LLM, factory=lambda _: Flaky(), config_model=FakeLLMConfig)
    res = await resources(
        {
            "providers": {
                "llm": {
                    "type": "flaky",
                    "policies": {"retry": {"max_attempts": 3, "initial_backoff_s": 0.001}},
                },
                "embeddings": {"type": "fake_embeddings", "dimensions": 32},
            }
        },
        registry=registry,
    )
    request = GenerationRequest(messages=(Message(role="user", content="hi"),))
    await res.llm.generate(request, build_run_context())
    assert len(attempts) == 3
    await res.aclose()


@pytest.mark.anyio
async def test_a_fallback_model_answers_when_the_primary_fails() -> None:
    registry = ComponentRegistry()
    registry.register(
        "down",
        kind=Kind.LLM,
        factory=lambda _: FakeLanguageModel(fail_with=TransientError("down", remedy="wait")),
        config_model=FakeLLMConfig,
    )
    res = await resources(
        {
            "providers": {
                "llm": {
                    "type": "down",
                    "policies": {
                        "retry": {"max_attempts": 1},
                        "fallback": {"type": "fake_llm", "model_id": "fake/backup"},
                    },
                },
                "embeddings": {"type": "fake_embeddings", "dimensions": 32},
            }
        },
        registry=registry,
    )
    request = GenerationRequest(messages=(Message(role="user", content="hi"),))
    result = await res.llm.generate(request, build_run_context())
    assert result.model_id == "fake/backup"
    await res.aclose()


@pytest.mark.anyio
async def test_a_fallback_embedder_of_another_width_is_refused() -> None:
    """Vectors from a different width are incomparable with the index."""
    with pytest.raises(ConfigError, match="fallback embedder"):
        await resources(
            {
                "providers": {
                    "embeddings": {
                        "type": "fake_embeddings",
                        "dimensions": 32,
                        "policies": {"fallback": {"type": "fake_embeddings", "dimensions": 16}},
                    }
                }
            }
        )


@pytest.mark.anyio
async def test_answer_query_completes_the_answer_with_run_wide_facts() -> None:
    res = await resources()
    answer = await answer_query(load_pipeline(res), "anything", res, run_id="run-x")

    assert isinstance(answer, Answer)
    assert answer.run_id == "run-x"
    assert answer.manifest.config_hash == res.snapshot.hash
    assert answer.manifest.index_epochs == {"primary": 0}
    assert answer.manifest.model_ids["embeddings"] == "fake/embedding-model"
    # Every step's usage, not just generation's: retrieval embedded the query.
    assert "retrieve" in answer.usage.by_step
    # Nothing is ingested, so the naive recipe abstains -- a policy, not an error.
    assert answer.abstained
    await res.aclose()


@pytest.mark.anyio
async def test_a_pipeline_not_ending_in_an_answer_is_a_contract_error() -> None:
    res = await resources()

    async def echo(data: str, ctx: Any) -> str:
        return data

    with pytest.raises(ContractError, match="not an Answer"):
        await answer_query(Pipeline("echo", [as_step(echo)]), "q", res)
    await res.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("no_colon_here", "module:function"),
        ("definitely_not_a_module_xyz:build", "could not be imported"),
        ("hardpoint.recipes.naive:missing", "no callable"),
    ],
)
async def test_a_bad_pipeline_factory_names_the_problem(target: str, message: str) -> None:
    res = await resources({"project": {"pipeline": target}})
    with pytest.raises(ConfigError, match=message) as exc_info:
        load_pipeline(res)
    assert exc_info.value.config_path == "project.pipeline"
    await res.aclose()


@pytest.mark.anyio
async def test_sources_are_built_on_demand_with_their_key_as_id(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    res = await resources({"sources": {"docs": {"type": "local_files", "root": str(tmp_path)}}})
    source = await build_source(res, "docs")
    assert source.id == "docs"
    with pytest.raises(ConfigError, match="No source named"):
        await build_source(res, "wiki")
    await res.aclose()


@pytest.mark.anyio
async def test_prompts_load_from_the_project_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "answer.md").write_text("{{ question }}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    res = await resources()
    rendered = await res.prompts.render("answer", {"question": "why?"})
    assert rendered.messages[0].content == "why?"
    await res.aclose()


@pytest.mark.anyio
async def test_run_context_carries_the_configured_budget() -> None:
    res = await resources({"budgets": {"request": {"max_llm_calls": 2, "deadline_s": 5}}})
    ctx = res.run_context()
    assert ctx.budget.max_llm_calls == 2
    remaining = ctx.deadline.remaining_s()
    assert remaining is not None
    assert 0 < remaining <= 5
    assert ctx.run_id.startswith("run_")
    await res.aclose()
