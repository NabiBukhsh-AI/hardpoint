"""Structured output with one repair attempt (ARCHITECTURE.md §18.2)."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from hardpoint.core.capabilities import ModelCapabilities
from hardpoint.core.errors import ContractError
from hardpoint.core.ports import GenerationRequest, Message
from hardpoint.generation import generate_structured
from hardpoint.observability.logging import JsonFormatter, run_logger
from hardpoint.testing import FakeLanguageModel, ScriptedResponse, build_run_context


class Reply(BaseModel):
    answer: str
    confidence: float


def ask() -> GenerationRequest:
    return GenerationRequest(messages=(Message(role="user", content="Reply as JSON."),))


@pytest.mark.anyio
async def test_a_valid_reply_takes_one_attempt() -> None:
    model = FakeLanguageModel([ScriptedResponse('{"answer": "yes", "confidence": 0.8}')])
    result = await generate_structured(model, ask(), Reply, build_run_context())
    assert result.value == Reply(answer="yes", confidence=0.8)
    assert result.attempts == 1
    assert model.calls[0].response_schema is not None, "declared support sends the schema"


@pytest.mark.anyio
async def test_an_invalid_reply_is_repaired_once_with_the_error_appended() -> None:
    model = FakeLanguageModel(
        [
            ScriptedResponse(
                '```json\n{"answer": "yes", "confidence": 0.5}\n```', match="error was"
            ),
            ScriptedResponse('{"answer": "yes"}'),
        ]
    )
    result = await generate_structured(model, ask(), Reply, build_run_context())
    assert result.attempts == 2
    assert result.usage.calls == 2
    repair = model.calls[1].messages
    assert repair[-2].role == "assistant"
    assert "confidence" in repair[-1].content


@pytest.mark.anyio
async def test_a_second_failure_raises_with_the_validation_error() -> None:
    model = FakeLanguageModel([ScriptedResponse("not json at all")])
    with pytest.raises(ContractError, match="did not produce a valid Reply") as exc_info:
        await generate_structured(model, ask(), Reply, build_run_context())
    assert exc_info.value.code == "generation.structured_output_invalid"
    assert model.call_count == 2, "one attempt plus one repair, no more"


@pytest.mark.anyio
async def test_no_schema_is_sent_to_a_model_that_does_not_declare_support() -> None:
    capabilities = ModelCapabilities(context_window_tokens=1000, supports_structured_output=False)
    model = FakeLanguageModel(
        [ScriptedResponse('{"answer": "a", "confidence": 1}')], capabilities=capabilities
    )
    await generate_structured(model, ask(), Reply, build_run_context())
    assert model.calls[0].response_schema is None


def test_the_run_logger_stamps_ids_and_the_json_formatter_writes_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="hardpoint.test"):
        run_logger("hardpoint.test", run_id="run-1", trace_id="t-1").info("hello %s", "there")
    (record,) = caplog.records
    rendered = JsonFormatter({"service": "svc"}).format(record)
    assert '"run_id": "run-1"' in rendered
    assert '"trace_id": "t-1"' in rendered
    assert '"message": "hello there"' in rendered
    assert '"service": "svc"' in rendered


def test_the_library_configures_no_logging_handlers() -> None:
    """INSTRUCTIONS.md §13.13: never basicConfig, never a handler, never the root logger."""
    import pathlib

    import hardpoint

    root = pathlib.Path(hardpoint.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "templates" not in path.parts
        and any(
            needle in path.read_text(encoding="utf-8")
            for needle in ("basicConfig(", "addHandler(", "getLogger()")
        )
        and path.name != "logging.py"
    ]
    assert offenders == []
