"""The M2 Definition of Done, against a generated ``rag-service`` (INSTRUCTIONS.md §7).

"A generated service streams answers, emits traces with correct parent/child
nesting across async boundaries, reports per-request cost, and passes a load
smoke test without file-descriptor or session leaks."

The service is generated from the template, ingested, and driven over ASGI with
many concurrent requests. Its providers are the real OpenAI-compatible adapters
talking HTTP to a loopback server -- so there are real connection pools to leak
-- and its index is the SQLite file index, so there are real files to leak.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import anyio.lowlevel
import httpx
import pytest

from hardpoint.cli.commands.ingest import ingest
from hardpoint.cli.commands.init import write_project
from hardpoint.core.config import load_config
from hardpoint.observability.tracing import CollectedSpan
from hardpoint.runtime import build_resources
from tests.contract.fake_openai_server import fake_openai_server

QUERIES = 60
STREAMS = 30


def forget_generated_modules() -> None:
    """Drop cached ``pipelines`` and ``service`` modules from other generated projects."""
    generated = {"service", "pipelines"}
    for name in [module for module in sys.modules if module.split(".")[0] in generated]:
        del sys.modules[name]


@pytest.fixture
def service_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    forget_generated_modules()
    root = tmp_path / "support-bot"
    write_project(root, "rag-service")
    monkeypatch.chdir(root)
    monkeypatch.syspath_prepend(str(root))
    with fake_openai_server() as (base_url, state):
        state.dimensions = 16
        state.reply = "Rotate a key by creating a new key and revoking the old one [1]."
        overrides = {
            "providers": {
                "llm": {"type": "openai_chat", "model": "fake-chat", "base_url": base_url},
                "embeddings": {
                    "type": "openai_embeddings",
                    "model": "fake-embed",
                    "dimensions": 16,
                    "base_url": base_url,
                },
            },
            "pricing": {
                "openai/fake-chat": {"input": 1.0, "output": 2.0},
                "openai/fake-embed": {"embed": 0.1},
            },
        }
        yield {"root": root, "overrides": overrides, "server": state}
    for name in [
        module for module in sys.modules if module.split(".")[0] in {"service", "pipelines"}
    ]:
        del sys.modules[name]


def open_descriptors() -> int | None:
    """Open file descriptors, where the platform lets us count them cheaply."""
    fd_dir = Path("/proc/self/fd")
    return len(list(fd_dir.iterdir())) if fd_dir.is_dir() else None


def trace_is_well_formed(root: CollectedSpan) -> list[str]:
    """Every step under the pipeline, every provider span under its own step."""
    problems: list[str] = []
    steps = {child.attributes.get("step.name"): child for child in root.children}
    if set(steps) != {"input_guards", "retrieve", "assemble_context", "generate"}:
        problems.append(f"steps {sorted(map(str, steps))}")
    retrieve_children = {span.name for span in steps.get("retrieve", root).children}
    if not {"hardpoint.embed", "hardpoint.index.query"} <= retrieve_children:
        problems.append(f"retrieve children {retrieve_children}")
    generate_spans = {span.name for span in steps.get("generate", root).walk()}
    if not {"hardpoint.llm", "hardpoint.guard"} <= generate_spans:
        problems.append(f"generate spans {generate_spans}")
    if any(span.trace != root.trace for span in root.walk()):
        problems.append("a span escaped its trace")
    return problems


@pytest.mark.anyio
async def test_the_generated_service_under_load(  # noqa: PLR0915 - one scenario, read top to bottom
    service_project: dict[str, Any],
) -> None:
    from service.app import create_app

    overrides = service_project["overrides"]

    # Ingest with the same providers the service will use.
    resolved = load_config("config", env="offline", overrides=overrides)
    ingest_res = await build_resources(resolved)
    report, code = await ingest(ingest_res, [])
    await ingest_res.aclose()
    assert code == 0, report

    descriptors_before = open_descriptors()
    tasks_before = len(asyncio.all_tasks())

    app = create_app(env="offline", overrides=overrides)
    async with app.router.lifespan_context(app):
        resources = app.state.resources
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
            answers: list[dict[str, Any]] = []
            streams: list[str] = []

            async def query(n: int) -> None:
                response = await client.post(
                    "/query",
                    json={"question": f"How do I rotate API key number {n}?"},
                    headers={"x-request-id": f"load-{n}"},
                )
                assert response.status_code == 200, response.text
                answers.append(response.json())

            async def stream(n: int) -> None:
                async with client.stream(
                    "POST", "/query/stream", json={"question": f"How do I rotate key {n}?"}
                ) as response:
                    assert response.status_code == 200
                    streams.append("".join([chunk async for chunk in response.aiter_text()]))

            async with anyio.create_task_group() as group:
                for n in range(QUERIES):
                    group.start_soon(query, n)
                for n in range(STREAMS):
                    group.start_soon(stream, n)

            ready = await client.get("/readyz")
            assert ready.json() == {"status": "ready"}

        # Streams answers: text deltas first, then one done event with citations.
        for body in streams:
            events = [block for block in body.split("\n\n") if block]
            kinds = [block.split("\n", 1)[0].removeprefix("event: ") for block in events]
            assert kinds[-1] == "done", kinds
            assert kinds.count("delta") > 1, "streamed in more than one piece"
            done = json.loads(events[-1].split("data: ", 1)[1])
            assert done["citations"]

        # Reports per-request cost: priced, positive, and never zero.
        assert len(answers) == QUERIES
        assert {answer["run_id"] for answer in answers} == {f"load-{n}" for n in range(QUERIES)}
        for answer in answers:
            cost = answer["usage"]["total_cost_usd"]
            assert cost is not None
            assert cost > 0

        # Traces nest correctly across async boundaries, under concurrency.
        roots = [span for span in resources.tracer.roots if span.name == "hardpoint.pipeline"]
        assert len(roots) == QUERIES + STREAMS
        malformed = {id(root): trace_is_well_formed(root) for root in roots}
        assert not any(malformed.values()), [p for p in malformed.values() if p][:3]

        llm, embedder = resources.llm.inner, resources.embedder.inner
        index = resources.index().inner

    # No session, file or task leaks once the service has shut down.
    assert llm._client.is_closed
    assert embedder._client.is_closed
    assert index._connection is None
    await anyio.lowlevel.checkpoint()
    assert len(asyncio.all_tasks()) <= tasks_before
    descriptors_after = open_descriptors()
    if descriptors_before is not None and descriptors_after is not None:
        assert descriptors_after <= descriptors_before + 2
