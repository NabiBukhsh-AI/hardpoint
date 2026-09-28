"""The service end to end, offline: ingest, then every route.

Runs with the fakes from `config/offline.yaml`, so no keys and no network.
"""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hardpoint.cli.main import app as cli

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    copy = tmp_path / "project"
    shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns(".hardpoint", "artefacts", ".env"))
    monkeypatch.chdir(copy)
    monkeypatch.syspath_prepend(str(copy))
    monkeypatch.setenv("HARDPOINT_ENV", "offline")
    result = CliRunner().invoke(cli, ["ingest", "run"])
    assert result.exit_code == 0, result.output

    from service.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_query_returns_a_cited_answer_and_echoes_the_request_id(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": "How do I rotate an API key?"}, headers={"x-request-id": "req-42"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["run_id"] == "req-42"
    assert response.headers["x-request-id"] == "req-42"
    assert "total_cost_usd" in body["usage"]


def test_query_stream_sends_deltas_then_a_done_event(client: TestClient) -> None:
    with client.stream("POST", "/query/stream", json={"question": "What are the rate limits?"}) as response:
        body = "".join(response.iter_text())
    events = [block for block in body.split("\n\n") if block]
    kinds = [block.split("\n")[0].removeprefix("event: ") for block in events]
    assert kinds[0] == "delta"
    assert kinds[-1] == "done"
    done = json.loads(events[-1].split("data: ", 1)[1])
    assert done["citations"]


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_metrics_are_exposed(client: TestClient) -> None:
    client.post("/query", json={"question": "How do refunds work?"})
    text = client.get("/metrics").text
    assert "hardpoint_requests_total" in text
