"""The M3 Definition of Done, through the real command on a generated project.

INSTRUCTIONS.md §8: "deliberately degrade retrieval (for example set top_k=1)
and confirm CI fails with a useful diff; restore and confirm it passes; confirm
a full suite replays from cassettes with zero provider calls."
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hardpoint.cli.commands.init import write_project
from hardpoint.cli.main import app
from tests.contract.fake_openai_server import fake_openai_server


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "demo"
    write_project(root, "rag-minimal")
    monkeypatch.chdir(root)
    for name in ("HARDPOINT_ENV", "OPENAI_API_KEY", "HARDPOINT__RETRIEVAL__TOP_K"):
        monkeypatch.delenv(name, raising=False)
    return root


def hardpoint(*args: str, code: int = 0) -> str:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == code, result.output
    return result.output


def test_degraded_retrieval_fails_the_gate_with_a_per_case_diff(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hardpoint("--env", "offline", "ingest", "run")
    assert "PASSED" in hardpoint("--env", "offline", "eval", "run", "--suite", "smoke")

    monkeypatch.setenv("HARDPOINT__RETRIEVAL__TOP_K", "1")
    failed = hardpoint("--env", "offline", "eval", "run", "--suite", "smoke", code=1)
    assert "FAILED" in failed
    assert "### Regressions against the baseline" in failed
    assert "### Cases that got worse" in failed
    assert "| rotate-key | How do I rotate an API key? | precision_at_5 |" in failed
    report = (project / "artefacts" / "eval" / "smoke.md").read_text(encoding="utf-8")
    assert "Cases that got worse" in report, "the diff is written for CI to publish"

    monkeypatch.delenv("HARDPOINT__RETRIEVAL__TOP_K")
    assert "PASSED" in hardpoint("--env", "offline", "eval", "run", "--suite", "smoke")


def test_a_cost_cap_refuses_to_start(project: Path) -> None:
    hardpoint("--env", "offline", "ingest", "run")
    overlay = project / "config" / "priced.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "llm": {"type": "openai_chat", "model": "gpt-4o", "api_key": "${env:KEY:-x}"},
                    "embeddings": {"type": "fake_embeddings"},
                },
                "indexes": {"primary": {"type": "sqlite", "path": ".hardpoint/offline-index.db"}},
                "ingestion": {"state": {"type": "sqlite", "path": ".hardpoint/offline-state.db"}},
            }
        ),
        encoding="utf-8",
    )
    refused = hardpoint(
        "--env", "priced", "eval", "run", "--suite", "smoke", "--max-cost", "0.0001", code=2
    )
    assert "above the cap" in refused
    assert "Nothing was run" in refused


def test_a_suite_replays_from_cassettes_with_zero_provider_calls(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with fake_openai_server() as (base_url, state):
        state.dimensions = 16
        overlay = {
            "providers": {
                "llm": {"type": "openai_chat", "model": "m", "base_url": base_url},
                "embeddings": {
                    "type": "openai_embeddings",
                    "model": "e",
                    "dimensions": 16,
                    "base_url": base_url,
                },
            },
            "pricing": {"openai/m": {"input": 1.0, "output": 1.0}, "openai/e": {"embed": 1.0}},
            "indexes": {"primary": {"type": "sqlite", "path": ".hardpoint/recorded-index.db"}},
            "ingestion": {"state": {"type": "sqlite", "path": ".hardpoint/recorded-state.db"}},
            # The loopback server's vectors are random, so retrieval quality is not
            # what this test is about; replaying faithfully is.
            "eval": {"thresholds": {"hit_rate_at_5": 0.0}},
        }
        (project / "config" / "recorded.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")

        hardpoint("--env", "recorded", "ingest", "run")
        before = len(state.requests)
        hardpoint(
            "--env", "recorded", "eval", "run", "--suite", "smoke",
            "--cassettes", "record", "--update-baseline",
        )  # fmt: skip
        assert len(state.requests) > before, "recording calls the providers"
    # The provider is gone: any call now would fail to connect.

    replayed = hardpoint(
        "--env", "recorded", "eval", "run", "--suite", "smoke", "--cassettes", "replay"
    )
    assert "PASSED" in replayed
    assert "| error_rate | 0.0000 | 0.0000 |" in replayed, "every case replayed, none errored"
    assert (project / "evals" / "cassettes" / "smoke.json").is_file()
