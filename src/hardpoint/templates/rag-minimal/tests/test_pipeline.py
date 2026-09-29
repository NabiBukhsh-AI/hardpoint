"""The whole path -- ingest, then ask -- with the offline fakes.

No keys and no network, so this runs in CI. It drives the real `hardpoint`
command against a copy of this project, so what it proves is what you run.
"""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hardpoint.cli.main import app

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway copy, so the index and manifest never land in the real project."""
    copy = tmp_path / "project"
    shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns(".hardpoint", "artefacts", ".env"))
    monkeypatch.chdir(copy)
    monkeypatch.setenv("HARDPOINT_ENV", "offline")
    return copy


def hardpoint(*args: str) -> str:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.stdout


def test_ingest_then_ask_returns_a_cited_answer(project: Path) -> None:
    hardpoint("ingest", "run")
    answer = json.loads(hardpoint("ask", "How do I rotate an API key?", "--json"))

    assert answer["citations"], "an answer must cite what it was built from"
    assert not answer["abstained"]
    assert any("api-keys" in citation["source_uri"] for citation in answer["citations"])


def test_reingesting_an_unchanged_corpus_embeds_nothing(project: Path) -> None:
    hardpoint("ingest", "run")
    second = hardpoint("ingest", "run")
    assert "embedded   0 chunks" in second


def test_the_explain_report_shows_what_happened(project: Path) -> None:
    hardpoint("ingest", "run")
    report = hardpoint("ask", "What happens when I am rate limited?", "--explain")
    for section in ("steps", "context (", "dropped (", "prompt (", "usage"):
        assert section in report


def test_doctor_passes(project: Path) -> None:
    assert "0 failed" in hardpoint("doctor")


def test_the_quality_gate_passes_against_the_committed_baseline(project: Path) -> None:
    hardpoint("ingest", "run")
    assert "PASSED" in hardpoint("eval", "run", "--suite", "smoke")
