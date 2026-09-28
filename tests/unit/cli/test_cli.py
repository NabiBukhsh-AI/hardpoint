"""The ``hardpoint`` command, driven the way a user drives it.

Every test works on a freshly generated ``rag-minimal`` project in ``tmp_path``
with the offline fakes, so the whole path -- init, ingest, ask, doctor -- runs
with zero network access (INSTRUCTIONS.md §6.6).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from hardpoint.adapters.index.sqlite import SqliteVectorIndex
from hardpoint.cli.commands.doctor import Check, run_doctor
from hardpoint.cli.commands.init import render_template_files, write_project
from hardpoint.cli.commands.project import CliState, read_dotenv
from hardpoint.cli.main import app
from hardpoint.core.config import REDACTED
from hardpoint.core.ports import IndexSpec
from hardpoint.core.registry import BUILTIN_COMPONENTS, BuiltinEntry, ComponentRegistry, Kind

SECRET = "sk-test-not-a-real-key-0123456789abcdef"  # noqa: S105 - looked for in output


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "demo"
    write_project(root, "rag-minimal")
    monkeypatch.chdir(root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HARDPOINT_ENV", raising=False)
    return root


def run(*args: str, code: int = 0) -> str:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == code, result.output
    return result.output


def offline(*args: str, code: int = 0) -> str:
    return run("--env", "offline", *args, code=code)


# --------------------------------------------------------------------------- #
# init                                                                        #
# --------------------------------------------------------------------------- #


def test_init_generates_the_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = run("init", "support-bot")
    root = tmp_path / "support-bot"

    assert "created support-bot" in output
    for expected in ("config/base.yaml", "pipelines/rag.py", "prompts/answer.md", ".gitignore",
                     ".env.example", "docs/api-keys.md", "tests/test_pipeline.py"):  # fmt: skip
        assert (root / expected).is_file(), expected
    base = (root / "config/base.yaml").read_text(encoding="utf-8")
    assert "name: support-bot" in base
    assert "{{" not in base, "every placeholder in config is substituted"
    assert "{{ question }}" in (root / "prompts/answer.md").read_text(encoding="utf-8"), (
        "prompt placeholders belong to the prompt renderer and must survive generation"
    )


def test_init_never_overwrites_a_users_files(project: Path) -> None:
    """ARCHITECTURE.md §24: the generator never rewrites user files."""
    (project / "pipelines" / "rag.py").write_text("# mine\n", encoding="utf-8")
    output = run("init", str(project), code=2)
    assert "not empty" in output
    assert (project / "pipelines" / "rag.py").read_text(encoding="utf-8") == "# mine\n"


def test_init_diff_shows_what_a_fresh_template_would_change(project: Path) -> None:
    assert "no differences" in run("init", str(project), "--diff")
    (project / "prompts" / "answer.md").write_text("changed\n", encoding="utf-8")
    diff = run("init", str(project), "--diff")
    assert "-changed" in diff
    assert "prompts/answer.md" in diff


def test_an_unknown_template_is_refused(tmp_path: Path) -> None:
    output = run("init", str(tmp_path / "x"), "--template", "nope", code=2)
    assert "Available templates" in output


def test_every_template_file_is_text_the_renderer_understands() -> None:
    files = render_template_files("rag-minimal", "p")
    assert "dot_gitignore" not in files
    assert ".gitignore" in files


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #


def test_config_show_redacts_secrets_and_names_origins(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    output = run("config", "show")
    assert SECRET not in output
    assert f'providers.llm.api_key = "{REDACTED}"' in output
    assert "retrieval.top_k = 8    # base_file" in output


def test_config_show_reads_the_dotenv_file(project: Path) -> None:
    (project / ".env").write_text(f"OPENAI_API_KEY={SECRET}\n", encoding="utf-8")
    output = run("config", "show", "--yaml")
    assert SECRET not in output
    assert REDACTED in output


def test_config_schema_covers_registered_components(project: Path) -> None:
    schema = json.loads(run("config", "schema"))
    definitions = schema["$defs"]
    assert "component_index_sqlite" in definitions
    assert definitions["component_llm_openai_chat"]["properties"]["type"] == {
        "const": "openai_chat"
    }
    assert definitions["component_llm_openai_chat"]["additionalProperties"] is False


def test_config_validate_catches_an_unknown_component_option(project: Path) -> None:
    assert "ok: env 'offline'" in offline("config", "validate")
    overlay = project / "config" / "offline.yaml"
    overlay.write_text(
        overlay.read_text(encoding="utf-8") + "\nsources:\n  docs: {type: local_files, rot: x}\n",
        encoding="utf-8",
    )
    output = offline("config", "validate", code=2)
    assert "rot" in output
    assert "Did you mean 'root'" in output


def test_a_config_error_prints_the_remedy_not_a_traceback(project: Path) -> None:
    output = run("ask", "hello", code=2)
    assert "OPENAI_API_KEY" in output
    assert "remedy" in output
    assert "Traceback" not in output


def test_dotenv_parsing() -> None:
    assert read_dotenv(Path("definitely-missing.env")) == {}


def test_dotenv_handles_comments_quotes_and_export(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# c\n\nexport A=1\nB='two'\nC=\"three\"\nnot a pair\n", encoding="utf-8")
    assert read_dotenv(path) == {"A": "1", "B": "two", "C": "three"}


# --------------------------------------------------------------------------- #
# components                                                                  #
# --------------------------------------------------------------------------- #


def test_components_list_shows_builtins_and_their_source(project: Path) -> None:
    output = run("components", "list", "--type", "index")
    assert "sqlite" in output
    assert "qdrant" in output
    assert "builtin" in output
    assert "openai_chat" not in output


def test_components_list_describe_prints_options(project: Path) -> None:
    output = run("components", "list", "--type", "index", "--describe")
    assert "collection:" in output
    assert "path:" in output


def test_components_list_resolved_shows_each_policy_chain(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    output = run("components", "list", "--resolved")
    assert "providers.llm" in output
    assert "timeout(total) -> retry -> timeout(per_attempt)" in output


def test_components_list_rejects_an_unknown_kind(project: Path) -> None:
    assert "unknown kind" in run("components", "list", "--type", "gadget", code=2)


# --------------------------------------------------------------------------- #
# ingest and ask                                                              #
# --------------------------------------------------------------------------- #


def test_ingest_plan_changes_nothing(project: Path) -> None:
    output = offline("ingest", "plan")
    assert "added      4" in output
    assert not (project / ".hardpoint" / "offline-index.db").exists() or "records" not in output
    # Nothing ingested, so status reports it.
    assert "never ingested" in offline("ingest", "status")


def test_ingest_run_then_rerun_embeds_nothing_the_second_time(project: Path) -> None:
    first = offline("ingest", "run")
    assert "ingest docs: ok" in first
    second = offline("ingest", "run")
    assert "embedded   0 chunks" in second
    status = offline("ingest", "status")
    assert "docs: ok" in status
    assert "index primary: epoch 2" in status


def test_ingest_run_exits_non_zero_when_something_was_quarantined(project: Path) -> None:
    (project / "docs" / "empty.md").write_text("   \n", encoding="utf-8")
    output = offline("ingest", "run", code=1)
    assert "quarantined" in output


def test_ask_returns_an_answer_with_citations(project: Path) -> None:
    """INSTRUCTIONS.md §6.6: ``ask`` returns an Answer with non-empty citations."""
    offline("ingest", "run")
    answer = json.loads(offline("ask", "How do I rotate an API key?", "--json"))

    assert answer["citations"]
    assert answer["citations"][0]["source_uri"].endswith("api-keys.md")
    assert "[1]" in answer["text"]
    assert answer["manifest"]["index_epochs"] == {"primary": 1}
    assert not answer["abstained"]


def test_ask_explain_prints_the_execution_report(project: Path) -> None:
    offline("ingest", "run")
    report = offline("ask", "What happens when I am rate limited?", "--explain")
    for section in ("steps", "retrieve", "context (", "dropped (", "prompt (answer@", "usage"):
        assert section in report, section
    assert "<retrieved_context>" in report


def test_explain_redacts_prompt_content_when_configured(project: Path) -> None:
    overlay = project / "config" / "offline.yaml"
    overlay.write_text(
        overlay.read_text(encoding="utf-8") + "\nobservability: {redact: [messages.content]}\n",
        encoding="utf-8",
    )
    offline("ingest", "run")
    report = offline("ask", "How do I rotate an API key?", "--explain")
    assert "[redacted]" in report
    assert "<retrieved_context>" not in report


def test_ask_on_an_empty_index_abstains_with_the_projects_wording(project: Path) -> None:
    output = offline("ask", "anything at all")
    assert "couldn't find anything in our documentation" in output
    assert "abstained" in output


# --------------------------------------------------------------------------- #
# doctor: the seeded misconfigurations (INSTRUCTIONS.md §6.6 and §14)         #
# --------------------------------------------------------------------------- #


def seed(project: Path, changes: dict[str, Any]) -> CliState:
    """Write an overlay that is the offline one plus ``changes``, and select it."""
    overlay = yaml.safe_load((project / "config" / "offline.yaml").read_text(encoding="utf-8"))

    def merge(target: dict[str, Any], update: dict[str, Any]) -> None:
        # The loader's rule: a block whose `type` changes is replaced, not merged.
        for key, value in update.items():
            existing = target.get(key)
            if (
                isinstance(value, dict)
                and isinstance(existing, dict)
                and value.get("type", existing.get("type")) == existing.get("type")
            ):
                merge(existing, value)
            else:
                target[key] = value

    merge(overlay, changes)
    (project / "config" / "seeded.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return CliState(env="seeded")


def failures(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks if check.status != "ok"}


@pytest.mark.anyio
async def test_doctor_passes_on_a_healthy_project(project: Path) -> None:
    checks = await run_doctor(CliState(env="offline"))
    assert failures(checks) == {}
    assert {"secret hygiene", "configuration", "pipeline", "index primary"} <= {
        check.name for check in checks
    }


@pytest.mark.anyio
async def test_doctor_catches_a_missing_environment_variable(project: Path) -> None:
    found = failures(await run_doctor(CliState()))
    check = found["environment OPENAI_API_KEY"]
    assert check.status == "fail"
    assert ".env" in check.remedy


@pytest.mark.anyio
async def test_doctor_catches_a_missing_extra(project: Path) -> None:
    registry = ComponentRegistry(
        [
            *BUILTIN_COMPONENTS,
            BuiltinEntry("vendor_index", Kind.INDEX, "tests.fixtures.needs_missing_extra",
                         "build", "Config", extra="vendor"),
        ]
    )  # fmt: skip
    state = seed(project, {"indexes": {"primary": {"type": "vendor_index"}}})
    check = failures(await run_doctor(state, registry=registry))["component indexes.primary"]
    assert check.status == "fail"
    assert "pip install 'hardpoint[vendor]'" in check.remedy


@pytest.mark.anyio
async def test_doctor_catches_a_dimension_mismatch(project: Path) -> None:
    index = SqliteVectorIndex(project / ".hardpoint" / "offline-index.db")
    await index.ensure(IndexSpec(name="primary", dimensions=8))
    await index.aclose()

    check = failures(await run_doctor(CliState(env="offline")))["dimensions primary"]
    assert check.status == "fail"
    assert "8-dimension" in check.detail
    assert "256" in check.detail


@pytest.mark.anyio
async def test_doctor_catches_an_unreachable_index(project: Path) -> None:
    with socket.socket() as probe:  # a port nothing is listening on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    state = seed(
        project,
        {
            "indexes": {
                "primary": {"type": "qdrant", "url": f"http://127.0.0.1:{port}", "timeout_s": 2}
            }
        },
    )
    check = failures(await run_doctor(state))["index primary"]
    assert check.status == "fail"
    assert "unreachable" in check.detail


@pytest.mark.anyio
async def test_doctor_catches_a_literal_secret_in_yaml(project: Path) -> None:
    base = project / "config" / "base.yaml"
    base.write_text(
        base.read_text(encoding="utf-8").replace("${env:OPENAI_API_KEY}", SECRET),
        encoding="utf-8",
    )
    check = failures(await run_doctor(CliState(env="offline")))["secret hygiene"]
    assert check.status == "fail"
    assert "api_key has a literal value" in check.detail
    assert "Rotate" in check.remedy


@pytest.mark.anyio
async def test_doctor_warns_on_version_skew(project: Path) -> None:
    state = seed(project, {"project": {"hardpoint_version": "9.9.0"}})
    check = failures(await run_doctor(state))["version"]
    assert check.status == "warn"
    assert "9.9.0" in check.detail


@pytest.mark.anyio
async def test_doctor_catches_a_missing_source_directory(project: Path) -> None:
    state = seed(project, {"sources": {"docs": {"type": "local_files", "root": "nowhere"}}})
    assert failures(await run_doctor(state))["source docs"].status == "fail"


@pytest.mark.anyio
async def test_doctor_does_not_mistake_token_budgets_for_secrets(project: Path) -> None:
    """``token_budget`` and ``max_tokens`` end in token-ish words and are not secrets."""
    checks = await run_doctor(CliState(env="offline"))
    assert next(c for c in checks if c.name == "secret hygiene").status == "ok"


def test_doctor_command_exits_non_zero_on_failure(project: Path) -> None:
    output = run("doctor", code=1)
    assert "[FAIL] environment OPENAI_API_KEY" in output
    assert "1 failed" in output


def test_doctor_live_probes_each_provider(project: Path) -> None:
    output = offline("doctor", "--live")
    assert "[ ok ] provider embeddings" in output
    assert "[ ok ] provider llm" in output


def test_serve_runs_the_project_service_with_a_graceful_shutdown(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hardpoint.cli.main as cli_main

    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeUvicorn:
        @staticmethod
        def run(target: str, **options: Any) -> None:
            calls.append((target, options))

    monkeypatch.setattr(cli_main, "_import_uvicorn", lambda: FakeUvicorn)
    offline("serve", "--port", "9123")

    ((target, options),) = calls
    assert target == "service.app:create_app"
    assert options["factory"] is True
    assert options["port"] == 9123
    assert options["timeout_graceful_shutdown"] == 60, "the request deadline, from config"
    assert os.environ["HARDPOINT_ENV"] == "offline"


def test_serve_without_the_extra_names_the_install_command(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hardpoint.cli.main as cli_main
    from hardpoint.core.errors import MissingDependencyError

    def missing() -> Any:
        raise MissingDependencyError("no uvicorn", remedy="pip install 'hardpoint[serve]'")

    monkeypatch.setattr(cli_main, "_import_uvicorn", missing)
    assert "hardpoint[serve]" in offline("serve", code=2)


def test_version_command() -> None:
    assert "contract" in run("version")
