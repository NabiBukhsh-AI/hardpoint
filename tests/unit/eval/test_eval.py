"""Evaluation: datasets, metrics, gate, judge, runner and cassettes (INSTRUCTIONS.md §8)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from hardpoint.core.config import resolve
from hardpoint.core.errors import ConfigError
from hardpoint.core.ids import document_id
from hardpoint.core.ports import GenerationRequest, Message
from hardpoint.eval import (
    CaseResult,
    EvalCase,
    EvalReport,
    EvalRunner,
    Judge,
    estimate_cost,
    evaluate_gate,
    load_baseline,
    load_dataset,
    render_markdown,
    resolve_document_ids,
    write_baseline,
    write_dataset,
)
from hardpoint.eval import runner as runner_module
from hardpoint.eval.metrics import (
    context_precision,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from hardpoint.generation import InMemoryPromptStore
from hardpoint.ingestion.chunkers.recursive import RecursiveChunker
from hardpoint.ingestion.parsers.text import MarkdownParser, TextParser
from hardpoint.ingestion.sync import SyncEngine
from hardpoint.runtime import build_resources, build_source
from hardpoint.testing import FakeLanguageModel, ScriptedResponse, build_run_context
from hardpoint.testing.cassettes import Cassette, CassetteMissError, wrap_resources

# --------------------------------------------------------------------------- #
# Datasets                                                                    #
# --------------------------------------------------------------------------- #


def test_datasets_round_trip_as_one_case_per_line(tmp_path: Path) -> None:
    cases = [
        EvalCase(id="a", query="first?", expected_document_ids=("docs:a.md",), tags=("smoke",)),
        EvalCase(id="b", query="second?"),
    ]
    path = write_dataset(tmp_path / "suite.jsonl", cases)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert load_dataset(path) == cases


def test_a_bad_line_is_named_by_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    path.write_text('{"id": "a", "query": "q"}\n{"id": "b"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="line 2"):
        load_dataset(path)


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    path.write_text('{"id": "a", "query": "q"}\n\n{"id": "a", "query": "r"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="reuses the case id 'a' from line 1"):
        load_dataset(path)


def test_a_missing_dataset_explains_the_format(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_dataset(tmp_path / "absent.jsonl")
    assert '"query"' in (exc_info.value.remedy or "")


def test_source_path_document_ids_resolve_to_ingestion_ids() -> None:
    case = EvalCase(id="a", query="q", expected_document_ids=("docs:api-keys.md", "doc_abc"))
    assert resolve_document_ids(case) == {document_id("docs", "api-keys.md"), "doc_abc"}


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def test_retrieval_metrics_by_hand() -> None:
    flags = [False, True, False, True]
    assert hit_rate_at_k(flags, 1) == 0.0
    assert hit_rate_at_k(flags, 2) == 1.0
    assert precision_at_k(flags, 4) == 0.5
    assert precision_at_k([True], 5) == 0.2, "retrieving fewer than k does not flatter precision"
    assert mrr(flags) == 0.5
    assert mrr([False, False]) == 0.0
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert context_precision([True, False, True, True]) == 0.75
    assert context_precision([]) == 0.0


def test_ndcg_rewards_order_and_counts_what_was_never_retrieved() -> None:
    assert ndcg_at_k([True, False], 2) == 1.0
    assert ndcg_at_k([False, True], 2) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k([True], 5, missing=1) < 1.0, "finding one of two expected is not perfect"
    assert ndcg_at_k([False, False], 2) == 0.0


# --------------------------------------------------------------------------- #
# Gate and report                                                             #
# --------------------------------------------------------------------------- #


def report(
    metrics: dict[str, float | None], cases: dict[str, dict[str, float]] | None = None
) -> EvalReport:
    return EvalReport(
        suite="s",
        hardpoint_version="0",
        config_hash="h" * 64,
        started_at="t0",
        finished_at="t1",
        k=5,
        cases=[
            CaseResult(case_id=case_id, query=f"q {case_id}", run_id="r", metrics=values)
            for case_id, values in (cases or {}).items()
        ],
        metrics=metrics,
    )


def test_thresholds_are_minimums_except_for_latency_cost_and_rates() -> None:
    result = evaluate_gate(
        report({"recall_at_5": 0.7, "p95_latency_ms": 900.0, "error_rate": 0.0}),
        thresholds={"recall_at_5": 0.8, "p95_latency_ms": 500.0, "error_rate": 0.0, "mrr": 0.5},
        baseline=None,
        tolerance=0.0,
    )
    assert not result.passed
    assert result.threshold_failures == [
        "mrr: not measured (threshold 0.5)",
        "p95_latency_ms: 900.0000 is above the maximum 500.0",
        "recall_at_5: 0.7000 is below the minimum 0.8",
    ]


def test_a_regression_beyond_tolerance_fails_with_the_cases_that_got_worse(tmp_path: Path) -> None:
    baseline_run = report(
        {"recall_at_5": 1.0, "p95_latency_ms": 10.0},
        {"a": {"recall_at_5": 1.0}, "b": {"recall_at_5": 1.0}},
    )
    path = write_baseline(tmp_path / "s.json", baseline_run)
    baseline = load_baseline(path)

    current = report(
        {"recall_at_5": 0.5, "p95_latency_ms": 99.0},
        {"a": {"recall_at_5": 0.0}, "b": {"recall_at_5": 1.0}},
    )
    result = evaluate_gate(current, thresholds={}, baseline=baseline, tolerance=0.02)

    assert not result.passed
    assert [(r.metric, r.baseline, r.current) for r in result.regressions] == [
        ("recall_at_5", 1.0, 0.5)
    ]
    assert [(r.case_id, r.delta) for r in result.case_regressions] == [("a", -1.0)]
    assert "p95_latency_ms" not in {r.metric for r in result.regressions}, (
        "latency is not baselined"
    )

    current.gate = result
    markdown = render_markdown(current, baseline["metrics"] if baseline else None)
    assert "FAILED" in markdown
    assert "| a | q a | recall_at_5 | 1.0000 | 0.0000 | -1.0000 |" in markdown


def test_within_tolerance_passes() -> None:
    baseline = {"metrics": {"recall_at_5": 0.80}, "cases": {}}
    result = evaluate_gate(
        report({"recall_at_5": 0.79}), thresholds={}, baseline=baseline, tolerance=0.02
    )
    assert result.passed


def test_judge_metrics_are_not_compared_across_different_judges() -> None:
    baseline = {"metrics": {"faithfulness": 0.9}, "judge": {"judge_model": "a"}, "cases": {}}
    current = report({"faithfulness": 0.1})
    current.judge = {"judge_model": "b"}
    result = evaluate_gate(current, thresholds={}, baseline=baseline, tolerance=0.0)
    assert result.passed
    assert any("judge" in note for note in result.notes)


def test_no_baseline_is_noted() -> None:
    result = evaluate_gate(report({}), thresholds={}, baseline=None, tolerance=0.0)
    assert result.passed
    assert load_baseline(Path("definitely/absent.json")) is None


# --------------------------------------------------------------------------- #
# Judge                                                                       #
# --------------------------------------------------------------------------- #


JUDGE_PROMPTS = {
    "judge_faithfulness": (
        "--- user ---\nRate faithfulness. Q: {{ question }} A: {{ answer }} C: {{ context }}"
    ),
    "judge_relevance": (
        "--- user ---\nRate relevance. Q: {{ question }} A: {{ answer }} R: {{ reference }}"
    ),
}


@pytest.mark.anyio
async def test_judge_scores_record_model_prompt_version_and_temperature() -> None:
    from hardpoint.core.models import Answer, RunManifest, Usage

    model = FakeLanguageModel(
        [
            ScriptedResponse('{"score": 0.9, "reason": "supported"}', match="faithfulness"),
            ScriptedResponse('{"score": 0.4, "reason": "tangential"}', match="relevance"),
        ],
        model_id="fake/judge",
    )
    prompts = InMemoryPromptStore(JUDGE_PROMPTS)
    judge = Judge(model, prompts)
    answer = Answer(
        text="A",
        usage=Usage(),
        run_id="r",
        manifest=RunManifest(hardpoint_version="0", config_hash="", pipeline_name="p"),
    )
    faithful = await judge.faithfulness("Q", answer, build_run_context())
    relevant = await judge.answer_relevance("Q", answer, build_run_context(), reference="R")

    assert (faithful.score, relevant.score) == (0.9, 0.4)
    assert faithful.judge_model == "fake/judge"
    assert faithful.temperature == 0.0
    assert faithful.prompt_version == await prompts.version_of("judge_faithfulness")
    assert model.calls[0].temperature == 0.0


@pytest.mark.anyio
async def test_a_missing_judge_prompt_says_where_it_belongs() -> None:
    from hardpoint.core.models import Answer, RunManifest, Usage

    judge = Judge(FakeLanguageModel(), InMemoryPromptStore())
    answer = Answer(
        text="A",
        usage=Usage(),
        run_id="r",
        manifest=RunManifest(hardpoint_version="0", config_hash="", pipeline_name="p"),
    )
    with pytest.raises(ConfigError) as exc_info:
        await judge.faithfulness("Q", answer, build_run_context())
    assert "prompts/judge_faithfulness.md" in (exc_info.value.remedy or "")


# --------------------------------------------------------------------------- #
# The runner: the production path, measured                                   #
# --------------------------------------------------------------------------- #


def corpus(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "docs" / "keys.md").write_text(
        "# Keys\n\nRotate a key by creating a new key and revoking the old key.\n", encoding="utf-8"
    )
    (root / "docs" / "billing.md").write_text(
        "# Billing\n\nInvoices are issued monthly and payable within thirty days.\n",
        encoding="utf-8",
    )


def offline_config(root: Path) -> dict[str, Any]:
    return {
        "project": {"pipeline": "hardpoint.recipes.naive:build"},
        "providers": {"llm": {"type": "fake_llm"}, "embeddings": {"type": "fake_embeddings"}},
        "indexes": {"primary": {"type": "sqlite", "path": str(root / "index.db")}},
        "sources": {"docs": {"type": "local_files", "root": str(root / "docs")}},
        "ingestion": {"state": {"type": "sqlite", "path": str(root / "state.db")}},
        "retrieval": {"top_k": 4},
    }


async def ingested(root: Path, overrides: dict[str, Any] | None = None) -> Any:
    corpus(root)
    res = await build_resources(
        resolve(base={**offline_config(root), **(overrides or {})}, environ={}),
        prompts=ANSWER_PROMPT,
    )
    engine = SyncEngine(
        source=await build_source(res, "docs"),
        state=res.state,
        index=res.index(),
        embedder=res.embedder,
        chunker=RecursiveChunker(),
        parsers=[MarkdownParser(), TextParser()],
    )
    await engine.run(res.run_context())
    return res


ANSWER_PROMPT = InMemoryPromptStore({"answer": "{{ context }}\n\nQuestion: {{ question }}"})

CASES = [
    EvalCase(id="keys", query="How do I rotate a key?", expected_document_ids=("docs:keys.md",)),
    EvalCase(
        id="billing", query="When are invoices payable?", expected_document_ids=("docs:billing.md",)
    ),
    EvalCase(id="unlabelled", query="Hello there?"),
]


@pytest.mark.anyio
async def test_the_runner_measures_the_production_path(tmp_path: Path) -> None:
    res = await ingested(tmp_path)
    runner = EvalRunner(res, k=2)
    result = await runner.run("suite", CASES)
    await res.aclose()

    keys = result.case("keys")
    assert keys.metrics["hit_rate_at_2"] == 1.0
    assert keys.metrics["mrr"] == 1.0
    assert keys.run_id == "eval-suite-keys"
    assert result.case("unlabelled").metrics == {}
    assert result.metrics["hit_rate_at_2"] == 1.0, (
        "only labelled cases count towards retrieval means"
    )
    assert result.metrics["error_rate"] == 0.0
    assert result.metrics["total_cost_usd"] == 0.0
    assert result.metrics["p95_latency_ms"] is not None


def test_the_runner_has_no_execution_path_of_its_own() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §8: the factory the service uses, never a second path."""
    source = inspect.getsource(runner_module)
    assert "load_pipeline(res)" in source
    assert "answer_query(" in source
    assert "Pipeline(" not in source, "the runner must not compose a pipeline of its own"
    assert "run_detailed" not in source, "nor drive one around answer_query"


@pytest.mark.anyio
async def test_the_runner_answers_exactly_as_answer_query_does(tmp_path: Path) -> None:
    from hardpoint.runtime import answer_query, load_pipeline

    res = await ingested(tmp_path)
    direct = await answer_query(load_pipeline(res), CASES[0].query, res)
    measured = (await EvalRunner(res).run("s", CASES[:1])).case("keys")
    await res.aclose()
    assert measured.answer == direct.text
    assert (
        measured.retrieved == [item.chunk.id for item in direct.context.items]
        if direct.context
        else []
    )


@pytest.mark.anyio
async def test_a_failing_case_is_recorded_not_raised(tmp_path: Path) -> None:
    res = await ingested(tmp_path, {"retrieval": {"top_k": 4, "no_context_policy": "raise"}})
    await res.index().delete(
        res.run_context(),
        filter=__import__("hardpoint.core.filters", fromlist=["F"]).F.field("document_id").exists(),
    )
    result = await EvalRunner(res).run("s", CASES[:1])
    await res.aclose()
    assert result.case("keys").error is not None
    assert result.metrics["error_rate"] == 1.0
    assert result.case("keys").metrics["hit_rate_at_5"] == 0.0


@pytest.mark.anyio
async def test_cost_estimates_refuse_to_guess(tmp_path: Path) -> None:
    res = await ingested(tmp_path)
    assert estimate_cost(res, CASES) == 0.0, "fakes are free"
    await res.aclose()

    priced = await build_resources(
        resolve(
            base={
                "providers": {
                    "llm": {"type": "openai_chat", "model": "gpt-4o-mini", "api_key": "x"},
                    "embeddings": {
                        "type": "openai_embeddings",
                        "model": "text-embedding-3-small",
                        "dimensions": 8,
                        "api_key": "x",
                    },
                }
            },
            environ={},
        )
    )
    estimate = estimate_cost(priced, CASES)
    assert estimate is not None
    assert estimate > 0
    assert estimate_cost(priced, CASES, judge=True) > estimate
    unpriced = priced.with_components(llm_=FakeLanguageModel(model_id="acme/secret-model"))
    assert estimate_cost(unpriced, CASES) is None
    await priced.aclose()


# --------------------------------------------------------------------------- #
# Cassettes                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_cassette_replays_a_suite_with_zero_provider_calls(tmp_path: Path) -> None:
    from tests.contract.fake_openai_server import fake_openai_server

    corpus(tmp_path)
    with fake_openai_server() as (base_url, state):
        state.dimensions = 8
        config = offline_config(tmp_path)
        config["providers"] = {
            "llm": {"type": "openai_chat", "model": "m", "base_url": base_url},
            "embeddings": {
                "type": "openai_embeddings",
                "model": "e",
                "dimensions": 8,
                "base_url": base_url,
            },
        }
        config["pricing"] = {"openai/m": {"input": 1, "output": 1}, "openai/e": {"embed": 1}}
        res = await build_resources(resolve(base=config, environ={}), prompts=ANSWER_PROMPT)
        engine = SyncEngine(
            source=await build_source(res, "docs"), state=res.state, index=res.index(),
            embedder=res.embedder, chunker=RecursiveChunker(), parsers=[MarkdownParser()],
        )  # fmt: skip
        await engine.run(res.run_context())

        cassette_path = tmp_path / "cassette.json"
        recording = Cassette(cassette_path, "record")
        recorded = await EvalRunner(wrap_resources(res, recording)).run("s", CASES)
        recording.save()
        calls_after_recording = len(state.requests)

        replaying = Cassette(cassette_path, "replay")
        replayed = await EvalRunner(wrap_resources(res, replaying)).run("s", CASES)
        await res.aclose()

    assert len(state.requests) == calls_after_recording, "replay made zero provider calls"
    assert replaying.hits > 0
    assert [c.answer for c in replayed.cases] == [c.answer for c in recorded.cases]
    assert replayed.metrics["recall_at_5"] == recorded.metrics["recall_at_5"]
    assert json.loads(cassette_path.read_text(encoding="utf-8"))


@pytest.mark.anyio
async def test_an_unrecorded_request_in_replay_mode_raises(tmp_path: Path) -> None:
    from hardpoint.testing.cassettes import CassetteLanguageModel

    model = CassetteLanguageModel(FakeLanguageModel(), Cassette(tmp_path / "c.json", "replay"))
    request = GenerationRequest(messages=(Message(role="user", content="new"),))
    with pytest.raises(CassetteMissError) as exc_info:
        await model.generate(request, build_run_context())
    assert "--cassettes record" in (exc_info.value.remedy or "")


@pytest.mark.anyio
async def test_streams_are_recorded_and_replayed(tmp_path: Path) -> None:
    from hardpoint.testing.cassettes import CassetteLanguageModel

    path = tmp_path / "c.json"
    request = GenerationRequest(messages=(Message(role="user", content="hi"),))
    recorder = Cassette(path, "record")
    live = FakeLanguageModel([ScriptedResponse("one two three")])
    recorded = [
        d async for d in CassetteLanguageModel(live, recorder).stream(request, build_run_context())
    ]
    recorder.save()

    silent = FakeLanguageModel(fail_with=RuntimeError("must not be called"))
    replayed = [
        d
        async for d in CassetteLanguageModel(silent, Cassette(path, "replay")).stream(
            request, build_run_context()
        )
    ]
    assert replayed == recorded
