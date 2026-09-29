"""``hardpoint eval run``: the quality gate.

Loads ``<datasets_dir>/<suite>.jsonl``, runs it through the project's own
pipeline factory (``EvalRunner``), gates the result on thresholds and on the
committed baseline, and writes a JSON and a Markdown report. Exits non-zero when
the gate fails, so CI fails with the per-case regression table in its log -- and
in the GitHub job summary, when there is one.

``--max-cost`` refuses to start a suite whose estimated cost exceeds the cap,
and refuses to start one it cannot estimate, rather than guessing.
``--cassettes replay`` runs from recordings with zero provider calls.
"""

import os
from pathlib import Path
from typing import Literal

import anyio

from hardpoint.core.errors import ConfigError
from hardpoint.core.registry import Kind
from hardpoint.eval.baseline import evaluate_gate, load_baseline, write_baseline
from hardpoint.eval.dataset import load_dataset
from hardpoint.eval.metrics.judge import Judge
from hardpoint.eval.report import EvalReport, render_markdown
from hardpoint.eval.runner import EvalRunner, estimate_cost
from hardpoint.runtime.resources import Resources
from hardpoint.testing.cassettes import Cassette, wrap_resources

__all__ = ["CassetteSetting", "run_suite"]

CassetteSetting = Literal["off", "record", "replay"]


async def _judge(res: Resources) -> Judge | None:
    settings = res.config.eval.judge
    if not settings.enabled:
        return None
    model = (
        await res.registry.create(
            Kind.LLM, settings.llm.type, settings.llm.options(), config_path="eval.judge.llm"
        )
        if settings.llm is not None
        else res.llm
    )
    return Judge(
        model,
        res.prompts,
        temperature=settings.temperature,
        faithfulness_prompt=settings.faithfulness_prompt,
        relevance_prompt=settings.relevance_prompt,
    )


async def run_suite(
    res: Resources,
    suite: str,
    *,
    tags: tuple[str, ...] = (),
    max_cost: float | None = None,
    cassettes: CassetteSetting = "off",
    update_baseline: bool = False,
    output: Path = Path("artefacts/eval"),
) -> tuple[EvalReport, str]:
    """Run a suite and gate it.

    Returns:
        The report, and its Markdown rendering.

    Raises:
        ConfigError: For a missing dataset, or a cost cap the suite would exceed
            or cannot be estimated against.
    """
    settings = res.config.eval
    cases = load_dataset(Path(settings.datasets_dir) / f"{suite}.jsonl")
    if tags:
        cases = [case for case in cases if set(tags) & set(case.tags)]
    judge = await _judge(res)

    cap = max_cost if max_cost is not None else settings.max_cost_usd
    if cap is not None and cassettes != "replay":
        estimate = estimate_cost(res, cases, judge=judge is not None)
        if estimate is None:
            raise ConfigError(
                f"Suite {suite!r} cannot be costed: a model it uses has no price, so a "
                f"cap of ${cap:.2f} cannot be enforced.",
                config_path="pricing",
                remedy="Add the model under `pricing:` in config/base.yaml, or run without a cap.",
            )
        if estimate > cap:
            raise ConfigError(
                f"Suite {suite!r} is estimated to cost ${estimate:.4f} for {len(cases)} "
                f"cases, above the cap of ${cap:.2f}. Nothing was run.",
                config_path="eval.max_cost_usd",
                remedy=(
                    "Raise --max-cost, run a subset with --tag, or replay from "
                    "cassettes with --cassettes replay."
                ),
            )

    cassette = None
    if cassettes != "off":
        cassette = Cassette(Path(settings.datasets_dir) / "cassettes" / f"{suite}.json", cassettes)
        res = wrap_resources(res, cassette)

    report = await EvalRunner(res, judge=judge).run(suite, cases)
    if cassette is not None and cassettes == "record":
        cassette.save()

    baseline_path = Path(settings.baselines_dir) / f"{suite}.json"
    baseline = None if update_baseline else load_baseline(baseline_path)
    report.gate = evaluate_gate(
        report, thresholds=settings.thresholds, baseline=baseline, tolerance=settings.tolerance
    )
    if update_baseline:
        write_baseline(baseline_path, report)
        report.gate.notes.append(f"Baseline written to {baseline_path}.")

    markdown = render_markdown(report, (baseline or {}).get("metrics"))
    await anyio.to_thread.run_sync(_write_outputs, report, markdown, output)
    return report, markdown


def _write_outputs(report: EvalReport, markdown: str, output: Path) -> None:
    """Write the JSON and Markdown reports, and the GitHub job summary when there is one."""
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{report.suite}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (output / f"{report.suite}.md").write_text(markdown, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(markdown)
