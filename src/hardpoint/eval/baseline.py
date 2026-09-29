"""Committed baselines and the quality gate (INSTRUCTIONS.md §8).

``evals/baselines/<suite>.json`` is a committed artefact: the aggregate metrics
and each case's retrieval metrics from the run everyone agreed was good. The
gate fails when

- any configured threshold is breached, or
- an aggregate quality metric fell more than ``tolerance`` below the baseline,

and in either case it lists the cases that got worse, so the failure says which
questions regressed and by how much rather than only that a number moved.

Latency and cost are gated by thresholds only, never against the baseline: they
move with the machine and the network, and a gate that fails on a slow CI
runner is a gate people learn to ignore. Judge metrics are compared only when
the judge -- model, prompt version and temperature -- is the one that produced
the baseline; otherwise the report says so instead of comparing incomparable
numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hardpoint.eval.report import (
    LOWER_IS_BETTER,
    CaseRegression,
    EvalReport,
    GateResult,
    Regression,
)

__all__ = ["evaluate_gate", "load_baseline", "write_baseline"]

_JUDGE_METRICS = frozenset({"faithfulness", "answer_relevance"})
_NOT_BASELINED = frozenset({"p50_latency_ms", "p95_latency_ms", "mean_cost_usd", "total_cost_usd"})


def write_baseline(path: str | Path, report: EvalReport) -> Path:
    """Write the baseline: aggregates, each case's metrics, and the judge used."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": report.suite,
        "hardpoint_version": report.hardpoint_version,
        "k": report.k,
        "metrics": report.metrics,
        "judge": report.judge,
        "cases": {case.case_id: case.metrics for case in report.cases},
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_baseline(path: str | Path) -> dict[str, Any] | None:
    """Read a baseline, or ``None`` when there is none yet."""
    source = Path(path)
    if not source.is_file():
        return None
    loaded: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return loaded


def _worse(metric: str, baseline: float, current: float, tolerance: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return current > baseline + tolerance
    return current < baseline - tolerance


def evaluate_gate(
    report: EvalReport,
    *,
    thresholds: Mapping[str, float],
    baseline: Mapping[str, Any] | None,
    tolerance: float,
) -> GateResult:
    """Decide whether the suite passes.

    Args:
        report: The run.
        thresholds: Metric to bound. Maximums for :data:`LOWER_IS_BETTER`
            metrics, minimums for the rest.
        baseline: The committed baseline, or ``None`` to gate on thresholds only.
        tolerance: How far an aggregate may fall below the baseline.

    Returns:
        The verdict and every reason for it.
    """
    failures = _threshold_failures(report, thresholds)
    notes: list[str] = []
    regressions: list[Regression] = []
    case_regressions: list[CaseRegression] = []
    if baseline is None:
        notes.append("No baseline is committed; the gate checked thresholds only.")
    else:
        same_judge = baseline.get("judge") == report.judge
        if not same_judge and (baseline.get("judge") or report.judge):
            notes.append(
                "The judge (model, prompt version or temperature) differs from the baseline's, "
                "so judge metrics were not compared."
            )
        regressions = _aggregate_regressions(report, baseline, tolerance, same_judge=same_judge)
        case_regressions = _case_regressions(report, baseline, tolerance)

    return GateResult(
        passed=not failures and not regressions,
        threshold_failures=failures,
        regressions=regressions,
        case_regressions=case_regressions,
        notes=notes,
    )


def _threshold_failures(report: EvalReport, thresholds: Mapping[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, bound in sorted(thresholds.items()):
        value = report.metrics.get(metric)
        if value is None:
            failures.append(f"{metric}: not measured (threshold {bound})")
        elif metric in LOWER_IS_BETTER and value > bound:
            failures.append(f"{metric}: {value:.4f} is above the maximum {bound}")
        elif metric not in LOWER_IS_BETTER and value < bound:
            failures.append(f"{metric}: {value:.4f} is below the minimum {bound}")
    return failures


def _aggregate_regressions(
    report: EvalReport, baseline: Mapping[str, Any], tolerance: float, *, same_judge: bool
) -> list[Regression]:
    regressions: list[Regression] = []
    for metric, base in sorted((baseline.get("metrics") or {}).items()):
        current = report.metrics.get(metric)
        skip = metric in _NOT_BASELINED or (metric in _JUDGE_METRICS and not same_judge)
        if skip or base is None or current is None:
            continue
        if _worse(metric, float(base), float(current), tolerance):
            regressions.append(Regression(metric=metric, baseline=base, current=current))
    return regressions


def _case_regressions(
    report: EvalReport, baseline: Mapping[str, Any], tolerance: float
) -> list[CaseRegression]:
    found: list[CaseRegression] = []
    baseline_cases: Mapping[str, Mapping[str, float]] = baseline.get("cases") or {}
    for case in report.cases:
        before = baseline_cases.get(case.case_id, {})
        for metric, current in sorted(case.metrics.items()):
            base = before.get(metric)
            if base is not None and _worse(metric, base, current, tolerance):
                found.append(
                    CaseRegression(
                        metric=metric,
                        baseline=base,
                        current=current,
                        case_id=case.case_id,
                        query=case.query,
                    )
                )
    return found
