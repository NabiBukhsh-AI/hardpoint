"""Datasets, the eval runner, metrics, baselines and reports.

Owns evaluation plumbing. Does not own the user's golden data, and never builds a
second execution path: the runner drives the production pipeline through the
project's own factory (ADR-011).
"""

from __future__ import annotations

from hardpoint.eval.baseline import evaluate_gate, load_baseline, write_baseline
from hardpoint.eval.dataset import EvalCase, load_dataset, resolve_document_ids, write_dataset
from hardpoint.eval.metrics.judge import Judge, JudgeScore
from hardpoint.eval.report import (
    LOWER_IS_BETTER,
    CaseRegression,
    CaseResult,
    EvalReport,
    GateResult,
    Regression,
    render_markdown,
)
from hardpoint.eval.runner import EvalRunner, estimate_cost

__all__ = [
    "LOWER_IS_BETTER",
    "CaseRegression",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "EvalRunner",
    "GateResult",
    "Judge",
    "JudgeScore",
    "Regression",
    "estimate_cost",
    "evaluate_gate",
    "load_baseline",
    "load_dataset",
    "render_markdown",
    "resolve_document_ids",
    "write_baseline",
    "write_dataset",
]
