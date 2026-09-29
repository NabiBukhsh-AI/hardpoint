"""Evaluation results: per case, aggregated, and rendered as JSON and Markdown.

The report is written as JSON for machines -- the baseline is a trimmed copy --
and as Markdown for the pull request, where the per-case regression table is
what makes a failing gate actionable: *which* questions got worse, and by how
much (INSTRUCTIONS.md §8).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.eval.metrics.judge import JudgeScore

__all__ = [
    "LOWER_IS_BETTER",
    "CaseRegression",
    "CaseResult",
    "EvalReport",
    "GateResult",
    "Regression",
    "render_markdown",
]

LOWER_IS_BETTER: Final = frozenset(
    {
        "p50_latency_ms",
        "p95_latency_ms",
        "mean_cost_usd",
        "total_cost_usd",
        "degradation_rate",
        "abstention_rate",
        "error_rate",
    }
)
"""Metrics where a smaller number is the better one. Everything else is higher-better."""

_MUTABLE = ConfigDict(extra="forbid")


class CaseResult(BaseModel):
    """One case's outcome."""

    model_config = _MUTABLE

    case_id: str
    query: str
    run_id: str
    answer: str = ""
    abstained: bool = False
    blocked: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None
    degradations: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    judge: list[JudgeScore] = Field(default_factory=list)


class Regression(BaseModel):
    """An aggregate metric that got worse than the baseline allows."""

    model_config = _MUTABLE

    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        """Current minus baseline."""
        return self.current - self.baseline


class CaseRegression(Regression):
    """One case's metric that got worse than the baseline allows."""

    case_id: str
    query: str = ""


class GateResult(BaseModel):
    """Whether the suite passes, and every reason it does not."""

    model_config = _MUTABLE

    passed: bool
    threshold_failures: list[str] = Field(default_factory=list)
    regressions: list[Regression] = Field(default_factory=list)
    case_regressions: list[CaseRegression] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """A whole suite run."""

    model_config = _MUTABLE

    suite: str
    hardpoint_version: str
    config_hash: str
    started_at: str
    finished_at: str
    k: int
    cases: list[CaseResult]
    metrics: dict[str, float | None]
    judge: dict[str, str | float] | None = None
    gate: GateResult | None = None

    def case(self, case_id: str) -> CaseResult:
        """Look a case up by id.

        Raises:
            KeyError: If the suite has no such case.
        """
        for result in self.cases:
            if result.case_id == case_id:
                return result
        raise KeyError(case_id)


def _format(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if abs(value) < 1000 else f"{value:.0f}"  # noqa: PLR2004


def render_markdown(report: EvalReport, baseline: dict[str, float | None] | None = None) -> str:
    """Render the report for a pull request comment or a CI summary."""
    gate = report.gate
    verdict = "no gate" if gate is None else "PASSED" if gate.passed else "FAILED"
    lines = [
        f"## Eval suite `{report.suite}`: {verdict}",
        "",
        f"{len(report.cases)} cases, k={report.k}, hardpoint {report.hardpoint_version}, "
        f"config `{report.config_hash[:12]}`",
        "",
        "| metric | current | baseline | delta |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sorted(report.metrics.items()):
        base = (baseline or {}).get(name)
        delta = "" if value is None or base is None else f"{value - base:+.4f}"
        lines.append(f"| {name} | {_format(value)} | {_format(base)} | {delta} |")

    if gate is not None and not gate.passed:
        if gate.threshold_failures:
            lines.extend(["", "### Threshold breaches", ""])
            lines.extend(f"- {failure}" for failure in gate.threshold_failures)
        if gate.regressions:
            lines.extend(["", "### Regressions against the baseline", ""])
            lines.extend(
                f"- `{r.metric}`: {_format(r.baseline)} -> {_format(r.current)} ({r.delta:+.4f})"
                for r in gate.regressions
            )
        if gate.case_regressions:
            lines.extend(
                [
                    "",
                    "### Cases that got worse",
                    "",
                    "| case | query | metric | baseline | current | delta |",
                    "|---|---|---|---:|---:|---:|",
                ]
            )
            lines.extend(
                f"| {r.case_id} | {r.query[:60]} | {r.metric} | {_format(r.baseline)} | "
                f"{_format(r.current)} | {r.delta:+.4f} |"
                for r in sorted(gate.case_regressions, key=lambda r: r.delta)
            )
    if gate is not None and gate.notes:
        lines.extend(["", *[f"> {note}" for note in gate.notes]])
    errors = [case for case in report.cases if case.error]
    if errors:
        lines.extend(["", "### Errors", ""])
        lines.extend(f"- {case.case_id}: {case.error}" for case in errors)
    return "\n".join(lines) + "\n"
