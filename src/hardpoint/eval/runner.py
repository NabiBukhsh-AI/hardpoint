"""The eval runner: the production pipeline, run over a dataset.

**No second execution path** (INSTRUCTIONS.md §8 and §13.10 **[LOCKED]**). The
runner builds the pipeline with :func:`load_pipeline` -- the project's own
factory, the one the service and ``hardpoint ask`` call -- and answers every case
with :func:`answer_query`, the function the service's routes call. If evaluation
ran anything else, it would be measuring a different system from the one that
ships (ADR-011).

What it adds is only measurement: retrieval metrics from each answer's context
bundle, optional judge scores, and operational statistics across the suite.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio

import hardpoint
from hardpoint.core.errors import HardpointError
from hardpoint.core.models import Answer
from hardpoint.core.tokens import estimate_tokens
from hardpoint.eval.dataset import EvalCase, resolve_document_ids
from hardpoint.eval.metrics.judge import Judge, JudgeScore
from hardpoint.eval.metrics.retrieval import (
    context_precision,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
)
from hardpoint.eval.report import CaseResult, EvalReport
from hardpoint.runtime.resources import answer_query, load_pipeline

if TYPE_CHECKING:
    from hardpoint.runtime.pipeline import Pipeline
    from hardpoint.runtime.resources import Resources

__all__ = ["EvalRunner", "estimate_cost"]

_ESTIMATED_COMPLETION_TOKENS = 400
_ESTIMATED_TEMPLATE_TOKENS = 200


class EvalRunner:
    """Runs a dataset through the project's pipeline and measures the results.

    Args:
        res: The configured resources -- the same object the service builds.
        judge: Scores faithfulness and relevance, when given.
        k: The ``@k`` cut-off. Defaults to ``eval.k``.
        concurrency: Cases at once. Defaults to ``eval.concurrency``.
    """

    def __init__(
        self,
        res: Resources,
        *,
        judge: Judge | None = None,
        k: int | None = None,
        concurrency: int | None = None,
    ) -> None:
        self.res = res
        self.judge = judge
        self.k = k or res.config.eval.k
        self.concurrency = concurrency or res.config.eval.concurrency
        # The project's factory, never a pipeline of the runner's own making.
        self.pipeline: Pipeline[Any, Any] = load_pipeline(res)

    async def run(self, suite: str, cases: Sequence[EvalCase]) -> EvalReport:
        """Run every case and aggregate."""
        started = datetime.now(UTC).isoformat()
        results: list[CaseResult | None] = [None] * len(cases)
        limiter = anyio.Semaphore(self.concurrency)

        async def one(position: int, case: EvalCase) -> None:
            async with limiter:
                results[position] = await self._case(suite, case)

        async with anyio.create_task_group() as group:
            for position, case in enumerate(cases):
                group.start_soon(one, position, case)

        finished = [result for result in results if result is not None]
        scores = [score for result in finished for score in result.judge]
        return EvalReport(
            suite=suite,
            hardpoint_version=hardpoint.__version__,
            config_hash=self.res.snapshot.hash,
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            k=self.k,
            cases=finished,
            metrics=self._aggregate(finished, [c.has_retrieval_labels for c in cases]),
            judge=_judge_metadata(scores),
        )

    async def _case(self, suite: str, case: EvalCase) -> CaseResult:
        run_id = f"eval-{suite}-{case.id}"
        started = time.perf_counter()
        try:
            answer = await answer_query(self.pipeline, case.query, self.res, run_id=run_id)
        except HardpointError as exc:
            return CaseResult(
                case_id=case.id,
                query=case.query,
                run_id=run_id,
                error=f"{exc.code}: {exc.message}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                metrics=self._retrieval_metrics(case, None),
            )
        latency = (time.perf_counter() - started) * 1000.0

        judged: list[JudgeScore] = []
        if self.judge is not None and not (answer.abstained or answer.blocked):
            ctx = self.res.run_context(run_id=f"{run_id}-judge")
            judged = [
                await self.judge.faithfulness(
                    case.query, answer, ctx, reference=case.reference_answer
                ),
                await self.judge.answer_relevance(
                    case.query, answer, ctx, reference=case.reference_answer
                ),
            ]

        return CaseResult(
            case_id=case.id,
            query=case.query,
            run_id=run_id,
            answer=answer.text,
            abstained=answer.abstained,
            blocked=answer.blocked,
            latency_ms=latency,
            cost_usd=answer.usage.total_cost_usd,
            degradations=[d.reason for d in answer.degradations if d.severity == "warn"],
            retrieved=[chunk_id for chunk_id, _ in _ranking(answer)],
            metrics=self._retrieval_metrics(case, answer),
            judge=judged,
        )

    def _retrieval_metrics(self, case: EvalCase, answer: Answer | None) -> dict[str, float]:
        """The deterministic metrics for one case. An errored case scores zero."""
        if not case.has_retrieval_labels:
            return {}
        k = self.k
        expected_docs = resolve_document_ids(case)
        expected_chunks = frozenset(case.expected_chunk_ids)
        ranking = _ranking(answer) if answer is not None else []

        def relevant(chunk_id: str, document_id: str | None) -> bool:
            return chunk_id in expected_chunks or document_id in expected_docs

        flags = [relevant(chunk, doc) for chunk, doc in ranking]
        found_docs = {doc for _, doc in ranking[:k] if doc in expected_docs}
        found_chunks = {chunk for chunk, _ in ranking[:k] if chunk in expected_chunks}
        expected_total = len(expected_docs) + len(expected_chunks)
        anywhere = {doc for _, doc in ranking} | {chunk for chunk, _ in ranking}
        missing = len((expected_docs | expected_chunks) - anywhere)
        included = (
            [relevant(item.chunk.id, item.chunk.document_id) for item in answer.context.items]
            if answer is not None and answer.context is not None
            else []
        )
        return {
            f"hit_rate_at_{k}": hit_rate_at_k(flags, k),
            f"precision_at_{k}": precision_at_k(flags, k),
            f"recall_at_{k}": (len(found_docs) + len(found_chunks)) / expected_total,
            "mrr": mrr(flags),
            f"ndcg_at_{k}": ndcg_at_k(flags, k, missing=missing),
            "context_precision": context_precision(included),
        }

    def _aggregate(
        self, results: Sequence[CaseResult], labelled: Sequence[bool]
    ) -> dict[str, float | None]:
        """Suite-level metrics: retrieval means, judge means, operational statistics."""
        metrics: dict[str, float | None] = {}
        scored = [result for result, has in zip(results, labelled, strict=True) if has]
        names = sorted({name for result in scored for name in result.metrics})
        for name in names:
            metrics[name] = statistics.fmean(result.metrics.get(name, 0.0) for result in scored)

        for judged_metric in ("faithfulness", "answer_relevance"):
            values = [s.score for r in results for s in r.judge if s.metric == judged_metric]
            if values:
                metrics[judged_metric] = statistics.fmean(values)

        count = len(results)
        latencies = sorted(result.latency_ms for result in results)
        costs = [result.cost_usd for result in results if result.error is None]
        total_cost = None if any(cost is None for cost in costs) else sum(c or 0.0 for c in costs)
        metrics.update(
            {
                "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_latency_ms": _percentile(latencies, 95),
                "total_cost_usd": total_cost,
                "mean_cost_usd": None if total_cost is None or not count else total_cost / count,
                "degradation_rate": _rate(results, lambda r: bool(r.degradations)),
                "abstention_rate": _rate(results, lambda r: r.abstained),
                "error_rate": _rate(results, lambda r: r.error is not None),
            }
        )
        return metrics


def _ranking(answer: Answer) -> list[tuple[str, str | None]]:
    """Everything retrieved, included or dropped, in descending score order."""
    if answer.context is None:
        return []
    items: list[tuple[float, str, str | None]] = [
        (item.score or 0.0, item.chunk.id, item.chunk.document_id) for item in answer.context.items
    ]
    items.extend(
        (record.score or 0.0, record.chunk_id, record.document_id)
        for record in answer.context.dropped
    )
    items.sort(key=lambda entry: -entry[0])
    return [(chunk_id, document_id) for _, chunk_id, document_id in items]


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


def _rate(results: Sequence[CaseResult], predicate: Any) -> float:
    return sum(1 for result in results if predicate(result)) / len(results) if results else 0.0


def _judge_metadata(scores: Sequence[JudgeScore]) -> dict[str, str | float] | None:
    if not scores:
        return None
    metadata: dict[str, str | float] = {
        "judge_model": scores[0].judge_model,
        "temperature": scores[0].temperature,
    }
    for score in scores:
        metadata[f"{score.metric}_prompt_version"] = score.prompt_version
    return metadata


def estimate_cost(
    res: Resources, cases: Sequence[EvalCase], *, judge: bool = False
) -> float | None:
    """An upper-bound estimate of what running the suite would cost.

    What ``hardpoint eval run --max-cost`` checks before spending anything. Each
    case is assumed to fill the context budget and produce a few hundred tokens;
    judging adds two calls per case over the same context.

    Returns:
        Dollars, or ``None`` when a model involved has no price -- in which case
        the caller should refuse a cost cap rather than guess.
    """
    config = res.config
    llm_price = res.pricing.price(res.llm.id) if not res.llm.id.startswith("fake/") else None
    embed_id = res.embedder.id
    embed_price = res.pricing.embed_price_per_million(embed_id)
    if (llm_price is None and not res.llm.id.startswith("fake/")) or embed_price is None:
        return None

    prompt_tokens = config.retrieval.context.token_budget + _ESTIMATED_TEMPLATE_TOKENS
    calls_per_case = 3 if judge else 1
    total = 0.0
    for case in cases:
        query_tokens = estimate_tokens(case.query)
        total += query_tokens * embed_price / 1_000_000
        if llm_price is not None:
            total += (
                calls_per_case
                * (
                    (prompt_tokens + query_tokens) * (llm_price.input or 0.0)
                    + _ESTIMATED_COMPLETION_TOKENS * (llm_price.output or 0.0)
                )
                / 1_000_000
            )
    return total
