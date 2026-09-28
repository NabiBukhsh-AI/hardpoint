"""``hardpoint ask``: one question through the project's pipeline.

Goes through :func:`load_pipeline` and :func:`answer_query`, the same path the
service and the eval runner take, so what ``ask`` shows is what production does.

``--explain`` prints the execution report ARCHITECTURE.md §21 describes: steps
and their timings, what was retrieved with scores, what was dropped from
context and why, the rendered prompt (redacted per configuration) and usage
with cost. It is the feature that turns "why did it answer that" from an
investigation into a read.
"""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from hardpoint.core.capabilities import ModelCapabilities
from hardpoint.core.context import RunContext
from hardpoint.core.models import Answer
from hardpoint.core.ports import (
    GenerationDelta,
    GenerationRequest,
    GenerationResult,
    LanguageModel,
    Message,
)
from hardpoint.runtime.resources import Resources, answer_query, load_pipeline

__all__ = ["CapturingModel", "ask", "render_explain"]

_REDACTED = "[redacted]"


class CapturingModel:
    """A ``LanguageModel`` that remembers what it was sent, for ``--explain``.

    Substituted into the resources for one command, visibly, rather than
    reached for through a hook.
    """

    def __init__(self, inner: LanguageModel) -> None:
        self.inner = inner
        self.id = inner.id
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Record the request, then delegate."""
        self.requests.append(req)
        return await self.inner.generate(req, ctx)

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Record the request, then delegate."""
        self.requests.append(req)
        return self.inner.stream(req, ctx)

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Delegate."""
        return await self.inner.count_tokens(messages)

    def capabilities(self) -> ModelCapabilities:
        """Delegate."""
        return self.inner.capabilities()

    async def aclose(self) -> None:
        """Close the inner model."""
        closer = getattr(self.inner, "aclose", None)
        if closer is not None:
            await closer()


async def ask(res: Resources, question: str) -> tuple[Answer, CapturingModel]:
    """Answer one question, capturing the prompt that was sent."""
    capturing = CapturingModel(res.llm)
    res = res.with_components(llm_=capturing)
    answer = await answer_query(load_pipeline(res), question, res)
    return answer, capturing


def _cost(value: float | None) -> str:
    return "unpriced" if value is None else f"${value:.6f}"


def render_explain(answer: Answer, captured: CapturingModel, *, redact_content: bool) -> str:
    """Render the execution report for one answer."""
    manifest = answer.manifest
    lines = [
        f"run       {answer.run_id}",
        f"pipeline  {manifest.pipeline_name}   config {manifest.config_hash[:12]}   "
        f"hardpoint {manifest.hardpoint_version}",
        f"models    {', '.join(f'{role}={model}' for role, model in manifest.model_ids.items())}",
        f"epochs    {', '.join(f'{k}={v}' for k, v in manifest.index_epochs.items()) or '-'}",
        "",
        "steps",
    ]
    for step, usage in answer.usage.by_step.items():
        details = [f"{usage.latency_ms:8.1f} ms"]
        if usage.calls:
            details.append(f"{usage.calls} call{'s' if usage.calls != 1 else ''}")
        if usage.prompt_tokens or usage.completion_tokens:
            details.append(f"{usage.prompt_tokens} in / {usage.completion_tokens} out")
        if usage.embed_tokens:
            details.append(f"{usage.embed_tokens} embed tokens")
        if usage.calls:
            details.append(_cost(usage.cost_usd) + (" (estimated)" if usage.estimated else ""))
        lines.append(f"  {step:<20} {'  '.join(details)}")

    context = answer.context
    items = context.items if context else []
    lines.extend(["", f"context ({len(items)} included)"])
    for item in items:
        score = f"{item.score:.4f}" if item.score is not None else "   -  "
        source = item.chunk.metadata.get("path") or item.chunk.document_id
        lines.append(f"  [{item.citation_key}] {score}  {item.chunk.id}  {source}")

    dropped = context.dropped if context else []
    lines.extend(["", f"dropped ({len(dropped)})"])
    for record in dropped:
        score = f"{record.score:.4f}" if record.score is not None else "   -  "
        lines.append(f"  {score}  {record.chunk_id}  {record.reason}: {record.detail or ''}")

    versions = ", ".join(f"{name}@{version}" for name, version in manifest.prompt_versions.items())
    lines.extend(["", f"prompt ({versions or 'not rendered'})"])
    if captured.requests:
        for message in captured.requests[-1].messages:
            lines.append(f"  --- {message.role} ---")
            body = _REDACTED if redact_content else message.content
            lines.extend(f"  {line}" for line in body.splitlines())
    else:
        lines.append("  (the model was not called)")

    totals = answer.usage
    lines.extend(
        [
            "",
            "usage",
            f"  {totals.total_calls} provider calls, {totals.total_tokens} tokens, "
            f"{_cost(totals.total_cost_usd)}, {totals.total_latency_ms:.1f} ms",
            "",
            "degradations",
        ]
    )
    lines.extend(f"  {d.severity}  {d.step}: {d.reason}  {d.detail}" for d in answer.degradations)
    if not answer.degradations:
        lines.append("  none")
    return "\n".join(lines)


def _readable(source: str) -> str:
    """Show a local file as a path relative to the project, not a ``file://`` URI."""
    if not source.startswith("file:"):
        return source
    path = Path(url2pathname(urlparse(source).path))
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def render_answer(answer: Answer) -> str:
    """Render the answer and its sources for the terminal."""
    lines = [answer.text, ""]
    for citation in answer.citations:
        source = _readable(citation.source_uri) if citation.source_uri else citation.document_id
        lines.append(f"  [{citation.citation_key}] {source}")
    if answer.abstained:
        lines.append("  (abstained: nothing relevant was retrieved)")
    return "\n".join(lines).rstrip()
