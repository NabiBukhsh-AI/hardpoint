"""The generation step.

Implements the ``Generate`` requirements in INSTRUCTIONS.md §6.4: streaming,
citations resolved from the ``ContextBundle``, and a populated ``RunManifest``.

## Retrieved content never reaches a system message **[LOCKED]**

INSTRUCTIONS.md §6.4 and §13.12. The system message carries the project's own
instructions and nothing else; the retrieved block goes in a user message,
delimited and labelled as data by ``ContextAssembler``.

This does not prevent prompt injection and the documentation says so plainly
(ARCHITECTURE.md §18.3). What it removes is the structural ambiguity: content a
model has been told is data, arriving in the role reserved for user input, is a
much weaker position for an injected instruction than the same text arriving in
the role reserved for the operator's own directions.

A test asserts it by scanning the rendered request, because this is the kind of
rule a later refactor breaks while making a prompt "tidier".

## One code path, streamed or not

:meth:`Generate.stream_events` yields text as it is produced and then the
complete ``Answer``. Calling the step consumes it; a pipeline run with
``on_delta`` passes the text on as it arrives. A streamed answer and a returned
one are therefore built by the same code and differ only in timing.

## The manifest is populated here because this is where the facts are

``Answer.manifest`` is what makes a run reproducible and a regression
bisectable (ARCHITECTURE.md §10). The model id and the prompt version are known
only at generation. The model id recorded is the one the adapter *reports*,
which differs from the configured one exactly when a fallback fired.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING

import hardpoint
from hardpoint.core.cache_keys import generation_key, params_hash
from hardpoint.core.models import Answer, Degradation, RunManifest, StepUsage, Usage
from hardpoint.core.ports import GenerationRequest, Message
from hardpoint.core.types import JsonValue
from hardpoint.retrieval.assemble import citations_for
from hardpoint.retrieval.retrievers import Assembled

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import LanguageModel, PromptStore

__all__ = ["Generate", "NoContextPolicy"]

NoContextPolicy = str
"""``abstain | answer_without_context | escalate | raise``.

Typed as a string alias rather than a Literal because the policy set is fixed by
configuration validation in ``core.config.schema``, and duplicating the literal
here would create two places to change it.
"""


class Generate:
    """Renders a prompt, calls the model, and builds the ``Answer``.

    Args:
        model: The language model.
        prompts: Where the prompt template lives. The library ships no prompt
            content (INSTRUCTIONS.md §13.11).
        prompt: The prompt's name in the store.
        pipeline_name: Recorded in the run manifest.
        no_context_policy: What to do when nothing was retrieved. Empty
            retrieval is a policy decision, not an error: modelling it as an
            exception forces try/except into every application
            (ARCHITECTURE.md §6.2).
        abstention_text: What an abstention says. Supplied by the caller,
            because the wording is product voice and belongs in the generated
            project (INSTRUCTIONS.md §7 **[LOCKED]**). ``None`` uses a neutral
            placeholder, not a product voice.
        stream: Whether to use the model's streaming interface when called as a
            step. The answer is identical either way.
        temperature: Sampling temperature, passed through.
        max_output_tokens: Cap on generated tokens.
        extra_variables: Additional template variables, merged under the ones
            this step supplies so a caller cannot accidentally shadow
            ``context`` or ``question``.
        cache: Serve repeated prompts from the shared cache. Off by default: a
            cached answer is a product decision, not a speed-up. The key holds
            the prompt version and the model, so editing the prompt or switching
            model never serves a stale answer (ARCHITECTURE.md §22.2).
        cache_ttl_s: How long a cached answer lives.
        name: The step's name.
    """

    def __init__(
        self,
        model: LanguageModel,
        prompts: PromptStore,
        *,
        prompt: str = "answer",
        pipeline_name: str = "rag",
        no_context_policy: NoContextPolicy = "abstain",
        abstention_text: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        extra_variables: Mapping[str, JsonValue] | None = None,
        cache: bool = False,
        cache_ttl_s: int | None = 3600,
        name: str = "generate",
    ) -> None:
        self.name = name
        self.model = model
        self.prompts = prompts
        self.prompt = prompt
        self.pipeline_name = pipeline_name
        self.no_context_policy = no_context_policy
        self.abstention_text = abstention_text or (
            "I could not find anything in the indexed documents that answers that."
        )
        self.stream = stream
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.extra_variables = dict(extra_variables or {})
        self.cache = cache
        self.cache_ttl_s = cache_ttl_s

    async def __call__(self, data: Assembled, ctx: RunContext) -> Answer:
        """Generate an answer for the assembled context.

        Args:
            data: The query and its assembled context.
            ctx: The run context.

        Returns:
            The answer, with citations, usage and a populated manifest.

        Raises:
            RetrievalError: If nothing was retrieved and the configured policy
                is ``raise``.
            ProviderError: Whatever the model adapter raised, mapped into the
                taxonomy.
        """
        final: Answer | None = None
        async for event in self.stream_events(data, ctx, stream=self.stream):
            if isinstance(event, Answer):
                final = event
        if final is None:  # pragma: no cover - stream_events always ends with an Answer
            raise RuntimeError("Generate.stream_events ended without an Answer")
        return final

    async def stream_events(  # noqa: PLR0912, PLR0915 - one linear path, kept whole
        self,
        data: Assembled,
        ctx: RunContext,
        *,
        stream: bool = True,
        feedback: str | None = None,
    ) -> AsyncIterator[str | Answer]:
        """Yield text as it is produced, then the complete ``Answer``, last.

        Args:
            data: The query and its assembled context.
            ctx: The run context.
            stream: Use the model's streaming interface. When false the model is
                called once and the whole text is yielded as one piece.
            feedback: Appended as a final user message: why a previous answer
                was rejected, for one regeneration (``GuardedGenerate``).

        Raises:
            RetrievalError: As for ``__call__``.
            ProviderError: As for ``__call__``.
        """
        degradations: list[Degradation] = []

        if data.context.is_empty:
            abstained = await self._handle_empty_context(data, ctx, degradations)
            if abstained is not None:
                yield abstained.text
                yield abstained
                return

        rendered = await self.prompts.render(
            self.prompt,
            {**self.extra_variables, "question": data.query, "context": data.context.rendered},
        )
        messages = self._messages(rendered.messages, data)
        if feedback:
            messages = (*messages, Message(role="user", content=feedback))
        request = GenerationRequest(
            messages=messages,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        key = self._cache_key(request, rendered.version) if self._caching(ctx) else None
        if key is not None:
            cached = await ctx.cache.get(key)
            outcome = "hit" if cached is not None else "miss"
            ctx.metrics.increment("hardpoint.cache.lookups", layer="generation", result=outcome)
            if cached is not None:
                entry = json.loads(cached)
                text, model_id = str(entry["text"]), str(entry["model_id"])
                yield text
                yield self._answer(
                    data,
                    ctx,
                    text=text,
                    model_id=model_id,
                    usage=StepUsage(),
                    prompt_version=rendered.version,
                    degradations=[],
                )
                return

        finish_reason: str | None = None
        if stream:
            pieces: list[str] = []
            usage: StepUsage | None = None
            deltas = self.model.stream(request, ctx)
            try:
                async for delta in deltas:
                    if delta.text:
                        pieces.append(delta.text)
                        yield delta.text
                    if delta.usage is not None:
                        usage = delta.usage
                    if delta.finish_reason is not None:
                        finish_reason = delta.finish_reason
            finally:
                # Closed here, in this task, so a span the model opened around
                # the stream ends in the context that opened it -- not later,
                # from whatever task finalises an abandoned generator.
                closer = getattr(deltas, "aclose", None)
                if closer is not None:
                    await closer()
            text, model_id = "".join(pieces), self.model.id
            if usage is None:
                # A provider that omits usage on the stream leaves an estimate,
                # never a zero, which would understate every streamed request.
                usage = StepUsage(
                    calls=1,
                    prompt_tokens=await self.model.count_tokens(request.messages),
                    completion_tokens=max(1, len(text) // 4),
                    estimated=True,
                )
        else:
            result = await self.model.generate(request, ctx)
            text, model_id, usage = result.text, result.model_id, result.usage
            finish_reason = result.finish_reason
            yield text

        if finish_reason == "length":
            degradations.append(
                Degradation(
                    step=self.name,
                    reason="output_truncated",
                    detail=(
                        "The model stopped at its output limit, so the answer is "
                        "incomplete. Raise max_output_tokens."
                    ),
                )
            )
        elif key is not None:
            entry_bytes = json.dumps({"text": text, "model_id": model_id}).encode("utf-8")
            await ctx.cache.set(key, entry_bytes, ttl_s=self.cache_ttl_s)

        ctx.usage.record(
            self.name,
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            estimated=usage.estimated,
        )
        yield self._answer(
            data,
            ctx,
            text=text,
            model_id=model_id,
            usage=usage,
            prompt_version=rendered.version,
            degradations=degradations,
        )

    async def regenerate(self, data: Assembled, ctx: RunContext, *, feedback: str) -> Answer:
        """Generate again, telling the model why its previous answer was rejected.

        Used once per request by ``GuardedGenerate`` for a guard whose action is
        ``retry`` (ARCHITECTURE.md §18.2). Not streamed: the first attempt has
        already been withheld, so there is nothing to stream into.
        """
        final: Answer | None = None
        async for event in self.stream_events(data, ctx, stream=False, feedback=feedback):
            if isinstance(event, Answer):
                final = event
        if final is None:  # pragma: no cover - stream_events always ends with an Answer
            raise RuntimeError("Generate.stream_events ended without an Answer")
        return final

    def _answer(
        self,
        data: Assembled,
        ctx: RunContext,
        *,
        text: str,
        model_id: str,
        usage: StepUsage,
        prompt_version: str,
        degradations: list[Degradation],
    ) -> Answer:
        return Answer(
            text=text,
            citations=citations_for(data.context),
            context=data.context,
            usage=Usage(by_step={self.name: usage}, total_cost_usd=usage.cost_usd),
            degradations=degradations,
            run_id=ctx.run_id,
            manifest=self._manifest(model_id, prompt_version),
        )

    # ----------------------------------------------------------------- #
    # Caching                                                           #
    # ----------------------------------------------------------------- #

    def _caching(self, ctx: RunContext) -> bool:
        return self.cache and ctx.cache.enabled("shared")

    def _cache_key(self, request: GenerationRequest, prompt_version: str) -> str:
        """``gen:{model_id}:{params_hash}:{prompt_version}:{sha256(rendered_prompt)}``."""
        rendered = "\n".join(f"{m.role}:{m.content}" for m in request.messages)
        parameters: dict[str, JsonValue] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        return generation_key(self.model.id, params_hash(parameters), prompt_version, rendered)

    # ----------------------------------------------------------------- #
    # Message construction                                              #
    # ----------------------------------------------------------------- #

    def _messages(self, rendered: tuple[Message, ...], data: Assembled) -> tuple[Message, ...]:
        """Place the rendered prompt's messages, keeping context out of ``system``.

        The prompt template may itself interpolate ``{{ context }}`` into a user
        message. When it has not, the context block is appended as its own user
        message, so a prompt that forgot it still gets it -- in the right role.
        """
        messages = list(rendered)
        block = data.context.rendered

        if any(block in message.content for message in messages):
            return tuple(messages)

        messages.append(Message(role="user", content=block))
        return tuple(messages)

    # ----------------------------------------------------------------- #
    # Empty retrieval                                                   #
    # ----------------------------------------------------------------- #

    async def _handle_empty_context(
        self, data: Assembled, ctx: RunContext, degradations: list[Degradation]
    ) -> Answer | None:
        """Apply the no-context policy. Returns an answer, or ``None`` to proceed.

        Raises:
            RetrievalError: When the policy is ``raise``.
        """
        if self.no_context_policy == "answer_without_context":
            degradations.append(
                Degradation(
                    step=self.name,
                    reason="no_context",
                    detail="Nothing was retrieved; the model answered unaided.",
                )
            )
            return None

        if self.no_context_policy == "raise":
            from hardpoint.core.errors import RetrievalError  # noqa: PLC0415 - error path only

            raise RetrievalError(
                f"Nothing was retrieved for {data.query[:80]!r}.",
                step=self.name,
                run_id=ctx.run_id,
                remedy=(
                    "Lower `retrieval.score_threshold`, raise `retrieval.top_k`, or "
                    "check the index has been ingested. Set "
                    "`retrieval.no_context_policy` to `abstain` to return a cited "
                    "refusal instead of raising."
                ),
            )

        # abstain, and escalate, which differ only in what the caller does next.
        reason = "abstained" if self.no_context_policy == "abstain" else "escalated"
        return Answer(
            text=self.abstention_text,
            abstained=True,
            citations=[],
            context=data.context,
            usage=Usage(),
            degradations=[
                Degradation(
                    step=self.name,
                    reason=reason,
                    detail="Nothing was retrieved, so no answer was attempted.",
                    severity="info",
                )
            ],
            run_id=ctx.run_id,
            manifest=self._manifest(self.model.id, ""),
        )

    def _manifest(self, model_id: str, prompt_version: str) -> RunManifest:
        """Build the run manifest.

        Args:
            model_id: What the adapter reported, which differs from what was
                configured exactly when a fallback fired.
            prompt_version: The rendered prompt's version, which is a content
                hash and therefore moves whenever the prompt text does.
        """
        return RunManifest(
            hardpoint_version=hardpoint.__version__,
            config_hash="",
            pipeline_name=self.pipeline_name,
            model_ids={"llm": model_id},
            prompt_versions={self.prompt: prompt_version} if prompt_version else {},
        )

    def __repr__(self) -> str:
        """Render the model and prompt."""
        return f"Generate(model={self.model.id!r}, prompt={self.prompt!r})"
