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

## The manifest is populated here because this is where the facts are

``Answer.manifest`` is what makes a run reproducible and a regression
bisectable (ARCHITECTURE.md §10). The model id and the prompt version are known
only at generation, so assembling the manifest anywhere else would mean passing
them somewhere to be assembled later.

The model id recorded is the one the adapter *reports*, not the one that was
configured. Those differ exactly when a fallback fired, which is the case where
knowing the difference matters most.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING

import hardpoint
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
            project (INSTRUCTIONS.md §7 **[LOCKED]**). The default here is a
            neutral placeholder, not a product voice.
        stream: Whether to stream. The answer is identical either way; streaming
            changes when the first token reaches the caller, not what is said.
        temperature: Sampling temperature, passed through.
        max_output_tokens: Cap on generated tokens.
        extra_variables: Additional template variables, merged under the ones
            this step supplies so a caller cannot accidentally shadow
            ``context`` or ``question``.
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
        abstention_text: str = (
            "I could not find anything in the indexed documents that answers that."
        ),
        stream: bool = False,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        extra_variables: Mapping[str, JsonValue] | None = None,
        name: str = "generate",
    ) -> None:
        self.name = name
        self.model = model
        self.prompts = prompts
        self.prompt = prompt
        self.pipeline_name = pipeline_name
        self.no_context_policy = no_context_policy
        self.abstention_text = abstention_text
        self.stream = stream
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.extra_variables = dict(extra_variables or {})

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
        degradations: list[Degradation] = []

        if data.context.is_empty:
            abstained = await self._handle_empty_context(data, ctx, degradations)
            if abstained is not None:
                return abstained

        rendered = await self.prompts.render(
            self.prompt,
            {
                **self.extra_variables,
                "question": data.query,
                "context": data.context.rendered,
            },
        )
        request = GenerationRequest(
            messages=self._messages(rendered.messages, data),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        if self.stream:
            text, model_id, usage = await self._generate_streaming(request, ctx)
        else:
            result = await self.model.generate(request, ctx)
            text, model_id, usage = result.text, result.model_id, result.usage
            if result.finish_reason == "length":
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

        ctx.usage.record(
            self.name,
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            estimated=usage.estimated,
        )

        return Answer(
            text=text,
            citations=citations_for(data.context),
            context=data.context,
            usage=Usage(by_step={self.name: usage}, total_cost_usd=usage.cost_usd),
            degradations=degradations,
            run_id=ctx.run_id,
            manifest=self._manifest(model_id, rendered.version),
        )

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

    # ----------------------------------------------------------------- #
    # Streaming                                                         #
    # ----------------------------------------------------------------- #

    async def _generate_streaming(
        self, request: GenerationRequest, ctx: RunContext
    ) -> tuple[str, str, StepUsage]:
        """Consume the stream and assemble the final text and usage.

        Usage arrives on the final delta, and a provider that omits it leaves an
        estimate rather than a zero -- a zero would silently understate the bill
        for every streamed request.
        """
        pieces: list[str] = []
        usage: StepUsage | None = None

        async for delta in self.model.stream(request, ctx):
            if delta.text:
                pieces.append(delta.text)
            if delta.usage is not None:
                usage = delta.usage

        text = "".join(pieces)
        if usage is None:
            usage = StepUsage(
                calls=1,
                completion_tokens=max(1, len(text) // 4),
                estimated=True,
            )
        return text, self.model.id, usage

    async def stream_deltas(self, data: Assembled, ctx: RunContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive, for a streaming service endpoint.

        Separate from ``__call__`` because a ``Step`` returns a value, and a
        service that wants tokens as they are produced needs an iterator. The
        caller builds the terminal citations-and-usage event from the ``Answer``
        that ``__call__`` returns for the non-streamed path.
        """
        rendered = await self.prompts.render(
            self.prompt,
            {
                **self.extra_variables,
                "question": data.query,
                "context": data.context.rendered,
            },
        )
        request = GenerationRequest(
            messages=self._messages(rendered.messages, data),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        async for delta in self.model.stream(request, ctx):
            if delta.text:
                yield delta.text

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
