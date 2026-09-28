"""An OpenAI-compatible chat adapter.

Implements the M1 language model adapter from INSTRUCTIONS.md §6.3.

## Why this speaks HTTP rather than using the OpenAI SDK

"OpenAI-compatible" is a wire protocol, not a library. Ollama, vLLM, LM Studio,
Together, Groq, Azure OpenAI and OpenAI itself all serve
``POST /chat/completions`` with the same shapes, so an adapter written against
the protocol works with all of them and a new one costs a ``base_url``.

``httpx`` is already a base dependency, so this adapter needs **no extra**:
``pip install hardpoint`` is immediately useful. And ARCHITECTURE.md §16.4 asks
that an adapter be rewritable against a new major SDK version without a
hardpoint breaking change -- not depending on an SDK at all satisfies that
completely.

## Transport only **[LOCKED]**

No retries, no backoff, no caching, no tracing decisions, no business logic
(INSTRUCTIONS.md §6.3). Every provider failure is mapped into the taxonomy by
``adapters._http``; whether it is worth retrying is decided by
``runtime/policies.py``.

## Capabilities are declared by the caller, honestly

An OpenAI-compatible endpoint could be anything from GPT-4 to a 1B local model,
and this adapter cannot know which. So capabilities are constructor arguments
with conservative defaults. Over-declaring is a bug the contract kit exists to
catch; under-declaring merely costs a capability that would have worked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hardpoint.adapters._http import (
    DEFAULT_TIMEOUT_S,
    as_int,
    as_json,
    build_client,
    map_response_error,
    map_transport_error,
)
from hardpoint.core.capabilities import ModelCapabilities
from hardpoint.core.errors import ContractError
from hardpoint.core.models import StepUsage
from hardpoint.core.ports import (
    GenerationDelta,
    GenerationRequest,
    GenerationResult,
    Message,
    ToolCall,
)
from hardpoint.core.tokens import estimate_tokens
from hardpoint.core.types import JsonValue, ModelId

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["OpenAIChatConfig", "OpenAICompatibleChat", "build"]

_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


class OpenAICompatibleChat:
    """A ``LanguageModel`` speaking the OpenAI chat-completions protocol.

    Args:
        model: The model name the endpoint expects.
        base_url: The API root. Defaults to OpenAI's; point it at any compatible
            server.
        api_key: Bearer token. Omitted entirely when ``None``, so a local
            endpoint is not sent an empty credential.
        context_window_tokens: Declared context window.
        max_output_tokens: Declared output cap.
        supports_tools: Whether to declare tool calling.
        supports_structured_output: Whether to declare schema-constrained output.
        supports_vision: Whether to declare image input.
        timeout_s: Transport timeout. The run deadline is the real bound; this
            only stops a hung socket outliving the process.
        client: A pre-built client, for tests and for callers that manage their
            own connection pool. When given, this adapter does not close it.
        model_id: What to report as the model id. Defaults to
            ``openai/<model>``, which is what the run manifest and the pricing
            table key on.
        extra_headers: Additional headers, for gateways that need them.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        context_window_tokens: int = 8192,
        max_output_tokens: int | None = None,
        supports_tools: bool = False,
        supports_structured_output: bool = False,
        supports_vision: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        model_id: ModelId | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.id = model_id or f"openai/{model}"
        self._owns_client = client is None
        self._client = client or build_client(
            base_url=base_url, api_key=api_key, headers=extra_headers, timeout_s=timeout_s
        )
        self._capabilities = ModelCapabilities(
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            supports_tools=supports_tools,
            supports_structured_output=supports_structured_output,
            supports_streaming=True,
            supports_vision=supports_vision,
        )

    def capabilities(self) -> ModelCapabilities:
        """Declare what this endpoint supports, as configured."""
        return self._capabilities

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller supplied it."""
        if self._owns_client:
            await self._client.aclose()

    # ----------------------------------------------------------------- #
    # Generation                                                        #
    # ----------------------------------------------------------------- #

    async def generate(self, req: GenerationRequest, ctx: RunContext) -> GenerationResult:
        """Call ``POST /chat/completions``.

        Raises:
            AuthError, RateLimitedError, TransientError, InvalidRequestError,
            ProviderTimeout: Mapped from the response status or the transport
                failure. Never a raw ``httpx`` exception.
            ContractError: If the response is well-formed JSON but not a
                chat-completion.
        """
        payload = self._payload(req, stream=False)
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except Exception as exc:
            raise map_transport_error(exc, component=self.id, operation="generate") from exc

        if response.is_error:
            raise map_response_error(response, component=self.id, operation="generate")

        body = as_json(response, component=self.id, operation="generate")
        return self._result_from(body, req)

    def stream(self, req: GenerationRequest, ctx: RunContext) -> AsyncIterator[GenerationDelta]:
        """Stream ``POST /chat/completions`` with ``stream: true``.

        Server-sent events, parsed line by line. Usage arrives on the final
        chunk when the endpoint sends ``stream_options.include_usage``; when it
        does not, the caller receives no usage delta and ``Generate`` estimates
        rather than reporting a zero.
        """
        payload = self._payload(req, stream=True)

        async def iterator() -> AsyncIterator[GenerationDelta]:
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    if response.is_error:
                        await response.aread()
                        raise map_response_error(response, component=self.id, operation="stream")
                    async for line in response.aiter_lines():
                        delta = self._delta_from_line(line)
                        if delta is not None:
                            yield delta
            except httpx.HTTPError as exc:
                raise map_transport_error(exc, component=self.id, operation="stream") from exc

        return iterator()

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate tokens using the documented heuristic.

        The chat-completions protocol exposes no tokenizer, and pulling in
        ``tiktoken`` would add a base dependency for an estimate that is wrong
        for every non-OpenAI endpoint this adapter also serves. So this is the
        ``core.tokens`` heuristic, which under-counts for code and non-Latin
        scripts and never returns zero.

        A caller who needs exact counts passes their own counter to
        ``ContextAssembler``.
        """
        return max(1, sum(estimate_tokens(message.content) for message in messages))

    # ----------------------------------------------------------------- #
    # Wire format                                                       #
    # ----------------------------------------------------------------- #

    def _payload(self, req: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        """Build the request body.

        ``provider_options`` is merged last so a caller can reach a knob this
        adapter does not model. Keys that collide with ones built here are the
        caller's decision to override, which is the point of an opaque escape
        hatch.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_payload(m) for m in req.messages],
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if req.max_output_tokens is not None:
            payload["max_tokens"] = req.max_output_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop:
            payload["stop"] = list(req.stop)
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": req.response_schema},
            }
        if req.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in req.tools
            ]
            payload["tool_choice"] = req.tool_choice

        payload.update(req.provider_options)
        return payload

    @staticmethod
    def _message_payload(message: Message) -> dict[str, Any]:
        """Render one message in the wire format."""
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return payload

    def _result_from(
        self, body: Mapping[str, JsonValue], req: GenerationRequest
    ) -> GenerationResult:
        """Build a ``GenerationResult`` from a completion body."""
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ContractError(
                f"{self.id}: the response contained no choices.",
                component=self.id,
                remedy=(
                    "The endpoint returned a well-formed body with nothing in it. "
                    "Check the model name is one this endpoint serves."
                ),
            )

        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = message.get("content") or "" if isinstance(message, dict) else ""
        raw_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

        return GenerationResult(
            text=str(text),
            tool_calls=self._tool_calls_from(message if isinstance(message, dict) else {}),
            finish_reason=_FINISH_REASONS.get(str(raw_reason), "stop"),  # type: ignore[arg-type]
            model_id=str(body.get("model") or self.model),
            usage=self._usage_from(body, req, str(text)),
            raw_response_id=str(body.get("id")) if body.get("id") else None,
        )

    @staticmethod
    def _tool_calls_from(message: Mapping[str, JsonValue]) -> tuple[ToolCall, ...]:
        """Extract tool calls, keeping arguments as the text the model produced."""
        raw = message.get("tool_calls")
        if not isinstance(raw, list):
            return ()
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or ""),
                )
            )
        return tuple(calls)

    def _usage_from(
        self, body: Mapping[str, JsonValue], req: GenerationRequest, text: str
    ) -> StepUsage:
        """Build usage, estimating when the provider omitted it.

        **Never a zero** (ARCHITECTURE.md §9.2). An endpoint that reports no
        token counts -- which several OpenAI-compatible servers do not -- gets an
        estimate flagged ``estimated=True``, so a cost figure downstream can be
        read with the right confidence instead of silently understating a bill.
        """
        usage = body.get("usage")
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            return StepUsage(
                calls=1,
                prompt_tokens=as_int(usage.get("prompt_tokens")),
                completion_tokens=as_int(usage.get("completion_tokens")),
                estimated=False,
            )

        return StepUsage(
            calls=1,
            prompt_tokens=sum(estimate_tokens(m.content) for m in req.messages),
            completion_tokens=max(1, estimate_tokens(text)),
            estimated=True,
        )

    def _delta_from_line(self, line: str) -> GenerationDelta | None:  # noqa: PLR0911
        """Parse one SSE line into a delta, or ``None`` for keep-alives and ``[DONE]``."""
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            return None

        try:
            chunk = json.loads(payload)
        except ValueError:
            # A malformed chunk mid-stream is not worth failing the whole
            # response over: the tokens already delivered are still correct.
            return None
        if not isinstance(chunk, dict):
            return None

        usage = chunk.get("usage")
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
                return GenerationDelta(
                    usage=StepUsage(
                        calls=1,
                        prompt_tokens=as_int(usage.get("prompt_tokens")),
                        completion_tokens=as_int(usage.get("completion_tokens")),
                        estimated=False,
                    )
                )
            return None

        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        delta = choice.get("delta")
        text = delta.get("content") if isinstance(delta, dict) else None
        raw_reason = choice.get("finish_reason")

        if text is None and raw_reason is None:
            return None

        return GenerationDelta(
            text=str(text) if text else "",
            finish_reason=_FINISH_REASONS.get(str(raw_reason)) if raw_reason else None,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Render the model id."""
        return f"OpenAICompatibleChat(model={self.model!r}, id={self.id!r})"


class OpenAIChatConfig(BaseModel):
    """Configuration for ``type: openai_chat``. Field meanings match the constructor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    context_window_tokens: int = Field(default=8192, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    model_id: str | None = None


def build(config: OpenAIChatConfig) -> OpenAICompatibleChat:
    """Registry factory for ``type: openai_chat``."""
    return OpenAICompatibleChat(**config.model_dump())
