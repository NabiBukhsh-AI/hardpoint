"""The ``LanguageModel`` conformance suite. INSTRUCTIONS.md §6.6.

An adapter is not "supported" until it passes this. Bind it in your own tests::

    from hardpoint.testing.contracts import language_model_contract

    TestMyChat = language_model_contract(lambda: MyChatAdapter(...))

The name must begin with ``Test``; see the ``vector_index`` kit's docstring for
why.

## What this checks, and why each one

The invariants here are the ones whose violation is *silent*:

- **``generate`` always returns usage, never zero tokens.** A zero is the shape
  of a cost report that understates a bill, and nothing errors.
- **The model id reported is the model that served the request.** It differs
  from the configured one exactly when a fallback fired, which is when knowing
  matters.
- **``count_tokens`` never returns zero for real text.** A zero lets a context
  assembler believe anything fits.
- **``capabilities()`` is answerable and self-consistent.** A model declaring an
  output cap larger than its context window has mis-declared something.
- **Streaming and non-streaming agree on the answer.** Streaming changes when
  the first token arrives, not what is said.
- **Errors are the taxonomy's, never the SDK's.** Checked only when the caller
  supplies a factory that can fail, since most cannot be made to on demand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hardpoint.core.errors import AuthError, MissingDependencyError
from hardpoint.core.ports import GenerationRequest, LanguageModel, Message

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["language_model_contract"]


def _ask(text: str = "Say the word hello and nothing else.") -> GenerationRequest:
    return GenerationRequest(
        messages=(
            Message(role="system", content="You are a terse assistant."),
            Message(role="user", content=text),
        ),
        max_output_tokens=64,
        temperature=0.0,
    )


def language_model_contract(
    factory: Callable[[], LanguageModel],
    *,
    unauthorised_factory: Callable[[], LanguageModel] | None = None,
    supports_streaming: bool = True,
) -> type:
    """Build a pytest class asserting a ``LanguageModel`` implementation conforms.

    Args:
        factory: Returns a ready model.
        unauthorised_factory: Returns one whose credentials will be rejected.
            When given, error-mapping tests are added; when not, they are *not
            generated*, rather than generated and skipped.
        supports_streaming: Whether to run the streaming tests. An adapter for a
            provider without streaming declares ``supports_streaming=False`` in
            its capabilities and passes this as ``False``.

    Returns:
        A test class. Bind it to a name beginning with ``Test``.
    """
    try:
        import pytest  # noqa: PLC0415 - see the vector_index kit's docstring
    except ModuleNotFoundError as exc:  # pragma: no cover - pytest is a dev dependency
        raise MissingDependencyError(
            "The contract kits build pytest test classes, and pytest is not installed.",
            component="language_model_contract",
            remedy="pip install pytest",
            cause=exc,
        ) from exc

    from hardpoint.testing.fixtures import build_run_context  # noqa: PLC0415

    class LanguageModelContract:
        """Behaviour every ``LanguageModel`` implementation must exhibit."""

        @pytest.fixture
        def ctx(self) -> RunContext:
            return build_run_context()

        @pytest.fixture
        def model(self) -> Any:
            return factory()

        def test_satisfies_the_protocol(self, model: LanguageModel) -> None:
            assert isinstance(model, LanguageModel)

        def test_reports_a_model_id(self, model: LanguageModel) -> None:
            """Recorded in the run manifest, so it cannot be empty."""
            assert model.id

        def test_declares_self_consistent_capabilities(self, model: LanguageModel) -> None:
            capabilities = model.capabilities()
            assert capabilities.context_window_tokens > 0
            if capabilities.max_output_tokens is not None:
                assert capabilities.max_output_tokens <= capabilities.context_window_tokens, (
                    "an output cap larger than the context window is a mis-declaration"
                )

        @pytest.mark.anyio
        async def test_generate_returns_text_and_a_model_id(
            self, model: LanguageModel, ctx: RunContext
        ) -> None:
            result = await model.generate(_ask(), ctx)
            assert isinstance(result.text, str)
            assert result.model_id, "the model that served the request must be named"

        @pytest.mark.anyio
        async def test_generate_always_reports_usage(
            self, model: LanguageModel, ctx: RunContext
        ) -> None:
            """**Never a zero.** A zero is a cost report that understates a bill.

            A provider that does not report token counts must be estimated for,
            with ``estimated=True``, rather than reported as free.
            """
            result = await model.generate(_ask(), ctx)

            assert result.usage.calls >= 1
            assert result.usage.prompt_tokens > 0, (
                "zero prompt tokens means the adapter neither read nor estimated them"
            )
            if result.usage.cost_usd is not None:
                assert result.usage.cost_usd >= 0

        @pytest.mark.anyio
        async def test_a_finish_reason_is_always_reported(
            self, model: LanguageModel, ctx: RunContext
        ) -> None:
            """``length`` means truncated, and a caller has to be able to tell."""
            result = await model.generate(_ask(), ctx)
            assert result.finish_reason in {
                "stop",
                "length",
                "tool_calls",
                "content_filter",
                "error",
            }

        @pytest.mark.anyio
        async def test_count_tokens_is_never_zero_for_real_text(self, model: LanguageModel) -> None:
            """A zero lets a context assembler believe anything fits."""
            counted = await model.count_tokens(
                [Message(role="user", content="A sentence with several words in it.")]
            )
            assert counted > 0

        @pytest.mark.anyio
        async def test_count_tokens_grows_with_length(self, model: LanguageModel) -> None:
            """A counter that ignored its input would pass the previous test."""
            short = await model.count_tokens([Message(role="user", content="hi")])
            longer = await model.count_tokens([Message(role="user", content="hi " * 200)])
            assert longer > short

        @pytest.mark.anyio
        async def test_streaming_yields_deltas(self, model: LanguageModel, ctx: RunContext) -> None:
            if not supports_streaming:
                pytest.skip("this adapter declares no streaming support")

            deltas = [delta async for delta in model.stream(_ask(), ctx)]
            assert deltas, "a stream must yield at least one delta"

        @pytest.mark.anyio
        async def test_streaming_and_generation_agree(
            self, model: LanguageModel, ctx: RunContext
        ) -> None:
            """Streaming changes when the first token arrives, not what is said.

            Compared on being non-empty rather than on exact equality: a real
            provider is not deterministic even at temperature zero, and a
            contract kit that demanded byte equality would fail against every
            live endpoint.
            """
            if not supports_streaming:
                pytest.skip("this adapter declares no streaming support")

            request = _ask()
            plain = await model.generate(request, ctx)
            streamed = "".join([delta.text async for delta in model.stream(request, ctx)])

            assert bool(streamed.strip()) == bool(plain.text.strip()), (
                "streaming and generation disagree about whether there is an answer"
            )

    if unauthorised_factory is not None:
        build_unauthorised = unauthorised_factory

        class LanguageModelContractWithAuth(LanguageModelContract):
            """The base contract plus error mapping."""

            @pytest.mark.anyio
            async def test_bad_credentials_raise_auth_error(self, ctx: RunContext) -> None:
                """Never a leaked SDK exception type.

                A caller forced to catch the provider's own exception class
                would be coupled to the provider, which is the coupling this
                library exists to remove.
                """
                with pytest.raises(AuthError) as exc_info:
                    await build_unauthorised().generate(_ask(), ctx)

                assert exc_info.value.retryable is False, (
                    "an auth failure is not retryable; retrying burns budget on the same rejection"
                )
                assert exc_info.value.remedy

        return LanguageModelContractWithAuth

    return LanguageModelContract
