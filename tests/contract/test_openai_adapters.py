"""The OpenAI-compatible adapters, against a real loopback server.

INSTRUCTIONS.md §6.6 requires the ``LanguageModel`` and ``EmbeddingModel``
contract kits to exist and pass. They are bound here twice: once to the fakes,
and once to the real adapters talking HTTP to a loopback server.

The second binding is the one that matters. A contract kit that only ever ran
against a fake proves the fake conforms, which nobody doubted. Running it
against the adapter over real sockets, real JSON and real SSE framing is what
proves the adapter speaks the protocol.

No traffic leaves the machine: the server binds to 127.0.0.1, which the suite's
network guard permits for exactly this reason.
"""

from __future__ import annotations

import pytest

from hardpoint.adapters.embeddings import OpenAICompatibleEmbeddings
from hardpoint.adapters.llm import OpenAICompatibleChat
from hardpoint.core.errors import (
    AuthError,
    ContractError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    TransientError,
)
from hardpoint.core.ports import GenerationRequest, Message
from hardpoint.testing import (
    FakeEmbeddingModel,
    FakeLanguageModel,
    build_run_context,
)
from hardpoint.testing.contracts import embedding_model_contract, language_model_contract
from tests.contract.fake_openai_server import fake_openai_server

pytestmark = pytest.mark.contract


# --------------------------------------------------------------------------- #
# The kits, bound to the fakes                                                #
# --------------------------------------------------------------------------- #


TestFakeLanguageModel = language_model_contract(FakeLanguageModel)
"""The fakes must pass their own kits (M0 DoD, and INSTRUCTIONS.md §6.6)."""


TestFakeEmbeddingModel = embedding_model_contract(
    lambda: FakeEmbeddingModel(dimensions=8), asymmetric=True
)
"""``FakeEmbeddingModel`` is deliberately asymmetric, so the kit checks that."""


# --------------------------------------------------------------------------- #
# The kits, bound to the real adapters over loopback HTTP                     #
# --------------------------------------------------------------------------- #
#
# The server has to outlive each test, so it is started once for the module. A
# per-test server would be tidier and would also make the contract kit's own
# fixtures spin one up per assertion, which is a lot of sockets for no gain.

_server = fake_openai_server()
_BASE_URL, _STATE = _server.__enter__()


def _shutdown() -> None:
    _server.__exit__(None, None, None)


import atexit  # noqa: E402 - registered after the server exists

atexit.register(_shutdown)


TestOpenAICompatibleChat = language_model_contract(
    lambda: OpenAICompatibleChat("fake-model", base_url=_BASE_URL, api_key="test-key")
)
"""The real adapter, over real HTTP."""


TestOpenAICompatibleEmbeddings = embedding_model_contract(
    lambda: OpenAICompatibleEmbeddings(
        "fake-embed", dimensions=8, base_url=_BASE_URL, api_key="test-key"
    )
)
"""The real adapter, over real HTTP."""


# --------------------------------------------------------------------------- #
# Transport and error mapping                                                 #
# --------------------------------------------------------------------------- #


def ask(text: str = "hello") -> GenerationRequest:
    return GenerationRequest(messages=(Message(role="user", content=text),))


@pytest.mark.anyio
async def test_the_adapter_sends_the_bearer_token() -> None:
    with fake_openai_server() as (base_url, state):
        model = OpenAICompatibleChat("m", base_url=base_url, api_key="secret-key")
        await model.generate(ask(), build_run_context())
        await model.aclose()

    assert state.auth_headers[-1] == "Bearer secret-key"


@pytest.mark.anyio
async def test_no_credential_is_sent_when_none_is_configured() -> None:
    """A local endpoint must not be handed an empty ``Authorization`` header."""
    with fake_openai_server() as (base_url, state):
        model = OpenAICompatibleChat("m", base_url=base_url)
        await model.generate(ask(), build_run_context())
        await model.aclose()

    assert state.auth_headers[-1] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (401, AuthError, False),
        (403, AuthError, False),
        (429, RateLimitedError, True),
        (400, InvalidRequestError, False),
        (404, InvalidRequestError, False),
        (422, InvalidRequestError, False),
        (500, TransientError, True),
        (503, TransientError, True),
        (418, ProviderError, False),
    ],
)
async def test_status_codes_map_to_the_taxonomy(
    status: int, expected: type[Exception], retryable: bool
) -> None:
    """**[LOCKED]** Never a raw ``httpx`` exception (INSTRUCTIONS.md §6.3).

    The classification is not cosmetic: it is what decides whether ``Retry``
    will act, so an auth failure classified as transient would burn the budget
    on four identical rejections.
    """
    with fake_openai_server() as (base_url, state):
        state.status = status
        model = OpenAICompatibleChat("m", base_url=base_url)

        with pytest.raises(expected) as exc_info:
            await model.generate(ask(), build_run_context())
        await model.aclose()

    error = exc_info.value
    assert getattr(error, "retryable", None) is retryable
    assert getattr(error, "remedy", None), "every provider error carries a remedy"


@pytest.mark.anyio
async def test_a_rate_limit_carries_the_servers_retry_after() -> None:
    """A server that says how long to wait knows better than any backoff curve."""
    with fake_openai_server() as (base_url, state):
        state.status = 429
        state.headers = {"retry-after": "12"}
        model = OpenAICompatibleChat("m", base_url=base_url)

        with pytest.raises(RateLimitedError) as exc_info:
            await model.generate(ask(), build_run_context())
        await model.aclose()

    assert exc_info.value.retry_after_s == 12.0


@pytest.mark.anyio
async def test_a_date_form_retry_after_is_ignored_rather_than_misparsed() -> None:
    """A wrong wait computed from a misparsed date is worse than no hint."""
    with fake_openai_server() as (base_url, state):
        state.status = 429
        state.headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        model = OpenAICompatibleChat("m", base_url=base_url)

        with pytest.raises(RateLimitedError) as exc_info:
            await model.generate(ask(), build_run_context())
        await model.aclose()

    assert exc_info.value.retry_after_s is None


@pytest.mark.anyio
async def test_the_providers_own_error_text_is_preserved() -> None:
    """ "maximum context length is 8192 tokens" loses its number if paraphrased."""
    with fake_openai_server() as (base_url, state):
        state.status = 400
        state.error_body = {
            "error": {"message": "This model's maximum context length is 8192 tokens."}
        }
        model = OpenAICompatibleChat("m", base_url=base_url)

        with pytest.raises(InvalidRequestError) as exc_info:
            await model.generate(ask(), build_run_context())
        await model.aclose()

    assert "8192 tokens" in str(exc_info.value)


@pytest.mark.anyio
async def test_an_unreachable_endpoint_is_classified_and_retryable() -> None:
    """Never a raw ``httpx`` exception, and always retryable.

    Accepts either classification: a closed port is refused on some platforms
    and silently dropped -- and so eventually timed out -- on others. Both are
    correct readings of "could not reach it", and both are retryable, which is
    the property that decides what ``Retry`` does. Pinning one would make this
    test a report on the host's TCP stack rather than on the adapter.
    """
    model = OpenAICompatibleChat("m", base_url="http://127.0.0.1:1/v1", timeout_s=1.0)

    with pytest.raises((TransientError, ProviderTimeout)) as exc_info:
        await model.generate(ask(), build_run_context())
    await model.aclose()

    assert exc_info.value.retryable is True
    assert exc_info.value.remedy


# --------------------------------------------------------------------------- #
# Usage                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_reported_usage_is_used_when_the_provider_sends_it() -> None:
    with fake_openai_server() as (base_url, _state):
        model = OpenAICompatibleChat("m", base_url=base_url)
        result = await model.generate(ask(), build_run_context())
        await model.aclose()

    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.estimated is False


@pytest.mark.anyio
async def test_usage_is_estimated_never_zero_when_the_provider_omits_it() -> None:
    """**A zero would silently understate the bill for every request.**

    Several OpenAI-compatible servers report no token counts at all.
    """
    with fake_openai_server() as (base_url, state):
        state.omit_usage = True
        model = OpenAICompatibleChat("m", base_url=base_url)
        result = await model.generate(ask("a longer question here"), build_run_context())
        await model.aclose()

    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.estimated is True


# --------------------------------------------------------------------------- #
# Streaming                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_streaming_reassembles_the_reply() -> None:
    with fake_openai_server() as (base_url, state):
        state.reply = "the refund window is thirty days"
        model = OpenAICompatibleChat("m", base_url=base_url)

        deltas = [d async for d in model.stream(ask(), build_run_context())]
        await model.aclose()

    assert "".join(d.text for d in deltas) == "the refund window is thirty days"
    assert any(d.finish_reason == "stop" for d in deltas)


@pytest.mark.anyio
async def test_streaming_reports_usage_on_the_final_chunk() -> None:
    with fake_openai_server() as (base_url, _state):
        model = OpenAICompatibleChat("m", base_url=base_url)
        deltas = [d async for d in model.stream(ask(), build_run_context())]
        await model.aclose()

    usage = [d.usage for d in deltas if d.usage is not None]
    assert usage, "the final chunk must carry usage"
    assert usage[-1].prompt_tokens == 11


@pytest.mark.anyio
async def test_a_streaming_error_status_is_mapped_too() -> None:
    """The error path is easy to leave unmapped in the streaming branch."""
    with fake_openai_server() as (base_url, state):
        state.status = 401
        model = OpenAICompatibleChat("m", base_url=base_url)

        with pytest.raises(AuthError):
            _ = [d async for d in model.stream(ask(), build_run_context())]
        await model.aclose()


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_embeddings_are_returned_in_input_order_even_when_shuffled() -> None:
    """**The silent corruption.**

    OpenAI-compatible responses carry an ``index`` and do not guarantee arrival
    order. An adapter that trusted arrival order would attach every chunk's
    vector to a neighbouring chunk, and retrieval would keep working while
    returning the wrong passages.
    """
    texts = ["alpha text", "beta text", "gamma text"]

    with fake_openai_server() as (base_url, state):
        model = OpenAICompatibleEmbeddings("e", dimensions=8, base_url=base_url)
        ctx = build_run_context()

        ordered = await model.embed(texts, "document", ctx)
        state.shuffle_embeddings = True
        shuffled = await model.embed(texts, "document", ctx)
        await model.aclose()

    assert shuffled.vectors == ordered.vectors, "the adapter must sort on the index field"


@pytest.mark.anyio
async def test_a_width_mismatch_is_a_contract_error_with_a_remedy() -> None:
    """Declared and produced disagreeing corrupts an index silently."""
    with fake_openai_server() as (base_url, state):
        state.dimensions = 16
        model = OpenAICompatibleEmbeddings("e", dimensions=8, base_url=base_url)

        with pytest.raises(ContractError) as exc_info:
            await model.embed(["text"], "document", build_run_context())
        await model.aclose()

    assert "16" in str(exc_info.value)
    assert exc_info.value.remedy is not None
    assert "two widths" in exc_info.value.remedy


@pytest.mark.anyio
async def test_an_empty_batch_makes_no_request() -> None:
    """A provider asked to embed nothing bills for it on some plans."""
    with fake_openai_server() as (base_url, state):
        model = OpenAICompatibleEmbeddings("e", dimensions=8, base_url=base_url)
        result = await model.embed([], "document", build_run_context())
        await model.aclose()

    assert result.vectors == ()
    assert state.requests == []


@pytest.mark.anyio
async def test_a_non_json_response_names_the_likely_cause() -> None:
    """A base URL pointing at a proxy is the common mistake."""
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://example.invalid/v1"
    )
    model = OpenAICompatibleChat("m", client=client)

    with pytest.raises(ProviderError) as exc_info:
        await model.generate(ask(), build_run_context())
    await client.aclose()

    assert "OpenAI-compatible" in str(exc_info.value)
    assert "gateway" in (exc_info.value.remedy or "")


# --------------------------------------------------------------------------- #
# Transport hygiene                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_one_client_is_reused_across_calls() -> None:
    """Building a client per request turns a batch into a batch of handshakes."""
    with fake_openai_server() as (base_url, _state):
        model = OpenAICompatibleChat("m", base_url=base_url)
        first = model._client
        await model.generate(ask(), build_run_context())
        await model.generate(ask(), build_run_context())
        assert model._client is first
        await model.aclose()


@pytest.mark.anyio
async def test_a_caller_supplied_client_is_not_closed_by_the_adapter() -> None:
    """The caller owns what the caller built."""
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://example.invalid/v1"
    )
    model = OpenAICompatibleChat("m", client=client)

    await model.aclose()
    assert not client.is_closed, "the adapter must not close a client it did not build"
    await client.aclose()


def test_no_provider_sdk_type_appears_in_the_adapter_signatures() -> None:
    """**[LOCKED]** INSTRUCTIONS.md §6.3.

    ``httpx`` is a base dependency and appears deliberately as the transport
    seam; what must never appear is a *provider's* SDK type, which would couple
    a caller to that provider.
    """
    import inspect

    for adapter in (OpenAICompatibleChat, OpenAICompatibleEmbeddings):
        for name, member in vars(adapter).items():
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            rendered = str(inspect.signature(member))
            for banned in ("openai.", "ChatCompletion", "CreateEmbeddingResponse"):
                assert banned not in rendered, f"{adapter.__name__}.{name} exposes {banned}"


def test_the_adapters_hold_no_retry_logic() -> None:
    """**[LOCKED]** Retry lives in exactly one place (INSTRUCTIONS.md §13.1).

    Checked against the source, because a retry loop added to an adapter would
    still pass every behavioural test here while making retry behaviour
    unconfigurable and invisible in traces.
    """
    import ast
    from pathlib import Path

    import hardpoint.adapters as adapters_package

    root = Path(adapters_package.__file__).resolve().parent
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                offenders.append(f"{path.name}: while loop")
            if isinstance(node, ast.Name) and node.id in {"sleep", "backoff", "retry"}:
                offenders.append(f"{path.name}: reference to {node.id!r}")

    assert not offenders, f"adapters must contain no retry machinery: {offenders}"
