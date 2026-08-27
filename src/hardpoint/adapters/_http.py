"""Shared HTTP transport for adapters, and the error mapping they all owe.

Every adapter must map provider exceptions into the ``core.errors`` taxonomy
(INSTRUCTIONS.md §6.3 **[LOCKED]**). Doing that once, here, is the difference
between one correct mapping and three subtly different ones -- the same argument
that puts retry in exactly one place.

## What this module refuses to do

**No retries, no backoff, no circuit breaking.** Those live in
``runtime/policies.py`` and nowhere else (INSTRUCTIONS.md §13.1). This module
raises a correctly *classified* error and stops; whether that error is worth
retrying is a policy decision made above it.

**No caching and no tracing decisions.** Transport only.

## Why the status mapping is the way it is

The classification is not cosmetic: it decides whether ``Retry`` will act.

- ``401``/``403`` is ``AuthError``, never retryable. Retrying a bad credential
  is four identical failures and a quarter of the budget.
- ``429`` is ``RateLimitedError``, carrying ``Retry-After`` when the server sent
  one, because a server that says how long to wait knows better than any curve.
- ``408``/``5xx`` is ``TransientError``. A gateway timeout is worth another go.
- ``400``/``404``/``422`` is ``InvalidRequestError``, never retryable: the same
  request will be rejected the same way. Context-too-long lands here, which is
  why the message keeps the provider's own text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import httpx

from hardpoint.core.errors import (
    AuthError,
    HardpointError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    TransientError,
)
from hardpoint.core.types import JsonValue

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "as_int",
    "as_json",
    "build_client",
    "map_response_error",
    "map_transport_error",
]

DEFAULT_TIMEOUT_S: Final = 60.0
"""Transport timeout when no deadline is in play.

Generous, because the deadline on ``RunContext`` is the real bound and this only
exists so a hung socket cannot outlive the process.
"""

_AUTH_STATUS: Final = frozenset({401, 403})
_RATE_LIMITED_STATUS: Final = 429
_SERVER_ERROR_FLOOR: Final = 500
_RETRYABLE_STATUS: Final = frozenset({408, 409, 500, 502, 503, 504})
_INVALID_STATUS: Final = frozenset({400, 404, 405, 413, 422})


def as_int(value: object, default: int = 0) -> int:
    """Coerce a wire value to an int, tolerating a provider that sends a string.

    Returns ``default`` for anything that will not convert, because a malformed
    token count should degrade the estimate rather than fail a response that is
    otherwise perfectly good.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def build_client(
    *,
    base_url: str,
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> httpx.AsyncClient:
    """Build a shared client for one adapter instance.

    One client per adapter, reused for every call, so connections are pooled
    (INSTRUCTIONS.md §12.6). Building one per request is the mistake that turns
    a batch of embeddings into a batch of TLS handshakes.

    Args:
        base_url: The API root, without a trailing path.
        api_key: Sent as a bearer token when given. Omitted entirely when not,
            so a local endpoint that wants no auth is not sent an empty header.
        headers: Extra headers.
        timeout_s: Transport timeout.

    Returns:
        A client the caller owns and should close.
    """
    merged = {"content-type": "application/json", **(headers or {})}
    if api_key:
        merged["authorization"] = f"Bearer {api_key}"

    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=merged,
        timeout=httpx.Timeout(timeout_s),
    )


def _detail(response: httpx.Response) -> str:
    """Extract the provider's own error text, falling back to the body.

    Kept verbatim because the useful part -- "this model's maximum context length
    is 8192 tokens" -- is the provider's wording, and paraphrasing it would lose
    the number.
    """
    try:
        payload: Any = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if isinstance(payload.get("message"), str):
            return str(payload["message"])
    return str(payload)[:500]


def _retry_after(response: httpx.Response) -> float | None:
    """Read ``Retry-After``, tolerating the header being absent or a date."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # An HTTP-date form. Rather than parse it, let the backoff curve decide:
        # a wrong wait computed from a misparsed date is worse than no hint.
        return None


def map_response_error(
    response: httpx.Response, *, component: str, operation: str
) -> HardpointError:
    """Turn a non-2xx response into the right taxonomy error.

    Args:
        response: The response. Not raised on here; the caller raises what comes
            back, so the mapping stays a pure function.
        component: The adapter's registry key, for the message.
        operation: What was being attempted.

    Returns:
        The classified error, never a raw ``HTTPStatusError``.
    """
    status = response.status_code
    detail = _detail(response)
    message = f"{component}: {operation} failed with HTTP {status}. {detail}"

    if status in _AUTH_STATUS:
        return AuthError(
            message,
            component=component,
            remedy=(
                "Check the API key in the environment and that it is authorised for "
                "this model. This is not retried: the same credential produces the "
                "same rejection."
            ),
        )

    if status == _RATE_LIMITED_STATUS:
        return RateLimitedError(
            message,
            retry_after_s=_retry_after(response),
            component=component,
            remedy=(
                "Lower `policies.rate_limit.requests_per_second` for this component, "
                "or raise the account's quota. Retry with backoff is already applied "
                "if `policies.retry` is configured."
            ),
        )

    if status in _INVALID_STATUS:
        return InvalidRequestError(
            message,
            component=component,
            remedy=(
                "The request itself was rejected, so retrying will not help. If the "
                "context was too long, lower `retrieval.context.token_budget` or "
                "`retrieval.top_k`."
            ),
        )

    if status in _RETRYABLE_STATUS or status >= _SERVER_ERROR_FLOOR:
        return TransientError(
            message,
            component=component,
            remedy="Usually resolves on retry. Configure `policies.retry` if it is not already on.",
        )

    return ProviderError(
        message,
        component=component,
        remedy="An unclassified provider status. Report it if it recurs.",
    )


def map_transport_error(exc: Exception, *, component: str, operation: str) -> HardpointError:
    """Turn an httpx transport failure into the right taxonomy error.

    A connection that never opened and a request that timed out are different
    facts, and both are retryable for different reasons, so both are classified
    rather than collapsed into a generic provider error.
    """
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeout(
            f"{component}: {operation} timed out.",
            component=component,
            remedy=(
                "Raise `policies.timeout.per_attempt_s`, or "
                "`budgets.request.deadline_s` if the run deadline is what ran out."
            ),
            cause=exc,
        )

    if isinstance(exc, httpx.TransportError):
        return TransientError(
            f"{component}: {operation} could not reach the endpoint. {exc}",
            component=component,
            remedy=(
                "Check the endpoint URL and that the service is reachable from here. "
                "`hardpoint doctor` checks provider reachability at startup."
            ),
            cause=exc,
        )

    return ProviderError(
        f"{component}: {operation} failed. {exc}",
        component=component,
        remedy="An unmapped transport failure. Report it with the original traceback.",
        cause=exc,
    )


def as_json(response: httpx.Response, *, component: str, operation: str) -> dict[str, JsonValue]:
    """Decode a JSON body, classifying a malformed one as a contract violation.

    A 200 carrying something that is not JSON means the endpoint is not what it
    claimed to be -- a proxy error page, most often -- and calling that a
    ``ProviderError`` rather than letting a ``JSONDecodeError`` escape is what
    keeps the taxonomy honest.
    """
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"{component}: {operation} returned a {response.status_code} that is not "
            f"JSON. The endpoint may not be OpenAI-compatible.",
            component=component,
            remedy=(
                "Check the base URL points at the API root, not at a proxy or a web "
                "page. The first 200 characters of the body were: "
                f"{response.text[:200]!r}"
            ),
            cause=exc,
        ) from exc

    if not isinstance(payload, dict):
        raise ProviderError(
            f"{component}: {operation} returned JSON that is not an object.",
            component=component,
            remedy="Check the base URL points at an OpenAI-compatible API root.",
        )
    return payload
