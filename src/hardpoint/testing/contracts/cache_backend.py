"""The ``CacheBackend`` conformance suite.

Bind it in your own tests::

    from hardpoint.testing.contracts import cache_backend_contract

    TestMyCache = cache_backend_contract(lambda: MyCache(...))

Small, because the port is small, but every assertion is one a real backend has
got wrong: returning ``b""`` for a miss, leaking a key past its prefix on
``delete_prefix``, or treating ``ttl_s=None`` as "expire immediately".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hardpoint.core.errors import MissingDependencyError
from hardpoint.core.ports import CacheBackend

__all__ = ["cache_backend_contract"]


def cache_backend_contract(
    factory: Callable[[], CacheBackend],
    *,
    expire: Callable[[CacheBackend, float], None] | None = None,
) -> type:
    """Build a pytest class asserting a ``CacheBackend`` implementation conforms.

    Args:
        factory: Returns an empty backend.
        expire: Advances the backend's clock by that many seconds. When given,
            expiry tests are added; a backend on real time cannot be tested for
            expiry without sleeping, and a sleeping suite is a slow one.

    Returns:
        A test class. Bind it to a name beginning with ``Test``.
    """
    try:
        import pytest  # noqa: PLC0415 - see the vector_index kit's docstring
    except ModuleNotFoundError as exc:  # pragma: no cover - pytest is a dev dependency
        raise MissingDependencyError(
            "The contract kits build pytest test classes, and pytest is not installed.",
            component="cache_backend_contract",
            remedy="pip install pytest",
            cause=exc,
        ) from exc

    class CacheBackendContract:
        """Behaviour every ``CacheBackend`` implementation must exhibit."""

        @pytest.fixture
        def cache(self) -> Any:
            return factory()

        def test_satisfies_the_protocol(self, cache: CacheBackend) -> None:
            assert isinstance(cache, CacheBackend)

        @pytest.mark.anyio
        async def test_a_miss_is_none_not_empty_bytes(self, cache: CacheBackend) -> None:
            """``b""`` is a legitimate cached value; a miss must be distinguishable."""
            assert await cache.get("absent") is None

        @pytest.mark.anyio
        async def test_set_then_get_round_trips_bytes(self, cache: CacheBackend) -> None:
            await cache.set("k", b"\x00binary\xff", None)
            assert await cache.get("k") == b"\x00binary\xff"

        @pytest.mark.anyio
        async def test_an_empty_value_is_a_hit(self, cache: CacheBackend) -> None:
            await cache.set("empty", b"", None)
            assert await cache.get("empty") == b""

        @pytest.mark.anyio
        async def test_set_replaces(self, cache: CacheBackend) -> None:
            await cache.set("k", b"one", None)
            await cache.set("k", b"two", None)
            assert await cache.get("k") == b"two"

        @pytest.mark.anyio
        async def test_delete_prefix_removes_only_that_prefix(self, cache: CacheBackend) -> None:
            """How an epoch bump would invalidate one layer without touching another."""
            await cache.set("retr:primary:1:a", b"1", None)
            await cache.set("retr:primary:1:b", b"2", None)
            await cache.set("embed:m:query:x", b"3", None)

            assert await cache.delete_prefix("retr:primary:1:") == 2
            assert await cache.get("retr:primary:1:a") is None
            assert await cache.get("embed:m:query:x") == b"3"

        @pytest.mark.anyio
        async def test_keys_with_separators_and_unicode_work(self, cache: CacheBackend) -> None:
            key = "gen:openai/gpt-4o:abc:v1:café/../∅"
            await cache.set(key, b"v", 60)
            assert await cache.get(key) == b"v"

    if expire is None:
        return CacheBackendContract

    advance = expire

    class CacheBackendContractWithExpiry(CacheBackendContract):
        """The base contract plus TTL behaviour."""

        @pytest.mark.anyio
        async def test_an_entry_expires_after_its_ttl(self, cache: CacheBackend) -> None:
            await cache.set("short", b"v", 10)
            advance(cache, 11)
            assert await cache.get("short") is None

        @pytest.mark.anyio
        async def test_an_entry_lives_until_its_ttl(self, cache: CacheBackend) -> None:
            await cache.set("short", b"v", 10)
            advance(cache, 5)
            assert await cache.get("short") == b"v"

        @pytest.mark.anyio
        async def test_no_ttl_never_expires(self, cache: CacheBackend) -> None:
            await cache.set("forever", b"v", None)
            advance(cache, 10**9)
            assert await cache.get("forever") == b"v"

    return CacheBackendContractWithExpiry
