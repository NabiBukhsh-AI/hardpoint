"""A ``CacheBackend`` on Redis, behind the ``redis`` extra.

For a multi-replica service, where every replica should share one embedding and
retrieval cache. ``redis`` is imported inside :func:`build`, never at module
import, so this module imports in a bare environment and a missing extra
surfaces as ``MissingDependencyError`` with the install command
(INSTRUCTIONS.md §3 **[LOCKED]**).

A cache outage must degrade latency, not availability (``core.ports``): a Redis
error on ``get`` is a miss and on ``set`` is ignored, so an unavailable cache
slows requests rather than failing them.
"""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import BaseModel, ConfigDict

from hardpoint.core.errors import MissingDependencyError

__all__ = ["RedisCache", "RedisCacheConfig", "build"]


class RedisCache:
    """A ``CacheBackend`` over a ``redis.asyncio`` client.

    Args:
        client: A ``redis.asyncio.Redis``. Typed ``Any`` because the SDK type
            must not appear in a hardpoint signature (INSTRUCTIONS.md §6.3).
        prefix: Namespace for every key, so one Redis can serve several projects.
    """

    def __init__(self, client: Any, *, prefix: str = "hardpoint:") -> None:
        self._client = client
        self.prefix = prefix

    async def get(self, key: str) -> bytes | None:
        """Return the value, or ``None`` on a miss or a Redis error."""
        try:
            value = await self._client.get(self.prefix + key)
        except Exception:  # a cache outage degrades latency, never availability
            return None
        return bytes(value) if value is not None else None

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        """Store a value; errors are ignored for the same reason as in ``get``."""
        with contextlib.suppress(Exception):
            await self._client.set(self.prefix + key, value, ex=ttl_s)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix, by ``SCAN`` rather than the blocking ``KEYS``."""
        deleted = 0
        async for key in self._client.scan_iter(match=f"{self.prefix}{prefix}*"):
            deleted += int(await self._client.delete(key))
        return deleted

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._client.aclose()

    def __repr__(self) -> str:
        """Render the prefix; the URL may hold a password."""
        return f"RedisCache(prefix={self.prefix!r})"


class RedisCacheConfig(BaseModel):
    """``cache.backend: {type: redis}``. Put credentials in the URL via ``${env:...}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = "redis://localhost:6379/0"
    prefix: str = "hardpoint:"


def build(config: RedisCacheConfig) -> RedisCache:
    """Registry factory for ``type: redis``. Imports ``redis`` here, not at module import.

    Raises:
        MissingDependencyError: If the ``redis`` extra is not installed.
    """
    try:
        from redis import asyncio as redis_asyncio  # noqa: PLC0415 - optional, lazily loaded
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "The Redis cache backend requires the 'redis' extra, which is not installed.",
            extra="redis",
            component="redis",
            config_path="cache.backend",
            remedy="pip install 'hardpoint[redis]'",
            cause=exc,
        ) from exc
    return RedisCache(redis_asyncio.from_url(config.url), prefix=config.prefix)
