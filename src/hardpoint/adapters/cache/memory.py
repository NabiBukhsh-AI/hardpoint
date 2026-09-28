"""An in-process ``CacheBackend`` with TTLs and a size bound.

The default shared cache for a single process, and the request-scoped cache
every run gets. Bounded, least-recently-used first, so a long-running service
cannot grow it without limit.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MemoryCache", "MemoryCacheConfig", "build"]


class MemoryCache:
    """A dict with expiry and an entry limit.

    Args:
        max_entries: Least recently used entries are evicted beyond this.
        clock: Monotonic seconds. Injected so a test can expire an entry
            without sleeping.
    """

    def __init__(
        self, *, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[bytes, float | None]] = OrderedDict()

    async def get(self, key: str) -> bytes | None:
        """Return the value, or ``None`` when absent or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires is not None and self._clock() >= expires:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        """Store a value; ``ttl_s=None`` never expires."""
        expires = None if ttl_s is None else self._clock() + ttl_s
        self._entries[key] = (value, expires)
        self._entries.move_to_end(key)
        # One entry was added, so at most one needs evicting.
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix."""
        doomed = [key for key in self._entries if key.startswith(prefix)]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def __len__(self) -> int:
        """How many entries are held, expired ones included until touched."""
        return len(self._entries)

    def __repr__(self) -> str:
        """Render the size."""
        return f"MemoryCache(entries={len(self._entries)}, max_entries={self.max_entries})"


class MemoryCacheConfig(BaseModel):
    """``cache.backend: {type: memory}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_entries: int = Field(default=10_000, ge=1)


def build(config: MemoryCacheConfig) -> MemoryCache:
    """Registry factory for ``type: memory`` under ``cache.backend``."""
    return MemoryCache(max_entries=config.max_entries)
