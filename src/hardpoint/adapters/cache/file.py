"""A ``CacheBackend`` on the local filesystem, shared across processes.

What a laptop or a single-host deployment uses when the CLI, the service and
the eval runner should share embeddings without running Redis. One file per
key, named by the key's SHA-256 so any key is a valid file name, holding an
expiry header and the value.

A cache outage must degrade latency, not availability (``core.ports``), so an
unreadable or corrupt entry is a miss, and a failed write is ignored.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Callable
from pathlib import Path

import anyio
from pydantic import BaseModel, ConfigDict

__all__ = ["FileCache", "FileCacheConfig", "build"]

_HEADER = struct.Struct(">d")
"""Expiry as a big-endian double of Unix time; ``0`` means never."""


class FileCache:
    """One file per key under a directory.

    Args:
        directory: Where entries live. Created on first write.
        clock: Wall-clock seconds. Wall clock rather than monotonic, because
            entries outlive the process that wrote them.
    """

    def __init__(self, directory: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self._clock = clock

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / digest[:2] / digest

    def _index(self) -> Path:
        return self.directory / "keys.tsv"

    async def get(self, key: str) -> bytes | None:
        """Return the value, or ``None`` when absent, expired or unreadable."""
        try:
            raw = await anyio.Path(self._path(key)).read_bytes()
        except OSError:
            return None
        if len(raw) < _HEADER.size:
            return None
        (expires,) = _HEADER.unpack_from(raw)
        if expires and self._clock() >= expires:
            return None
        return raw[_HEADER.size :]

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:
        """Store a value, atomically: written to a temporary file, then renamed."""
        path = self._path(key)
        expires = 0.0 if ttl_s is None else self._clock() + ttl_s
        try:
            await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            await anyio.Path(temporary).write_bytes(_HEADER.pack(expires) + value)
            await anyio.Path(temporary).replace(path)
            async with await anyio.open_file(self._index(), "a", encoding="utf-8") as index:
                await index.write(f"{path.name}\t{key}\n")
        except OSError:
            return

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix, using the key index file."""
        index = anyio.Path(self._index())
        try:
            lines = (await index.read_text(encoding="utf-8")).splitlines()
        except OSError:
            return 0
        kept: list[str] = []
        doomed: set[str] = set()
        for line in lines:
            _, _, key = line.partition("\t")
            if key.startswith(prefix):
                doomed.add(key)
            else:
                kept.append(line)
        deleted = 0
        for key in doomed:
            try:
                await anyio.Path(self._path(key)).unlink()
                deleted += 1
            except OSError:
                continue
        await index.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        return deleted

    def __repr__(self) -> str:
        """Render the directory."""
        return f"FileCache(directory={str(self.directory)!r})"


class FileCacheConfig(BaseModel):
    """``cache.backend: {type: file}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: str = ".hardpoint/cache"


def build(config: FileCacheConfig) -> FileCache:
    """Registry factory for ``type: file`` under ``cache.backend``."""
    return FileCache(config.directory)
