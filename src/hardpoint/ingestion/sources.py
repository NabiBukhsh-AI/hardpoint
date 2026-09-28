"""Where documents come from.

Implements INSTRUCTIONS.md §6.2. A source answers two questions: what documents
exist, and what are the bytes of one of them.

## Listing has to be cheap

The sync engine lists the *whole* source on every run to compute a diff, so
listing must not fetch content. That is what ``SourceEntry.revision`` is for: an
etag, an mtime, a commit sha -- anything the source can report without
downloading, that changes when the content changes.

A source that could only report a revision by reading the file would make every
run cost a full download, which is the cost incremental ingestion exists to
avoid.

## Why ``Source`` is not in ``core.ports``

It lives next to the implementations and the sync engine that drives it
(INSTRUCTIONS.md §6.2). ``core.ports`` holds the boundaries a *pipeline* crosses;
this is a boundary only ingestion crosses.
"""

from __future__ import annotations

import mimetypes
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio
from pydantic import BaseModel, ConfigDict

from hardpoint.core.errors import IngestionError
from hardpoint.core.ports import SourceBlob, SourceEntry
from hardpoint.core.types import JsonValue

__all__ = [
    "DEFAULT_MEDIA_TYPE",
    "LocalFileSource",
    "LocalFilesConfig",
    "Source",
    "build",
    "media_type_for",
]

DEFAULT_MEDIA_TYPE = "application/octet-stream"
"""What an unrecognised extension reports as, so a parser can refuse it."""

# mimetypes does not know Markdown on every platform, and getting it wrong sends
# a structured document to the plain-text parser, which silently loses every
# heading the chunker would have respected.
_EXTRA_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdx": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".text": "text/plain",
    ".log": "text/plain",
}


def media_type_for(path: str | Path) -> str:
    """Return the IANA media type for a path, used to select a parser.

    Args:
        path: The file path or URI.

    Returns:
        A media type, or :data:`DEFAULT_MEDIA_TYPE` when the extension is
        unrecognised. Never ``None``, so a caller never has to guard.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _EXTRA_MEDIA_TYPES:
        return _EXTRA_MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or DEFAULT_MEDIA_TYPE


@runtime_checkable
class Source(Protocol):
    """Something documents can be ingested from.

    Attributes:
        id: Identifier for this configured source. Part of every document's
            identity, so the same file reachable through two sources is two
            documents with separate manifests and separate deletes.
    """

    id: str

    def list(self) -> AsyncIterator[SourceEntry]:
        """Yield every document currently in the source.

        Declared as returning an ``AsyncIterator`` rather than as ``async def``,
        because that is how an async generator types: an implementation writes
        ``async def list(self): yield ...`` and callers iterate without awaiting
        first. INSTRUCTIONS.md §6.2 writes ``async def``, which would oblige
        every caller to ``await source.list()`` before iterating and which no
        implementation would naturally write.

        Must not fetch content. See the module docstring.
        """
        ...

    async def fetch(self, entry: SourceEntry) -> SourceBlob:
        """Read one document's bytes.

        Raises:
            IngestionError: If the document cannot be read. The sync engine
                quarantines it and continues.
        """
        ...


class LocalFileSource:
    """Documents from a directory on the local filesystem.

    The source the ``rag-minimal`` template starts with, and the one every test
    fixture corpus uses.

    Revision is ``mtime_ns:size``. Both change when a file is edited, neither
    requires reading it, and the pair catches a case an mtime alone misses -- a
    file restored from backup with its timestamp preserved but its length
    changed. It can still miss an edit that preserves both, which is exactly why
    the sync engine verifies ``content_hash`` after fetching rather than
    trusting the revision.

    Args:
        root: Directory to walk.
        source_id: Identifier for this configured source.
        patterns: Glob patterns to include, relative to ``root``.
        exclude: Glob patterns to skip. Applied after ``patterns``.
        follow_symlinks: Whether to include symlinked files. Defaults to
            ``False``, because a symlink into a parent directory turns a corpus
            walk into a much larger one than anybody intended.

    Raises:
        IngestionError: On construction, if ``root`` is not a directory. Failing
            at construction beats listing an empty corpus and cheerfully
            reporting that nothing changed.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        source_id: str = "files",
        patterns: Sequence[str] = ("**/*.md", "**/*.txt"),
        exclude: Sequence[str] = (),
        follow_symlinks: bool = False,
    ) -> None:
        self.id = source_id
        # Absolute, so `uri` is a real file URI whatever the working directory.
        self.root = Path(root).resolve()
        self.patterns = tuple(patterns)
        self.exclude = tuple(exclude)
        self.follow_symlinks = follow_symlinks

        if not self.root.is_dir():
            raise IngestionError(
                f"The source directory {self.root} does not exist or is not a directory.",
                source_id=source_id,
                remedy=(
                    f"Create {self.root}, or point the source's `root` at the "
                    f"directory holding the documents to ingest."
                ),
            )

    async def list(self) -> AsyncIterator[SourceEntry]:
        """Yield an entry per matching file, in sorted path order.

        Sorted so two runs over an unchanged corpus produce the same order.
        Filesystem walk order is not stable across platforms, and an unstable
        order makes one ingest report impossible to diff against the next.
        """
        for path in self._matching_paths():
            stat = path.stat()
            relative = path.relative_to(self.root).as_posix()
            yield SourceEntry(
                id=relative,
                uri=path.as_uri(),
                revision=f"{stat.st_mtime_ns}:{stat.st_size}",
                size_bytes=stat.st_size,
                media_type=media_type_for(path),
                metadata=self._metadata_for(relative),
            )

    def _matching_paths(self) -> Sequence[Path]:
        """Return every included, non-excluded file, sorted and de-duplicated.

        Annotated ``Sequence[Path]`` rather than ``list[Path]``: the port names
        a method ``list``, which shadows the builtin inside this class body, so
        ``list[Path]`` here resolves to the method rather than to the type.
        """
        found: set[Path] = set()
        for pattern in self.patterns:
            found.update(path for path in self.root.glob(pattern) if path.is_file())

        for pattern in self.exclude:
            found.difference_update(self.root.glob(pattern))

        if not self.follow_symlinks:
            found = {path for path in found if not path.is_symlink()}

        return sorted(found)

    @staticmethod
    def _metadata_for(relative: str) -> dict[str, JsonValue]:
        """Derive filterable metadata from a path.

        The folder a document sits in is the most commonly used filter in a
        small corpus, so it is carried without the user configuring anything.
        """
        parent = Path(relative).parent.as_posix()
        return {"path": relative, "folder": "" if parent == "." else parent}

    async def fetch(self, entry: SourceEntry) -> SourceBlob:
        """Read a file's bytes.

        Read through ``anyio`` rather than ``Path.read_bytes`` so that ingesting
        a large corpus does not block the event loop on each file
        (INSTRUCTIONS.md §12.6).

        Raises:
            IngestionError: If the file cannot be read -- removed between the
                listing and the fetch, or permission denied. The sync engine
                quarantines it and continues rather than failing the run.
        """
        path = self.root / entry.id
        try:
            data = await anyio.Path(path).read_bytes()
        except OSError as exc:
            raise IngestionError(
                f"Could not read {path}: {exc.strerror or exc}.",
                document_id=entry.id,
                source_id=self.id,
                remedy=(
                    "Check the file still exists and is readable. A file removed "
                    "between listing and fetching is quarantined rather than fatal, "
                    "and the next run treats it as deleted."
                ),
                cause=exc,
            ) from exc

        return SourceBlob(
            entry=entry,
            data=data,
            media_type=entry.media_type or media_type_for(path),
        )

    def __repr__(self) -> str:
        """Render the source id and the directory it walks."""
        return f"LocalFileSource(id={self.id!r}, root={str(self.root)!r})"


class LocalFilesConfig(BaseModel):
    """Configuration for ``type: local_files`` under ``sources``.

    ``source_id`` defaults to the key the source is configured under, which is
    what makes it part of every document's identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str = "docs"
    source_id: str = "files"
    patterns: tuple[str, ...] = ("**/*.md", "**/*.txt")
    exclude: tuple[str, ...] = ()
    follow_symlinks: bool = False


def build(config: LocalFilesConfig) -> LocalFileSource:
    """Registry factory for ``type: local_files``."""
    return LocalFileSource(
        config.root,
        source_id=config.source_id,
        patterns=config.patterns,
        exclude=config.exclude,
        follow_symlinks=config.follow_symlinks,
    )
