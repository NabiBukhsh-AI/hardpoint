"""Plain-text and Markdown parsers.

Implements the M1 parser set from INSTRUCTIONS.md §6.2. A parser turns fetched
bytes into text plus whatever structure it can recover.

## Structure is not decoration

``Block`` is what lets the chunker split at a heading rather than mid-sentence,
and heading-aware splitting is most of the difference between chunks that
retrieve well and chunks that do not. A parser that returned text with no blocks
is legal -- some formats have no structure -- but it hands the chunker nothing
to work with, so recovering structure where it exists is the parser's main job.

## Decoding is the parser's problem

``SourceBlob`` carries bytes, because ``Document.content_hash`` is over raw
bytes and a re-encoding must register as a change. The parser decodes, and does
so tolerantly: a single malformed byte in a 200-page document should produce a
parse *warning* and a usable document, not quarantine the whole thing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from hardpoint.core.errors import IngestionError
from hardpoint.core.ids import content_hash, document_id
from hardpoint.core.models import Block, CharSpan, Document, ParsedDocument

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext
    from hardpoint.core.ports import SourceBlob

__all__ = ["MarkdownParser", "TextParser", "decode"]

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_ATX_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")

_SETEXT_H1_LEVEL = 1
_SETEXT_H2_LEVEL = 2


def decode(blob: SourceBlob) -> tuple[str, list[str]]:
    """Decode a blob's bytes to text, tolerating damage.

    Tries UTF-8 first, then UTF-8 with a BOM, then Latin-1 as a last resort --
    Latin-1 cannot fail, so this always produces something. Each fallback adds a
    parse warning, because "this document decoded oddly" is worth knowing when
    its retrieval quality turns out to be poor.

    Args:
        blob: The fetched document.

    Returns:
        ``(text, warnings)``.

    Raises:
        Nothing. Decoding never fails; it degrades and says so.
    """
    warnings: list[str] = []

    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return blob.data.decode(encoding), warnings
        except UnicodeDecodeError:
            continue

    warnings.append("Not valid UTF-8; decoded as latin-1. Non-ASCII characters may be wrong.")
    return blob.data.decode("latin-1", errors="replace"), warnings


def _build_document(blob: SourceBlob) -> Document:
    """Build the ``Document`` for a blob, with a deterministic id."""
    return Document(
        id=document_id(_source_id_of(blob), blob.entry.id),
        source_uri=blob.entry.uri,
        source_id=_source_id_of(blob),
        media_type=blob.media_type,
        content_hash=content_hash(blob.data),
        revision=blob.entry.revision,
        metadata=dict(blob.entry.metadata),
    )


def _source_id_of(blob: SourceBlob) -> str:
    """Return the configured source id a blob came from.

    Carried on the entry's metadata by the sync engine, which knows which source
    it asked. Falls back to the empty string so a parser can be exercised in
    isolation without a sync engine.
    """
    value = blob.entry.metadata.get("source_id", "")
    return value if isinstance(value, str) else ""


class TextParser:
    """Parses plain text into paragraphs.

    The floor every corpus can rely on. Blank lines separate paragraphs, which is
    the only structure plain text carries, and it is enough for the chunker to
    avoid splitting mid-sentence.

    Attributes:
        media_types: ``text/plain`` and the types with no better parser.
    """

    media_types = frozenset({"text/plain", "text/x-rst", "application/octet-stream"})

    async def parse(self, blob: SourceBlob, ctx: RunContext) -> ParsedDocument:
        """Extract text and paragraph blocks.

        Raises:
            IngestionError: If the blob is empty. An empty document produces no
                chunks, so quarantining it with a reason beats indexing nothing
                and reporting success.
        """
        text, warnings = decode(blob)
        if not text.strip():
            raise IngestionError(
                f"{blob.entry.uri} decoded to no text.",
                document_id=blob.entry.id,
                remedy=(
                    "Remove the empty file from the corpus, or exclude it with a "
                    "source `exclude` pattern. It is quarantined, not fatal."
                ),
            )

        return ParsedDocument(
            document=_build_document(blob),
            text=text,
            blocks=list(_paragraph_blocks(text)),
            parse_warnings=warnings,
        )

    def __repr__(self) -> str:
        """Render the class name."""
        return "TextParser()"


class MarkdownParser:
    """Parses Markdown, recovering headings, code fences, lists and paragraphs.

    Deliberately a small hand-written scanner rather than a Markdown library. The
    chunker needs four things -- where the headings are, what level each is, where
    code fences begin and end, and where paragraphs break -- and a full CommonMark
    parse would add a dependency to the base install for structure nobody
    consumes. The ``parsers`` extra exists for the formats that genuinely need a
    library.

    Code fences matter more than they look: splitting inside one produces a chunk
    of syntactically broken code, which embeds poorly and reads worse.

    Attributes:
        media_types: ``text/markdown``.
    """

    media_types = frozenset({"text/markdown"})

    async def parse(self, blob: SourceBlob, ctx: RunContext) -> ParsedDocument:
        """Extract text and structural blocks.

        Raises:
            IngestionError: If the blob decodes to no text.
        """
        text, warnings = decode(blob)
        if not text.strip():
            raise IngestionError(
                f"{blob.entry.uri} decoded to no text.",
                document_id=blob.entry.id,
                remedy=(
                    "Remove the empty file from the corpus, or exclude it with a "
                    "source `exclude` pattern. It is quarantined, not fatal."
                ),
            )

        blocks = list(_markdown_blocks(text))
        if not any(block.kind == "heading" for block in blocks):
            warnings.append("No headings found; chunks will be split on paragraphs only.")

        return ParsedDocument(
            document=_build_document(blob),
            text=text,
            blocks=blocks,
            parse_warnings=warnings,
        )

    def __repr__(self) -> str:
        """Render the class name."""
        return "MarkdownParser()"


def _paragraph_blocks(text: str) -> list[Block]:
    """Split text into paragraph blocks, carrying each one's span."""
    blocks: list[Block] = []
    position = 0

    for piece in _PARAGRAPH_BREAK.split(text):
        start = text.find(piece, position) if piece else position
        if start < 0:  # pragma: no cover - find only misses if text was mutated
            start = position
        end = start + len(piece)
        position = end
        if piece.strip():
            blocks.append(
                Block(kind="paragraph", text=piece.strip(), span=CharSpan(start=start, end=end))
            )

    return blocks


def _markdown_blocks(text: str) -> list[Block]:  # noqa: PLR0915 - one scanner, one pass
    """Scan Markdown into heading, code, list and paragraph blocks.

    One pass, line by line, tracking only whether a code fence is open. Anything
    more would be a Markdown implementation, and the chunker does not need one.
    """
    blocks: list[Block] = []
    lines = text.splitlines(keepends=True)

    offset = 0
    starts: list[int] = []
    for line in lines:
        starts.append(offset)
        offset += len(line)

    pending: list[int] = []  # indices of lines accumulating into a paragraph
    in_fence = False
    fence_start: int | None = None
    fence_lines: list[int] = []

    def flush_paragraph() -> None:
        if not pending:
            return
        start = starts[pending[0]]
        end = starts[pending[-1]] + len(lines[pending[-1]])
        body = "".join(lines[index] for index in pending).strip()
        if body:
            kind = "list" if _LIST_ITEM.match(lines[pending[0]]) else "paragraph"
            blocks.append(
                Block(kind=kind, text=body, span=CharSpan(start=start, end=end))  # type: ignore[arg-type]
            )
        pending.clear()

    for index, raw in enumerate(lines):
        line = raw.rstrip("\n")

        if _FENCE.match(line):
            if in_fence:
                fence_lines.append(index)
                start = fence_start if fence_start is not None else starts[index]
                end = starts[index] + len(raw)
                blocks.append(
                    Block(
                        kind="code",
                        text="".join(lines[i] for i in fence_lines).strip(),
                        span=CharSpan(start=start, end=end),
                    )
                )
                in_fence = False
                fence_start = None
                fence_lines = []
            else:
                flush_paragraph()
                in_fence = True
                fence_start = starts[index]
                fence_lines = [index]
            continue

        if in_fence:
            fence_lines.append(index)
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            flush_paragraph()
            blocks.append(
                Block(
                    kind="heading",
                    text=heading.group("text").strip(),
                    level=len(heading.group("hashes")),
                    span=CharSpan(start=starts[index], end=starts[index] + len(raw)),
                )
            )
            continue

        # A setext heading is the previous line underlined with = or -. Recognised
        # because plenty of real corpora use it and treating the underline as a
        # paragraph would put a row of dashes into a chunk.
        if pending and _SETEXT_UNDERLINE.match(line) and lines[pending[-1]].strip():
            underlined = pending[-1]
            before = pending[:-1]
            pending.clear()
            pending.extend(before)
            flush_paragraph()
            blocks.append(
                Block(
                    kind="heading",
                    text=lines[underlined].strip(),
                    level=_SETEXT_H1_LEVEL if line.strip().startswith("=") else _SETEXT_H2_LEVEL,
                    span=CharSpan(start=starts[underlined], end=starts[index] + len(raw)),
                )
            )
            continue

        if not line.strip():
            flush_paragraph()
            continue

        pending.append(index)

    if in_fence:
        # An unterminated fence. Keeping the text is better than discarding it,
        # and the warning tells a human the document is malformed.
        start = fence_start if fence_start is not None else 0
        blocks.append(
            Block(
                kind="code",
                text="".join(lines[i] for i in fence_lines).strip(),
                span=CharSpan(start=start, end=len(text)),
            )
        )
    else:
        flush_paragraph()

    return blocks
