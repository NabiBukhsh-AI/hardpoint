"""A structure-aware recursive chunker.

Implements the M1 chunker from INSTRUCTIONS.md §6.2: respects headings and
paragraph boundaries, with a target size and an overlap configured in tokens.

## Why structure comes before size

Splitting purely by size cuts sentences in half, separates a heading from the
text it introduces, and slices code fences into syntactically broken fragments.
Each of those embeds badly, and the damage is invisible: retrieval quality drops
with no error anywhere.

So the split points are chosen in descending order of preference:

1. Between blocks, at a heading. A heading starts a new topic, which is the best
   possible place to cut.
2. Between blocks, at a paragraph boundary.
3. Between sentences within an over-long block.
4. Between words, as a last resort.

Only the last of these loses meaning, and it only happens for a single block
that is larger than the target on its own.

## Heading context travels with the chunk

Each chunk is prefixed with the heading path it sits under -- ``Billing >
Refunds`` -- because a chunk reading "within 30 days of purchase" retrieves for
almost nothing on its own, and retrieves correctly once it carries the heading
that says what it is about. The prefix counts against the token budget, so it is
kept to the heading trail and nothing else.

## Overlap

Consecutive chunks repeat the last ``overlap_tokens`` of the previous chunk, so
an answer spanning a boundary is not lost by whichever side missed it. Overlap
costs storage and embedding spend linearly, which is why it defaults to a small
fraction of the target rather than to a quarter of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hardpoint.core.ids import chunk_id
from hardpoint.core.models import Block, CharSpan, Chunk, ParsedDocument, TrustLevel
from hardpoint.core.tokens import TokenCounter, estimate_tokens

if TYPE_CHECKING:
    from hardpoint.core.context import RunContext

__all__ = ["HEADING_SEPARATOR", "RecursiveChunker"]

HEADING_SEPARATOR = " > "
"""Joins the heading trail into a chunk's context prefix."""

# Sentence-ish. Deliberately not a full sentence tokenizer: this only decides
# where to cut a block that is already too large, and being slightly wrong there
# costs a slightly awkward boundary rather than a wrong answer.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Fewer than two chunks leaves nothing to merge an undersized one into.
_MERGEABLE_MINIMUM = 2


@dataclass(frozen=True)
class _Piece:
    """A unit of text the chunker may place, with where it came from."""

    text: str
    span: CharSpan
    headings: tuple[str, ...]
    tokens: int
    is_heading: bool = False


class RecursiveChunker:
    """Splits a parsed document into chunks, respecting its structure.

    Args:
        target_tokens: The size to aim for. Chunks come out at or below this
            except where a single indivisible piece exceeds it.
        overlap_tokens: How much of the previous chunk to repeat at the start of
            the next. Must be smaller than ``target_tokens``.
        min_tokens: Chunks smaller than this are merged into their neighbour
            where possible. A two-word chunk retrieves noise.
        include_heading_path: Whether to prefix each chunk with its heading
            trail. On by default; see the module docstring.
        count_tokens: How to measure. Defaults to the documented heuristic in
            ``core.tokens``; pass a model's tokenizer for accuracy.

    Raises:
        ValueError: If ``overlap_tokens`` is not smaller than ``target_tokens``.
            Equal or larger would make each chunk contain the whole of the
            previous one, and the chunker would never advance.
    """

    def __init__(
        self,
        *,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 16,
        include_heading_path: bool = True,
        count_tokens: TokenCounter = estimate_tokens,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be smaller than target_tokens "
                f"({target_tokens}); otherwise every chunk would contain the whole of "
                f"the previous one and chunking would never advance."
            )
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.include_heading_path = include_heading_path
        self.count_tokens = count_tokens

    async def chunk(self, doc: ParsedDocument, ctx: RunContext) -> list[Chunk]:
        """Split a document into chunks, in document order.

        Chunk ids come from ``ids.chunk_id``, so re-chunking unchanged text
        produces unchanged ids and the sync engine re-embeds nothing.

        Args:
            doc: The parsed document.
            ctx: The run context. Unused; present because the port declares it.

        Returns:
            The document's chunks, numbered from zero in document order.
        """
        pieces = list(self._pieces(doc))
        grouped = self._group(pieces)
        return [
            self._build(doc, index, body, span, headings)
            for index, (body, span, headings) in enumerate(grouped)
        ]

    # ----------------------------------------------------------------- #
    # Splitting                                                         #
    # ----------------------------------------------------------------- #

    def _pieces(self, doc: ParsedDocument) -> list[_Piece]:
        """Turn a document's blocks into placeable pieces, tracking the heading trail.

        A document with no blocks -- a parser that recovered no structure -- is
        treated as one piece per paragraph, so the chunker degrades to
        paragraph splitting rather than to nothing.
        """
        blocks = doc.blocks or self._synthetic_blocks(doc.text)
        pieces: list[_Piece] = []
        trail: list[str] = []

        for block in blocks:
            if block.kind == "heading":
                level = block.level or 1
                del trail[level - 1 :]
                trail.append(block.text)
                # The heading itself is placed, so a chunk starting at a heading
                # opens with it rather than with the text beneath it.
                pieces.append(self._piece(block.text, block.span, tuple(trail), is_heading=True))
                continue

            for text, span in self._split_oversized(block):
                pieces.append(self._piece(text, span, tuple(trail)))

        return pieces

    def _piece(
        self,
        text: str,
        span: CharSpan,
        headings: tuple[str, ...],
        *,
        is_heading: bool = False,
    ) -> _Piece:
        return _Piece(
            text=text,
            span=span,
            headings=headings,
            tokens=self.count_tokens(text),
            is_heading=is_heading,
        )

    @staticmethod
    def _synthetic_blocks(text: str) -> list[Block]:
        """Treat a structureless document as a sequence of paragraphs."""
        blocks: list[Block] = []
        position = 0
        for piece in re.split(r"\n[ \t]*\n", text):
            start = text.find(piece, position) if piece else position
            if start < 0:  # pragma: no cover - only if text was mutated underneath
                start = position
            position = start + len(piece)
            if piece.strip():
                blocks.append(
                    Block(
                        kind="paragraph",
                        text=piece.strip(),
                        span=CharSpan(start=start, end=position),
                    )
                )
        return blocks

    def _split_oversized(self, block: Block) -> list[tuple[str, CharSpan]]:
        """Split a block that exceeds the target, by sentence and then by word.

        A code block is never split by sentence: a fence cut in half is
        syntactically broken and embeds as noise. It is split by line instead,
        which at least leaves each fragment readable.
        """
        if self.count_tokens(block.text) <= self.target_tokens:
            return [(block.text, block.span)]

        separator = "\n" if block.kind == "code" else " "
        units = block.text.splitlines() if block.kind == "code" else _SENTENCE_END.split(block.text)
        units = [unit for unit in units if unit.strip()]

        parts: list[tuple[str, CharSpan]] = []
        current: list[str] = []
        for unit in units:
            candidate = separator.join([*current, unit])
            if current and self.count_tokens(candidate) > self.target_tokens:
                parts.append((separator.join(current), block.span))
                current = [unit]
            else:
                current.append(unit)

        if current:
            parts.append((separator.join(current), block.span))

        # A single unit still over the target: split on words, the last resort.
        final: list[tuple[str, CharSpan]] = []
        for text, span in parts:
            if self.count_tokens(text) <= self.target_tokens:
                final.append((text, span))
                continue
            final.extend((piece, span) for piece in self._split_words(text))
        return final

    def _split_words(self, text: str) -> list[str]:
        """Split on whitespace, the only split that always terminates."""
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            if current and self.count_tokens(" ".join([*current, word])) > self.target_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    # ----------------------------------------------------------------- #
    # Grouping                                                          #
    # ----------------------------------------------------------------- #

    def _group(self, pieces: list[_Piece]) -> list[tuple[str, CharSpan, tuple[str, ...]]]:
        """Pack pieces into chunks at or below the target, with overlap.

        A heading always starts a new chunk when one is already in progress,
        because a heading marks a topic change and the best available split
        point is exactly there.
        """
        chunks: list[tuple[str, CharSpan, tuple[str, ...]]] = []
        current: list[_Piece] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            body = "\n\n".join(piece.text for piece in current)
            span = CharSpan(start=current[0].span.start, end=current[-1].span.end)
            chunks.append((body, span, current[0].headings))
            current = self._carry_overlap(current)
            current_tokens = sum(piece.tokens for piece in current)

        for piece in pieces:
            # Any heading starts a new chunk when one is already in progress. An
            # earlier version inferred this from the heading trail's length, which
            # is backwards: a deeper heading has a *longer* trail than the text
            # above it, so `## Refunds` under `# Billing` never split and a whole
            # document collapsed into one chunk labelled with its first heading.
            starts_a_topic = bool(current) and piece.is_heading
            over_target = bool(current) and current_tokens + piece.tokens > self.target_tokens
            if starts_a_topic or over_target:
                flush()

            current.append(piece)
            current_tokens += piece.tokens

        if current:
            body = "\n\n".join(piece.text for piece in current)
            span = CharSpan(start=current[0].span.start, end=current[-1].span.end)
            chunks.append((body, span, current[0].headings))

        return self._merge_undersized(chunks)

    def _carry_overlap(self, placed: list[_Piece]) -> list[_Piece]:
        """Return the trailing pieces to repeat at the start of the next chunk."""
        if self.overlap_tokens <= 0:
            return []
        carried: list[_Piece] = []
        total = 0
        for piece in reversed(placed):
            if total + piece.tokens > self.overlap_tokens:
                break
            carried.insert(0, piece)
            total += piece.tokens
        return carried

    def _merge_undersized(
        self, chunks: list[tuple[str, CharSpan, tuple[str, ...]]]
    ) -> list[tuple[str, CharSpan, tuple[str, ...]]]:
        """Fold a chunk below ``min_tokens`` into its neighbour.

        A two-word chunk retrieves noise: it matches on a single term and
        contributes nothing to an answer. Merging is preferable to dropping,
        because the text still belongs to the document.

        **Merging never crosses a heading boundary.** A short section folded into
        the one above it inherits that section's heading path, so a chunk about
        disputes ends up labelled as being about refunds -- and the heading path
        is exactly what makes it retrievable. A short chunk with the right label
        beats a merged one with the wrong label, and the validator's
        ``TooShortRule`` is the backstop for chunks that are genuinely useless.
        """
        if self.min_tokens <= 0 or len(chunks) < _MERGEABLE_MINIMUM:
            return chunks

        merged: list[tuple[str, CharSpan, tuple[str, ...]]] = []
        for body, span, headings in chunks:
            undersized = self.count_tokens(body) < self.min_tokens
            same_section = bool(merged) and merged[-1][2] == headings
            if undersized and same_section:
                previous_body, previous_span, previous_headings = merged[-1]
                merged[-1] = (
                    f"{previous_body}\n\n{body}",
                    CharSpan(start=previous_span.start, end=span.end),
                    previous_headings,
                )
                continue
            merged.append((body, span, headings))
        return merged

    # ----------------------------------------------------------------- #
    # Building                                                          #
    # ----------------------------------------------------------------- #

    def _build(
        self,
        doc: ParsedDocument,
        index: int,
        body: str,
        span: CharSpan,
        headings: tuple[str, ...],
    ) -> Chunk:
        """Build a chunk, prefixing the heading trail and deriving its id.

        A chunk that begins at a heading already opens with that heading, so the
        prefix carries only its *ancestors*. Prefixing the full trail would
        repeat the heading immediately below itself, which reads badly and
        charges the token budget twice for the same words.
        """
        text = body
        if self.include_heading_path and headings:
            trail = headings[:-1] if body.startswith(headings[-1]) else headings
            prefix = HEADING_SEPARATOR.join(trail)
            if prefix and not body.startswith(prefix):
                text = f"{prefix}\n\n{body}"

        return Chunk(
            id=chunk_id(doc.document.id, index, text),
            document_id=doc.document.id,
            index=index,
            text=text,
            span=span,
            trust=TrustLevel.UNTRUSTED,
            metadata={
                **doc.document.metadata,
                "heading_path": HEADING_SEPARATOR.join(headings),
                "source_uri": doc.document.source_uri,
            },
            token_count=self.count_tokens(text),
        )

    def __repr__(self) -> str:
        """Render the sizes that determine the output."""
        return (
            f"RecursiveChunker(target_tokens={self.target_tokens}, "
            f"overlap_tokens={self.overlap_tokens})"
        )
