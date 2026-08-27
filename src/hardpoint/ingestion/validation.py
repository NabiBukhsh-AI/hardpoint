"""Chunk validation: a real stage, not a filter comprehension.

Implements the M1 validator set from INSTRUCTIONS.md §6.2. Empty chunks, chunks
below a minimum token count, boilerplate, and chunks exceeding the embedding
model's limit are dropped or quarantined **with a reason**
(ARCHITECTURE.md §14).

## Why rejections are reported rather than filtered

A chunk silently dropped is a chunk that will never be retrieved, and nobody
will know why the answer was incomplete. Every rejection here carries the rule
that rejected it and the text that was rejected, and the sync engine writes them
to a quarantine artefact a human can read.

The single most useful signal in that artefact is a document that produced a
hundred rejections: it means the parser chose wrong, or the chunker's target is
mis-set, and neither would otherwise surface at all.

## Drop or quarantine

A rejection either *drops* the chunk and keeps the document, or *quarantines*
the whole document. Dropping is right for a chunk that is merely useless -- a
row of dashes, a page number. Quarantining is right when the rejection suggests
the document was parsed wrongly, because indexing the rest of it would be
indexing something nobody has looked at.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from hardpoint.core.ids import normalise_text, text_hash
from hardpoint.core.models import Chunk
from hardpoint.core.tokens import TokenCounter, estimate_tokens

__all__ = [
    "BoilerplateRule",
    "ChunkValidator",
    "DuplicateRule",
    "EmptyRule",
    "Rejection",
    "TooLongRule",
    "TooShortRule",
    "ValidationReport",
    "ValidationRule",
    "default_rules",
]

Disposition = Literal["drop", "quarantine"]


@dataclass(frozen=True)
class Rejection:
    """One chunk a rule refused, and why.

    Args:
        chunk_id: The chunk's deterministic id.
        document_id: The document it came from.
        rule: The rule that rejected it, so an artefact can be grouped by cause.
        reason: A human-readable explanation.
        disposition: ``drop`` to discard the chunk, ``quarantine`` to reject the
            whole document.
        excerpt: The start of the offending text, so a human can see what was
            rejected without opening the source.
    """

    chunk_id: str
    document_id: str
    rule: str
    reason: str
    disposition: Disposition = "drop"
    excerpt: str = ""

    def __repr__(self) -> str:
        """Render the rule and the chunk it rejected."""
        return f"Rejection(rule={self.rule!r}, chunk={self.chunk_id!r})"


@dataclass(frozen=True)
class ValidationReport:
    """What validation kept and what it refused.

    Args:
        kept: The chunks that survived, in document order.
        rejections: Everything refused, in the order refused.
    """

    kept: tuple[Chunk, ...]
    rejections: tuple[Rejection, ...]

    @property
    def quarantines_document(self) -> bool:
        """Whether any rejection was severe enough to reject the whole document."""
        return any(rejection.disposition == "quarantine" for rejection in self.rejections)

    def __repr__(self) -> str:
        """Render the counts."""
        return f"ValidationReport(kept={len(self.kept)}, rejections={len(self.rejections)})"


class ValidationRule(Protocol):
    """A rule that may refuse a chunk.

    ``name`` is declared read-only rather than as a plain attribute: a bare
    ``name: str`` on a Protocol requires a *settable* attribute, which a frozen
    dataclass field is not, and the rules are frozen because they hold only
    configuration.
    """

    @property
    def name(self) -> str:
        """How the rule appears in a quarantine artefact."""
        ...

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Return a rejection, or ``None`` to accept.

        Args:
            chunk: The chunk under test.
            seen: Normalised text hashes of chunks already accepted in this
                document. Passed in rather than held on the rule so a rule stays
                stateless and one validator can be reused across documents.
        """
        ...


def _excerpt(text: str, limit: int = 120) -> str:
    """Return the start of a text, for a rejection record."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


@dataclass(frozen=True)
class EmptyRule:
    """Refuses a chunk that is empty or only whitespace.

    Always a drop rather than a quarantine: a stray blank chunk says nothing
    about whether the rest of the document parsed correctly.
    """

    name: str = "empty"

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Reject a chunk with no visible content."""
        if chunk.text.strip():
            return None
        return Rejection(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            rule=self.name,
            reason="The chunk contains no text.",
        )


@dataclass(frozen=True)
class TooShortRule:
    """Refuses a chunk below a minimum token count.

    A three-token chunk matches on a single term and contributes nothing to an
    answer, so it is retrieval noise that also costs an embedding.

    Args:
        min_tokens: The floor.
        count_tokens: How to measure.
    """

    min_tokens: int = 8
    count_tokens: TokenCounter = estimate_tokens
    name: str = "too_short"

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Reject a chunk shorter than the floor."""
        tokens = chunk.token_count
        if tokens is None:
            tokens = self.count_tokens(chunk.text)
        if tokens >= self.min_tokens:
            return None
        return Rejection(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            rule=self.name,
            reason=f"{tokens} tokens is below the {self.min_tokens}-token floor.",
            excerpt=_excerpt(chunk.text),
        )


@dataclass(frozen=True)
class TooLongRule:
    """Refuses a chunk above the embedding model's input limit.

    A **quarantine**, not a drop. A chunk over the limit means the chunker's
    target and the model's limit disagree, which affects the whole document and
    probably the whole corpus. Dropping it would index most of a document and
    hide the misconfiguration.

    Args:
        max_tokens: The ceiling, normally the embedding model's input limit.
        count_tokens: How to measure.
    """

    max_tokens: int = 8192
    count_tokens: TokenCounter = estimate_tokens
    name: str = "too_long"

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Reject a chunk longer than the ceiling."""
        tokens = chunk.token_count
        if tokens is None:
            tokens = self.count_tokens(chunk.text)
        if tokens <= self.max_tokens:
            return None
        return Rejection(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            rule=self.name,
            reason=(
                f"{tokens} tokens exceeds the {self.max_tokens}-token limit. Lower "
                f"the chunker's target_tokens, or configure an embedding model with "
                f"a larger input limit."
            ),
            disposition="quarantine",
            excerpt=_excerpt(chunk.text),
        )


@dataclass(frozen=True)
class DuplicateRule:
    """Refuses a chunk whose normalised text was already accepted.

    Compared on normalised text, so a repeated licence header formatted two ways
    is still recognised as one. Duplicates are common -- boilerplate footers,
    repeated tables -- and each one costs an embedding and then competes with
    the original in every result set.
    """

    name: str = "duplicate"

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Reject a chunk identical to one already kept in this document."""
        digest = text_hash(chunk.text)
        if digest not in seen:
            return None
        return Rejection(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            rule=self.name,
            reason="Identical text was already indexed for this document.",
            excerpt=_excerpt(chunk.text),
        )


@dataclass(frozen=True)
class BoilerplateRule:
    """Refuses a chunk that is structurally content-free.

    Catches the page numbers, rules of dashes and navigation crumbs that survive
    parsing. Judged by the ratio of alphanumeric characters rather than by a
    pattern list, because the shapes vary by format and a list would never be
    complete.

    Args:
        min_alphanumeric_ratio: Below this proportion of letters and digits, a
            chunk is treated as content-free.
    """

    min_alphanumeric_ratio: float = 0.25
    name: str = "boilerplate"

    def check(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        """Reject a chunk with too little alphanumeric content."""
        stripped = "".join(chunk.text.split())
        if not stripped:
            return None  # EmptyRule owns this case, and owning it twice is noise.

        ratio = sum(character.isalnum() for character in stripped) / len(stripped)
        if ratio >= self.min_alphanumeric_ratio:
            return None
        return Rejection(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            rule=self.name,
            reason=(
                f"Only {ratio:.0%} of the characters are alphanumeric, below the "
                f"{self.min_alphanumeric_ratio:.0%} floor."
            ),
            excerpt=_excerpt(chunk.text),
        )


def default_rules(
    *,
    min_tokens: int = 8,
    max_tokens: int = 8192,
    count_tokens: TokenCounter = estimate_tokens,
) -> tuple[ValidationRule, ...]:
    """Return the M1 rule set, in the order they should run.

    Ordered cheapest and most decisive first: an empty chunk should be reported
    as empty rather than as boilerplate, and a duplicate should not also be
    reported as too short.
    """
    rules: tuple[ValidationRule, ...] = (
        EmptyRule(),
        TooLongRule(max_tokens=max_tokens, count_tokens=count_tokens),
        TooShortRule(min_tokens=min_tokens, count_tokens=count_tokens),
        BoilerplateRule(),
        DuplicateRule(),
    )
    return rules


class ChunkValidator:
    """Runs the rules over a document's chunks.

    Args:
        rules: The rules to apply, in order. Defaults to :func:`default_rules`.

    Raises:
        Nothing. Validation reports; it does not raise.
    """

    def __init__(self, rules: Sequence[ValidationRule] | None = None) -> None:
        self.rules = tuple(rules) if rules is not None else default_rules()

    def validate(self, chunks: Iterable[Chunk]) -> ValidationReport:
        """Partition chunks into kept and rejected.

        The first rule to reject a chunk wins, and later rules are not consulted:
        a chunk reported under two rules is twice the artefact for one problem,
        and the first rule is the most specific by construction.

        Args:
            chunks: The document's chunks, in document order.

        Returns:
            What was kept and what was refused.
        """
        kept: list[Chunk] = []
        rejections: list[Rejection] = []
        seen: set[str] = set()

        for chunk in chunks:
            rejection = self._first_rejection(chunk, seen)
            if rejection is not None:
                rejections.append(rejection)
                continue
            seen.add(text_hash(chunk.text))
            kept.append(chunk)

        return ValidationReport(kept=tuple(kept), rejections=tuple(rejections))

    def _first_rejection(self, chunk: Chunk, seen: set[str]) -> Rejection | None:
        for rule in self.rules:
            rejection = rule.check(chunk, seen)
            if rejection is not None:
                return rejection
        return None

    def __repr__(self) -> str:
        """Render the rule names, in order."""
        return f"ChunkValidator(rules=[{', '.join(rule.name for rule in self.rules)}])"


def normalised_for_comparison(text: str) -> str:
    """Return the form duplicate detection compares on.

    Exposed so a caller writing its own duplicate rule agrees with this one
    about what "the same text" means.
    """
    return normalise_text(text)
