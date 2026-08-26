r"""Deterministic identity and hashing (INSTRUCTIONS.md §5.2, **[LOCKED]**).

Content-addressed identity is what makes idempotent re-ingestion, incremental
re-embedding, correct deletes and safe caching possible (ADR-009). Every
function here is a pure function of its arguments and must produce the same
output on every machine, in every process, on every supported Python version.

Consequently this module never uses :func:`hash`, :func:`uuid.uuid4`, the wall
clock, the locale, or any iteration order that is not explicitly sorted
(INSTRUCTIONS.md §13.6).

## The algorithm, stated so it can be reimplemented

Every derived value is ``sha256`` over a byte string built the same way:

1. A domain label, so that two different kinds of identifier can never collide
   even when their inputs coincide. The label carries a version suffix.
2. The label and each argument joined with a single NUL byte, ``b"\x00"``.
3. Text arguments encoded as UTF-8. Integers rendered in decimal, ASCII.

NUL is used as the separator because it cannot occur in a Python ``str``'s UTF-8
encoding of any character that appears in a URI or in normalised text, so the
joined form is unambiguous.

Changing any of this changes every identifier in every existing index. The
golden fixture at ``tests/fixtures/ids_golden.json`` exists so that such a
change fails loudly instead of silently orphaning a corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from hardpoint.core.types import JsonValue

__all__ = [
    "CHUNK_ID_PREFIX",
    "DOCUMENT_ID_PREFIX",
    "ID_HEX_LENGTH",
    "chunk_id",
    "content_hash",
    "document_id",
    "normalise_text",
    "stable_hash",
    "text_hash",
]

DOCUMENT_ID_PREFIX = "doc_"
"""Prefix on every document identifier, so an id is readable in a log."""

CHUNK_ID_PREFIX = "chk_"
"""Prefix on every chunk identifier."""

ID_HEX_LENGTH = 24
"""Hex characters of digest kept in an identifier.

24 hex characters is 96 bits. At one billion chunks the probability of any
collision is below 1e-10, which is several orders of magnitude below the
probability of a silent disk error over the same corpus, while keeping ids short
enough to read in a trace.
"""

_SEPARATOR = b"\x00"
_DOCUMENT_DOMAIN = b"hardpoint/document/v1"
_CHUNK_DOMAIN = b"hardpoint/chunk/v1"
_TEXT_DOMAIN = b"hardpoint/text/v1"
_STABLE_DOMAIN = b"hardpoint/stable/v1"

_WHITESPACE_RUN = re.compile(r"\s+")


def _digest(domain: bytes, *parts: bytes) -> str:
    """Return the full hex sha256 of ``domain`` and ``parts`` joined by NUL."""
    hasher = hashlib.sha256()
    hasher.update(domain)
    for part in parts:
        hasher.update(_SEPARATOR)
        hasher.update(part)
    return hasher.hexdigest()


def normalise_text(text: str) -> str:
    """Normalise text so that cosmetically different inputs hash identically.

    Applies Unicode NFKC normalisation, collapses every run of whitespace to a
    single space, and strips the ends. NFKC is what folds a non-breaking space,
    a full-width digit or a ligature onto its ordinary form, so a document
    re-exported by a different tool does not look like a rewrite.

    This does **not** lowercase, strip punctuation, or remove diacritics. Those
    change meaning, and chunk text is shown to a model and cited to a user.

    Args:
        text: Arbitrary text.

    Returns:
        The normalised form. Empty input returns an empty string.

    Raises:
        Nothing.
    """
    return _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def content_hash(data: bytes) -> str:
    """Return the sha256 hex digest of raw bytes.

    Used for ``Document.content_hash``, which is what lets ingestion skip a
    document whose source revision changed but whose bytes did not.

    Args:
        data: The raw bytes as fetched from the source, before any decoding.

    Returns:
        A 64-character lowercase hex digest.

    Raises:
        Nothing.
    """
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of ``text`` after normalisation.

    This is the value the ingestion manifest stores per chunk to decide whether
    a chunk needs re-embedding. It is separate from :func:`content_hash` because
    it is defined over normalised text rather than raw bytes.

    Args:
        text: The chunk text.

    Returns:
        A 64-character lowercase hex digest.

    Raises:
        Nothing.
    """
    return _digest(_TEXT_DOMAIN, normalise_text(text).encode("utf-8"))


def document_id(source_id: str, source_uri: str) -> str:
    """Derive a stable document id from the source that produced it.

    The pair is used rather than the URI alone so that the same file reachable
    through two configured sources yields two documents, which is what makes
    per-source manifests and per-source deletes correct.

    Args:
        source_id: Identifier of the configured source, for example ``"docs"``.
        source_uri: The document's URI within that source.

    Returns:
        ``"doc_"`` followed by 24 hex characters.

    Raises:
        Nothing.
    """
    digest = _digest(_DOCUMENT_DOMAIN, source_id.encode("utf-8"), source_uri.encode("utf-8"))
    return f"{DOCUMENT_ID_PREFIX}{digest[:ID_HEX_LENGTH]}"


def chunk_id(document_id: str, index: int, text: str) -> str:
    """Derive a stable chunk id from its document, position and normalised text.

    Identity is content-addressed over all three (ADR-009). Including the index
    means two identical paragraphs in one document remain distinct chunks;
    including the normalised text means an edited paragraph becomes a new id, so
    the sync engine re-embeds it and retires the old one.

    Args:
        document_id: The owning document's id.
        index: Zero-based position of the chunk within the document.
        text: The chunk text, normalised internally before hashing.

    Returns:
        ``"chk_"`` followed by 24 hex characters.

    Raises:
        ValueError: If ``index`` is negative.
    """
    if index < 0:
        raise ValueError(f"chunk index must be non-negative, got {index}")
    digest = _digest(
        _CHUNK_DOMAIN,
        document_id.encode("utf-8"),
        str(index).encode("ascii"),
        normalise_text(text).encode("utf-8"),
    )
    return f"{CHUNK_ID_PREFIX}{digest[:ID_HEX_LENGTH]}"


def stable_hash(obj: JsonValue) -> str:
    """Hash a JSON-compatible value canonically.

    Object keys are sorted, separators carry no whitespace, and non-ASCII
    characters are emitted literally as UTF-8, so two structurally equal values
    hash identically regardless of how they were constructed. Used for the
    config snapshot hash and for cache keys over structured parameters.

    ``1`` and ``1.0`` hash differently, deliberately: JSON distinguishes them,
    and a cache key that conflated them would collide across genuinely different
    requests.

    Args:
        obj: Any JSON-compatible value.

    Returns:
        A 64-character lowercase hex digest.

    Raises:
        ValueError: If the value contains ``NaN`` or an infinity, which have no
            canonical JSON form.
        TypeError: If the value contains a type JSON cannot represent.
    """
    canonical = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _digest(_STABLE_DOMAIN, canonical.encode("utf-8"))
