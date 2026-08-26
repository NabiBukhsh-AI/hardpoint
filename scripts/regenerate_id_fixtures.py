"""Regenerate the deterministic-id golden fixture.

The fixture at ``tests/fixtures/ids_golden.json`` pins the output of every
function in ``hardpoint.core.ids`` for a fixed set of inputs. Its purpose is to
make an accidental change to the identity algorithm fail loudly rather than
silently orphan every chunk in every existing index.

**Running this script is a deliberate act.** It should happen only when the
identity algorithm is being changed on purpose, and that change must be
accompanied by a domain-label version bump in ``ids.py``, a CHANGELOG entry, and
a migration note explaining that existing indexes must be rebuilt.

Regenerating it to make a failing test pass is the mistake the fixture exists to
prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hardpoint.core import ids

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ids_golden.json"

# Inputs chosen to exercise the cases most likely to break: ASCII, non-ASCII,
# NFKC-foldable forms, whitespace variants, empty strings, and separator
# characters that a naive concatenation would let collide.
NORMALISE_CASES: list[str] = [
    "",
    "hello",
    "  leading and trailing  ",
    "collapse\t\tinner\n\nwhitespace",
    "non breaking space",
    "ﬁ ligature and ４ fullwidth digit",
    "café",  # already NFC
    "café",  # NFD, folds to the same string under NFKC
    "ünïcödé mixed em space",
    "line\r\nendings\rvary",
]

DOCUMENT_CASES: list[tuple[str, str]] = [
    ("docs", "guide.md"),
    ("docs", "guide.md#section"),
    ("docs", "a/b.md"),
    ("handbook", "guide.md"),
    # The next pair would collide under naive concatenation without a separator.
    ("a", "bc"),
    ("ab", "c"),
    ("", ""),
    ("docs", "文書.md"),
]

CHUNK_CASES: list[tuple[str, int, str]] = [
    ("doc_000000000000000000000000", 0, "hello world"),
    ("doc_000000000000000000000000", 1, "hello world"),
    ("doc_000000000000000000000000", 0, "  hello   world  "),
    ("doc_000000000000000000000000", 0, "hello worlds"),
    ("doc_111111111111111111111111", 0, "hello world"),
    ("doc_000000000000000000000000", 42, ""),
    ("doc_000000000000000000000000", 0, "ünïcödé chunk"),
]

CONTENT_CASES: list[bytes] = [
    b"",
    b"hello world",
    b"\x00\x01\x02",
    "café".encode(),
]

STABLE_CASES: list[object] = [
    None,
    True,
    0,
    1,
    1.0,
    "",
    "hello",
    [],
    [1, 2, 3],
    {},
    {"a": 1, "b": 2},
    {"b": 2, "a": 1},  # same value as above, must hash identically
    {"nested": {"z": [1, {"y": None}], "a": "café"}},
]


def build() -> dict[str, object]:
    """Compute every golden value from the current implementation."""
    return {
        "_comment": (
            "Golden fixture for hardpoint.core.ids. Regenerating this file changes "
            "every identifier in every existing index. See "
            "scripts/regenerate_id_fixtures.py before touching it."
        ),
        "normalise_text": [
            {"text": case, "expected": ids.normalise_text(case)} for case in NORMALISE_CASES
        ],
        "document_id": [
            {"source_id": s, "source_uri": u, "expected": ids.document_id(s, u)}
            for s, u in DOCUMENT_CASES
        ],
        "chunk_id": [
            {
                "document_id": d,
                "index": i,
                "text": t,
                "expected": ids.chunk_id(d, i, t),
            }
            for d, i, t in CHUNK_CASES
        ],
        "content_hash": [
            {"data_hex": data.hex(), "expected": ids.content_hash(data)} for data in CONTENT_CASES
        ],
        "text_hash": [{"text": case, "expected": ids.text_hash(case)} for case in NORMALISE_CASES],
        "stable_hash": [
            {"value": case, "expected": ids.stable_hash(case)}  # type: ignore[arg-type]
            for case in STABLE_CASES
        ],
    }


def main() -> int:
    """Write the fixture, refusing unless ``--yes-i-mean-it`` was passed."""
    if "--yes-i-mean-it" not in sys.argv:
        sys.stderr.write(
            "Refusing to regenerate the identity golden fixture.\n\n"
            "Rewriting it changes every chunk id in every existing index, and every\n"
            "index built with the old algorithm becomes unreachable.\n\n"
            "If the algorithm is genuinely changing, bump the domain labels in\n"
            "src/hardpoint/core/ids.py, add a CHANGELOG entry with a migration note,\n"
            "then re-run:\n\n"
            "    uv run python scripts/regenerate_id_fixtures.py --yes-i-mean-it\n"
        )
        return 2

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(build(), indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"wrote {FIXTURE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
