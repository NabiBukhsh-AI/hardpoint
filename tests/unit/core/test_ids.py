"""Deterministic identity. **[LOCKED]** INSTRUCTIONS.md §5.2.

Three layers of protection, because chunk identity is the one thing in this
library that cannot be fixed after the fact: a change to it silently orphans
every chunk in every existing index rather than raising anything.

1. **Golden fixture.** ``tests/fixtures/ids_golden.json`` pins known
   input/output pairs. Any change to the algorithm fails these.
2. **Independent recomputation.** The golden fixture alone is not enough,
   because an agent or a hurried human can regenerate it in the same commit
   that changes the algorithm. So a second implementation, written here
   directly from the byte recipe in the module docstring, must agree with the
   real one. Changing the algorithm now requires changing three things that
   were deliberately written to disagree.
3. **Property tests.** Stability across processes, domain separation, and the
   absence of ``hash()``/``uuid``/clock in the source.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from hardpoint.core import ids

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "ids_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Golden fixture                                                           #
# --------------------------------------------------------------------------- #


def test_fixture_exists_and_is_populated(golden: dict[str, Any]) -> None:
    for section in (
        "normalise_text",
        "document_id",
        "chunk_id",
        "content_hash",
        "text_hash",
        "stable_hash",
    ):
        assert golden[section], f"golden fixture section {section!r} is empty"


def test_golden_normalise_text(golden: dict[str, Any]) -> None:
    for case in golden["normalise_text"]:
        assert ids.normalise_text(case["text"]) == case["expected"], case


def test_golden_document_id(golden: dict[str, Any]) -> None:
    for case in golden["document_id"]:
        assert ids.document_id(case["source_id"], case["source_uri"]) == case["expected"], case


def test_golden_chunk_id(golden: dict[str, Any]) -> None:
    for case in golden["chunk_id"]:
        actual = ids.chunk_id(case["document_id"], case["index"], case["text"])
        assert actual == case["expected"], case


def test_golden_content_hash(golden: dict[str, Any]) -> None:
    for case in golden["content_hash"]:
        assert ids.content_hash(bytes.fromhex(case["data_hex"])) == case["expected"], case


def test_golden_text_hash(golden: dict[str, Any]) -> None:
    for case in golden["text_hash"]:
        assert ids.text_hash(case["text"]) == case["expected"], case


def test_golden_stable_hash(golden: dict[str, Any]) -> None:
    for case in golden["stable_hash"]:
        assert ids.stable_hash(case["value"]) == case["expected"], case


# --------------------------------------------------------------------------- #
# 2. Independent recomputation from the documented recipe                     #
# --------------------------------------------------------------------------- #
#
# Written from the module docstring, not from the implementation. If these
# disagree with hardpoint.core.ids, one of the two is wrong and the golden
# fixture cannot arbitrate.


def reference_digest(domain: bytes, *parts: bytes) -> str:
    """sha256 over the domain label and each part, joined by a single NUL."""
    payload = domain
    for part in parts:
        payload += b"\x00" + part
    return hashlib.sha256(payload).hexdigest()


def reference_normalise(text: str) -> str:
    """NFKC, collapse whitespace runs to one space, strip."""
    folded = unicodedata.normalize("NFKC", text)
    return " ".join(folded.split())


def test_reference_normalise_agrees() -> None:
    samples = [
        "",
        "plain",
        "  spaced  ",
        "tabs\tand\nnewlines",
        "ﬁ ４",
        "café",
        " nbsp ",
        " em space",
    ]
    for sample in samples:
        assert ids.normalise_text(sample) == reference_normalise(sample), repr(sample)


def test_reference_document_id_agrees() -> None:
    for source_id, source_uri in [("docs", "a.md"), ("", ""), ("a", "bc"), ("ab", "c")]:
        expected = reference_digest(
            b"hardpoint/document/v1",
            source_id.encode("utf-8"),
            source_uri.encode("utf-8"),
        )[: ids.ID_HEX_LENGTH]
        assert ids.document_id(source_id, source_uri) == f"doc_{expected}"


def test_reference_chunk_id_agrees() -> None:
    for document_id, index, text in [
        ("doc_abc", 0, "hello"),
        ("doc_abc", 7, "  hello  world "),
        ("doc_xyz", 0, "ünïcödé"),
    ]:
        expected = reference_digest(
            b"hardpoint/chunk/v1",
            document_id.encode("utf-8"),
            str(index).encode("ascii"),
            reference_normalise(text).encode("utf-8"),
        )[: ids.ID_HEX_LENGTH]
        assert ids.chunk_id(document_id, index, text) == f"chk_{expected}"


def test_reference_content_hash_agrees() -> None:
    for data in [b"", b"hello", bytes(range(256))]:
        assert ids.content_hash(data) == hashlib.sha256(data).hexdigest()


def test_reference_stable_hash_agrees() -> None:
    for value in [None, 1, "x", [1, "a"], {"b": 1, "a": 2}]:
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        expected = reference_digest(b"hardpoint/stable/v1", canonical.encode("utf-8"))
        assert ids.stable_hash(value) == expected


# --------------------------------------------------------------------------- #
# 3. Properties                                                               #
# --------------------------------------------------------------------------- #


def test_ids_are_stable_across_processes() -> None:
    """A fresh interpreter must derive the same ids.

    This is what ``hash()`` would break: PYTHONHASHSEED randomisation makes
    ``hash()`` differ between processes, so an id built on it would change on
    every restart. The subprocess runs with hash randomisation forced on.
    """
    program = (
        "import json;from hardpoint.core import ids;"
        "print(json.dumps([ids.document_id('docs','a.md'),"
        "ids.chunk_id('doc_x',3,'hello world'),"
        "ids.text_hash('hello'),"
        "ids.stable_hash({'a':1,'b':[2,3]})]))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "", "SYSTEMROOT": ""},
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"ids differed across hash seeds: {outputs}"


def test_document_and_chunk_domains_do_not_collide() -> None:
    """A document id and a chunk id built from the same bytes must differ.

    Without the domain label they would be the same digest with a different
    prefix, and a truncation bug would let one be mistaken for the other.
    """
    doc = ids.document_id("a", "b")
    chunk = ids.chunk_id("a", 0, "b")
    assert doc.removeprefix("doc_") != chunk.removeprefix("chk_")


def test_separator_prevents_argument_smearing() -> None:
    """('a','bc') and ('ab','c') must not produce the same document id.

    Naive concatenation would make them identical, which would silently merge
    two different documents into one manifest row.
    """
    assert ids.document_id("a", "bc") != ids.document_id("ab", "c")


def test_chunk_id_changes_with_each_component() -> None:
    base = ids.chunk_id("doc_a", 0, "text")
    assert ids.chunk_id("doc_b", 0, "text") != base
    assert ids.chunk_id("doc_a", 1, "text") != base
    assert ids.chunk_id("doc_a", 0, "other") != base


def test_chunk_id_ignores_cosmetic_whitespace_and_unicode_form() -> None:
    """An edit that changes nothing visible must not force a re-embed.

    Re-embedding a corpus because an exporter changed its line endings is the
    kind of avoidable bill this normalisation exists to prevent.
    """
    base = ids.chunk_id("doc_a", 0, "hello world")
    assert ids.chunk_id("doc_a", 0, "  hello   world  ") == base
    assert ids.chunk_id("doc_a", 0, "hello\tworld") == base
    assert ids.chunk_id("doc_a", 0, "hello world") == base


def test_chunk_id_is_sensitive_to_real_edits() -> None:
    base = ids.chunk_id("doc_a", 0, "hello world")
    assert ids.chunk_id("doc_a", 0, "hello worlds") != base
    assert ids.chunk_id("doc_a", 0, "Hello world") != base, "case is meaning, not cosmetics"


def test_id_shapes() -> None:
    doc = ids.document_id("s", "u")
    chunk = ids.chunk_id(doc, 0, "t")
    assert doc.startswith(ids.DOCUMENT_ID_PREFIX)
    assert chunk.startswith(ids.CHUNK_ID_PREFIX)
    assert len(doc) == len(ids.DOCUMENT_ID_PREFIX) + ids.ID_HEX_LENGTH
    assert len(chunk) == len(ids.CHUNK_ID_PREFIX) + ids.ID_HEX_LENGTH
    assert all(c in "0123456789abcdef" for c in doc.removeprefix(ids.DOCUMENT_ID_PREFIX))


def test_negative_chunk_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ids.chunk_id("doc_a", -1, "text")


def test_stable_hash_is_insensitive_to_key_order() -> None:
    assert ids.stable_hash({"a": 1, "b": 2}) == ids.stable_hash({"b": 2, "a": 1})


def test_stable_hash_distinguishes_int_from_float() -> None:
    """1 and 1.0 must hash differently; a cache key that conflated them would collide."""
    assert ids.stable_hash(1) != ids.stable_hash(1.0)


def test_stable_hash_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match=r"[Nn]a[Nn]|not JSON compliant"):
        ids.stable_hash(float("nan"))
    with pytest.raises(ValueError, match=r"[Ii]nf|not JSON compliant"):
        ids.stable_hash(float("inf"))


def test_stable_hash_rejects_unrepresentable_types() -> None:
    with pytest.raises(TypeError):
        ids.stable_hash({"when": object()})  # type: ignore[dict-item]


def test_content_hash_is_plain_sha256_hex() -> None:
    assert len(ids.content_hash(b"x")) == 64
    assert ids.content_hash(b"") == hashlib.sha256(b"").hexdigest()


FORBIDDEN_MODULES = frozenset({"uuid", "time", "datetime", "random", "secrets", "os"})


def test_ids_module_calls_no_unstable_builtin() -> None:
    """The source must never call ``hash()``.

    INSTRUCTIONS.md §13.6 **[LOCKED]**. Checked by parsing the module rather
    than by string search, so that ``content_hash`` is not a false positive and
    a call hidden in a nested function is not a false negative. The failure mode
    guarded against is a future edit, not the current code.
    """
    tree = ast.parse(Path(ids.__file__).read_text(encoding="utf-8"))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "hash" not in calls, (
        "ids.py called the built-in hash(), which is randomised per process and "
        "would make identifiers unstable across restarts."
    )


def test_ids_module_imports_no_source_of_nondeterminism() -> None:
    """The source must not import the clock, a random source, or the environment."""
    tree = ast.parse(Path(ids.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    offenders = sorted(imported & FORBIDDEN_MODULES)
    assert not offenders, (
        f"ids.py imported {offenders}, which cannot appear in a module whose "
        "outputs must be identical on every machine and in every process."
    )
