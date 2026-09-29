"""Evaluation datasets: ``EvalCase`` and JSONL (INSTRUCTIONS.md §8 **[LOCKED]**).

One case per line, so a pull request that adds three cases shows three added
lines and a reviewer can read them.

## Saying which documents are relevant

``expected_document_ids`` accepts real document ids (``doc_...``) and, because
nobody can compute a SHA-256 in their head, the form ``<source>:<path>`` --
``docs:api-keys.md`` -- which resolves through ``core.ids.document_id`` to the
same deterministic id ingestion assigned. Retrieval metrics then count a
retrieved chunk as relevant when its document is expected, or when the chunk id
itself is in ``expected_chunk_ids``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hardpoint.core.errors import ConfigError
from hardpoint.core.ids import document_id
from hardpoint.core.types import JsonValue

__all__ = ["EvalCase", "load_dataset", "resolve_document_ids", "write_dataset"]


class EvalCase(BaseModel):
    """One question and what a good answer to it looks like.

    Args:
        id: Stable identifier. Baselines and regression tables key on it.
        query: The question, exactly as a user would ask it.
        expected_chunk_ids: Chunks that should be retrieved.
        expected_document_ids: Documents that should be retrieved, as ids or as
            ``<source>:<path>``.
        reference_answer: A good answer, for judge metrics and for reading.
        metadata: Anything else, for slicing results.
        tags: Labels for filtering a suite, such as ``smoke`` or ``billing``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_chunk_ids: tuple[str, ...] = ()
    expected_document_ids: tuple[str, ...] = ()
    reference_answer: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def has_retrieval_labels(self) -> bool:
        """Whether retrieval metrics can be computed for this case."""
        return bool(self.expected_chunk_ids or self.expected_document_ids)


def resolve_document_ids(case: EvalCase) -> frozenset[str]:
    """The case's expected documents as real ids, resolving ``source:path`` forms."""
    resolved: set[str] = set()
    for value in case.expected_document_ids:
        source, separator, path = value.partition(":")
        if value.startswith("doc_") or not separator:
            resolved.add(value)
        else:
            resolved.add(document_id(source, path))
    return frozenset(resolved)


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Read a JSONL dataset.

    Raises:
        ConfigError: For a missing file, a line that is not a valid case -- named
            by file and line -- or two cases sharing an id.
    """
    source = Path(path)
    if not source.is_file():
        raise ConfigError(
            f"No evaluation dataset at {source}.",
            config_path="eval.datasets_dir",
            remedy=(
                f"Create {source} with one JSON object per line, for example\n"
                '{"id": "rotate-key", "query": "How do I rotate an API key?", '
                '"expected_document_ids": ["docs:api-keys.md"]}'
            ),
        )

    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise ConfigError(
                f"{source} line {number} is not a valid evaluation case: {exc}",
                config_path=str(source),
                remedy="Each line must be one JSON object with at least `id` and `query`.",
                cause=exc,
            ) from exc
        if case.id in seen:
            raise ConfigError(
                f"{source} line {number} reuses the case id {case.id!r} from line {seen[case.id]}.",
                config_path=str(source),
                remedy="Case ids key baselines and regression tables, so each must be unique.",
            )
        seen[case.id] = number
        cases.append(case)
    return cases


def write_dataset(path: str | Path, cases: Iterable[EvalCase]) -> Path:
    """Write cases as JSONL, one per line, keys sorted so diffs stay small."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: Sequence[str] = [
        json.dumps(case.model_dump(mode="json", exclude_defaults=True), sort_keys=True)
        for case in cases
    ]
    destination.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return destination
