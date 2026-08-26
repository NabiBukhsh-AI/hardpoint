"""The metadata filter tree. **[LOCKED]** INSTRUCTIONS.md §5.3.

Two things must hold, and both are load-bearing for vector store portability:

1. **The tree is closed and serialisable.** It round-trips through JSON with no
   loss, so a filter can be logged, cached in a key, and reconstructed.
2. **An unsupported operator raises, never silently drops.** A filter that
   quietly stops filtering is how one tenant sees another tenant's documents,
   and it is exactly the failure a permissive adapter would produce.

The operator semantics tested here are normative. Every adapter must reproduce
them, and the ``VectorIndex`` contract kit checks that it does.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from hardpoint.core.errors import UnsupportedFilterError
from hardpoint.core.filters import (
    COMPARISON_OPS,
    STRUCTURAL_OPS,
    And,
    Comparison,
    F,
    Not,
    Or,
    matches,
    validate_supported,
)

ROW: dict[str, Any] = {
    "tenant": "acme",
    "year": 2024,
    "score": 0.75,
    "public": True,
    "tags": ["billing", "urgent"],
    "title": "Invoice handling",
    "empty": None,
}


# --------------------------------------------------------------------------- #
# Operator semantics                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (F.field("tenant").eq("acme"), True),
        (F.field("tenant").eq("other"), False),
        (F.field("tenant").ne("other"), True),
        (F.field("tenant").ne("acme"), False),
        (F.field("year").gt(2023), True),
        (F.field("year").gt(2024), False),
        (F.field("year").gte(2024), True),
        (F.field("year").lt(2025), True),
        (F.field("year").lte(2024), True),
        (F.field("score").gt(0.5), True),
        (F.field("score").lt(0.5), False),
        (F.field("tenant").in_(["acme", "other"]), True),
        (F.field("tenant").in_(["other"]), False),
        (F.field("tenant").nin(["other"]), True),
        (F.field("tenant").nin(["acme"]), False),
        (F.field("tags").contains("urgent"), True),
        (F.field("tags").contains("missing"), False),
        (F.field("title").contains("Invoice"), True),
        (F.field("title").contains("invoice"), False),
        (F.field("tenant").exists(), True),
        (F.field("nope").exists(), False),
        (F.field("empty").exists(), True),
    ],
)
def test_operator_semantics(expression: Any, expected: bool) -> None:
    assert matches(expression, ROW) is expected


def test_absent_field_never_satisfies_a_comparison() -> None:
    """Only ``exists`` observes absence. This is what stops a typo widening a filter.

    A missing tenant key must not make ``tenant != "other"`` true, or a
    mis-spelled metadata key would silently return every other tenant's rows.
    """
    for op in sorted(COMPARISON_OPS):
        value: Any = ["x"] if op in {"in", "nin"} else "x"
        expression = Comparison(field="missing", op=op, value=value)  # type: ignore[arg-type]
        assert matches(expression, ROW) is False, op


def test_null_valued_field_exists_but_does_not_equal_a_value() -> None:
    assert matches(F.field("empty").exists(), ROW) is True
    assert matches(F.field("empty").eq(None), ROW) is True
    assert matches(F.field("empty").eq("x"), ROW) is False


def test_bool_does_not_order_as_a_number() -> None:
    """``True > 0`` is a Python accident, not a filter semantic.

    Left unhandled, a boolean flag would compare as 1 and quietly satisfy
    numeric range filters it has nothing to do with.
    """
    assert matches(F.field("public").gt(0), ROW) is False
    assert matches(F.field("public").eq(True), ROW) is True


def test_incomparable_types_do_not_match_and_do_not_raise() -> None:
    """Heterogeneous metadata is normal; a query is not the place to discover it."""
    assert matches(F.field("tenant").gt(5), ROW) is False
    assert matches(F.field("year").lt("2020"), ROW) is False


def test_in_requires_a_list_value() -> None:
    assert matches(Comparison(field="tenant", op="in", value="acme"), ROW) is False
    assert matches(Comparison(field="tenant", op="nin", value="acme"), ROW) is True


# --------------------------------------------------------------------------- #
# Composition                                                                 #
# --------------------------------------------------------------------------- #


def test_and_or_not_compose() -> None:
    expression = (F.field("tenant").eq("acme") & F.field("year").gte(2023)) | F.field("public").eq(
        False
    )
    assert matches(expression, ROW) is True
    assert matches(~expression, ROW) is False


def test_empty_and_matches_everything_empty_or_matches_nothing() -> None:
    """The identity elements. A filter builder that accumulates zero clauses
    must widen to everything for ``And`` and narrow to nothing for ``Or``.
    """
    assert matches(And(), ROW) is True
    assert matches(Or(), ROW) is False


def test_chained_conjunction_stays_flat() -> None:
    """``a & b & c`` is one three-clause And, not a nested pair.

    Keeps the serialised form matching what the user wrote, which matters
    because the serialised form ends up in cache keys and in traces.
    """
    expression = F.field("a").eq(1) & F.field("b").eq(2) & F.field("c").eq(3)
    assert isinstance(expression, And)
    assert len(expression.clauses) == 3
    assert not any(isinstance(clause, And) for clause in expression.clauses)


def test_chained_disjunction_stays_flat() -> None:
    expression = F.field("a").eq(1) | F.field("b").eq(2) | F.field("c").eq(3)
    assert isinstance(expression, Or)
    assert len(expression.clauses) == 3


def test_mixed_operators_do_not_flatten_across_kinds() -> None:
    expression = (F.field("a").eq(1) | F.field("b").eq(2)) & F.field("c").eq(3)
    assert isinstance(expression, And)
    assert any(isinstance(clause, Or) for clause in expression.clauses)


def test_builder_helpers() -> None:
    clauses = (F.field("a").eq(1), F.field("b").eq(2))
    assert isinstance(F.all_of(*clauses), And)
    assert isinstance(F.any_of(*clauses), Or)
    assert isinstance(F.none_of(*clauses), Not)
    assert matches(F.none_of(F.field("tenant").eq("other")), ROW) is True


def test_f_is_a_namespace_not_an_object() -> None:
    with pytest.raises(TypeError, match="namespace"):
        F()


# --------------------------------------------------------------------------- #
# Serialisation                                                               #
# --------------------------------------------------------------------------- #


def test_round_trips_through_json() -> None:
    """A filter must survive a cache key, a trace attribute and a config file."""
    original = (
        F.field("tenant").eq("acme")
        & (F.field("year").gte(2023) | F.field("tags").contains("urgent"))
        & ~F.field("archived").exists()
    )
    restored = And.model_validate_json(original.model_dump_json())
    assert restored == original
    assert matches(restored, ROW) == matches(original, ROW)


def test_discriminator_is_present_on_every_node() -> None:
    dumped = (F.field("a").eq(1) & F.field("b").exists()).model_dump()
    assert dumped["type"] == "and"
    assert {clause["type"] for clause in dumped["clauses"]} == {"comparison", "exists"}


def test_nodes_are_frozen() -> None:
    node = F.field("a").eq(1)
    with pytest.raises(ValidationError, match=r"frozen|immutable"):
        node.field = "b"  # type: ignore[misc]


def test_nodes_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Comparison(field="a", op="eq", value=1, sneaky=True)  # type: ignore[call-arg]


def test_invalid_operator_is_rejected_at_construction() -> None:
    """A typo in an operator fails at construction, not as an empty result set."""
    with pytest.raises(ValidationError, match="op"):
        Comparison(field="a", op="approximately", value=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# required_ops and capability validation                                      #
# --------------------------------------------------------------------------- #


def test_required_ops_reports_leaf_and_structural_operators() -> None:
    expression = F.field("a").eq(1) & (F.field("b").gte(2) | ~F.field("c").exists())
    assert expression.required_ops() == {"and", "eq", "or", "gte", "not", "exists"}


def test_required_ops_of_an_empty_junction() -> None:
    assert And().required_ops() == {"and"}
    assert Or().required_ops() == {"or"}


def test_validate_supported_passes_when_every_operator_is_available() -> None:
    expression = F.field("a").eq(1) & F.field("b").gte(2)
    assert validate_supported(expression, COMPARISON_OPS | STRUCTURAL_OPS, "full") is None


def test_unsupported_operator_raises_and_names_operator_and_backend() -> None:
    """**[LOCKED]** Never silently drop an unsupported clause."""
    expression = F.field("tags").contains("urgent")
    with pytest.raises(UnsupportedFilterError) as exc_info:
        validate_supported(expression, {"eq", "ne"}, "toy_index")

    error = exc_info.value
    assert error.operator == "contains"
    assert error.backend == "toy_index"
    rendered = str(error)
    assert "contains" in rendered
    assert "toy_index" in rendered
    assert error.remedy is not None
    assert "eq" in error.remedy


def test_unsupported_structural_operator_also_raises() -> None:
    """A backend with every comparison but no ``or`` must still be caught."""
    expression = F.field("a").eq(1) | F.field("b").eq(2)
    with pytest.raises(UnsupportedFilterError) as exc_info:
        validate_supported(expression, COMPARISON_OPS, "no_disjunction")
    assert exc_info.value.operator == "or"


def test_validate_supported_reports_a_backend_that_supports_nothing() -> None:
    with pytest.raises(UnsupportedFilterError) as exc_info:
        validate_supported(F.field("a").eq(1), set(), "empty")
    assert "(none)" in (exc_info.value.remedy or "")


def test_operator_sets_are_disjoint_and_complete() -> None:
    assert COMPARISON_OPS.isdisjoint(STRUCTURAL_OPS)
    assert len(COMPARISON_OPS) == 9
    assert len(STRUCTURAL_OPS) == 4
