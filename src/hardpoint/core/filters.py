"""Metadata filters: a closed, serialisable expression tree (**[LOCKED]**).

Implements INSTRUCTIONS.md §5.3. This is the single most important detail for
vector store portability, because "filters work differently" is what actually
blocks migrations (ARCHITECTURE.md §9.2).

The tree is *closed*: there are five node types and nine comparison operators,
and there is no escape hatch for a backend-specific expression. An adapter
translates the tree into its backend's own filter language, and an operator the
backend cannot express raises :class:`UnsupportedFilterError` at query
construction. A clause is **never** silently dropped: a filter that quietly
stops filtering is how one tenant sees another tenant's documents.

This module also owns the *meaning* of each operator, in :func:`matches`. That
makes the semantics normative rather than per-adapter folklore: the reference
implementation in ``hardpoint.testing.InMemoryVectorIndex`` uses it, and the
``VectorIndex`` contract kit checks every adapter against the same expectations.

## Semantics

A filter is evaluated against a flat metadata mapping.

- A field that is absent never satisfies a comparison. Only ``exists`` observes
  absence, and ``not(exists(f))`` is how you ask for it.
- ``eq`` and ``ne`` compare for equality. ``ne`` on an absent field is false,
  not true: absence is not inequality.
- ``gt``, ``gte``, ``lt``, ``lte`` order numbers against numbers and strings
  against strings. Any other pairing does not match, rather than raising,
  because heterogeneous metadata is normal and a query is not the place to
  discover it. ``bool`` is not a number here, so ``True > 0`` does not match.
- ``in`` and ``nin`` test membership of the *value*, which must be a list.
- ``contains`` tests that a string field contains the value as a substring, or
  that a list field has the value as an element.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.errors import UnsupportedFilterError
from hardpoint.core.types import JsonValue

__all__ = [
    "COMPARISON_OPS",
    "STRUCTURAL_OPS",
    "And",
    "Comparison",
    "ComparisonOp",
    "Exists",
    "F",
    "Filter",
    "FilterOp",
    "Not",
    "Or",
    "matches",
    "validate_supported",
]

ComparisonOp: TypeAlias = Literal["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains"]
"""The nine leaf operators. Fixed by INSTRUCTIONS.md §5.3."""

StructuralOp: TypeAlias = Literal["and", "or", "not", "exists"]
"""The four non-leaf operators. A backend may support comparisons but not ``or``."""

FilterOp: TypeAlias = Union[ComparisonOp, StructuralOp]  # noqa: UP007 - Literal union alias
"""Every operator name that can appear in :meth:`required_ops`."""

COMPARISON_OPS: frozenset[str] = frozenset(
    {"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains"}
)
"""Runtime-inspectable set of the comparison operators."""

STRUCTURAL_OPS: frozenset[str] = frozenset({"and", "or", "not", "exists"})
"""Runtime-inspectable set of the structural operators."""


class _Node(BaseModel):
    """Shared behaviour for the five filter node types.

    This is a base for one closed expression tree, not a general component base
    class: it exists so the tree can be serialised and combined, and nothing
    outside this module inherits from it (INSTRUCTIONS.md §13.4).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def required_ops(self) -> frozenset[str]:
        """Return every operator this expression uses, including structural ones.

        An adapter compares this against its declared ``IndexCapabilities`` to
        decide whether it can express the filter at all.
        """
        raise NotImplementedError  # pragma: no cover - every subclass overrides

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Evaluate this expression against a metadata mapping."""
        raise NotImplementedError  # pragma: no cover - every subclass overrides

    def __and__(self, other: Filter) -> And:
        """Combine two expressions with a conjunction, flattening nested ``And``."""
        return And(clauses=[*_flatten(self, And), *_flatten(other, And)])

    def __or__(self, other: Filter) -> Or:
        """Combine two expressions with a disjunction, flattening nested ``Or``."""
        return Or(clauses=[*_flatten(self, Or), *_flatten(other, Or)])

    def __invert__(self) -> Not:
        """Negate this expression."""
        return Not(clause=self)  # type: ignore[arg-type]  # self is always a Filter member


class Comparison(_Node):
    """A leaf comparison between a metadata field and a literal value.

    Args:
        field: The metadata key to read.
        op: One of the nine comparison operators.
        value: The literal to compare against. For ``in`` and ``nin`` it must be
            a list.
    """

    type: Literal["comparison"] = "comparison"
    field: str
    op: ComparisonOp
    value: JsonValue = None

    def required_ops(self) -> frozenset[str]:
        """Return the single operator this leaf uses."""
        return frozenset({self.op})

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Evaluate this comparison. An absent field never matches."""
        if self.field not in metadata:
            return False
        return _compare(metadata[self.field], self.op, self.value)


class Exists(_Node):
    """Tests whether a metadata field is present.

    Presence, not truthiness: a field set to ``null``, ``0`` or ``""`` exists.
    """

    type: Literal["exists"] = "exists"
    field: str

    def required_ops(self) -> frozenset[str]:
        """Return ``{"exists"}``."""
        return frozenset({"exists"})

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Return whether the field is present in the mapping."""
        return self.field in metadata


class And(_Node):
    """Conjunction. An empty clause list matches everything."""

    type: Literal["and"] = "and"
    clauses: Sequence[Filter] = ()

    def required_ops(self) -> frozenset[str]:
        """Return ``{"and"}`` plus the operators of every clause."""
        return frozenset({"and"}).union(*(c.required_ops() for c in self.clauses), frozenset())

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Return whether every clause matches."""
        return all(clause.matches(metadata) for clause in self.clauses)


class Or(_Node):
    """Disjunction. An empty clause list matches nothing."""

    type: Literal["or"] = "or"
    clauses: Sequence[Filter] = ()

    def required_ops(self) -> frozenset[str]:
        """Return ``{"or"}`` plus the operators of every clause."""
        return frozenset({"or"}).union(*(c.required_ops() for c in self.clauses), frozenset())

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Return whether at least one clause matches."""
        return any(clause.matches(metadata) for clause in self.clauses)


class Not(_Node):
    """Negation of a single expression."""

    type: Literal["not"] = "not"
    clause: Filter

    def required_ops(self) -> frozenset[str]:
        """Return ``{"not"}`` plus the operators of the negated expression."""
        return frozenset({"not"}) | self.clause.required_ops()

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Return the negation of the inner expression."""
        return not self.clause.matches(metadata)


Filter: TypeAlias = Annotated[
    Union[Comparison, Exists, And, Or, Not],  # noqa: UP007 - discriminated union needs Union
    Field(discriminator="type"),
]
"""The filter expression tree. Discriminated on ``type`` so it round-trips JSON."""

And.model_rebuild()
Or.model_rebuild()
Not.model_rebuild()


def _flatten(node: Filter | _Node, kind: type[And] | type[Or]) -> list[Filter]:
    """Return ``node``'s clauses if it is the same kind of junction, else ``[node]``.

    Keeps ``a & b & c`` a three-clause ``And`` rather than a nested pair, so the
    serialised form matches what the user wrote.
    """
    if isinstance(node, kind):
        return list(node.clauses)
    return [node]  # type: ignore[list-item]  # node is always a Filter member


# --------------------------------------------------------------------------- #
# Operator semantics                                                          #
# --------------------------------------------------------------------------- #


def _is_number(value: object) -> bool:
    """Return whether the value orders as a number. ``bool`` deliberately does not."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _ordered(left: JsonValue, right: JsonValue) -> int | None:
    """Return -1, 0 or 1 for comparable pairs, or ``None`` when not comparable."""
    if _is_number(left) and _is_number(right):
        numeric_left, numeric_right = float(left), float(right)  # type: ignore[arg-type]
        if numeric_left == numeric_right:
            return 0
        return -1 if numeric_left < numeric_right else 1
    if isinstance(left, str) and isinstance(right, str):
        if left == right:
            return 0
        return -1 if left < right else 1
    return None


def _compare(actual: JsonValue, op: ComparisonOp, expected: JsonValue) -> bool:  # noqa: PLR0911
    """Apply one comparison operator. Never raises on a type mismatch."""
    if op == "eq":
        return bool(actual == expected)
    if op == "ne":
        return bool(actual != expected)
    if op in {"in", "nin"}:
        member = isinstance(expected, list) and actual in expected
        return member if op == "in" else not member
    if op == "contains":
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        return isinstance(actual, list) and expected in actual

    order = _ordered(actual, expected)
    if order is None:
        return False
    if op == "gt":
        return order > 0
    if op == "gte":
        return order >= 0
    if op == "lt":
        return order < 0
    return order <= 0  # "lte"


def matches(node: Filter, metadata: Mapping[str, JsonValue]) -> bool:
    """Evaluate a filter against a metadata mapping.

    This is the normative definition of what every operator means. Adapters must
    translate to a backend query with the same behaviour, and the ``VectorIndex``
    contract kit checks that they do.

    Args:
        node: The filter expression.
        metadata: A flat mapping of metadata keys to JSON values.

    Returns:
        Whether the metadata satisfies the expression.

    Raises:
        Nothing. A type mismatch does not match; it does not raise.
    """
    return node.matches(metadata)


def validate_supported(node: Filter, supported: Collection[str], backend: str) -> None:
    """Raise if the filter uses an operator the backend cannot express.

    Call this at query construction, before any network call, so the failure
    names the operator rather than surfacing as an empty result set.

    Args:
        node: The filter expression.
        supported: The operator names the backend supports, normally
            ``IndexCapabilities.filter_ops``.
        backend: The backend's registry key, for the message.

    Returns:
        ``None`` when every operator is supported.

    Raises:
        UnsupportedFilterError: Naming the first unsupported operator, the
            backend, and the operators the backend does support.
    """
    supported_set = frozenset(supported)
    required = node.required_ops()
    missing = sorted(required - supported_set)
    if not missing:
        return

    operator = missing[0]
    raise UnsupportedFilterError(
        f"The filter uses the {operator!r} operator, which the {backend!r} index does not support.",
        operator=operator,
        backend=backend,
        supported=supported_set,
        component=backend,
        remedy=(
            f"Rewrite the filter without {', '.join(repr(m) for m in missing)}, or "
            f"use an index that supports it. {backend!r} supports: "
            f"{', '.join(sorted(supported_set)) or '(none)'}."
        ),
    )


class _FieldRef:
    """A field name awaiting an operator. Produced by :meth:`F.field`."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def _cmp(self, op: ComparisonOp, value: JsonValue) -> Comparison:
        return Comparison(field=self._name, op=op, value=value)

    def eq(self, value: JsonValue) -> Comparison:
        """Field equals the value."""
        return self._cmp("eq", value)

    def ne(self, value: JsonValue) -> Comparison:
        """Field is present and does not equal the value."""
        return self._cmp("ne", value)

    def in_(self, values: Sequence[JsonValue]) -> Comparison:
        """Field is one of the values. Named ``in_`` because ``in`` is a keyword."""
        return self._cmp("in", list(values))

    def nin(self, values: Sequence[JsonValue]) -> Comparison:
        """Field is present and is not one of the values."""
        return self._cmp("nin", list(values))

    def gt(self, value: JsonValue) -> Comparison:
        """Field orders strictly after the value."""
        return self._cmp("gt", value)

    def gte(self, value: JsonValue) -> Comparison:
        """Field orders at or after the value."""
        return self._cmp("gte", value)

    def lt(self, value: JsonValue) -> Comparison:
        """Field orders strictly before the value."""
        return self._cmp("lt", value)

    def lte(self, value: JsonValue) -> Comparison:
        """Field orders at or before the value."""
        return self._cmp("lte", value)

    def contains(self, value: JsonValue) -> Comparison:
        """Field is a string containing the value, or a list holding it."""
        return self._cmp("contains", value)

    def exists(self) -> Exists:
        """Field is present, whatever its value."""
        return Exists(field=self._name)


class F:
    """Builder entry point for filter expressions.

    Example:
        >>> flt = F.field("tenant").eq("acme") & F.field("year").gte(2023)
        >>> flt.matches({"tenant": "acme", "year": 2024})
        True

    A namespace rather than an instantiable class: it holds no state and exists
    so that expressions read as ``F.field(...)`` at the call site.
    """

    def __init__(self) -> None:
        """Refuse construction; ``F`` is a namespace, not an object."""
        raise TypeError("F is a namespace of builders and is not instantiated")

    @staticmethod
    def field(name: str) -> _FieldRef:
        """Start an expression on a metadata field.

        Args:
            name: The metadata key.

        Returns:
            A reference exposing one method per comparison operator.
        """
        return _FieldRef(name)

    @staticmethod
    def all_of(*clauses: Filter) -> And:
        """Conjunction of every clause. Zero clauses match everything."""
        return And(clauses=list(clauses))

    @staticmethod
    def any_of(*clauses: Filter) -> Or:
        """Disjunction of every clause. Zero clauses match nothing."""
        return Or(clauses=list(clauses))

    @staticmethod
    def none_of(*clauses: Filter) -> Not:
        """Negation of the disjunction of every clause."""
        return Not(clause=Or(clauses=list(clauses)))
