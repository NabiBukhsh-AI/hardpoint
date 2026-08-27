"""``InMemoryVectorIndex`` against the ``VectorIndex`` contract kit. **[LOCKED]**

INSTRUCTIONS.md §5.8 requires the in-memory index to be the reference
implementation for the contract suite, and the M0 Definition of Done requires
the fakes to pass their own kits.

This file is also the worked example a third-party adapter author copies: bind
the kit to a factory, name the result with a ``Test`` prefix, done.

The kit is bound three times, because a contract kit that has only ever been run
against one configuration has not been shown to discriminate:

- the default index, declaring every filter operator;
- an index declaring only a subset, which must make the undeclared-operator test
  actually fire rather than skip;
- an index that rejects credentials, which adds the error-mapping tests.
"""

from __future__ import annotations

import pytest

from hardpoint.core.capabilities import IndexCapabilities
from hardpoint.core.errors import UnsupportedFilterError
from hardpoint.core.filters import F
from hardpoint.core.ports import IndexSpec, VectorQuery
from hardpoint.testing import InMemoryVectorIndex, build_run_context, vector_index_contract

pytestmark = pytest.mark.contract


TestInMemoryVectorIndex = vector_index_contract(InMemoryVectorIndex)
"""The default configuration: every filter operator, namespaces, cosine."""


TestInMemoryVectorIndexDotProduct = vector_index_contract(lambda: InMemoryVectorIndex(metric="dot"))
"""A different distance metric must not change any behavioural guarantee."""


TestLimitedVectorIndex = vector_index_contract(
    lambda: InMemoryVectorIndex(
        capabilities=IndexCapabilities(
            filter_ops=frozenset({"eq", "and", "exists", "gt", "gte", "lt", "lte", "in"}),
            supports_namespaces=False,
            supports_delete_by_filter=True,
        )
    )
)
"""A backend that cannot express every operator.

This binding is what proves the kit's undeclared-operator test can fire. Against
the fully capable index that test skips, and a suite whose most important
assertion only ever skips is not an assertion.
"""


TestUnauthorisedVectorIndex = vector_index_contract(
    InMemoryVectorIndex,
    unauthorised_factory=lambda: InMemoryVectorIndex(unauthorised=True),
)
"""Adds the error-mapping tests, since this fake can simulate a rejected credential."""


# --------------------------------------------------------------------------- #
# The kit itself must discriminate                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_the_limited_index_really_does_reject_an_undeclared_operator() -> None:
    """Asserted directly, not only through the kit.

    If the kit's undeclared-operator test silently skipped for every binding,
    the suite would still be green while checking nothing. This proves the
    behaviour independently of how the kit selects its test case.
    """
    index = InMemoryVectorIndex(capabilities=IndexCapabilities(filter_ops=frozenset({"eq"})))
    ctx = build_run_context()
    await index.ensure(IndexSpec(name=index.name, dimensions=8))

    with pytest.raises(UnsupportedFilterError) as exc_info:
        await index.query(
            VectorQuery(vector=(1.0,) + (0.0,) * 7, filter=F.field("tags").contains("x")),
            ctx,
        )
    assert exc_info.value.operator == "contains"
    assert exc_info.value.backend == index.name


@pytest.mark.anyio
async def test_delete_by_filter_also_validates_operators() -> None:
    """A delete is more destructive than a query, so it must not be laxer.

    An unsupported operator that was silently dropped here would widen the
    delete to every record the remaining clauses matched.
    """
    index = InMemoryVectorIndex(capabilities=IndexCapabilities(filter_ops=frozenset({"eq"})))
    ctx = build_run_context()

    with pytest.raises(UnsupportedFilterError):
        await index.delete(ctx, filter=F.field("tags").contains("x"))
