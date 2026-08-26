"""Declared capabilities. ARCHITECTURE.md §9.2.

Capabilities exist so that provider switching fails loudly at composition time
instead of quietly at request time. The tests that matter are the ones proving
``require`` actually raises, and that the error says what to do about it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hardpoint.core.capabilities import ALL_FILTER_OPS, IndexCapabilities, ModelCapabilities
from hardpoint.core.errors import CapabilityError
from hardpoint.core.filters import COMPARISON_OPS, STRUCTURAL_OPS


def full_model() -> ModelCapabilities:
    return ModelCapabilities(
        context_window_tokens=128_000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_vision=True,
    )


def minimal_model() -> ModelCapabilities:
    return ModelCapabilities(context_window_tokens=8192)


# --------------------------------------------------------------------------- #
# ModelCapabilities                                                           #
# --------------------------------------------------------------------------- #


def test_capabilities_default_to_absent() -> None:
    """Under-declaring is safe; over-declaring is a bug. So the defaults are False."""
    capabilities = minimal_model()
    assert capabilities.supports_tools is False
    assert capabilities.supports_structured_output is False
    assert capabilities.supports_streaming is False
    assert capabilities.supports_vision is False
    assert capabilities.max_output_tokens is None


def test_require_passes_when_everything_is_supported() -> None:
    assert full_model().require("openai/gpt-4o", tools=True, streaming=True) is None


def test_require_passes_when_nothing_is_asked_for() -> None:
    assert minimal_model().require("toy/model") is None


def test_require_raises_naming_every_missing_capability() -> None:
    """A step needing tools must fail at startup, not on the first user request."""
    with pytest.raises(CapabilityError) as exc_info:
        minimal_model().require("toy/model", tools=True, structured_output=True)

    rendered = str(exc_info.value)
    assert "toy/model" in rendered
    assert "tools" in rendered
    assert "structured output" in rendered
    assert exc_info.value.code == "capability.unsupported"


def test_capability_error_remedy_covers_the_adapter_being_wrong() -> None:
    """Under-declaration by an adapter is a real cause, so the remedy names it."""
    with pytest.raises(CapabilityError) as exc_info:
        minimal_model().require("toy/model", vision=True)
    remedy = exc_info.value.remedy or ""
    assert "capabilities()" in remedy
    assert "under-declaring" in remedy


def test_require_only_complains_about_what_was_asked_for() -> None:
    capabilities = ModelCapabilities(context_window_tokens=1000, supports_tools=True)
    with pytest.raises(CapabilityError) as exc_info:
        capabilities.require("m", tools=True, streaming=True)
    assert "tools" not in str(exc_info.value)
    assert "streaming" in str(exc_info.value)


def test_fits_bounds_the_context_window() -> None:
    capabilities = ModelCapabilities(context_window_tokens=100)
    assert capabilities.fits(100) is True
    assert capabilities.fits(101) is False


def test_context_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ModelCapabilities(context_window_tokens=0)
    with pytest.raises(ValidationError):
        ModelCapabilities(context_window_tokens=10, max_output_tokens=0)


# --------------------------------------------------------------------------- #
# IndexCapabilities                                                           #
# --------------------------------------------------------------------------- #


def test_index_declares_no_filter_operators_by_default() -> None:
    """An index that has not declared its operators supports none of them.

    The safe default: a permissive default would let an adapter accept a filter
    it cannot express and return unfiltered rows.
    """
    assert IndexCapabilities().filter_ops == frozenset()


def test_all_filter_ops_covers_the_closed_tree() -> None:
    assert ALL_FILTER_OPS == COMPARISON_OPS | STRUCTURAL_OPS
    assert len(ALL_FILTER_OPS) == 13


def test_unsupported_ops_reports_the_difference() -> None:
    capabilities = IndexCapabilities(filter_ops=frozenset({"eq", "and"}))
    assert capabilities.unsupported_ops({"eq", "and"}) == frozenset()
    assert capabilities.unsupported_ops({"eq", "contains", "or"}) == frozenset({"contains", "or"})


def test_delete_consistency_is_declared_and_closed() -> None:
    """The sync engine and the contract kit both need to know."""
    assert IndexCapabilities().delete_consistency == "consistent"
    assert IndexCapabilities(delete_consistency="eventual").delete_consistency == "eventual"
    with pytest.raises(ValidationError):
        IndexCapabilities(delete_consistency="probably")  # type: ignore[arg-type]


def test_delete_by_filter_defaults_to_supported() -> None:
    """Ingestion removes a document's chunks by filter; most backends allow it."""
    assert IndexCapabilities().supports_delete_by_filter is True


def test_capabilities_are_frozen_and_forbid_unknown_fields() -> None:
    capabilities = IndexCapabilities()
    with pytest.raises(ValidationError):
        capabilities.supports_hybrid = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        IndexCapabilities(supports_magic=True)  # type: ignore[call-arg]
