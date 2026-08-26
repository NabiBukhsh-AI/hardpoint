"""The exception taxonomy. INSTRUCTIONS.md §5.5, ARCHITECTURE.md §18.1, §26.3.

The **[LOCKED]** requirement is a test asserting that every concrete exception
class defines a ``code``, and that ``remedy`` is present for every
``ConfigError`` subclass. Both are enforced by discovering subclasses
reflectively rather than by listing them, so a class added later is covered
without anyone remembering to add it here.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from hardpoint.core import errors
from hardpoint.core.errors import (
    AuthError,
    BudgetExceeded,
    CapabilityError,
    ConfigError,
    ContractError,
    GuardViolation,
    HardpointError,
    IngestionError,
    InvalidRequestError,
    MissingDependencyError,
    MissingEnvironmentVariableError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    TransientError,
    UnsupportedFilterError,
)


def all_error_classes() -> list[type[HardpointError]]:
    """Every HardpointError subclass exported by the module."""
    return [
        obj
        for obj in vars(errors).values()
        if inspect.isclass(obj) and issubclass(obj, HardpointError)
    ]


def construct(cls: type[HardpointError]) -> HardpointError:
    """Build an instance of any error class, supplying its required keywords."""
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(cls.__init__).parameters
    defaults: dict[str, Any] = {
        "remedy": "do the thing",
        "variable": "SOME_VAR",
        "operator": "contains",
        "backend": "toy",
        "guard": "schema",
        "reason": "invalid",
        "limit": "max_cost_usd",
        "limit_value": 1.0,
        "observed": 2.0,
    }
    for name, parameter in parameters.items():
        if name in {"self", "message"}:
            continue
        if parameter.default is inspect.Parameter.empty and name in defaults:
            kwargs[name] = defaults[name]
    return cls("something failed", **kwargs)


# --------------------------------------------------------------------------- #
# The locked assertions                                                       #
# --------------------------------------------------------------------------- #


def test_every_error_class_defines_a_code() -> None:
    """**[LOCKED]** Every concrete exception class defines a ``code``."""
    for cls in all_error_classes():
        assert cls.default_code, f"{cls.__name__} has no default_code"
        assert construct(cls).code == cls.default_code, cls.__name__


def test_error_codes_are_unique() -> None:
    """Two classes sharing a code would make the code useless for dispatch."""
    seen: dict[str, str] = {}
    for cls in all_error_classes():
        code = cls.default_code
        if cls.default_code == HardpointError.default_code and cls is not HardpointError:
            continue  # inherits the base code deliberately
        assert code not in seen, f"{cls.__name__} reuses the code of {seen.get(code)}"
        seen[code] = cls.__name__


def test_every_config_error_requires_a_remedy() -> None:
    """**[LOCKED]** ``remedy`` is required for every ``ConfigError`` subclass.

    Asserted against the signature, not merely against an instance: a subclass
    that made ``remedy`` optional would pass an instance check by being
    constructed carefully in this file, and then ship an unactionable error.
    """
    config_errors = [cls for cls in all_error_classes() if issubclass(cls, ConfigError)]
    assert config_errors, "no ConfigError subclasses were discovered"

    for cls in config_errors:
        parameter = inspect.signature(cls.__init__).parameters.get("remedy")
        assert parameter is not None, f"{cls.__name__} does not accept a remedy"
        assert parameter.default is inspect.Parameter.empty, (
            f"{cls.__name__} makes remedy optional; a configuration error the "
            "operator cannot act on is a bug in the error."
        )
        assert construct(cls).remedy, f"{cls.__name__} produced an empty remedy"


def test_config_error_cannot_be_built_without_a_remedy() -> None:
    with pytest.raises(TypeError):
        ConfigError("bad config")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Message rendering (ARCHITECTURE.md §26.3)                                   #
# --------------------------------------------------------------------------- #


def test_message_answers_what_where_why_and_what_next() -> None:
    error = MissingDependencyError(
        "Component 'qdrant' requires the 'qdrant' extra.",
        extra="qdrant",
        component="qdrant",
        config_path="indexes.primary.type",
        remedy="pip install 'hardpoint[qdrant]'",
    )
    rendered = str(error)

    assert "Component 'qdrant' requires the 'qdrant' extra." in rendered  # what failed
    assert "qdrant" in rendered  # which component
    assert "indexes.primary.type" in rendered  # which config path
    assert "config.missing_dependency" in rendered  # why, machine-readably
    assert "pip install 'hardpoint[qdrant]'" in rendered  # what to do next


def test_message_omits_unknown_context() -> None:
    """A bare error must not render a wall of ``None``s."""
    rendered = str(ContractError("adapter returned no usage"))
    assert "None" not in rendered
    assert "component" not in rendered
    assert "contract.violated" in rendered


def test_repr_is_single_line_for_tracebacks() -> None:
    rendered = repr(ContractError("boom"))
    assert "\n" not in rendered
    assert "ContractError" in rendered


def test_context_returns_only_populated_fields() -> None:
    context = TransientError("upstream 503", component="openai_chat", step="generate").context()
    assert context == {
        "code": "provider.transient",
        "component": "openai_chat",
        "step": "generate",
        "retryable": "true",
    }


# --------------------------------------------------------------------------- #
# Taxonomy shape and retryability                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (TransientError, True),
        (RateLimitedError, True),
        (ProviderTimeout, True),
        (AuthError, False),
        (InvalidRequestError, False),
        (ContractError, False),
        (CapabilityError, False),
        (IngestionError, False),
        (BudgetExceeded, False),
    ],
)
def test_retryability_matches_the_failure_matrix(cls: type[HardpointError], expected: bool) -> None:
    """ARCHITECTURE.md §18.1 fixes which failures are retryable in principle."""
    assert construct(cls).retryable is expected


def test_retryability_can_be_overridden_per_instance() -> None:
    assert AuthError("expired token", retryable=True).retryable is True


def test_taxonomy_parentage() -> None:
    """The inheritance tree is part of the contract: callers catch by branch."""
    assert issubclass(TransientError, ProviderError)
    assert issubclass(RateLimitedError, ProviderError)
    assert issubclass(AuthError, ProviderError)
    assert issubclass(InvalidRequestError, ProviderError)
    assert issubclass(ProviderTimeout, ProviderError)
    assert issubclass(UnsupportedFilterError, CapabilityError)
    assert issubclass(MissingDependencyError, ConfigError)
    assert issubclass(MissingEnvironmentVariableError, ConfigError)
    assert all(issubclass(cls, HardpointError) for cls in all_error_classes())


def test_provider_errors_are_catchable_as_one_branch() -> None:
    """An adapter's unmapped failure must still be catchable as ProviderError."""
    with pytest.raises(ProviderError):
        raise RateLimitedError("429", retry_after_s=1.5)


# --------------------------------------------------------------------------- #
# Structured payloads                                                         #
# --------------------------------------------------------------------------- #


def test_rate_limited_carries_the_servers_own_backoff() -> None:
    """Retry must be able to honour Retry-After rather than guessing."""
    assert RateLimitedError("429", retry_after_s=2.5).retry_after_s == 2.5
    assert RateLimitedError("429").retry_after_s is None


def test_unsupported_filter_names_operator_and_backend() -> None:
    error = UnsupportedFilterError(
        "no contains", operator="contains", backend="toy", supported=frozenset({"eq"})
    )
    assert error.operator == "contains"
    assert error.backend == "toy"
    assert error.supported == frozenset({"eq"})


def test_ingestion_error_carries_the_document() -> None:
    """A quarantine report must point at a document, not a stack frame."""
    error = IngestionError("could not parse", document_id="doc_abc", source_id="docs")
    assert error.document_id == "doc_abc"
    assert error.source_id == "docs"


def test_guard_violation_carries_guard_and_reason() -> None:
    error = GuardViolation("blocked", guard="groundedness", reason="unsupported_claim")
    assert (error.guard, error.reason) == ("groundedness", "unsupported_claim")


def test_budget_exceeded_carries_the_limit_that_was_passed() -> None:
    error = BudgetExceeded("over budget", limit="max_cost_usd", limit_value=0.15, observed=0.21)
    assert (error.limit, error.limit_value, error.observed) == ("max_cost_usd", 0.15, 0.21)


def test_missing_environment_variable_names_the_variable() -> None:
    error = MissingEnvironmentVariableError(
        "QDRANT_URL is not set",
        variable="QDRANT_URL",
        config_path="indexes.primary.url",
        remedy="export QDRANT_URL=...",
    )
    assert error.variable == "QDRANT_URL"
    assert "QDRANT_URL" in str(error)


def test_cause_is_attached_for_chained_tracebacks() -> None:
    original = RuntimeError("socket closed")
    error = TransientError("upstream dropped the connection", cause=original)
    assert error.__cause__ is original


def test_warning_base_exists_for_non_fatal_signals() -> None:
    """An unpriced model warns; it does not raise and does not stay silent."""
    assert issubclass(errors.HardpointWarning, UserWarning)
