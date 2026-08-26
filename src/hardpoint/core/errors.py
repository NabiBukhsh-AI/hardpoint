"""The exception taxonomy.

Implements ARCHITECTURE.md §18.1. Every exception in this module carries a
machine-readable ``code``, knows whether it is retryable, and renders a message
that answers the four questions required by ARCHITECTURE.md §26.3: what failed,
which component and config path, why, and what to do next.

Adapters are required to map provider exceptions into this taxonomy. An
exception an adapter cannot classify becomes a plain :class:`ProviderError` with
the original attached as ``__cause__`` -- never a leaked SDK exception type.

This module owns error *shape*. It owns no policy: whether a retryable error is
actually retried is decided in ``runtime/policies.py``, not here.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "AuthError",
    "BudgetExceeded",
    "CapabilityError",
    "ConfigError",
    "ContractError",
    "DuplicateComponentError",
    "GuardViolation",
    "HardpointError",
    "HardpointWarning",
    "IngestionError",
    "InvalidConfigError",
    "InvalidRequestError",
    "LoopLimitExceeded",
    "MissingDependencyError",
    "MissingEnvironmentVariableError",
    "ProviderError",
    "ProviderTimeout",
    "RateLimitedError",
    "RetrievalError",
    "TransientError",
    "UnknownComponentError",
    "UnsupportedFilterError",
]


class HardpointWarning(UserWarning):
    """Base class for warnings the library emits.

    Warnings are used where an error would be wrong but silence would be worse:
    an unpriced model, an unknown ``provider_options`` key, a component
    declaring a contract version newer than this release.
    """


class HardpointError(Exception):
    """Base class for every error the library raises.

    Subclasses set :attr:`default_code` and :attr:`default_retryable`; callers
    may override either per instance. ``str(error)`` renders the message
    followed by whatever context is known and, when present, the remedy.

    This class does not decide whether an error is recoverable in a given
    pipeline; it records whether the *kind* of failure is retryable in
    principle. Policy lives in ``runtime/policies.py``.

    Args:
        message: What failed and why, in one sentence.
        code: Machine-readable code. Defaults to the class's ``default_code``.
        component: Registry key of the component that failed, when known.
        config_path: Dotted path into the configuration that selected the
            failing component, for example ``indexes.primary.type``.
        step: Name of the pipeline step that was running, when known.
        run_id: Identifier of the run, when known.
        retryable: Whether retrying could plausibly succeed. Defaults to the
            class's ``default_retryable``.
        remedy: What the operator should do next. Required on
            :class:`ConfigError` and its subclasses.
        cause: The underlying exception, attached as ``__cause__``.

    Raises:
        Nothing. Constructing an error never fails.
    """

    default_code: ClassVar[str] = "hardpoint.error"
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        retryable: bool | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code if code is not None else self.default_code
        self.component = component
        self.config_path = config_path
        self.step = step
        self.run_id = run_id
        self.retryable = retryable if retryable is not None else self.default_retryable
        self.remedy = remedy
        if cause is not None:
            self.__cause__ = cause

    def context(self) -> dict[str, str]:
        """Return the known context fields, for structured logging and spans.

        Only populated fields appear, so a log record is not padded with nulls.
        """
        fields = {
            "code": self.code,
            "component": self.component,
            "config_path": self.config_path,
            "step": self.step,
            "run_id": self.run_id,
            "retryable": "true" if self.retryable else "false",
        }
        return {key: value for key, value in fields.items() if value is not None}

    def __str__(self) -> str:
        """Render what failed, where, why, and what to do next."""
        lines = [self.message, ""]
        labels = (
            ("code", self.code),
            ("component", self.component),
            ("config", self.config_path),
            ("step", self.step),
            ("run", self.run_id),
            ("retryable", "yes" if self.retryable else "no"),
        )
        lines.extend(f"  {label + ':':<11}{value}" for label, value in labels if value is not None)
        if self.remedy:
            lines.extend(["", "  remedy:"])
            lines.extend(f"    {line}" for line in self.remedy.splitlines())
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a short, single-line form for tracebacks and test output."""
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


class ConfigError(HardpointError):
    """Invalid configuration, a missing environment variable, or an unknown component.

    ``remedy`` is **required** on this branch of the taxonomy
    (INSTRUCTIONS.md §5.5 **[LOCKED]**): a configuration problem the operator
    cannot act on is a bug in the error, not just in the configuration. Where
    the fix is a shell command, the remedy prints the command
    (INSTRUCTIONS.md §12.3).
    """

    default_code: ClassVar[str] = "config.invalid"

    def __init__(
        self,
        message: str,
        *,
        remedy: str,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            component=component,
            config_path=config_path,
            step=step,
            run_id=run_id,
            retryable=False,
            remedy=remedy,
            cause=cause,
        )


class InvalidConfigError(ConfigError):
    """A configuration value failed validation, or an unknown key was present.

    Unknown keys are always an error, never ignored
    (INSTRUCTIONS.md §13.7). The remedy names the closest valid key.
    """

    default_code: ClassVar[str] = "config.invalid_value"


class MissingEnvironmentVariableError(ConfigError):
    """A ``${env:VAR}`` reference resolved to nothing and had no default."""

    default_code: ClassVar[str] = "config.missing_env"

    def __init__(
        self,
        message: str,
        *,
        variable: str,
        remedy: str,
        code: str | None = None,
        config_path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, remedy=remedy, code=code, config_path=config_path, cause=cause)
        self.variable = variable


class UnknownComponentError(ConfigError):
    """A ``type:`` discriminator named a component the registry does not know.

    The remedy carries the nearest registered keys of the same kind, because a
    typo is the overwhelmingly common cause (ARCHITECTURE.md §30).
    """

    default_code: ClassVar[str] = "config.unknown_component"


class DuplicateComponentError(ConfigError):
    """Two registrations claimed the same key at the same precedence level.

    A collision is an error naming both sources, never a silent override
    (ARCHITECTURE.md §17.1).
    """

    default_code: ClassVar[str] = "config.duplicate_component"


class MissingDependencyError(ConfigError):
    """A component was requested whose optional dependency is not installed.

    Raised instead of letting a raw ``ModuleNotFoundError`` escape. The remedy
    is the exact install command (ARCHITECTURE.md §16.3).
    """

    default_code: ClassVar[str] = "config.missing_dependency"

    def __init__(
        self,
        message: str,
        *,
        remedy: str,
        extra: str | None = None,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            remedy=remedy,
            code=code,
            component=component,
            config_path=config_path,
            cause=cause,
        )
        self.extra = extra


# --------------------------------------------------------------------------- #
# Contracts and capabilities                                                  #
# --------------------------------------------------------------------------- #


class ContractError(HardpointError):
    """An implementation violated a port invariant.

    Raised by the library against an adapter, not by an adapter against a
    provider: a ``LanguageModel`` that returns no ``Usage``, a ``VectorIndex``
    whose ``query`` returns scores out of order. The contract test kits exist to
    surface these before production does.
    """

    default_code: ClassVar[str] = "contract.violated"


class CapabilityError(HardpointError):
    """A model or index lacks a capability the composition requires.

    Raised at composition time where possible, at call time otherwise
    (ARCHITECTURE.md §9.2).
    """

    default_code: ClassVar[str] = "capability.unsupported"


class UnsupportedFilterError(CapabilityError):
    """A metadata filter used an operator the backend does not support.

    Raised at query construction, naming the operator and the backend. The
    clause is never silently dropped: a filter that quietly stops filtering is
    how a tenant sees another tenant's documents (INSTRUCTIONS.md §5.3
    **[LOCKED]**).
    """

    default_code: ClassVar[str] = "capability.unsupported_filter"

    def __init__(
        self,
        message: str,
        *,
        operator: str,
        backend: str,
        supported: frozenset[str] | None = None,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            component=component,
            config_path=config_path,
            remedy=remedy,
            cause=cause,
        )
        self.operator = operator
        self.backend = backend
        self.supported = supported


# --------------------------------------------------------------------------- #
# Providers                                                                   #
# --------------------------------------------------------------------------- #


class ProviderError(HardpointError):
    """An external provider failed in a way the adapter could not classify.

    Adapters map what they recognise onto the subclasses below. Anything else
    arrives here with the original exception as ``__cause__``.
    """

    default_code: ClassVar[str] = "provider.error"


class TransientError(ProviderError):
    """A transient provider failure: a 5xx, a connection reset, a dropped stream."""

    default_code: ClassVar[str] = "provider.transient"
    default_retryable: ClassVar[bool] = True


class RateLimitedError(ProviderError):
    """The provider rejected the call for rate or quota reasons.

    Carries ``retry_after_s`` when the provider supplied it, so that
    ``runtime.policies.Retry`` can honour the server's own backoff instead of
    guessing.
    """

    default_code: ClassVar[str] = "provider.rate_limited"
    default_retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            component=component,
            config_path=config_path,
            step=step,
            run_id=run_id,
            remedy=remedy,
            cause=cause,
        )
        self.retry_after_s = retry_after_s


class AuthError(ProviderError):
    """The provider rejected the credentials. Never retryable."""

    default_code: ClassVar[str] = "provider.auth"


class InvalidRequestError(ProviderError):
    """The provider rejected the request itself: malformed, or context too long.

    Never retryable: the same request will be rejected again.
    """

    default_code: ClassVar[str] = "provider.invalid_request"


class ProviderTimeout(ProviderError):  # noqa: N818 - name fixed by ARCHITECTURE.md §18.1
    """A provider call exceeded its timeout."""

    default_code: ClassVar[str] = "provider.timeout"
    default_retryable: ClassVar[bool] = True


# --------------------------------------------------------------------------- #
# Domain                                                                      #
# --------------------------------------------------------------------------- #


class IngestionError(HardpointError):
    """A document failed to parse, chunk, or validate.

    Carries ``document_id`` so a quarantine report can point at the offending
    document rather than at a stack frame.
    """

    default_code: ClassVar[str] = "ingestion.failed"

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        source_id: str | None = None,
        code: str | None = None,
        component: str | None = None,
        config_path: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        retryable: bool | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            component=component,
            config_path=config_path,
            step=step,
            run_id=run_id,
            retryable=retryable,
            remedy=remedy,
            cause=cause,
        )
        self.document_id = document_id
        self.source_id = source_id


class RetrievalError(HardpointError):
    """Retrieval failed.

    Note that retrieving *nothing* is not an error. Empty retrieval is a policy
    decision expressed by ``no_context_policy`` (ARCHITECTURE.md §6.2), because
    modelling it as an exception forces try/except into every application.
    """

    default_code: ClassVar[str] = "retrieval.failed"


class GuardViolation(HardpointError):  # noqa: N818 - name fixed by ARCHITECTURE.md §18.1
    """A guard blocked the input or the output.

    Carries the guard that fired and its reason, so the caller can distinguish a
    schema failure from a groundedness failure without parsing a message.
    """

    default_code: ClassVar[str] = "guard.violation"

    def __init__(
        self,
        message: str,
        *,
        guard: str,
        reason: str,
        code: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, code=code, step=step, run_id=run_id, remedy=remedy, cause=cause)
        self.guard = guard
        self.reason = reason


class BudgetExceeded(HardpointError):  # noqa: N818 - name fixed by ARCHITECTURE.md §18.1
    """A run passed one of its budget limits: cost, tokens, LLM calls, or deadline."""

    default_code: ClassVar[str] = "budget.exceeded"

    def __init__(
        self,
        message: str,
        *,
        limit: str,
        limit_value: float,
        observed: float,
        code: str | None = None,
        step: str | None = None,
        run_id: str | None = None,
        remedy: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, code=code, step=step, run_id=run_id, remedy=remedy, cause=cause)
        self.limit = limit
        self.limit_value = limit_value
        self.observed = observed


class LoopLimitExceeded(HardpointError):  # noqa: N818 - name fixed by ARCHITECTURE.md §18.1
    """A control loop hit an iteration or tool-call limit.

    A loop returns its best current state with a ``Degradation`` by default
    rather than raising (INSTRUCTIONS.md §10 **[LOCKED]**); this error exists for
    the configurations that ask to fail hard instead.
    """

    default_code: ClassVar[str] = "loop.limit_exceeded"
