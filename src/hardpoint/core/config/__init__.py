"""Five-layer configuration resolution, validation, snapshotting and redaction.

Owns the merge order defined in ARCHITECTURE.md §15.1 -- library defaults,
``base.yaml``, ``{env}.yaml``, ``HARDPOINT__`` environment overrides, then
explicit code overrides -- plus ``${env:VAR}`` interpolation, secret redaction
and the hashed, immutable ``ConfigSnapshot``.

Owns no control flow: configuration selects and parameterises components, it
never expresses step ordering (ADR-005).
"""

from __future__ import annotations

from hardpoint.core.config.loader import (
    ENV_PREFIX,
    ENV_SEPARATOR,
    ResolvedConfig,
    load_config,
    parse_env_overrides,
    resolve,
)
from hardpoint.core.config.schema import (
    CONFIG_VERSION,
    BudgetsConfig,
    CircuitBreakerPolicyConfig,
    ComponentSpec,
    ContextConfig,
    EvalConfig,
    GuardsConfig,
    GuardSpec,
    HardpointConfig,
    IngestionConfig,
    ObservabilityConfig,
    PluginsConfig,
    PolicyConfig,
    ProvidersConfig,
    RateLimitPolicyConfig,
    RequestBudgetConfig,
    RerankConfig,
    RetrievalConfig,
    RetryableKind,
    RetryPolicyConfig,
    TimeoutPolicyConfig,
)
from hardpoint.core.config.snapshot import REDACTED, ConfigSnapshot, Layer, flatten, redact

__all__ = [
    "CONFIG_VERSION",
    "ENV_PREFIX",
    "ENV_SEPARATOR",
    "REDACTED",
    "BudgetsConfig",
    "CircuitBreakerPolicyConfig",
    "ComponentSpec",
    "ConfigSnapshot",
    "ContextConfig",
    "EvalConfig",
    "GuardSpec",
    "GuardsConfig",
    "HardpointConfig",
    "IngestionConfig",
    "Layer",
    "ObservabilityConfig",
    "PluginsConfig",
    "PolicyConfig",
    "ProvidersConfig",
    "RateLimitPolicyConfig",
    "RequestBudgetConfig",
    "RerankConfig",
    "ResolvedConfig",
    "RetrievalConfig",
    "RetryPolicyConfig",
    "RetryableKind",
    "TimeoutPolicyConfig",
    "flatten",
    "load_config",
    "parse_env_overrides",
    "redact",
    "resolve",
]
