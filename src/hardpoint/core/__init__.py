"""Contracts, data models, errors, configuration, identity and the registry.

``core`` is the foundation every other layer is written against. It owns the
port Protocols, the Pydantic data models that cross every boundary, the error
taxonomy, ``RunContext``, configuration loading and validation, the component
registry, and deterministic id and hash derivation.

It owns no I/O, no provider knowledge and no execution. It imports nothing from
``hardpoint`` outside ``core``, and its third-party surface is limited to
``pydantic``, ``anyio`` and the standard library (ARCHITECTURE.md §8.1 R1,
enforced by the "core is independent" import-linter contract).

Everything re-exported here is public API. Anything else in ``core`` is internal
and may change in a minor release (ARCHITECTURE.md §24).
"""

from __future__ import annotations

from hardpoint.core.capabilities import ALL_FILTER_OPS, IndexCapabilities, ModelCapabilities
from hardpoint.core.errors import (
    AuthError,
    BudgetExceeded,
    CapabilityError,
    ConfigError,
    ContractError,
    DuplicateComponentError,
    GuardViolation,
    HardpointError,
    HardpointWarning,
    IngestionError,
    InvalidConfigError,
    InvalidRequestError,
    LoopLimitExceeded,
    MissingDependencyError,
    MissingEnvironmentVariableError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    RetrievalError,
    TransientError,
    UnknownComponentError,
    UnsupportedFilterError,
)
from hardpoint.core.filters import (
    COMPARISON_OPS,
    STRUCTURAL_OPS,
    And,
    Comparison,
    ComparisonOp,
    Exists,
    F,
    Filter,
    FilterOp,
    Not,
    Or,
    matches,
    validate_supported,
)
from hardpoint.core.ids import (
    chunk_id,
    content_hash,
    document_id,
    normalise_text,
    stable_hash,
    text_hash,
)
from hardpoint.core.models import (
    Answer,
    Block,
    CharSpan,
    Chunk,
    Citation,
    ContextBundle,
    ContextItem,
    Degradation,
    Document,
    DropRecord,
    ParsedDocument,
    RetrievedChunk,
    RunManifest,
    StepUsage,
    TrustLevel,
    Usage,
)
from hardpoint.core.types import JsonValue, ModelId, RunId

__all__ = [
    "ALL_FILTER_OPS",
    "COMPARISON_OPS",
    "STRUCTURAL_OPS",
    "And",
    "Answer",
    "AuthError",
    "Block",
    "BudgetExceeded",
    "CapabilityError",
    "CharSpan",
    "Chunk",
    "Citation",
    "Comparison",
    "ComparisonOp",
    "ConfigError",
    "ContextBundle",
    "ContextItem",
    "ContractError",
    "Degradation",
    "Document",
    "DropRecord",
    "DuplicateComponentError",
    "Exists",
    "F",
    "Filter",
    "FilterOp",
    "GuardViolation",
    "HardpointError",
    "HardpointWarning",
    "IndexCapabilities",
    "IngestionError",
    "InvalidConfigError",
    "InvalidRequestError",
    "JsonValue",
    "LoopLimitExceeded",
    "MissingDependencyError",
    "MissingEnvironmentVariableError",
    "ModelCapabilities",
    "ModelId",
    "Not",
    "Or",
    "ParsedDocument",
    "ProviderError",
    "ProviderTimeout",
    "RateLimitedError",
    "RetrievalError",
    "RetrievedChunk",
    "RunId",
    "RunManifest",
    "StepUsage",
    "TransientError",
    "TrustLevel",
    "UnknownComponentError",
    "UnsupportedFilterError",
    "Usage",
    "chunk_id",
    "content_hash",
    "document_id",
    "matches",
    "normalise_text",
    "stable_hash",
    "text_hash",
    "validate_supported",
]
