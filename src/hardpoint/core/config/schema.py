"""Root configuration models.

Implements the shape in ARCHITECTURE.md §15.2. Configuration *selects and
parameterises* components; it never expresses control flow, branching or step
ordering (ADR-005). There are no conditionals here, no loops, and no cross-file
references, because every declarative pipeline format that grew those became a
badly designed programming language with no debugger.

## Where ``extra="forbid"`` applies, and where it cannot

Every model here forbids unknown keys, so a typo is an error with a suggestion
rather than a setting that silently does nothing (INSTRUCTIONS.md §13.7).

:class:`ComponentSpec` is the one exception, and it is an exception by design
rather than by omission. A component block carries keys that only that component
knows about -- ``collection`` for one index, ``deployment`` for another -- so
this layer cannot know the valid set. Enforcement is not skipped, it moves: the
registry validates a block's options against that component's own
``config_model``, which *is* ``extra="forbid"``. A test asserts that
``ComponentSpec`` is the only model here permitted to allow extras.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.types import JsonValue

__all__ = [
    "CONFIG_VERSION",
    "BudgetsConfig",
    "CircuitBreakerPolicyConfig",
    "ComponentSpec",
    "ContextConfig",
    "EvalConfig",
    "GuardSpec",
    "GuardsConfig",
    "HardpointConfig",
    "IngestionConfig",
    "ObservabilityConfig",
    "PluginsConfig",
    "PolicyConfig",
    "ProvidersConfig",
    "RateLimitPolicyConfig",
    "RequestBudgetConfig",
    "RerankConfig",
    "RetrievalConfig",
    "RetryPolicyConfig",
    "RetryableKind",
    "TimeoutPolicyConfig",
]

CONFIG_VERSION = 1
"""The only ``version:`` this release accepts. Bumped on a breaking shape change."""

_STRICT = ConfigDict(frozen=True, extra="forbid")

RetryableKind = Literal["rate_limited", "transient", "timeout"]
"""Error kinds a retry policy may be told to act on.

A closed set rather than free text, so ``on: [rate_limitted]`` fails validation
instead of quietly disabling retries.
"""


def _default_policies() -> PolicyConfig:
    """Return an all-defaults policy set.

    A function rather than ``default_factory=PolicyConfig`` because
    ``ComponentSpec`` and ``PolicyConfig`` reference each other: a policy may
    name a fallback component, and a component carries policies. The name is
    resolved when the factory is called, not when this class is defined, which
    is what breaks the cycle without a lambda.
    """
    return PolicyConfig()


class ComponentSpec(BaseModel):
    """A ``type:`` discriminator plus that component's own options.

    The one model in this module that allows unknown keys, because the valid set
    belongs to the selected component and is enforced by the registry against
    that component's ``config_model``. See the module docstring.

    Args:
        type: The registry key, for example ``"openai_chat"`` or ``"qdrant"``.
        policies: Retry, timeout, circuit breaker, rate limit and fallback
            wrappers applied to this component at construction time.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    policies: PolicyConfig = Field(default_factory=_default_policies)

    def options(self) -> dict[str, JsonValue]:
        """Return the component-specific keys, excluding ``type`` and ``policies``.

        This is what the registry validates against the component's own config
        model.
        """
        extra = self.model_extra or {}
        return dict(extra)


class RetryPolicyConfig(BaseModel):
    """Exponential backoff with full jitter.

    Implemented once, in ``runtime/policies.py``, and never inside an adapter
    (INSTRUCTIONS.md §13.1 **[LOCKED]**).

    Args:
        max_attempts: Total attempts including the first. ``1`` disables retry.
        on: Which error kinds to retry. A non-retryable error is never retried
            regardless of what appears here.
        initial_backoff_s: Base delay before the second attempt.
        max_backoff_s: Ceiling on the computed delay.
        respect_retry_after: Honour a provider-supplied ``Retry-After`` in
            preference to the computed delay.
    """

    model_config = _STRICT

    max_attempts: int = Field(default=3, ge=1, le=10)
    on: tuple[RetryableKind, ...] = ("rate_limited", "transient", "timeout")
    initial_backoff_s: float = Field(default=0.5, gt=0)
    max_backoff_s: float = Field(default=30.0, gt=0)
    respect_retry_after: bool = True


class TimeoutPolicyConfig(BaseModel):
    """Per-attempt and total time limits.

    Both default to ``None``, meaning the deadline on ``RunContext`` governs.
    A per-attempt timeout larger than the remaining deadline is clamped, not
    honoured: the deadline always wins.
    """

    model_config = _STRICT

    per_attempt_s: float | None = Field(default=None, gt=0)
    total_s: float | None = Field(default=None, gt=0)


class CircuitBreakerPolicyConfig(BaseModel):
    """Stop calling a failing dependency for a cooldown period."""

    model_config = _STRICT

    enabled: bool = False
    failure_threshold: int = Field(default=5, ge=1)
    cooldown_s: float = Field(default=30.0, gt=0)


class RateLimitPolicyConfig(BaseModel):
    """Client-side concurrency and request-rate ceilings for one component."""

    model_config = _STRICT

    max_concurrent: int | None = Field(default=None, ge=1)
    requests_per_second: float | None = Field(default=None, gt=0)


class PolicyConfig(BaseModel):
    """The policy wrappers applied to a component (ADR-004).

    Policies are decorators applied to any port implementation or step,
    configured here and composed at construction time. Centralising them is the
    difference between one retry implementation and five subtly different ones.

    Args:
        retry: Backoff behaviour.
        timeout: Time limits.
        circuit_breaker: Failure isolation.
        rate_limit: Client-side ceilings.
        fallback: An alternative component to try when this one fails. This is
            the mechanism behind model fallback and reranker degradation.
    """

    model_config = _STRICT

    retry: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)
    timeout: TimeoutPolicyConfig = Field(default_factory=TimeoutPolicyConfig)
    circuit_breaker: CircuitBreakerPolicyConfig = Field(default_factory=CircuitBreakerPolicyConfig)
    rate_limit: RateLimitPolicyConfig = Field(default_factory=RateLimitPolicyConfig)
    fallback: ComponentSpec | None = None


class ProvidersConfig(BaseModel):
    """Which model providers to use for each role."""

    model_config = _STRICT

    llm: ComponentSpec | None = None
    embeddings: ComponentSpec | None = None
    reranker: ComponentSpec | None = None


class RerankConfig(BaseModel):
    """Reranking, an optional quality stage.

    ``on_failure`` defaults to ``skip`` because reranking is optional: the
    architecture's rule is availability-preserving degradation for optional
    quality stages, hard failure for correctness-critical ones
    (ARCHITECTURE.md §18.2). A skipped rerank records a ``Degradation``.
    """

    model_config = _STRICT

    enabled: bool = False
    type: str | None = None
    top_k: int = Field(default=8, ge=1)
    on_failure: Literal["skip", "fail"] = "skip"


class ContextConfig(BaseModel):
    """Context assembly: the token budget, the ordering, and the citation scheme.

    Args:
        token_budget: Maximum tokens of assembled context. Everything that does
            not fit is recorded in ``ContextBundle.dropped`` with a reason.
        ordering: ``relevance`` is plain descending score. ``document_order``
            restores source order. ``relevance_with_edges`` places the strongest
            items at the start and end, mitigating lost-in-the-middle.
        citation_style: How citation keys are rendered in the prompt.
        include_metadata: Whether chunk metadata is rendered alongside the text.
    """

    model_config = _STRICT

    token_budget: int = Field(default=4000, ge=1)
    ordering: Literal["relevance", "document_order", "relevance_with_edges"] = "relevance"
    citation_style: Literal["numeric", "source_key"] = "numeric"
    include_metadata: bool = False


class RetrievalConfig(BaseModel):
    """Retrieval parameters.

    ``no_context_policy`` is the reason empty retrieval is not an exception.
    Modelling it as an error forced try/except into every application, so it is
    a configured policy instead (ARCHITECTURE.md §6.2). The abstention *text* is
    a template in the generated project, never in the library
    (INSTRUCTIONS.md §7 **[LOCKED]**).
    """

    model_config = _STRICT

    top_k: int = Field(default=20, ge=1)
    score_threshold: float | None = None
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    no_context_policy: Literal["abstain", "answer_without_context", "escalate", "raise"] = "abstain"


class GuardSpec(BaseModel):
    """One guard and what it does when it fires.

    Allows component-specific options for the same reason as
    :class:`ComponentSpec`; the guard's own config model enforces them.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    action: Literal["allow", "flag", "redact", "block", "retry"] = "flag"

    def options(self) -> dict[str, JsonValue]:
        """Return the guard-specific keys, excluding ``type`` and ``action``."""
        return dict(self.model_extra or {})


class GuardsConfig(BaseModel):
    """Input and output guards, in the order they run.

    Ordering is explicit here and in the composition code, rather than implicit
    in an interceptor chain, so that reading it tells you what happens.
    """

    model_config = _STRICT

    input: tuple[GuardSpec, ...] = ()
    output: tuple[GuardSpec, ...] = ()


class ObservabilityConfig(BaseModel):
    """Tracing, metrics and redaction.

    ``redact`` lists field paths whose content is stripped from span attributes.
    Traces may contain sensitive content, so the generated project's production
    overlay turns this on for message content and chunk text. Recording
    everything by default would be a compliance trap (ARCHITECTURE.md §19).
    """

    model_config = _STRICT

    tracer: ComponentSpec | None = None
    metrics: ComponentSpec | None = None
    redact: tuple[str, ...] = ()


class RequestBudgetConfig(BaseModel):
    """Per-request ceilings. Exceeding one raises ``BudgetExceeded``.

    Every field defaults to ``None``, meaning unbounded. Budgets are opt-in
    because a default ceiling would silently truncate a legitimate workload;
    the generated project sets real values.
    """

    model_config = _STRICT

    max_cost_usd: float | None = Field(default=None, gt=0)
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    deadline_s: float | None = Field(default=None, gt=0)


class BudgetsConfig(BaseModel):
    """Budget settings by scope."""

    model_config = _STRICT

    request: RequestBudgetConfig = Field(default_factory=RequestBudgetConfig)


class IngestionConfig(BaseModel):
    """Ingestion throughput and failure handling.

    Ingestion never runs inside the request path (INSTRUCTIONS.md §13.18), so
    these settings govern a job, not a request.

    Args:
        embed_batch_size: Texts per embedding call.
        concurrency: Documents processed at once.
        fail_fast: Abort the run on the first document failure instead of
            quarantining it and continuing.
        quarantine_path: Where the artefact listing rejected documents and
            chunks is written, so a human can look at them.
    """

    model_config = _STRICT

    embed_batch_size: int = Field(default=128, ge=1)
    concurrency: int = Field(default=4, ge=1)
    fail_fast: bool = False
    quarantine_path: str = "artefacts/quarantine.jsonl"


class EvalConfig(BaseModel):
    """Quality gates.

    Args:
        thresholds: Metric name to minimum acceptable value. A breach fails CI
            with a per-case regression table.
        max_cost_usd: Refuse to start a suite whose estimated cost exceeds this.
    """

    model_config = _STRICT

    thresholds: dict[str, float] = Field(default_factory=dict)
    max_cost_usd: float | None = Field(default=None, gt=0)


class PluginsConfig(BaseModel):
    """Third-party component discovery.

    ``discover`` defaults to ``False``. Entry-point scanning at import time was
    rejected outright: it makes resolution non-deterministic, hides where a
    component came from, and makes a run unreproducible (ARCHITECTURE.md §6.3).
    Turning it on is an explicit, inspectable choice.
    """

    model_config = _STRICT

    discover: bool = False


class HardpointConfig(BaseModel):
    """The root configuration model.

    Args:
        version: Config schema version. Must equal :data:`CONFIG_VERSION`.
        providers: Model providers per role.
        indexes: Named vector indexes.
        sources: Named ingestion sources.
        retrieval: Retrieval parameters.
        guards: Input and output guards.
        observability: Tracing, metrics and redaction.
        budgets: Per-request ceilings.
        ingestion: Ingestion throughput and failure handling.
        eval: Quality gates.
        plugins: Third-party component discovery.
    """

    model_config = _STRICT

    version: int = CONFIG_VERSION
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    indexes: dict[str, ComponentSpec] = Field(default_factory=dict)
    sources: dict[str, ComponentSpec] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    guards: GuardsConfig = Field(default_factory=GuardsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)


ComponentSpec.model_rebuild()
PolicyConfig.model_rebuild()
