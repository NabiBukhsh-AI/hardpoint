"""Execution runtime: pipelines, policies and bounded concurrency.

Owns ``Step`` composition, deadline and budget enforcement, degradation
collection, and the single implementation of retry, timeout, circuit breaking,
rate limiting and fallback. Does not own the domain semantics of retrieval or
generation.

Two runtime shapes exist and no more (ADR-003): everything linear is a
``Pipeline``, everything with a feedback loop is a ``ControlLoop`` (M5).
Branching is ordinary Python inside a step.
"""

from __future__ import annotations

from hardpoint.runtime.concurrency import Failure, gather_bounded, gather_tolerant
from hardpoint.runtime.pipeline import Pipeline, PipelineRun
from hardpoint.runtime.policies import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    Fallback,
    PolicyChain,
    RateLimit,
    Retry,
    Timeout,
    backoff_delay,
    is_retryable,
)
from hardpoint.runtime.step import (
    FailurePolicy,
    PipelineInfo,
    Step,
    StepInfo,
    StepResult,
    as_step,
    step_type_name,
    unwrap,
)

__all__ = [
    "BreakerState",
    "CircuitBreaker",
    "CircuitOpenError",
    "Failure",
    "FailurePolicy",
    "Fallback",
    "Pipeline",
    "PipelineInfo",
    "PipelineRun",
    "PolicyChain",
    "RateLimit",
    "Retry",
    "Step",
    "StepInfo",
    "StepResult",
    "Timeout",
    "as_step",
    "backoff_delay",
    "gather_bounded",
    "gather_tolerant",
    "is_retryable",
    "step_type_name",
    "unwrap",
]
