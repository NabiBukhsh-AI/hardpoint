"""Guard steps returning an explicit ``GuardResult``.

Owns the guard step contract and the built-in checks: schema validation,
citation groundedness, an injection heuristic and an input shape check. Does not
own classifier models or safety policy, and is not a claim of prevention
(ARCHITECTURE.md §18.3, §19).
"""

from __future__ import annotations

from hardpoint.guards.base import (
    DEFAULT_REFUSAL,
    GuardAction,
    GuardCheck,
    GuardedGenerate,
    GuardResult,
    InputGuard,
    OutputGuard,
)
from hardpoint.guards.groundedness import GroundednessGuard, claims, grounding
from hardpoint.guards.injection import DEFAULT_PATTERNS, InjectionHeuristic, InputShapeGuard
from hardpoint.guards.schema import SchemaGuard, parse_json

__all__ = [
    "DEFAULT_PATTERNS",
    "DEFAULT_REFUSAL",
    "GroundednessGuard",
    "GuardAction",
    "GuardCheck",
    "GuardResult",
    "GuardedGenerate",
    "InjectionHeuristic",
    "InputGuard",
    "InputShapeGuard",
    "OutputGuard",
    "SchemaGuard",
    "claims",
    "grounding",
    "parse_json",
]
