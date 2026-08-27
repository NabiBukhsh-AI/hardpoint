"""Primitive type aliases shared across ``core``.

This module exists so that ``models`` and ``filters`` can share a vocabulary
without either importing the other. It defines names only; it holds no logic and
performs no I/O.
"""

from __future__ import annotations

from typing import TypeAlias

from pydantic import JsonValue

__all__ = ["CONTRACT_VERSION", "JsonValue", "ModelId", "RunId"]

CONTRACT_VERSION = "1.0"
"""Version of the port Protocols, independent of the distribution version.

Ports change only when this increments, which is rare and always accompanied by
a migration note. Third-party components declare the contract version they
target so the registry can warn on a mismatch (ARCHITECTURE.md §17.3).

Defined here rather than in ``hardpoint/__init__.py`` so that ``core`` can read
it without importing the root package, which would become a cycle as soon as the
root starts re-exporting core names as public API.
"""

ModelId: TypeAlias = str
"""Identifier for a model, by convention ``"<provider>/<model>"``.

Examples: ``"openai/gpt-4o-mini"``, ``"cohere/rerank-v3"``. The convention is
what the pricing table and the run manifest key on, but nothing validates it:
adapters report whatever their provider calls the model, and an unrecognised id
costs ``None`` rather than a wrong number (ARCHITECTURE.md §21).

A plain alias rather than a class, per the decision rule in INSTRUCTIONS.md §17:
prefer a function, or here a value, over an abstraction that earns nothing.
"""

RunId: TypeAlias = str
"""Identifier for one execution of a pipeline or loop, unique per run.

Carried on ``RunContext``, on every error, and on every log record.
"""

# ``JsonValue`` is re-exported from pydantic rather than redefined. Pydantic's
# version is a recursive ``TypeAliasType`` that its validation machinery already
# understands, so a hand-rolled recursive alias would only be a worse copy.
