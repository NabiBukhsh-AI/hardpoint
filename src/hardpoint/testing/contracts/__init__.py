"""Parametrised conformance suites, one per port.

A third party proves an implementation conforms by binding a factory to the kit::

    from hardpoint.testing.contracts import vector_index_contract

    TestMyIndex = vector_index_contract(lambda: MyIndex(...))

An adapter is not "supported" until it passes its kit (ARCHITECTURE.md §23).

Nothing here imports ``pytest`` at module scope. The kits build their test
classes inside a function, so importing this package in an environment with only
the five base dependencies installed still works (INSTRUCTIONS.md §3
**[LOCKED]**).
"""

from __future__ import annotations

from hardpoint.testing.contracts.vector_index import vector_index_contract

__all__ = ["vector_index_contract"]
