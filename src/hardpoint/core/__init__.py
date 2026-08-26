"""Contracts, data models, errors, configuration, identity and the registry.

``core`` is the foundation every other layer is written against. It owns the
port Protocols, the Pydantic data models that cross every boundary, the error
taxonomy, ``RunContext``, configuration loading and validation, the component
registry, and deterministic id and hash derivation.

It owns no I/O, no provider knowledge and no execution. It imports nothing from
``hardpoint`` outside ``core``, and its third-party surface is limited to
``pydantic``, ``anyio`` and the standard library (ARCHITECTURE.md §8.1 R1,
enforced by the "core is independent" import-linter contract).
"""
