"""Fakes, contract test kits and deterministic fixtures.

Everything here is public API: third parties import the contract kits to prove
their own adapters conform. Nothing here belongs on a production path.

Imports ``core`` and ``runtime`` only, so that a test kit never drags an
optional dependency into a consumer's environment (ARCHITECTURE.md §8.1 R6).
"""
