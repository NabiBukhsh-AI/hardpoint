"""Five-layer configuration resolution, validation, snapshotting and redaction.

Owns the merge order defined in ARCHITECTURE.md §15.1 -- library defaults,
``base.yaml``, ``{env}.yaml``, ``HARDPOINT__`` environment overrides, then
explicit code overrides -- plus ``${env:VAR}`` interpolation, secret redaction
and the hashed, immutable ``ConfigSnapshot``.

Owns no control flow: configuration selects and parameterises components, it
never expresses step ordering (ADR-005).
"""
