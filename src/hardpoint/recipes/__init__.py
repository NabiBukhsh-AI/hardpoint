"""Readable pre-composed pipelines, used as reference and as generator payload.

Recipes are examples, not the only way to compose. Each is a ``build(res)``
function a project can point ``project.pipeline`` at, or copy and edit -- which
is the intended use, because the composition belongs to the project
(ARCHITECTURE.md §16, "must remain concrete and visible").

They may import any domain module but never an adapter and never the CLI.
"""
