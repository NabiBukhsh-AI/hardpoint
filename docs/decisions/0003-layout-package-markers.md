# 0003 — Docstring-only package markers for unbuilt modules

**Status:** accepted · **Milestone:** M0
**Governs:** INSTRUCTIONS.md §2 (layout) against §3 (locked import-linter
contracts) and AGENT_PROMPT.md Prompt A ("do not scaffold M1 modules")

## Chosen

Every package named in the §2 layout exists from M0 as a directory whose
`__init__.py` contains **only** a module docstring stating the package's purpose,
its boundaries, and the milestone that populates it. No functions, no classes,
no stubs, no imports.

## Reasoning

The four import-linter contracts in §3 are `[LOCKED]` and reproduced verbatim.
They name `hardpoint.runtime`, `hardpoint.adapters`, `hardpoint.eval` and others
that M0 does not build. import-linter resolves `source_modules` and
`forbidden_modules` against the real import graph and errors on a module that is
absent, so with no such packages the locked contracts cannot run at all — and
the M0 Definition of Done requires `lint-imports` to pass all four.

The alternatives were to weaken a locked contract or to defer layer enforcement
to a later milestone. Both are worse: §3 says explicitly that the enforcement,
not the tool, is what is locked, and ARCHITECTURE.md §8.1 makes the point that
these rules rot within two releases if they are not enforced from day one.

A docstring-only `__init__.py` is layout, not implementation. It satisfies the
§12.5 requirement that every public module carry a docstring giving purpose and
boundaries, and it commits no design decisions that a later milestone would have
to undo.

## Rejected

- **Trimming the contracts to modules that exist today.** Modifies a `[LOCKED]`
  block, and the trimmed rule would have to be widened by hand at each
  milestone — exactly the drift the lock exists to prevent.
- **Deferring `lint-imports` to M1.** Fails the M0 DoD, and lets a layer
  violation into `core` before anything is enforcing the layers.
- **Writing placeholder implementations.** Forbidden by Prompt A and by §0.5.
