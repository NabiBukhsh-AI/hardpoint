# Backlog

Work that is deliberately deferred. Nothing here is a `TODO` left in merged code
(INSTRUCTIONS.md §0.5); each item names the milestone that owns it.

| Item | Milestone | Note |
|---|---|---|
| Console script entry point | M1 | `[project.scripts]` in `pyproject.toml` is commented out until `cli/main.py` exists, so that installing the package cannot produce a `hardpoint` command that raises `ModuleNotFoundError`. Uncomment when M1 lands the CLI. |
| Docs site (MkDocs Material) | M6 | INSTRUCTIONS.md §12.5 requires a docs site with a getting-started path under fifteen minutes. M0 ships `docs/` as plain Markdown only. |
| Public API snapshot test | M6 | ARCHITECTURE.md §24 requires a snapshot test that fails when the public surface changes without an intentional update. Deferred until the surface stops moving every milestone. |
| Pairwise extras install matrix | M6 | ARCHITECTURE.md §16.4. A weekly CI matrix installing each extra alone and commonly co-installed pairs. |
| Framework overhead benchmark | M2 | INSTRUCTIONS.md §12.6 requires a benchmark asserting under 15 ms p95 for a six-step pipeline against fakes. Needs `Pipeline`, which is M1. |
| `LanguageModel` and `EmbeddingModel` contract kits | M1 | M0 ships the `VectorIndex` kit only, per INSTRUCTIONS.md §5.8. The other two are named in the M1 DoD (§6.6). |
| Synchronous facade for `ComponentRegistry.create` | when needed | `create` is async only, per the async-first rule (ADR-007) and the decision rule in INSTRUCTIONS.md §17.5: leave it out until something needs it. The CLI wraps it with `anyio.run`. |
| Cassette record/replay (`testing/cassettes.py`) | M3 | INSTRUCTIONS.md §8. Requires real adapters to record against. |
| `[tool.hardpoint.components]` in the project's `pyproject.toml` | M6 | ARCHITECTURE.md §17.1 lists this as its own precedence level, but INSTRUCTIONS.md §5.6 fixes `Registration.source` to exactly `builtin | project | entrypoint`. Rather than invent a fourth source value that contradicts the locked dataclass, pyproject-declared components will register at the `project` level alongside entry-point discovery in M6. |
| Entry-point component discovery | M6 | The registry's `source="entrypoint"` precedence level is modelled in M0 and gated behind `plugins.discover`; the scan itself lands in M6 (INSTRUCTIONS.md §11). |
