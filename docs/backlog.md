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
| Cassette record/replay (`testing/cassettes.py`) | M3 | INSTRUCTIONS.md §8. Requires real adapters to record against. |
| Entry-point component discovery | M6 | The registry's `source="entrypoint"` precedence level is modelled in M0 and gated behind `plugins.discover`; the scan itself lands in M6 (INSTRUCTIONS.md §11). |
