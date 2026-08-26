# 0001 — Toolchain and packaging

**Status:** accepted · **Milestone:** M0 · **Governs:** INSTRUCTIONS.md §1 `[FREE]` rows

## Chosen

- **uv + hatchling + `src/` layout.** uv gives reproducible, fast environments
  and native `[dependency-groups]` support. Hatchling is the lightest PEP 517
  backend that reads the version from a single source in `__init__.py`.
- **ruff for both lint and format**, line length 100, a broad rule selection
  including `D` (Google docstrings, required by §4), `T20` (no `print` outside
  the CLI, required by §12.4) and `ANN` (complete public annotations, §12.1).
- **pytest with the anyio plugin bundled in `anyio`** rather than
  `pytest-asyncio`, so the async test runner matches the async runtime the
  library is locked to.
- **Dev interpreter 3.11**, the floor of `requires-python`, so 3.12-only syntax
  cannot creep in unnoticed. CI additionally runs 3.12 and 3.13.

## Rejected

- **Poetry / PDM.** Both work. uv is faster and its lockfile is already the
  ecosystem default for new projects; nothing here depends on the choice.
- **setuptools.** More configuration for no gain in a pure-Python package.
- **black + isort + flake8.** Three tools where ruff is one, with the same
  output.
- **Developing on the newest interpreter.** It hides accidental use of syntax
  that the declared floor does not support until a user reports it.
