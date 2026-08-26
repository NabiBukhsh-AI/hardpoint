# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`CONTRACT_VERSION` is versioned separately from the distribution version. Port
Protocols change only when `CONTRACT_VERSION` increments, and every increment
carries a migration note here.

## [Unreleased]

### Added

- Repository scaffold: `pyproject.toml` with the locked five-dependency base
  install, extras per ARCHITECTURE.md §16.2, ruff/mypy/pytest/coverage
  configuration, and the four locked import-linter contracts.
- CI workflow running lint, format check, strict type check, import-lint and
  tests on Python 3.11, 3.12 and 3.13, plus a bare-environment import job.
- `hardpoint.__version__` and `hardpoint.CONTRACT_VERSION`.

[Unreleased]: https://github.com/NabiBukhsh-AI/hardpoint/compare/HEAD...HEAD
