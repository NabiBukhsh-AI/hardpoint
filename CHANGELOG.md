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
- `core.types`: `JsonValue`, `ModelId`, `RunId`.
- `core.errors`: the exception taxonomy from ARCHITECTURE.md §18.1. Every error
  carries a code and renders what failed, where, why and what to do next; every
  `ConfigError` requires a remedy, enforced against the constructor signature.
- `core.ids`: deterministic `document_id`, `chunk_id`, `content_hash`,
  `text_hash`, `stable_hash` and `normalise_text`, with a golden fixture and an
  independently written reference implementation in the tests.
- `core.filters`: the closed, serialisable metadata filter tree with the `F`
  builder, normative operator semantics in `matches`, and `validate_supported`
  raising `UnsupportedFilterError` rather than dropping a clause.
- `core.capabilities`: `ModelCapabilities` and `IndexCapabilities`, with
  `require` failing at composition time.
- `core.models`: the boundary data models, frozen and `extra="forbid"`.
- `core.config`: five-layer resolution (defaults, `base.yaml`, `{env}.yaml`,
  `HARDPOINT__` environment overrides, code overrides) with per-key origin
  tracking, `${env:VAR}` and `${env:VAR:-default}` interpolation that names the
  file and line on a miss, secret redaction, and a hashed immutable
  `ConfigSnapshot` whose only unredacted accessor is `reveal()`.
- `core.registry`: `ComponentRegistry` with a lazy built-in table, deterministic
  project > entrypoint > builtin precedence, collision errors naming both
  sources, edit-distance suggestions for an unknown key, per-component option
  validation, and `MissingDependencyError` carrying the exact install command.

[Unreleased]: https://github.com/NabiBukhsh-AI/hardpoint/compare/HEAD...HEAD
