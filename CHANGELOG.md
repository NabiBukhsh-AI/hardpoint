# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`CONTRACT_VERSION` is versioned separately from the distribution version. Port
Protocols change only when `CONTRACT_VERSION` increments, and every increment
carries a migration note here.

## [Unreleased]

### Fixed

- `ConfigSnapshot.get()` returned unredacted data when the path addressed a
  section rather than a leaf, so reading a whole config block exposed any
  secret inside it. It now reads the redacted view at every depth.
- `ConfigSnapshot` unpickled with an empty `hash`, which would have let a run
  manifest record `config_hash=""` rather than failing.
- `parse_env_overrides` sorted by raw variable name, so with case-differing
  names a shorter `HARDPOINT__A__B` could silently replace the section built by
  `HARDPOINT__A__B__C` instead of reporting the conflict.
- `RecordingTracer` kept one shared span stack, producing a wrong parent/child
  tree whenever two tasks opened spans concurrently. Nesting is now task-local.
- `_deep_merge` left origin entries for keys an overlay had replaced, so
  `config show` could annotate paths that no longer exist.
- `Deadline.check` reported `limit_value=0.0, observed=0.0`; it now reports how
  far past the deadline the run is.
- `LocalFileSource` crashed on a relative `root` (`Path.as_uri` refuses one), so
  `root: docs` -- the natural configuration -- failed on the first file. The
  root is now resolved to an absolute path.
- `RecursiveChunker` carried overlap across heading boundaries, putting the
  previous section's tail *above* the next heading so the chunk took the
  previous section's heading path. Overlap now applies only to size splits
  inside a section.
- `IngestReport.cost_usd` started as `None` and was only ever summed when it was
  not, so every run reported "unpriced". It now starts at zero and becomes
  `None` only when an unpriced embedding happened.
- `SyncEngine` sent every chunk of a document in one embedding request, ignoring
  `ingestion.embed_batch_size`.
- The config loader parsed YAML 1.1 booleans, so the `on:` key in
  `retry: {on: [...]}` -- ARCHITECTURE.md §15.2's own example -- loaded as
  `True`. Only `true`/`false` are booleans now.
- `${env:...}` inside a mapping inside a list (a guard spec) was not
  interpolated, and a secret there would have been marked at a path the
  redactor never visits. Interpolation is now fully recursive, and a secret
  inside a list redacts the whole list.
- Loaded config values kept a private `str` subclass, so `yaml.safe_dump` of a
  snapshot failed.
- SQLite connections in tests were never closed, which Python 3.13 reports as a
  `ResourceWarning` and the suite treats as an error.

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
- `core.context`: `RunContext` with its locked nine-attribute scope, plus
  `Deadline` (absolute, monotonic, only ever tightening), `Budget`,
  `UsageAccumulator` and `CacheHandle`.
- `core.ports`: every provider Protocol -- `LanguageModel`, `EmbeddingModel`,
  `VectorIndex`, `Reranker`, `DocumentParser`, `Chunker`, `CacheBackend`,
  `Tracer`, `Span`, `MetricSink`, `Tool`, `PromptStore`, `StateStore`,
  `BlobStore` -- with the request and result types that form their contracts.
- `hardpoint.testing`: `FakeLanguageModel`, `FakeEmbeddingModel`,
  `InMemoryVectorIndex`, `FakeReranker`, `FakeCache`, `RecordingTracer`,
  `RecordingMetricSink`, and `build_run_context` plus deterministic model
  builders.
- `hardpoint.testing.contracts.vector_index_contract`: the `VectorIndex`
  conformance suite, exported for third-party adapter authors.
  `InMemoryVectorIndex` passes it in four configurations.
- `runtime`: `Step`, `StepResult`, `FailurePolicy` and `as_step`; `Pipeline`
  with `run_sync`, `explain` and `describe`; `gather_bounded` and
  `gather_tolerant`; and the single implementation of `Retry`, `Timeout`,
  `CircuitBreaker`, `RateLimit` and `Fallback`, composed by `PolicyChain` in a
  fixed order.
- `observability`: `NoOpTracer` and `NoOpMetricSink`, the defaults that let a
  `RunContext` be built outside a test.
- `core.tokens`: `estimate_tokens`, the documented fallback both the chunker and
  the context assembler measure with when no tokenizer is available.
- `ingestion`: the `Source` protocol and `LocalFileSource`; `TextParser` and
  `MarkdownParser`; `RecursiveChunker`, which splits on headings and paragraph
  boundaries and carries the heading path into each chunk; and `ChunkValidator`
  with the empty, too-short, too-long, boilerplate and duplicate rules.
- `ingestion.state.SqliteStateStore`: the manifest, with the versioned schema
  from INSTRUCTIONS.md §6.2 and per-document atomic commits.
- `ingestion.sync.SyncEngine`: the idempotent, incremental, delete-correct and
  resumable sync engine, with `plan()` for `--plan`.
- `ingestion.report`: `IngestReport` and the JSONL quarantine artefact.
- `core.ports.IndexMeta` and `StateStore.index_meta` / `record_index_meta`,
  completing the port against the `index_meta` table the schema specifies.
- `retrieval`: `VectorRetriever` and `ContextAssembler`, with token budgeting,
  three orderings, citation keys, recorded drops, and a delimited context block
  labelled as untrusted data.
- `generation`: `Generate` with streaming, citations and a populated
  `RunManifest`; `PromptTemplate`, `InMemoryPromptStore` and `FilePromptStore`,
  with prompt versions derived from a content hash.
- `adapters`: `OpenAICompatibleChat` and `OpenAICompatibleEmbeddings`, speaking
  the OpenAI wire protocol over `httpx` and requiring no extra, with every
  provider failure mapped into the error taxonomy.
- `testing.contracts.language_model_contract` and `embedding_model_contract`,
  exported for third-party adapter authors.
- `adapters.index.QdrantIndex`, over Qdrant's REST API and requiring no extra,
  with faithful filter translation and `contains` deliberately undeclared.
- `adapters.index.SqliteVectorIndex` (`type: sqlite`): a vector index in one
  file, standard library only, passing the full `VectorIndex` contract kit. What
  `rag-minimal` starts with.
- The built-in component table: `openai_chat`, `openai_embeddings`, `qdrant`,
  `sqlite`, `local_files`, the `sqlite` state store, and `fake_llm`,
  `fake_embeddings`, `memory` and `fake_reranker` for offline runs. Each
  adapter module carries its `<Component>Config` and a `build` factory.
- `runtime.Resources`, `build_resources`, `load_pipeline` and `answer_query`:
  configuration to policy-wrapped components, and the one path from a question
  to an `Answer` that the CLI, the service and the eval runner all share.
- `runtime.wrapped`: `PolicyLanguageModel`, `PolicyEmbeddingModel`,
  `PolicyVectorIndex` and `PolicyReranker`, applying each component's configured
  retry, timeout, breaker, rate limit and fallback.
- `recipes.naive`: retrieve, assemble, generate.
- The `hardpoint` command: `init` (with `--diff`), `config show|schema|validate`,
  `components list` (with `--describe` and `--resolved`), `ingest run|plan|status`,
  `ask` (with `--explain` and `--json`), `doctor` (with `--live`) and `version`.
- `templates/rag-minimal`: a project that ingests and answers offline with no
  keys (`--env offline`) and against any OpenAI-compatible endpoint otherwise,
  with its own passing test suite.
- Config: `project` (name, `hardpoint_version`, `pipeline`, `prompts_dir`),
  `ingestion.index`, `ingestion.state` and `ingestion.chunking`. A component
  block whose `type:` changes in an overlay is replaced rather than merged.
- `ContextItem.score` and `DropRecord.score`, so `ask --explain` shows why each
  chunk was used or dropped.
- `FakeEmbeddingModel(lexical=True)` and `FakeLanguageModel(extractive=True)`,
  which make offline retrieval find the relevant passage and offline answers
  quote it.

[Unreleased]: https://github.com/NabiBukhsh-AI/hardpoint/compare/HEAD...HEAD
