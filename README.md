# hardpoint

**Contracts, runtime, ingestion and evaluation for production RAG and agentic systems.**

[![CI](https://github.com/NabiBukhsh-AI/hardpoint/actions/workflows/ci.yml/badge.svg)](https://github.com/NabiBukhsh-AI/hardpoint/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> **Status: pre-alpha, under active construction.** Milestone M0 (foundations) is
> the current scope. The public API is not yet stable. See
> [CHANGELOG.md](CHANGELOG.md) and [docs/backlog.md](docs/backlog.md).

---

## What this is

`hardpoint` is two things that ship together and are deliberately kept separate.

1. **A runtime library.** Stable contracts, a small execution runtime, provider
   adapters, an ingestion sync engine, evaluation and observability primitives.
   You import it. You do not read it.
2. **A project generator.** A CLI that scaffolds a working repository: pipeline
   composition, prompts, config, eval datasets, ingestion jobs, the service
   entry point, tests, CI and containers. You own it. You read and edit it.

The line between them is the single most important rule in the design:

> **The modification boundary.** Code an engineer will predictably need to read
> and change lives in the *generated project*. Code an engineer will call but
> should not need to understand internally lives in the *library*. Anything
> ambiguous defaults to the generated project.

## Why

The dominant failure mode of AI application frameworks is not missing features.
It is that the interesting logic — retrieval strategy, prompts, context
assembly — gets buried behind configuration surfaces and inheritance chains, and
debugging becomes archaeology.

`hardpoint` inverts that. The interesting logic is ordinary Python in your
repository. The boring logic — retries, transport, tracing, cost accounting,
idempotent indexing, eval plumbing — is behind a versioned API.

What a composed pipeline is meant to look like:

```python
# pipelines/support_qa.py
def build(res: Resources) -> Pipeline:
    return Pipeline(
        name="support_qa",
        steps=[
            NormaliseQuery(),
            MultiQueryExpansion(res.llm, n=3, prompt=res.prompts["expand"]),
            ParallelRetrieve([
                VectorRetriever(res.indexes["primary"], res.embedder, top_k=40),
                SparseRetriever(res.indexes["primary"], top_k=40),
            ], fusion=ReciprocalRankFusion(k=60)),
            RerankStep(res.reranker, top_k=8, on_failure="skip"),
            ParentExpansion(res.documents),
            ContextAssembler(token_budget=6000, ordering="relevance_with_edges"),
            Generate(res.llm, prompt=res.prompts["answer"], stream=True),
            GroundednessGuard(min_supported_ratio=0.7, action="flag"),
        ],
    )
```

Nothing is hidden. Reading it tells you exactly what happens and in what order,
and every element is replaceable by a class you write yourself.

## Design principles

1. **Narrow ports, wide userland.** Abstract the boundary to the outside world.
   Do not abstract the user's domain logic.
2. **Contracts before implementations.** Every port ships with a Protocol, a
   fake, and a parametrised conformance suite in the same release.
3. **No import-time side effects.** Importing `hardpoint` starts no clients,
   reads no environment, opens no sockets.
4. **Explicit over implicit.** No global state, no ambient context, no
   auto-wiring container, no monkey patching.
5. **Typed at the seams.** Pydantic models on every boundary, `mypy --strict`
   across the package.
6. **Failure is a first-class output.** Degradation is data in the result
   object, not a line in a log.
7. **Cost and latency are data, not folklore.** Every run yields a `Usage`
   record attributed per step.
8. **Reproducibility is a feature.** Config snapshot, prompt version, model id
   and index epoch are recorded on every run.
9. **Optional means optional.** Missing extras raise an actionable error, never
   a raw `ImportError` traceback.
10. **Prefer deletion.** A capability that cannot justify its maintenance cost
    is removed, not deprecated into permanence.

## Architecture at a glance

| Layer | Owns | Does not own |
|---|---|---|
| `core` | Ports, data models, errors, `RunContext`, config, registry, deterministic ids | Any I/O, any provider knowledge, any execution |
| `runtime` | `Step`, `Pipeline`, `ControlLoop`, policies, budgets, deadlines | Domain semantics of retrieval or generation |
| `ingestion` | Sources, parsers, chunkers, the sync engine, state, index epochs | Vector store transport |
| `retrieval` | Retriever composition, fusion, refinement, context assembly | Which vector store you use |
| `generation` | Prompt rendering, LLM steps, structured output | Prompt content |
| `guards` | Guard contract and built-in guards | Classifier models, safety policy |
| `eval` | Datasets, runner, metrics, baselines, gates | Your golden data |
| `observability` | Tracer/MetricSink ports, usage aggregation, pricing | Being a backend |
| `adapters` | Transport only, per provider | Retries, tracing, business logic |
| `testing` | Fakes, contract kits, deterministic fixtures | Being importable in production paths |

Layer rules are enforced in CI by `import-linter`, not by convention. `core`
imports nothing from `hardpoint` outside `core`; no domain module imports an
adapter; adapters are leaves.

## Installation

Not yet published. Once released:

```bash
pip install hardpoint                 # base: pydantic, anyio, httpx, pyyaml, typer
pip install 'hardpoint[qdrant]'       # plus one vector index adapter
pip install 'hardpoint[openai,otel]'  # plus a provider and tracing
```

The base install pulls no provider SDKs, no ML frameworks and no OpenTelemetry.
Optional integrations live behind extras and are imported lazily, inside the
factory that needs them, never at module import time.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NabiBukhsh-AI/hardpoint.git
cd hardpoint
uv sync

uv run ruff check                    # lint
uv run ruff format --check           # format
uv run mypy --strict src/hardpoint   # types
uv run lint-imports                  # layer rules
uv run pytest                        # tests
```

All five must be clean. CI runs them on Python 3.11, 3.12 and 3.13, plus a
bare-environment job that installs only the base dependencies and asserts every
`hardpoint.*` module imports without an optional dependency present.

## Contributing

Read [`docs/decisions/`](docs/decisions/) before proposing a change to a design
choice; the alternatives that were considered and rejected are recorded there.
Additions to `core` must state which port or invariant they serve.

## License

[Apache-2.0](LICENSE)
