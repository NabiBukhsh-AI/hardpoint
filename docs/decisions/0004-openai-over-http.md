# 0004 — OpenAI-compatible adapters over httpx, not the OpenAI SDK

**Status:** accepted · **Milestone:** M1
**Governs:** INSTRUCTIONS.md §6.3 ("one OpenAI-compatible chat adapter, one
OpenAI-compatible embeddings adapter"), ARCHITECTURE.md §16.2 and §16.4

## Chosen

`OpenAICompatibleChat` and `OpenAICompatibleEmbeddings` speak the OpenAI wire
protocol directly over `httpx`. They require **no extra**.

## Reasoning

"OpenAI-compatible" names a wire protocol, not a library. OpenAI, Ollama, vLLM,
LM Studio, Together, Groq and Azure OpenAI all serve `POST /chat/completions`
and `POST /embeddings` with the same request and response shapes. An adapter
written against the protocol supports all of them, and a new provider costs a
`base_url` rather than a new adapter.

Three consequences follow, each of which the architecture asks for elsewhere:

- **`pip install hardpoint` is immediately useful.** `httpx` is already one of
  the five base dependencies, so the smallest useful system — an
  OpenAI-compatible endpoint plus `InMemoryVectorIndex` — needs no extras at all.
- **ARCHITECTURE.md §16.4 is satisfied completely.** It asks that an adapter be
  rewritable against a new major SDK version without a hardpoint breaking
  change. Not depending on an SDK removes the question.
- **The surface is smaller.** §6.3 asks for the adapter "chosen for the smallest
  surface"; two POST endpoints and an SSE parser is smaller than an SDK plus its
  transitive dependencies.

## Rejected

- **The `openai` SDK behind the `openai` extra.** Adds a dependency, its
  transitive tree and its release cadence, in exchange for the same two
  endpoints. It also narrows the adapter to OpenAI-shaped auth and retry
  behaviour that this library must override anyway, since retry lives in exactly
  one place.
- **One adapter per provider.** The whole point of a shared wire protocol is
  that this is unnecessary. A provider that genuinely diverges — Anthropic's
  messages API, Bedrock's signing — gets its own adapter and its own extra, and
  the extras stay declared in `pyproject.toml` for exactly that.

## Consequence

The `openai` extra remains declared but unused. It is kept because a future
native adapter may want it, and because removing a published extra is a
breaking change while leaving one unused is not.
