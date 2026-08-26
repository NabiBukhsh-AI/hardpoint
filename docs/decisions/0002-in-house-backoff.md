# 0002 — In-house backoff instead of tenacity

**Status:** accepted · **Milestone:** M0 (decision) / M1 (implementation)
**Governs:** ARCHITECTURE.md §16.1, which leaves this open: "`tenacity` (or a
small in-house backoff, decided at implementation)"

## Chosen

A small in-house backoff inside `runtime/policies.py`. `tenacity` is **not** a
base dependency.

## Reasoning

INSTRUCTIONS.md §1 is `[LOCKED]`: "Base runtime dependencies must remain:
`pydantic`, `anyio`, `httpx`, `pyyaml`, `typer`." Adding `tenacity` would need
architectural review, and the two documents conflict on this point only because
ARCHITECTURE.md deferred the call. The locked list wins.

The behaviour actually required by §6.1 is narrow: exponential backoff with full
jitter, a retryable-error predicate driven by the error taxonomy, `Retry-After`
honoured when the error carries it, and a span event per attempt. `tenacity`'s
value is its breadth of strategies, none of which are wanted here, and its
retry-state model would have to be adapted to emit our span events anyway.

## Rejected

- **`tenacity` as a base dependency.** Breaks a locked constraint to save
  roughly forty lines.
- **`tenacity` behind an extra.** Retry is not optional; it is on the default
  path for every provider call. An optional dependency on the default path is
  the worst of both.
