"""Execution runtime: pipelines, control loops, dispatchers, and policies.

Owns Step composition, deadline and budget enforcement, degradation collection,
and the single implementation of retry, timeout, circuit breaking, rate limiting
and fallback. Does not own the domain semantics of retrieval or generation.

Populated in milestone M1.
"""
