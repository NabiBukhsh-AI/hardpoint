"""Provider adapters. Transport only.

Each adapter maps one external system onto a core port. Adapters never implement
retries, caching or tracing decisions, never expose a provider SDK type in a
public signature, and never import another adapter or any domain module.

Populated from milestone M1 onward.
"""
