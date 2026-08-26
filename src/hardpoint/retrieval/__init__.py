"""Retriever composition, fusion, refinement, and context assembly.

Owns the composable steps from which a retrieval strategy is built. Does not own
which vector store you use, and never imports an adapter.

Populated in milestones M1 and M4.
"""
