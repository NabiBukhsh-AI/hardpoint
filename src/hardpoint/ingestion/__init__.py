"""Sources, parsing, chunking, validation, and the idempotent sync engine.

Owns change detection, incremental re-embedding, deletion handling, ingestion
state, and index epoch management. Does not own vector store transport, which is
an adapter concern.

Populated in milestone M1.
"""
