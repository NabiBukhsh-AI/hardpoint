"""Tools, tool permissions, and the agent turn step.

Owns the Tool protocol, the registry that renders provider-agnostic tool schemas,
and the authorisation gate consulted before a tool executes. Does not own graph
orchestration.

Populated in milestone M5.
"""
