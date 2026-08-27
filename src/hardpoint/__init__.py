"""hardpoint: contracts and runtime for production RAG and agentic systems.

This module is the public API surface. Only names re-exported here and from each
subpackage's ``__init__`` are public; everything else is internal and may change
in a minor release (ARCHITECTURE.md §24).

Importing ``hardpoint`` starts no clients, reads no environment, opens no
sockets and registers no plugins. That property is a design principle
(ARCHITECTURE.md §5.3) and is asserted by a test.
"""

from __future__ import annotations

__version__ = "0.0.0"
"""Distribution version. Single source of truth; hatchling reads it from here."""

from hardpoint.core.types import CONTRACT_VERSION  # noqa: E402 - must follow __version__

__all__ = ["CONTRACT_VERSION", "__version__"]
