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

CONTRACT_VERSION = "1.0"
"""Version of the port Protocols, independent of ``__version__``.

Ports change only when this increments, which is rare and always accompanied by
a migration note. Third-party components declare the contract version they
target so the registry can warn on a mismatch (ARCHITECTURE.md §17.3).
"""

__all__ = ["CONTRACT_VERSION", "__version__"]
