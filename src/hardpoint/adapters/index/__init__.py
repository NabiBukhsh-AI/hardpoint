"""VectorIndex adapters.

M1 ships one, speaking Qdrant's REST API over ``httpx``. Needs no extra, and a
local Qdrant container is the fastest way to run the vertical slice against a
real index.

``InMemoryVectorIndex`` in ``hardpoint.testing`` is the other complete
implementation, and is a conformance target rather than a toy: it passes the
same contract kit.
"""

from __future__ import annotations

from hardpoint.adapters.index.qdrant import QDRANT_FILTER_OPS, QdrantIndex, to_qdrant_filter

__all__ = ["QDRANT_FILTER_OPS", "QdrantIndex", "to_qdrant_filter"]
