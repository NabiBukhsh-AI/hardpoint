"""Declared capabilities of models and indexes.

Provider switching stays safe only if steps *query* what an implementation can
do rather than assuming it (ARCHITECTURE.md §9.2). These models are how an
implementation makes that declaration, and how a composition can fail at
construction time with a useful message instead of at 3 a.m. with a provider
error.

Declaring a capability an implementation does not have is a bug the contract
kits are written to catch. Declaring fewer capabilities than an implementation
has is merely conservative, and is always safe.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.errors import CapabilityError
from hardpoint.core.filters import COMPARISON_OPS, STRUCTURAL_OPS
from hardpoint.core.types import ModelId

__all__ = ["ALL_FILTER_OPS", "IndexCapabilities", "ModelCapabilities"]

ALL_FILTER_OPS: frozenset[str] = COMPARISON_OPS | STRUCTURAL_OPS
"""Every operator in the closed filter tree, for an index that supports them all."""


class ModelCapabilities(BaseModel):
    """What a language model supports.

    Args:
        context_window_tokens: Total tokens the model accepts across input and
            output. Used by ``ContextAssembler`` to bound the token budget.
        max_output_tokens: Cap on generated tokens, when the provider states one.
        supports_tools: Whether the model can be given tool definitions.
        supports_structured_output: Whether the model can be constrained to a
            schema, as opposed to being asked nicely in the prompt.
        supports_streaming: Whether ``stream`` yields incremental deltas.
        supports_vision: Whether image parts are accepted in messages.
        supports_prompt_cache: Whether ``CacheHint`` on a message is meaningful.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False
    supports_prompt_cache: bool = False

    def require(
        self,
        model_id: ModelId,
        *,
        tools: bool = False,
        structured_output: bool = False,
        streaming: bool = False,
        vision: bool = False,
    ) -> None:
        """Raise unless the model supports every requested capability.

        Call this at composition time. A step that needs tool calling should
        discover that the configured model cannot do it while the process is
        starting, not on the first user request.

        Args:
            model_id: The model's id, for the error message.
            tools: Require tool calling.
            structured_output: Require schema-constrained output.
            streaming: Require incremental streaming.
            vision: Require image input.

        Returns:
            ``None`` when every requested capability is present.

        Raises:
            CapabilityError: Naming every missing capability and the model.
        """
        wanted = {
            "tools": (tools, self.supports_tools),
            "structured output": (structured_output, self.supports_structured_output),
            "streaming": (streaming, self.supports_streaming),
            "vision": (vision, self.supports_vision),
        }
        missing = [name for name, (asked, has) in wanted.items() if asked and not has]
        if not missing:
            return

        raise CapabilityError(
            f"Model {model_id!r} does not support {', '.join(missing)}.",
            component=model_id,
            remedy=(
                f"Configure a model that supports {', '.join(missing)}, or remove the "
                f"step that requires it. If {model_id!r} does support it, the adapter's "
                f"capabilities() is under-declaring and should be corrected."
            ),
        )

    def fits(self, tokens: int) -> bool:
        """Return whether a token count fits inside the context window."""
        return tokens <= self.context_window_tokens


class IndexCapabilities(BaseModel):
    """What a vector index supports.

    ``filter_ops`` is the load-bearing field. It is what
    ``filters.validate_supported`` checks a query against, and it is why a
    migration between backends surfaces as a named error at query construction
    rather than as a filter that silently stopped filtering.

    Args:
        filter_ops: Operator names from the closed filter tree that this index
            can express.
        supports_hybrid: Whether dense and sparse can be combined in one query.
        supports_sparse: Whether sparse or keyword vectors are supported.
        supports_namespaces: Whether records can be partitioned by namespace.
        supports_delete_by_filter: Whether ``delete`` accepts a filter as well
            as explicit ids. Ingestion needs this to remove a document's chunks.
        max_metadata_bytes: Cap on serialised metadata per record, when the
            backend states one.
        delete_consistency: Whether a delete is visible to the next query
            (``consistent``) or only eventually (``eventual``). Contract kits
            and the sync engine both need to know.
        max_top_k: Largest ``top_k`` the backend will honour, when capped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filter_ops: frozenset[str] = frozenset()
    supports_hybrid: bool = False
    supports_sparse: bool = False
    supports_namespaces: bool = False
    supports_delete_by_filter: bool = True
    max_metadata_bytes: int | None = Field(default=None, gt=0)
    delete_consistency: Literal["consistent", "eventual"] = "consistent"
    max_top_k: int | None = Field(default=None, gt=0)

    def unsupported_ops(self, required: Collection[str]) -> frozenset[str]:
        """Return the required operators this index cannot express."""
        return frozenset(required) - self.filter_ops
