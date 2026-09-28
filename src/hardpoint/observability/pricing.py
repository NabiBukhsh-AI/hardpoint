"""Cost accounting from a shipped, overridable price table.

ARCHITECTURE.md §21 and INSTRUCTIONS.md §7. ``pricing.yaml`` ships with the
library; ``pricing:`` in project configuration overrides or extends it per model.

**An unknown model is unpriced, never free.** Its cost is ``None`` -- which makes
every total it contributes to ``None`` too -- and a warning is emitted once per
model per table, which in practice means once per process
(INSTRUCTIONS.md §7 **[LOCKED]**, §13.8). A silently wrong zero is how a bill
surprises somebody at the end of the month.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from importlib.resources import files
from typing import Final

import yaml

from hardpoint.core.config.schema import ModelPriceConfig
from hardpoint.core.errors import HardpointWarning
from hardpoint.core.models import StepUsage

__all__ = ["PricingTable", "UnpricedModelWarning"]

_LOGGER = logging.getLogger("hardpoint.observability.pricing")
_PER_MILLION: Final = 1_000_000


class UnpricedModelWarning(HardpointWarning):
    """A provider call was made with a model the price table does not know."""


class PricingTable:
    """Model id to price, with the shipped table under project overrides.

    Args:
        overrides: ``pricing:`` from configuration. A model listed here replaces
            the shipped entry entirely.
        include_shipped: Whether to start from ``pricing.yaml``.
    """

    def __init__(
        self,
        overrides: Mapping[str, ModelPriceConfig] | None = None,
        *,
        include_shipped: bool = True,
    ) -> None:
        self.prices: dict[str, ModelPriceConfig] = dict(_shipped() if include_shipped else {})
        self.prices.update(overrides or {})
        self._warned: set[str] = set()

    def price(self, model_id: str) -> ModelPriceConfig | None:
        """Return a model's price, or ``None`` when it is unknown."""
        return self.prices.get(model_id)

    def cost(self, model_id: str, usage: StepUsage) -> float | None:
        """Price one call's usage, or ``None`` -- with a one-time warning -- when unknown.

        Fake models (``fake/...``) cost nothing and are not warned about: they
        never leave the process.
        """
        if model_id.startswith("fake/"):
            return 0.0
        price = self.price(model_id)
        if price is None:
            self._warn(model_id)
            return None
        return (
            usage.prompt_tokens * (price.input or 0.0)
            + usage.completion_tokens * (price.output or 0.0)
            + usage.embed_tokens * (price.embed or 0.0)
        ) / _PER_MILLION + usage.calls * (price.per_call or 0.0)

    def embed_price_per_million(self, model_id: str) -> float | None:
        """The embedding price, for ``ingest run --plan``; ``None`` when unknown."""
        if model_id.startswith("fake/"):
            return 0.0
        price = self.price(model_id)
        return None if price is None or price.embed is None else price.embed

    def _warn(self, model_id: str) -> None:
        if model_id in self._warned:
            return
        self._warned.add(model_id)
        message = (
            f"No price is known for model {model_id!r}; its cost is reported as unknown "
            f"(null), not zero. Add it under `pricing:` in config/base.yaml, for "
            f"example `pricing: {{{model_id}: {{input: 0.5, output: 1.5}}}}` "
            f"(US dollars per million tokens)."
        )
        _LOGGER.warning(message)
        warnings.warn(message, UnpricedModelWarning, stacklevel=3)

    def __repr__(self) -> str:
        """Render how many models are priced."""
        return f"PricingTable(models={len(self.prices)})"


def _shipped() -> dict[str, ModelPriceConfig]:
    """Load ``pricing.yaml`` from the package."""
    raw = yaml.safe_load(files("hardpoint.observability").joinpath("pricing.yaml").read_text())
    models = raw.get("models", {}) if isinstance(raw, dict) else {}
    return {str(model): ModelPriceConfig.model_validate(price) for model, price in models.items()}
