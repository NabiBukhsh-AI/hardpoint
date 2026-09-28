"""``hardpoint config schema``: one JSON Schema for the whole configuration.

ARCHITECTURE.md §15.3: an editor pointed at this schema autocompletes and
validates YAML in place, including each component's own options. The root
models' schema comes from Pydantic; every component-typed slot is then replaced
by a choice over the registered components of that kind, each a closed object
with ``type`` fixed to its key.
"""

from typing import Any

from hardpoint.core.config.schema import HardpointConfig, PolicyConfig
from hardpoint.core.errors import HardpointError
from hardpoint.core.registry import ComponentRegistry, Kind

__all__ = ["config_schema"]

# Where each kind's components are configured, as (definition, property) pairs
# in Pydantic's generated schema, plus whether the slot is a name -> spec map.
_SLOTS: dict[Kind, list[tuple[str, str, bool]]] = {
    Kind.LLM: [("ProvidersConfig", "llm", False)],
    Kind.EMBEDDINGS: [("ProvidersConfig", "embeddings", False)],
    Kind.RERANKER: [("ProvidersConfig", "reranker", False)],
    Kind.INDEX: [("HardpointConfig", "indexes", True)],
    Kind.SOURCE: [("HardpointConfig", "sources", True)],
    Kind.STATE: [("IngestionConfig", "state", False)],
    Kind.TRACER: [("ObservabilityConfig", "tracer", False)],
    Kind.METRICS: [("ObservabilityConfig", "metrics", False)],
}


def _component_schema(key: str, model_schema: dict[str, Any]) -> dict[str, Any]:
    properties = dict(model_schema.get("properties", {}))
    properties["type"] = {"const": key}
    properties["policies"] = {"$ref": "#/$defs/PolicyConfig"}
    return {
        "type": "object",
        "title": model_schema.get("title", key),
        "properties": properties,
        "required": sorted({"type", *model_schema.get("required", [])}),
        "additionalProperties": False,
    }


def config_schema(registry: ComponentRegistry) -> dict[str, Any]:
    """Build the schema. Components whose extra is not installed are left out.

    Returns:
        A JSON Schema document.
    """
    schema = HardpointConfig.model_json_schema()
    definitions: dict[str, Any] = schema.setdefault("$defs", {})
    policies = PolicyConfig.model_json_schema()
    for name, nested in policies.pop("$defs", {}).items():
        definitions.setdefault(name, nested)
    definitions.setdefault("PolicyConfig", policies)

    for kind, slots in _SLOTS.items():
        choices: list[dict[str, Any]] = []
        for key in registry.keys(kind):
            try:
                model = registry.resolve(kind, key).config_model
            except HardpointError:
                continue
            component = model.model_json_schema()
            for name, nested in component.pop("$defs", {}).items():
                definitions.setdefault(name, nested)
            reference = f"component_{kind.value}_{key}"
            definitions[reference] = _component_schema(key, component)
            choices.append({"$ref": f"#/$defs/{reference}"})
        if not choices:
            continue

        for owner, prop, is_map in slots:
            target = schema if owner == "HardpointConfig" else definitions.get(owner, {})
            properties = target.get("properties", {})
            if prop not in properties:
                continue
            choice = {"anyOf": choices}
            properties[prop] = (
                {"type": "object", "additionalProperties": choice}
                if is_map
                else {"anyOf": [*choices, {"type": "null"}], "default": None}
            )
    return schema
