"""The component registry. INSTRUCTIONS.md §5.6, ARCHITECTURE.md §17.

Two locked behaviours dominate this file:

- **Built-ins are lazy.** Importing the registry imports no adapter. Proved in a
  subprocess, because ``sys.modules`` in a test session says nothing about what
  a fresh process would do.
- **A missing extra is a ``MissingDependencyError`` carrying the exact install
  command,** never a raw ``ModuleNotFoundError``. Proved end to end against a
  real module written to disk that imports a package which genuinely is not
  installed, rather than by mocking the import system.

The registry also has to tell that failure apart from "the adapter module does
not exist", which means the opposite thing and would send a user to install an
extra that could not help.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from hardpoint.core.errors import (
    ContractError,
    DuplicateComponentError,
    HardpointWarning,
    InvalidConfigError,
    MissingDependencyError,
    UnknownComponentError,
)
from hardpoint.core.registry import (
    BUILTIN_COMPONENTS,
    BuiltinEntry,
    ComponentRegistry,
    Kind,
)
from hardpoint.core.types import CONTRACT_VERSION


class ToyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "http://localhost"
    top_k: int = 10


class Toy:
    def __init__(self, config: ToyConfig) -> None:
        self.config = config


def build_toy(config: BaseModel) -> Toy:
    assert isinstance(config, ToyConfig)
    return Toy(config)


async def build_toy_async(config: BaseModel) -> Toy:
    assert isinstance(config, ToyConfig)
    return Toy(config)


@pytest.fixture
def registry() -> ComponentRegistry:
    return ComponentRegistry()


# --------------------------------------------------------------------------- #
# Laziness  [LOCKED]                                                          #
# --------------------------------------------------------------------------- #


def test_importing_the_registry_imports_no_adapter() -> None:
    """**[LOCKED]** Built-ins are a table of module paths, not imports.

    Checked in a fresh interpreter: what this test session has already imported
    says nothing about what ``import hardpoint.core.registry`` does on its own.
    """
    program = textwrap.dedent(
        """
        import sys
        import hardpoint.core.registry as registry
        registry.ComponentRegistry()
        adapters = [name for name in sys.modules if name.startswith("hardpoint.adapters.")]
        print(",".join(sorted(adapters)))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", (
        "constructing a registry imported adapter modules: " + result.stdout
    )


def test_builtin_table_holds_strings_not_callables() -> None:
    """The table cannot hold a factory without having imported the module first."""
    for entry in BUILTIN_COMPONENTS:
        assert isinstance(entry.module, str)
        assert isinstance(entry.factory, str)
        assert isinstance(entry.config_model, str)


def test_builtin_table_is_internally_consistent() -> None:
    """Keys are unique per kind and follow the naming convention (§4).

    Vacuous while the table is empty in M0, and gains teeth the moment M1 adds
    its first adapter. That is the point of writing it now.
    """
    seen: set[tuple[Kind, str]] = set()
    for entry in BUILTIN_COMPONENTS:
        identity = (entry.kind, entry.key)
        assert identity not in seen, f"duplicate built-in {entry.key!r} for {entry.kind}"
        seen.add(identity)
        assert entry.key == entry.key.lower()
        assert " " not in entry.key
        assert entry.module.startswith("hardpoint.adapters.")


def test_keys_and_describe_import_nothing(registry: ComponentRegistry) -> None:
    """``components list`` must work without the extras installed.

    Otherwise the command that tells you which extra you need would itself
    require that extra.
    """
    registry_with_table = ComponentRegistry(
        [
            BuiltinEntry(
                key="pretend_index",
                kind=Kind.INDEX,
                module="hardpoint.adapters.index.does_not_exist",
                factory="build",
                config_model="Config",
                extra="qdrant",
            )
        ]
    )
    assert registry_with_table.keys(Kind.INDEX) == ["pretend_index"]
    assert registry_with_table.describe(Kind.INDEX) == [("pretend_index", "builtin", "qdrant")]
    assert "hardpoint.adapters.index.does_not_exist" not in sys.modules


# --------------------------------------------------------------------------- #
# Missing dependency  [LOCKED]                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture
def adapter_on_path(tmp_path: Path) -> Iterator[str]:
    """Write a package containing an adapter that imports an absent third-party package.

    A real module on a real path, so the test exercises the actual import
    machinery rather than a patched version of it.
    """
    package = tmp_path / "toy_adapters"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "needs_extra.py").write_text(
        "import definitely_not_installed_xyz  # noqa: F401\n", encoding="utf-8"
    )
    (package / "healthy.py").write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel, ConfigDict


            class Config(BaseModel):
                model_config = ConfigDict(extra="forbid")
                url: str = "http://localhost"


            def build(config):
                return {"built": config.url}
            """
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        yield "toy_adapters"
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name.startswith("toy_adapters"):
                del sys.modules[name]


def test_missing_extra_raises_with_the_exact_install_command(adapter_on_path: str) -> None:
    """**[LOCKED]** Never a raw ModuleNotFoundError (ARCHITECTURE.md §16.3)."""
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="qdrant",
                kind=Kind.INDEX,
                module=f"{adapter_on_path}.needs_extra",
                factory="build",
                config_model="Config",
                extra="qdrant",
            )
        ]
    )

    with pytest.raises(MissingDependencyError) as exc_info:
        registry.resolve(Kind.INDEX, "qdrant", config_path="indexes.primary.type")

    error = exc_info.value
    assert error.extra == "qdrant"
    assert error.remedy == "pip install 'hardpoint[qdrant]'"
    rendered = str(error)
    assert "pip install 'hardpoint[qdrant]'" in rendered
    assert "indexes.primary.type" in rendered, "the message must say what asked for it"
    assert isinstance(error.__cause__, ModuleNotFoundError)


def test_missing_dependency_without_a_declared_extra_names_the_package(
    adapter_on_path: str,
) -> None:
    """Some components need a package that is not behind a named extra."""
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="loose",
                kind=Kind.INDEX,
                module=f"{adapter_on_path}.needs_extra",
                factory="build",
                config_model="Config",
                extra=None,
            )
        ]
    )
    with pytest.raises(MissingDependencyError) as exc_info:
        registry.resolve(Kind.INDEX, "loose")
    assert exc_info.value.remedy == "pip install definitely_not_installed_xyz"


def test_an_absent_adapter_module_is_reported_as_a_bug_not_a_missing_extra() -> None:
    """These two failures mean opposite things.

    Telling a user to ``pip install 'hardpoint[qdrant]'`` when the adapter module
    itself is missing sends them down a dead end: the install would succeed and
    the error would not change.
    """
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="ghost",
                kind=Kind.INDEX,
                module="hardpoint.adapters.index.ghost",
                factory="build",
                config_model="Config",
                extra="qdrant",
            )
        ]
    )
    with pytest.raises(ContractError) as exc_info:
        registry.resolve(Kind.INDEX, "ghost")

    rendered = str(exc_info.value)
    assert "bug in hardpoint" in rendered
    assert "pip install" not in rendered
    assert exc_info.value.remedy is not None
    assert "issues" in exc_info.value.remedy


def test_an_adapter_missing_its_declared_factory_is_reported_as_a_bug(
    adapter_on_path: str,
) -> None:
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="wrong_names",
                kind=Kind.INDEX,
                module=f"{adapter_on_path}.healthy",
                factory="build_something_else",
                config_model="Config",
            )
        ]
    )
    with pytest.raises(ContractError, match="does not define"):
        registry.resolve(Kind.INDEX, "wrong_names")


def test_a_healthy_builtin_resolves_and_constructs(adapter_on_path: str) -> None:
    """The happy path of lazy loading, so the error tests are not the only coverage."""
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="healthy",
                kind=Kind.INDEX,
                module=f"{adapter_on_path}.healthy",
                factory="build",
                config_model="Config",
            )
        ]
    )
    registration = registry.resolve(Kind.INDEX, "healthy")
    assert registration.source == "builtin"
    assert registration.module == f"{adapter_on_path}.healthy"
    assert callable(registration.factory)


# --------------------------------------------------------------------------- #
# Unknown keys                                                                #
# --------------------------------------------------------------------------- #


def test_unknown_key_suggests_the_closest_registered_key(registry: ComponentRegistry) -> None:
    """A typo is the overwhelmingly common cause (ARCHITECTURE.md §30)."""
    registry.register(
        "cohere_rerank", kind=Kind.RERANKER, factory=build_toy, config_model=ToyConfig
    )
    with pytest.raises(UnknownComponentError) as exc_info:
        registry.resolve(Kind.RERANKER, "cohere_reranker", config_path="retrieval.rerank.type")

    rendered = str(exc_info.value)
    assert "'cohere_rerank'" in rendered
    assert "retrieval.rerank.type" in rendered
    assert "components list" in rendered


def test_unknown_key_with_nothing_close_still_lists_what_exists(
    registry: ComponentRegistry,
) -> None:
    registry.register("alpha", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)
    with pytest.raises(UnknownComponentError) as exc_info:
        registry.resolve(Kind.LLM, "zzzzzzzz")
    assert "alpha" in (exc_info.value.remedy or "")


def test_unknown_key_when_nothing_is_registered_says_so(registry: ComponentRegistry) -> None:
    with pytest.raises(UnknownComponentError) as exc_info:
        registry.resolve(Kind.LLM, "anything")
    assert "No llm components are registered" in (exc_info.value.remedy or "")


def test_keys_are_namespaced_per_kind(registry: ComponentRegistry) -> None:
    """``openai`` can be both an LLM and an embedding model."""
    registry.register("openai", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)
    registry.register("openai", kind=Kind.EMBEDDINGS, factory=build_toy, config_model=ToyConfig)
    assert registry.keys(Kind.LLM) == ["openai"]
    assert registry.keys(Kind.EMBEDDINGS) == ["openai"]
    assert registry.keys(Kind.INDEX) == []


# --------------------------------------------------------------------------- #
# Precedence and collisions                                                   #
# --------------------------------------------------------------------------- #


def test_duplicate_key_at_the_same_level_names_both_sources(
    registry: ComponentRegistry,
) -> None:
    """A collision is an error, never a silent override (ARCHITECTURE.md §17.1)."""
    registry.register(
        "dup", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig, origin="first_plugin"
    )
    with pytest.raises(DuplicateComponentError) as exc_info:
        registry.register(
            "dup",
            kind=Kind.LLM,
            factory=build_toy,
            config_model=ToyConfig,
            origin="second_plugin",
        )
    rendered = str(exc_info.value)
    assert "first_plugin" in rendered
    assert "second_plugin" in rendered


def test_a_project_registration_shadows_a_builtin(adapter_on_path: str) -> None:
    """The intended mechanism for replacing a shipped component."""
    registry = ComponentRegistry(
        [
            BuiltinEntry(
                key="healthy",
                kind=Kind.INDEX,
                module=f"{adapter_on_path}.healthy",
                factory="build",
                config_model="Config",
            )
        ]
    )
    assert registry.source_of(Kind.INDEX, "healthy") == "builtin"

    registry.register("healthy", kind=Kind.INDEX, factory=build_toy, config_model=ToyConfig)

    assert registry.source_of(Kind.INDEX, "healthy") == "project"
    assert registry.resolve(Kind.INDEX, "healthy").factory is build_toy


def test_project_beats_entrypoint_beats_builtin(registry: ComponentRegistry) -> None:
    registry.register(
        "x", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig, source="entrypoint"
    )
    assert registry.source_of(Kind.LLM, "x") == "entrypoint"

    registry.register(
        "x", kind=Kind.LLM, factory=build_toy_async, config_model=ToyConfig, source="project"
    )
    assert registry.source_of(Kind.LLM, "x") == "project"
    assert registry.resolve(Kind.LLM, "x").factory is build_toy_async


def test_source_of_returns_none_for_an_unknown_key(registry: ComponentRegistry) -> None:
    assert registry.source_of(Kind.LLM, "nope") is None


def test_registering_after_resolution_invalidates_the_cache(
    registry: ComponentRegistry,
) -> None:
    """Otherwise a late project registration would be silently ignored."""
    registry.register(
        "y", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig, source="entrypoint"
    )
    assert registry.resolve(Kind.LLM, "y").factory is build_toy

    registry.register(
        "y", kind=Kind.LLM, factory=build_toy_async, config_model=ToyConfig, source="project"
    )
    assert registry.resolve(Kind.LLM, "y").factory is build_toy_async


# --------------------------------------------------------------------------- #
# Option validation                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_create_validates_options_and_builds(registry: ComponentRegistry) -> None:
    registry.register("toy", kind=Kind.INDEX, factory=build_toy, config_model=ToyConfig)
    built = await registry.create(Kind.INDEX, "toy", {"url": "http://h", "top_k": 3})
    assert isinstance(built, Toy)
    assert built.config.url == "http://h"
    assert built.config.top_k == 3


@pytest.mark.anyio
async def test_create_awaits_an_async_factory(registry: ComponentRegistry) -> None:
    """A factory may need to open a connection, so it may be a coroutine function."""
    registry.register("toy", kind=Kind.INDEX, factory=build_toy_async, config_model=ToyConfig)
    built = await registry.create(Kind.INDEX, "toy", {})
    assert isinstance(built, Toy)


@pytest.mark.anyio
async def test_create_applies_the_config_models_defaults(registry: ComponentRegistry) -> None:
    registry.register("toy", kind=Kind.INDEX, factory=build_toy, config_model=ToyConfig)
    built = await registry.create(Kind.INDEX, "toy")
    assert built.config.url == "http://localhost"


def test_unknown_option_is_an_error_naming_the_closest_valid_one(
    registry: ComponentRegistry,
) -> None:
    """This is where the enforcement ``ComponentSpec`` cannot do actually lands.

    ``ComponentSpec`` allows extras because it cannot know a component's valid
    keys. The component's own model does, and it forbids them.
    """
    registration = registry.register(
        "toy", kind=Kind.INDEX, factory=build_toy, config_model=ToyConfig
    )
    with pytest.raises(InvalidConfigError) as exc_info:
        registry.validate_options(registration, {"urll": "http://h"}, config_path="indexes.primary")

    rendered = str(exc_info.value)
    assert "unknown option" in rendered
    assert "'url'" in rendered
    assert "indexes.primary" in rendered
    assert "top_k" in (exc_info.value.remedy or ""), "the remedy lists the valid options"


def test_invalid_option_value_is_reported_with_its_path(registry: ComponentRegistry) -> None:
    registration = registry.register(
        "toy", kind=Kind.INDEX, factory=build_toy, config_model=ToyConfig
    )
    with pytest.raises(InvalidConfigError, match="top_k"):
        registry.validate_options(registration, {"top_k": "many"})


# --------------------------------------------------------------------------- #
# Contract version                                                            #
# --------------------------------------------------------------------------- #


def test_a_minor_contract_mismatch_warns(registry: ComponentRegistry) -> None:
    major = CONTRACT_VERSION.split(".", 1)[0]
    with pytest.warns(HardpointWarning, match="contract version"):
        registry.register(
            "old",
            kind=Kind.LLM,
            factory=build_toy,
            config_model=ToyConfig,
            contract_version=f"{major}.999",
        )


def test_a_major_contract_mismatch_is_a_hard_error(registry: ComponentRegistry) -> None:
    """Across a major contract break the ports are not compatible."""
    with pytest.raises(ContractError) as exc_info:
        registry.register(
            "future",
            kind=Kind.LLM,
            factory=build_toy,
            config_model=ToyConfig,
            contract_version="99.0",
        )
    assert exc_info.value.remedy is not None
    assert "CHANGELOG" in exc_info.value.remedy


def test_a_matching_contract_version_is_silent(registry: ComponentRegistry) -> None:
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        registry.register("ok", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)


# --------------------------------------------------------------------------- #
# Entry-point discovery                                                       #
# --------------------------------------------------------------------------- #


class FakeEntryPoint:
    def __init__(self, name: str, specs: list[Any]) -> None:
        self.name = name
        self._specs = specs

    def load(self) -> list[Any]:
        return self._specs


def spec(key: str) -> Any:
    from hardpoint.core.registry import Registration

    return Registration(
        key=key,
        kind=Kind.LLM,
        config_model=ToyConfig,
        factory=build_toy,
        extra=None,
        module="third_party",
        source="entrypoint",
    )


def test_discovery_registers_at_the_entrypoint_level(registry: ComponentRegistry) -> None:
    registered = registry.discover([FakeEntryPoint("vendor_a", [spec("a")])])
    assert registered == ["a"]
    assert registry.source_of(Kind.LLM, "a") == "entrypoint"


def test_discovery_is_deterministic_in_distribution_name_order(
    registry: ComponentRegistry,
) -> None:
    """Two distributions claiming one key must collide the same way every run.

    Filesystem order is not stable across machines, so without sorting a build
    could pass on one runner and fail on another.
    """
    entry_points = [
        FakeEntryPoint("zebra", [spec("z")]),
        FakeEntryPoint("alpha", [spec("a")]),
    ]
    assert registry.discover(entry_points) == ["a", "z"]


def test_two_distributions_claiming_one_key_collide(registry: ComponentRegistry) -> None:
    with pytest.raises(DuplicateComponentError) as exc_info:
        registry.discover(
            [FakeEntryPoint("vendor_a", [spec("same")]), FakeEntryPoint("vendor_b", [spec("same")])]
        )
    rendered = str(exc_info.value)
    assert "vendor_a" in rendered
    assert "vendor_b" in rendered


def test_nothing_is_discovered_unless_discovery_is_called(registry: ComponentRegistry) -> None:
    """Import-time auto-discovery was rejected (ARCHITECTURE.md §6.3).

    Discovery is a method a caller invokes when ``plugins.discover`` is true, not
    something that happens because a package is installed.
    """
    assert registry.keys(Kind.LLM) == []


# --------------------------------------------------------------------------- #
# Housekeeping                                                                #
# --------------------------------------------------------------------------- #


def test_registry_is_not_a_singleton() -> None:
    """No module-level registry, no global mutable state (INSTRUCTIONS.md §4)."""
    import hardpoint.core.registry as registry_module

    globals_that_are_registries = [
        name
        for name, value in vars(registry_module).items()
        if isinstance(value, ComponentRegistry)
    ]
    assert not globals_that_are_registries, (
        f"a module-level registry is global mutable state: {globals_that_are_registries}"
    )

    first, second = ComponentRegistry(), ComponentRegistry()
    first.register("only_here", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)
    assert second.keys(Kind.LLM) == []


def test_with_builtins_copies_project_registrations(registry: ComponentRegistry) -> None:
    registry.register("mine", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)
    clone = registry.with_builtins(
        [
            BuiltinEntry(
                key="theirs",
                kind=Kind.LLM,
                module="hardpoint.adapters.llm.x",
                factory="build",
                config_model="Config",
            )
        ]
    )
    assert clone.keys(Kind.LLM) == ["mine", "theirs"]
    assert registry.keys(Kind.LLM) == ["mine"]


def test_repr_reports_counts_without_importing(registry: ComponentRegistry) -> None:
    registry.register("a", kind=Kind.LLM, factory=build_toy, config_model=ToyConfig)
    assert "registered=1" in repr(registry)


def test_kind_covers_every_port_named_in_the_instructions() -> None:
    """INSTRUCTIONS.md §5.6 fixes this list; a missing kind is unrepresentable."""
    assert {kind.value for kind in Kind} == {
        "llm",
        "embeddings",
        "index",
        "reranker",
        "parser",
        "chunker",
        "cache",
        "state",
        "tracer",
        "metrics",
        "source",
        "guard",
        "metric",
        "tool",
    }
