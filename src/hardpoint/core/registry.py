"""The component registry: deterministic, inspectable, lazily loaded.

Implements INSTRUCTIONS.md §5.6 and ARCHITECTURE.md §17.

## Three things this module refuses to do

**It does not import adapters at import time.** Built-ins live in a static table
of *module path strings*. A module is imported the first time a component of
that type is actually resolved. This is what keeps ``pip install hardpoint`` free
of provider SDKs, and it is **[LOCKED]**.

**It does not discover plugins implicitly.** Entry-point scanning happens only
when ``plugins.discover`` is switched on. Auto-discovery at import time was
rejected outright: it makes resolution non-deterministic, hides where a
component came from, and makes a run unreproducible (ARCHITECTURE.md §6.3).

**It is not a singleton.** There is no module-level registry and no global
mutable state (INSTRUCTIONS.md §4). A caller constructs a
:class:`ComponentRegistry`, registers what it wants, and passes it where it is
needed. "Where did this object come from" then has an answer.

## Resolution order

Highest precedence first (ARCHITECTURE.md §17.1):

1. ``project`` -- registered explicitly by the user's project.
2. ``entrypoint`` -- a third-party distribution, only when discovery is enabled.
3. ``builtin`` -- shipped with hardpoint.

A collision *within* one level is an error naming both sources, never a silent
override. A registration at a higher level shadowing a lower one is the intended
mechanism, and ``source_of`` reports which won.

## The two ways a lazy import can fail

They mean opposite things and must not be conflated:

- The adapter module imports a third-party package that is not installed. That
  is a missing extra, and it raises :class:`MissingDependencyError` carrying the
  exact ``pip install`` command and the config path that asked for it. Never a
  raw ``ModuleNotFoundError`` traceback (ARCHITECTURE.md §16.3).
- The adapter module itself is absent. That is a bug in hardpoint, not something
  the user can install, so it raises :class:`ContractError` and says so. Telling
  a user to install an extra that would not fix their problem is worse than the
  original traceback.
"""

from __future__ import annotations

import difflib
import importlib
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from inspect import isawaitable
from typing import Any, Final, Literal

from pydantic import BaseModel, ValidationError

from hardpoint.core.errors import (
    ContractError,
    DuplicateComponentError,
    HardpointWarning,
    InvalidConfigError,
    MissingDependencyError,
    UnknownComponentError,
)
from hardpoint.core.types import CONTRACT_VERSION, JsonValue

__all__ = [
    "BUILTIN_COMPONENTS",
    "BuiltinEntry",
    "ComponentRegistry",
    "Kind",
    "Registration",
    "Source",
]

Source = Literal["builtin", "project", "entrypoint"]
"""Where a registration came from. Fixed by INSTRUCTIONS.md §5.6."""

_PRECEDENCE: Final[dict[Source, int]] = {"project": 0, "entrypoint": 1, "builtin": 2}
"""Lower wins. ARCHITECTURE.md §17.1."""

ENTRY_POINT_GROUP: Final = "hardpoint.components"
"""The entry-point group third-party distributions expose."""


class Kind(StrEnum):
    """What a component is, for namespacing and for ``components list --type``.

    Keys are unique per kind, not globally, so an LLM and an embedding model may
    both be called ``openai``.

    Attributes:
        LLM: A ``LanguageModel``.
        EMBEDDINGS: An ``EmbeddingModel``.
        INDEX: A ``VectorIndex``.
        RERANKER: A ``Reranker``.
        PARSER: A ``DocumentParser``.
        CHUNKER: A ``Chunker``.
        CACHE: A ``CacheBackend``.
        STATE: A ``StateStore`` for ingestion manifests.
        TRACER: A ``Tracer``.
        METRICS: A ``MetricSink``, the telemetry port.
        SOURCE: An ingestion ``Source``.
        GUARD: A guard step.
        METRIC: An evaluation metric. Distinct from ``METRICS``, which is
            telemetry; this one scores an eval case.
        TOOL: A ``Tool`` an agent may call.
    """

    LLM = "llm"
    EMBEDDINGS = "embeddings"
    INDEX = "index"
    RERANKER = "reranker"
    PARSER = "parser"
    CHUNKER = "chunker"
    CACHE = "cache"
    STATE = "state"
    TRACER = "tracer"
    METRICS = "metrics"
    SOURCE = "source"
    GUARD = "guard"
    METRIC = "metric"
    TOOL = "tool"


@dataclass(frozen=True)
class BuiltinEntry:
    """One row of the static built-in table.

    Deliberately *not* a :class:`Registration`: it holds a module path string
    rather than a factory, which is the whole point. Nothing here is imported
    until a component of this type is resolved.

    Args:
        key: Registry key, lowercase ``<vendor>_<kind>``.
        kind: What the component is.
        module: Dotted path of the adapter module, imported on first use.
        factory: Name of the factory callable inside that module.
        config_model: Name of the Pydantic config model inside that module.
        extra: The pip extra that installs this component's dependency, or
            ``None`` when it needs nothing beyond the base install.
        contract_version: The port contract version the adapter targets.
    """

    key: str
    kind: Kind
    module: str
    factory: str
    config_model: str
    extra: str | None = None
    contract_version: str = CONTRACT_VERSION


# The static built-in table. **[LOCKED]** mechanism: lazy module paths, never an
# import at module import time.
#
# It is empty in M0 because M0 ships no adapters, and an entry pointing at a
# module that does not exist would resolve to "this is a bug in hardpoint" rather
# than to anything useful. Each milestone appends its own adapters; the
# machinery, the precedence rules and the error behaviour are complete and
# tested now, against registrations made by the tests themselves.
BUILTIN_COMPONENTS: Final[tuple[BuiltinEntry, ...]] = ()


@dataclass(frozen=True)
class Registration:
    """A resolved component, ready to construct.

    Args:
        key: Registry key.
        kind: What the component is.
        config_model: Pydantic model validating this component's options.
        factory: Builds the component from its validated config. May be a
            coroutine function.
        extra: The pip extra required, when there is one.
        module: Where the component was loaded from.
        source: Which precedence level supplied it.
        contract_version: The port contract version it targets.
    """

    key: str
    kind: Kind
    config_model: type[BaseModel]
    # The factory returns a port implementation, whose type varies by kind and
    # cannot be expressed here without a union over every port. `Any` at a
    # genuine dynamic boundary, per INSTRUCTIONS.md §12.1.
    factory: Callable[[BaseModel], Any]
    extra: str | None
    module: str
    source: Source
    contract_version: str = CONTRACT_VERSION


class ComponentRegistry:
    """Resolves a ``type:`` discriminator to a constructible component.

    Not a singleton and not global. Construct one, register what the project
    adds, and pass it to whatever builds resources.

    Args:
        builtins: The static table to seed from. Defaults to
            :data:`BUILTIN_COMPONENTS`. Injectable so tests do not need the real
            adapter set.

    Raises:
        Nothing on construction. Nothing is imported here.
    """

    def __init__(self, builtins: Sequence[BuiltinEntry] = BUILTIN_COMPONENTS) -> None:
        self._builtins: dict[tuple[Kind, str], BuiltinEntry] = {}
        self._registrations: dict[tuple[Kind, str, Source], Registration] = {}
        self._origins: dict[tuple[Kind, str, Source], str] = {}
        self._resolved: dict[tuple[Kind, str], Registration] = {}

        for entry in builtins:
            self._builtins[(entry.kind, entry.key)] = entry

    # ----------------------------------------------------------------- #
    # Registration                                                      #
    # ----------------------------------------------------------------- #

    def register(
        self,
        key: str,
        *,
        kind: Kind,
        factory: Callable[[BaseModel], Any],
        config_model: type[BaseModel],
        extra: str | None = None,
        source: Source = "project",
        contract_version: str = CONTRACT_VERSION,
        origin: str | None = None,
    ) -> Registration:
        """Register a component under a key.

        The common case does not need this at all: write a class implementing the
        Protocol and pass the instance in Python. Registration is for when
        operations must be able to switch the implementation per environment
        through ``type:`` in YAML (ARCHITECTURE.md §17.2).

        Args:
            key: Lowercase ``<vendor>_<kind>``, unique within its kind and level.
            kind: What the component is.
            factory: Builds the component from its validated config. May be a
                coroutine function.
            config_model: Pydantic model validating the options. Should set
                ``extra="forbid"`` so an unknown option is an error.
            extra: The pip extra required, when there is one.
            source: Precedence level. Defaults to ``project``, the highest.
            contract_version: The port contract version targeted.
            origin: Human-readable description of where this came from, used in
                a collision message. Defaults to the factory's module.

        Returns:
            The registration that was stored.

        Raises:
            DuplicateComponentError: If the key is already taken at this level,
                naming both sources.
        """
        identity = (kind, key, source)
        where = origin or str(getattr(factory, "__module__", None) or "<unknown>")

        if identity in self._registrations:
            raise DuplicateComponentError(
                f"Two components claim the key {key!r} for kind {kind.value!r} at the "
                f"{source!r} level: {self._origins.get(identity, '<unknown>')} and {where}.",
                component=key,
                remedy=(
                    f"Rename one of them. Registry keys are unique per kind, so "
                    f"{key!r} can exist once as a {kind.value!r} component. "
                    f"`hardpoint components list --source` shows what is registered."
                ),
            )

        self._check_contract_version(key, contract_version)

        registration = Registration(
            key=key,
            kind=kind,
            config_model=config_model,
            factory=factory,
            extra=extra,
            module=where,
            source=source,
            contract_version=contract_version,
        )
        self._registrations[identity] = registration
        self._origins[identity] = where
        self._resolved.pop((kind, key), None)
        return registration

    def _check_contract_version(self, key: str, declared: str) -> None:
        """Warn on a minor mismatch, refuse a major one (ARCHITECTURE.md §17.3)."""
        if declared == CONTRACT_VERSION:
            return

        declared_major = declared.split(".", 1)[0]
        current_major = CONTRACT_VERSION.split(".", 1)[0]
        if declared_major != current_major:
            raise ContractError(
                f"Component {key!r} targets port contract version {declared!r}, but "
                f"this release implements {CONTRACT_VERSION!r}. Across a major "
                f"contract version the ports are not compatible.",
                component=key,
                remedy=(
                    f"Upgrade the component to contract {CONTRACT_VERSION!r}, or pin a "
                    f"hardpoint release implementing contract {declared!r}. The "
                    f"CHANGELOG records what changed between contract versions."
                ),
            )

        warnings.warn(
            f"Component {key!r} targets port contract version {declared!r} while this "
            f"release implements {CONTRACT_VERSION!r}. Minor contract versions are "
            f"compatible, but the component may not use everything available.",
            HardpointWarning,
            stacklevel=3,
        )

    # ----------------------------------------------------------------- #
    # Inspection                                                        #
    # ----------------------------------------------------------------- #

    def keys(self, kind: Kind) -> list[str]:
        """Return every registered key of a kind, sorted, without importing anything."""
        registered = {key for (k, key, _) in self._registrations if k == kind}
        built_in = {key for (k, key) in self._builtins if k == kind}
        return sorted(registered | built_in)

    def source_of(self, kind: Kind, key: str) -> Source | None:
        """Return which level would win for a key, or ``None`` if it is unknown.

        Answers "where did this component come from" without constructing it,
        which is what ``components list --source`` prints.
        """
        for source in sorted(_PRECEDENCE, key=lambda s: _PRECEDENCE[s]):
            if (kind, key, source) in self._registrations:
                return source
        if (kind, key) in self._builtins:
            return "builtin"
        return None

    def describe(self, kind: Kind) -> list[tuple[str, Source, str | None]]:
        """Return ``(key, winning source, required extra)`` for every key of a kind.

        Imports nothing: the extra is read from the static table, which is
        exactly why the table carries it.
        """
        rows: list[tuple[str, Source, str | None]] = []
        for key in self.keys(kind):
            source = self.source_of(kind, key)
            if source is None:  # pragma: no cover - keys() only yields known keys
                continue
            extra = self._extra_for(kind, key, source)
            rows.append((key, source, extra))
        return rows

    def _extra_for(self, kind: Kind, key: str, source: Source) -> str | None:
        if source == "builtin":
            entry = self._builtins.get((kind, key))
            return entry.extra if entry else None
        registration = self._registrations.get((kind, key, source))
        return registration.extra if registration else None

    # ----------------------------------------------------------------- #
    # Resolution                                                        #
    # ----------------------------------------------------------------- #

    def resolve(self, kind: Kind, key: str, *, config_path: str | None = None) -> Registration:
        """Resolve a key to a registration, importing its module on first use.

        Args:
            kind: What the component is.
            key: The ``type:`` discriminator from configuration.
            config_path: Where the key was configured, for the error message.

        Returns:
            The winning registration, with a real factory and config model.

        Raises:
            UnknownComponentError: If nothing is registered under the key,
                suggesting the closest registered key of that kind.
            MissingDependencyError: If the adapter's third-party dependency is
                not installed, carrying the exact install command.
            ContractError: If the adapter module itself is absent, which is a
                bug in hardpoint rather than something a user can install.
        """
        cached = self._resolved.get((kind, key))
        if cached is not None:
            return cached

        for source in sorted(_PRECEDENCE, key=lambda s: _PRECEDENCE[s]):
            registration = self._registrations.get((kind, key, source))
            if registration is not None:
                self._resolved[(kind, key)] = registration
                return registration

        entry = self._builtins.get((kind, key))
        if entry is None:
            raise self._unknown(kind, key, config_path)

        registration = self._load(entry, config_path)
        self._resolved[(kind, key)] = registration
        return registration

    def _unknown(self, kind: Kind, key: str, config_path: str | None) -> UnknownComponentError:
        """Build the error for a key nothing is registered under."""
        known = self.keys(kind)
        close = difflib.get_close_matches(key, known, n=3, cutoff=0.5)
        if close:
            suggestion = f"Did you mean {', '.join(repr(name) for name in close)}?"
        elif known:
            suggestion = f"Registered {kind.value} components: {', '.join(known)}."
        else:
            suggestion = (
                f"No {kind.value} components are registered. Register one on the "
                f"registry you pass to the resource factory, or install the extra "
                f"that provides it."
            )

        return UnknownComponentError(
            f"No {kind.value} component is registered under the key {key!r}.",
            component=key,
            config_path=config_path,
            remedy=(
                f"{suggestion}\n"
                f"`hardpoint components list --type {kind.value}` prints every "
                f"registered key and where it came from."
            ),
        )

    def _load(self, entry: BuiltinEntry, config_path: str | None) -> Registration:
        """Import a built-in's module and build its Registration.

        This is the only place an adapter module is imported, and it happens on
        first use rather than at import time.
        """
        try:
            module = importlib.import_module(entry.module)
        except ModuleNotFoundError as exc:
            raise self._import_failure(entry, exc, config_path) from exc

        try:
            factory = getattr(module, entry.factory)
            config_model = getattr(module, entry.config_model)
        except AttributeError as exc:
            raise ContractError(
                f"The adapter module {entry.module!r} does not define "
                f"{exc.name!r}, which the built-in table says it should. This is a "
                f"bug in hardpoint, not in your configuration.",
                component=entry.key,
                config_path=config_path,
                remedy=(
                    "Please report this with the hardpoint version, at "
                    "https://github.com/NabiBukhsh-AI/hardpoint/issues"
                ),
                cause=exc,
            ) from exc

        return Registration(
            key=entry.key,
            kind=entry.kind,
            config_model=config_model,
            factory=factory,
            extra=entry.extra,
            module=entry.module,
            source="builtin",
            contract_version=entry.contract_version,
        )

    def _import_failure(
        self, entry: BuiltinEntry, exc: ModuleNotFoundError, config_path: str | None
    ) -> Exception:
        """Distinguish a missing extra from a missing adapter module.

        ``exc.name`` is the module that could not be found. If it is the adapter
        itself, no ``pip install`` will help and saying otherwise sends the user
        down a dead end.
        """
        missing = exc.name or ""
        adapter_is_absent = missing == entry.module or entry.module.startswith(f"{missing}.")

        if adapter_is_absent:
            return ContractError(
                f"The adapter module {entry.module!r} for component {entry.key!r} "
                f"could not be imported: it does not exist. This is a bug in "
                f"hardpoint, not a missing dependency.",
                component=entry.key,
                config_path=config_path,
                remedy=(
                    "Please report this with the hardpoint version, at "
                    "https://github.com/NabiBukhsh-AI/hardpoint/issues"
                ),
                cause=exc,
            )

        if entry.extra is None:
            return MissingDependencyError(
                f"Component {entry.key!r} ({entry.module}) needs the Python package "
                f"{missing!r}, which is not installed.",
                component=entry.key,
                config_path=config_path,
                remedy=f"pip install {missing}",
                cause=exc,
            )

        return MissingDependencyError(
            f"Component {entry.key!r} ({entry.module}) requires the "
            f"{entry.extra!r} extra, which is not installed.",
            extra=entry.extra,
            component=entry.key,
            config_path=config_path,
            remedy=f"pip install 'hardpoint[{entry.extra}]'",
            cause=exc,
        )

    # ----------------------------------------------------------------- #
    # Construction                                                      #
    # ----------------------------------------------------------------- #

    def validate_options(
        self,
        registration: Registration,
        options: Mapping[str, JsonValue],
        *,
        config_path: str | None = None,
    ) -> BaseModel:
        """Validate a component block's options against the component's own model.

        This is where the ``extra="forbid"`` enforcement that ``ComponentSpec``
        cannot do lands. An unknown option is an error naming the closest valid
        one, never a setting that silently does nothing.

        Args:
            registration: The resolved component.
            options: The component-specific keys, from ``ComponentSpec.options``.
            config_path: Where the block was configured.

        Returns:
            The validated config model instance.

        Raises:
            InvalidConfigError: On an unknown or invalid option.
        """
        try:
            return registration.config_model.model_validate(dict(options))
        except ValidationError as exc:
            valid = sorted(registration.config_model.model_fields)
            problems: list[str] = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                if error["type"] == "extra_forbidden":
                    close = difflib.get_close_matches(location, valid, n=1, cutoff=0.6)
                    hint = f" Did you mean {close[0]!r}?" if close else ""
                    problems.append(f"  {location}: unknown option for {registration.key!r}.{hint}")
                else:
                    problems.append(f"  {location}: {error['msg']}")

            raise InvalidConfigError(
                f"Component {registration.key!r} was configured with invalid options:\n"
                + "\n".join(problems),
                component=registration.key,
                config_path=config_path,
                remedy=(
                    f"Valid options for {registration.key!r}: {', '.join(valid) or '(none)'}.\n"
                    f"`hardpoint components list --type {registration.kind.value} --describe` "
                    f"prints each component's config model."
                ),
                cause=exc,
            ) from exc

    async def create(
        self,
        kind: Kind,
        key: str,
        options: Mapping[str, JsonValue] | None = None,
        *,
        config_path: str | None = None,
    ) -> Any:
        """Resolve, validate and construct a component.

        Async because a factory may need to open a connection, and the whole
        library is async-first (ADR-007). A synchronous factory is supported and
        is simply not awaited.

        Args:
            kind: What the component is.
            key: The ``type:`` discriminator.
            options: The component-specific configuration keys.
            config_path: Where the block was configured, for error messages.

        Returns:
            The constructed component.

        Raises:
            UnknownComponentError: If the key is not registered.
            MissingDependencyError: If an extra is missing.
            InvalidConfigError: If the options are invalid.
        """
        registration = self.resolve(kind, key, config_path=config_path)
        config = self.validate_options(registration, options or {}, config_path=config_path)
        built = registration.factory(config)
        if isawaitable(built):
            return await built
        return built

    # ----------------------------------------------------------------- #
    # Entry-point discovery                                             #
    # ----------------------------------------------------------------- #

    def discover(self, entry_points: Iterable[Any]) -> list[str]:
        """Register components advertised by third-party distributions.

        Called only when ``plugins.discover`` is true. Entry points are taken as
        an argument rather than scanned here so that the scan itself stays in the
        CLI, where knowing about the installed environment belongs, and so this
        method is testable without installing anything.

        Results are processed in sorted order by name, so two distributions
        advertising the same key collide deterministically rather than according
        to filesystem order.

        Args:
            entry_points: Objects with ``name`` and ``load()``, as
                ``importlib.metadata`` yields.

        Returns:
            The keys that were registered, in the order they were processed.

        Raises:
            DuplicateComponentError: If two distributions claim one key.
        """
        registered: list[str] = []
        for entry_point in sorted(entry_points, key=lambda ep: str(ep.name)):
            loaded = entry_point.load()
            for spec in loaded() if callable(loaded) else loaded:
                self.register(
                    spec.key,
                    kind=spec.kind,
                    factory=spec.factory,
                    config_model=spec.config_model,
                    extra=spec.extra,
                    source="entrypoint",
                    contract_version=spec.contract_version,
                    origin=str(entry_point.name),
                )
                registered.append(spec.key)
        return registered

    def with_builtins(self, builtins: Sequence[BuiltinEntry]) -> ComponentRegistry:
        """Return a copy of this registry seeded from a different built-in table.

        Used by tests and by ``doctor`` to reason about a component set without
        mutating a live registry.
        """
        clone = ComponentRegistry(builtins)
        for identity, registration in self._registrations.items():
            clone._registrations[identity] = replace(registration)
            clone._origins[identity] = self._origins[identity]
        return clone

    def __repr__(self) -> str:
        """Report how much is registered, without importing anything."""
        return (
            f"ComponentRegistry(builtins={len(self._builtins)}, "
            f"registered={len(self._registrations)})"
        )
