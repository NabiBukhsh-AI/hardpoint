"""The immutable, hashed result of configuration resolution.

Implements the ``ConfigSnapshot`` half of INSTRUCTIONS.md §5.7. A snapshot is
what a run carries: the resolved values, which layer each key came from, which
keys hold secrets, and a hash that identifies the configuration in a run
manifest.

## Two views, deliberately

A snapshot holds resolved values that include real secrets, because a factory
building a client needs the real API key. It also holds a *redacted* view, which
is what every dump, log, trace attribute and error message uses.

The redacted view is the default everywhere. ``repr`` is redacted,
:meth:`to_dict` is redacted, and the only way to obtain a secret is
:meth:`reveal`, which is named so that reading the call tells you what is
happening. There is no path by which a secret is printed by accident.

## Why the hash is taken over the redacted content

INSTRUCTIONS.md §5.7 specifies the hash over the resolved-*and-redacted*
content, which has a consequence worth stating: **rotating a secret does not
change the config hash.** That is the desirable behaviour. A run manifest from
before a key rotation stays comparable with one from after it, and an eval
baseline is not invalidated by an operational event that changed no behaviour.

Environment identity is not lost as a result: the environment name and the set
of secret-bearing paths are both part of the hashed content, so a staging
snapshot and a production snapshot hash differently even when their redacted
values coincide.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hardpoint.core.ids import stable_hash
from hardpoint.core.types import JsonValue

__all__ = ["REDACTED", "ConfigSnapshot", "Layer", "flatten", "redact"]

REDACTED = "***"
"""What a secret renders as. Chosen to be obviously a placeholder, not a value."""


class Layer(StrEnum):
    """Where a configuration value came from.

    The five layers of ARCHITECTURE.md §15.1, in resolution order. Last writer
    wins, and ``config show`` annotates each key with the layer that won, which
    is what turns "why is this setting wrong" into a one-line answer.

    Attributes:
        DEFAULTS: The typed default on the config model, in code.
        BASE_FILE: ``config/base.yaml``.
        ENV_FILE: ``config/{env}.yaml``.
        ENV_VARS: A ``HARDPOINT__SECTION__KEY`` environment variable.
        OVERRIDES: An explicit override passed in code.
    """

    DEFAULTS = "defaults"
    BASE_FILE = "base_file"
    ENV_FILE = "env_file"
    ENV_VARS = "env_vars"
    OVERRIDES = "overrides"


def flatten(data: Mapping[str, JsonValue], prefix: str = "") -> Iterator[tuple[str, JsonValue]]:
    """Yield ``(dotted_path, value)`` for every leaf in a nested mapping.

    Lists are leaves: a path into a list element would be unstable under any
    edit, and no layer addresses one.

    Args:
        data: A nested mapping.
        prefix: Path prefix, used in recursion.

    Yields:
        Pairs of dotted path and leaf value.
    """
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, path)
        else:
            yield path, value


def redact(data: Mapping[str, JsonValue], secret_paths: frozenset[str]) -> dict[str, JsonValue]:
    """Return a deep copy of ``data`` with every secret-bearing path replaced.

    Args:
        data: The resolved configuration.
        secret_paths: Dotted paths whose value derived from ``${env:...}``.

    Returns:
        A new nested dict in which each secret path holds :data:`REDACTED`.
    """

    def walk(node: Mapping[str, JsonValue], prefix: str) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in secret_paths:
                result[key] = REDACTED
            elif isinstance(value, dict):
                result[key] = walk(value, path)
            else:
                result[key] = value
        return result

    return walk(data, "")


def _traverse(node: Mapping[str, JsonValue], path: str, default: JsonValue) -> JsonValue:
    """Follow a dotted path into a nested mapping, returning ``default`` on a miss."""
    current: JsonValue = dict(node)
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable, hashed configuration for one process or one run.

    Args:
        env: The environment name that was resolved, for example ``"prod"``.
        data: The resolved values, secrets included. Never dumped.
        origins: Dotted path to the layer that supplied the winning value.
        secret_paths: Dotted paths whose value came from ``${env:...}``.

    Attributes:
        hash: Content hash over the redacted configuration, the environment
            name and the secret-bearing paths. Recorded in every run manifest.

    Raises:
        Nothing. Construction is pure.
    """

    env: str
    data: Mapping[str, JsonValue]
    origins: Mapping[str, Layer] = field(default_factory=dict)
    secret_paths: frozenset[str] = frozenset()

    # Derived in __post_init__ rather than passed in, so a caller cannot hand
    # over a redacted view that disagrees with the data or a hash that lies.
    redacted: Mapping[str, JsonValue] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    hash: str = field(init=False, compare=False, default="")

    def __post_init__(self) -> None:
        """Compute the redacted view and the hash once, at construction."""
        redacted = redact(self.data, self.secret_paths)
        # Widened to list[JsonValue] because JsonValue is invariant in its
        # element type and list[str] is not a subtype of it.
        secret_list: list[JsonValue] = list(sorted(self.secret_paths))  # noqa: C413
        hashed: dict[str, JsonValue] = {
            "env": self.env,
            "config": redacted,
            "secret_paths": secret_list,
        }
        object.__setattr__(self, "redacted", redacted)
        object.__setattr__(self, "hash", stable_hash(hashed))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the redacted configuration as a plain dict.

        Redacted, not resolved. This is what ``config show``, a trace attribute
        and a log record all call, so none of them can leak a secret by default.
        """
        return redact(self.data, self.secret_paths)

    def get(self, path: str, default: JsonValue = None) -> JsonValue:
        """Read a resolved value by dotted path, redacting secrets.

        Reads the *redacted* view rather than checking whether this exact path
        is secret. That distinction is the whole correctness of this method: an
        exact-path check leaves ``get("indexes.primary")`` returning the section
        as a raw dict with a live API key inside it, which is a leak through the
        accessor documented as the safe one.

        Args:
            path: Dotted path, for example ``providers.llm.model``. May address
                a section as well as a leaf.
            default: Returned when the path is absent.

        Returns:
            The value with every secret at or beneath it replaced by
            :data:`REDACTED`, or ``default`` when the path is absent.
        """
        return _traverse(self.redacted, path, default)

    def reveal(self, path: str, default: JsonValue = None) -> JsonValue:
        """Read a resolved value by dotted path **without** redacting it.

        The only way to obtain a secret from a snapshot. Named so that a reader
        of the calling code can see that a secret is being handled, and so that
        a search for ``reveal(`` finds every place one escapes.

        Call this in a component factory, immediately before passing the value
        to a client constructor. Never log the result.

        Args:
            path: Dotted path.
            default: Returned when the path is absent.

        Returns:
            The real value, or the whole section when ``path`` addresses one.
        """
        return _traverse(self.data, path, default)

    def origin(self, path: str) -> Layer:
        """Return the layer that supplied the winning value for a path.

        Paths no layer touched report :attr:`Layer.DEFAULTS`, because the value
        came from the typed default on the config model.
        """
        return self.origins.get(path, Layer.DEFAULTS)

    def is_secret(self, path: str) -> bool:
        """Return whether a path's value came from ``${env:...}``."""
        return path in self.secret_paths

    def paths(self) -> list[str]:
        """Return every dotted leaf path, sorted."""
        return sorted(path for path, _ in flatten(self.data))

    def __repr__(self) -> str:
        """Render the redacted configuration. Never the resolved one."""
        return (
            f"ConfigSnapshot(env={self.env!r}, hash={self.hash!r}, "
            f"keys={len(self.paths())}, secrets={len(self.secret_paths)})"
        )

    def __str__(self) -> str:
        """Alias of :meth:`__repr__`, so an f-string cannot leak a secret."""
        return repr(self)

    def __getstate__(self) -> dict[str, Any]:
        """Return the redacted state, so pickling cannot exfiltrate a secret.

        Every field is included, not only the four that identify the snapshot:
        omitting ``hash`` left an unpickled snapshot reporting an empty string
        for it, which is worse than failing outright because a run manifest
        would happily record ``config_hash=""``.

        A revived snapshot carries the redacted values, so :meth:`reveal`
        returns :data:`REDACTED` rather than a secret. That is intended -- a
        pickled configuration is a configuration that has left the process --
        and it means a snapshot must be re-resolved, not unpickled, wherever a
        real credential is needed.
        """
        redacted = self.to_dict()
        return {
            "env": self.env,
            "data": redacted,
            "origins": dict(self.origins),
            "secret_paths": self.secret_paths,
            "redacted": redacted,
            "hash": self.hash,
        }
