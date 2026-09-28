"""Five-layer configuration resolution (INSTRUCTIONS.md §5.7, **[LOCKED]**).

Implements ARCHITECTURE.md §15.1. There are exactly five layers and no others:

1. Library defaults, typed, in code -- the defaults on the models in ``schema``.
2. ``config/base.yaml``
3. ``config/{env}.yaml``
4. ``HARDPOINT__SECTION__KEY`` environment variables
5. Explicit overrides passed in code

Last writer wins. There are no conditionals, no loops, no includes and no
cross-file references. An earlier draft of this design had all of them, and it
was approaching a small programming language with no debugger
(ARCHITECTURE.md §6.3).

## ``${env:...}`` and why a line number matters

A value may reference an environment variable as ``${env:VAR}`` or
``${env:VAR:-default}``. References are resolved during validation, and a
missing required variable raises ``MissingEnvironmentVariableError`` naming the
variable, the file, and the line it was referenced on.

Getting that line number is why this module loads YAML through a positional
loader instead of ``yaml.safe_load``. "Missing environment variable QDRANT_URL"
is a shrug; "config/prod.yaml line 14 references QDRANT_URL, which is not set"
is a fix. The cost is one loader subclass.

## Secrets

Any value that actually consumed an environment value through ``${env:...}`` is
marked secret at its dotted path, and the snapshot redacts it in every dump,
log, trace attribute and error message.

Within that, the rule is deliberately blunt: ``${env:LLM_MODEL}`` is redacted
just like ``${env:OPENAI_API_KEY}``, because this layer cannot tell them apart
and guessing wrong in the permissive direction leaks a key. Redacting a model id
costs nothing that matters, because ``RunManifest.model_ids`` is populated from
what the adapter reports at run time rather than from a config dump.

A reference that fell back to its literal default is *not* marked. That default
is written in a tracked YAML file, so redacting it would hide nothing from
anyone who can read the repository while making ``config show`` less useful.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from hardpoint.core.config.schema import CONFIG_VERSION, HardpointConfig
from hardpoint.core.config.snapshot import ConfigSnapshot, Layer, flatten
from hardpoint.core.errors import InvalidConfigError, MissingEnvironmentVariableError
from hardpoint.core.types import JsonValue

__all__ = [
    "ENV_PREFIX",
    "ENV_SEPARATOR",
    "ResolvedConfig",
    "load_config",
    "parse_env_overrides",
    "resolve",
]

ENV_PREFIX: Final = "HARDPOINT"
"""Prefix identifying an environment override."""

ENV_SEPARATOR: Final = "__"
"""Double underscore, so a single underscore stays part of a key name.

``HARDPOINT__RETRIEVAL__TOP_K`` addresses ``retrieval.top_k``, not
``retrieval.top.k``.
"""

# ${env:VAR} or ${env:VAR:-default}. The default may contain anything but "}".
_ENV_REFERENCE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

_DEFAULT_ENV: Final = "dev"

_BOOL_TAG: Final = "tag:yaml.org,2002:bool"
_YAML12_BOOL: Final = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


class _LocatedStr(str):
    """A string that remembers where in which file it was written.

    Exists only so that a ``${env:...}`` failure can name a file and a line.
    Behaves as an ordinary ``str`` everywhere else, including in comparisons,
    hashing and YAML round-tripping.
    """

    source: str
    line: int

    def __new__(cls, value: str, source: str = "<unknown>", line: int = 0) -> _LocatedStr:
        """Create the string and attach its origin."""
        located = super().__new__(cls, value)
        located.source = source
        located.line = line
        return located

    def where(self) -> str:
        """Render the origin as ``file line N``, for an error message."""
        return f"{self.source} line {self.line}"


def _positional_loader(source: str) -> type[yaml.SafeLoader]:
    """Build a SafeLoader subclass that tags every scalar string with its position."""

    class PositionalLoader(yaml.SafeLoader):
        """A SafeLoader that returns :class:`_LocatedStr` for scalar strings."""

    def construct_located_str(loader: yaml.SafeLoader, node: yaml.Node) -> str:
        value = loader.construct_scalar(node)  # type: ignore[arg-type]  # always a ScalarNode
        return _LocatedStr(str(value), source=source, line=node.start_mark.line + 1)

    PositionalLoader.add_constructor("tag:yaml.org,2002:str", construct_located_str)

    # YAML 1.2 booleans: only true/false. Under PyYAML's YAML 1.1 default, the
    # key in ARCHITECTURE.md §15.2's own example -- `retry: {on: [transient]}` --
    # loads as the boolean True, and `country: no` as False.
    PositionalLoader.yaml_implicit_resolvers = {
        first: [(tag, pattern) for tag, pattern in resolvers if tag != _BOOL_TAG]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    PositionalLoader.add_implicit_resolver(_BOOL_TAG, _YAML12_BOOL, list("tTfF"))
    return PositionalLoader


def _load_yaml_file(path: Path) -> dict[str, JsonValue]:
    """Load one YAML file, preserving scalar positions.

    Args:
        path: The file to read.

    Returns:
        The parsed mapping, or an empty dict for an empty file.

    Raises:
        InvalidConfigError: If the file is not valid YAML, or its top level is
            not a mapping.
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.load(text, Loader=_positional_loader(str(path)))  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise InvalidConfigError(
            f"{path} is not valid YAML: {exc}",
            config_path=str(path),
            remedy=(
                f"Fix the YAML syntax in {path}. Most often this is an indentation "
                "error or an unquoted value containing a colon."
            ),
            cause=exc,
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InvalidConfigError(
            f"{path} must contain a mapping at the top level, found {type(data).__name__}.",
            config_path=str(path),
            remedy=f"Rewrite {path} so its top level is a mapping of section names to values.",
        )
    return data


def _deep_merge(
    base: Mapping[str, JsonValue],
    overlay: Mapping[str, JsonValue],
    layer: Layer,
    origins: MutableMapping[str, Layer],
    prefix: str = "",
) -> dict[str, JsonValue]:
    """Merge ``overlay`` onto ``base``, recording which layer won for each leaf.

    Mappings merge key by key. Every other value, lists included, replaces
    wholesale. A list that merged element-wise would make ``guards.output``
    impossible to shorten in a production overlay, which is exactly the thing an
    overlay exists to do.

    One exception among mappings: a component block whose ``type:`` changes is
    replaced, not merged. Its options belong to the component, so an offline
    overlay switching ``openai_chat`` to ``fake_llm`` must not inherit
    ``api_key`` and ``base_url`` -- they would be unknown options for the new
    component, and would demand environment variables it never reads.
    """
    result: dict[str, JsonValue] = dict(base)
    for key, value in overlay.items():
        path = f"{prefix}.{key}" if prefix else key
        existing = result.get(key)
        switches_component = (
            isinstance(value, dict)
            and isinstance(existing, dict)
            and "type" in value
            and "type" in existing
            and value["type"] != existing["type"]
        )
        if isinstance(value, dict) and isinstance(existing, dict) and not switches_component:
            result[key] = _deep_merge(existing, value, layer, origins, path)
            continue

        # Wholesale replacement. Any origin recorded for a path *under* this one
        # now describes a key that no longer exists, and `config show` would
        # print it as though it were live. Purge before recording the new one.
        if isinstance(existing, dict):
            for stale in [p for p in origins if p == path or p.startswith(f"{path}.")]:
                del origins[stale]

        result[key] = value
        if isinstance(value, dict):
            for leaf, _ in flatten(value, path):
                origins[leaf] = layer
        else:
            origins[path] = layer
    return result


def parse_env_overrides(environ: Mapping[str, str]) -> dict[str, JsonValue]:
    """Turn ``HARDPOINT__SECTION__KEY`` variables into a nested mapping.

    Section names are case-insensitive and lowercased. Values are parsed as YAML
    scalars, so ``HARDPOINT__RETRIEVAL__TOP_K=40`` yields the integer ``40`` and
    ``HARDPOINT__PLUGINS__DISCOVER=true`` yields the boolean ``True``. A value
    that is not valid YAML is kept as the literal string.

    Args:
        environ: The environment to read. Passed in rather than read from
            ``os.environ`` here, so that tests never mutate the real one.

    Returns:
        A nested mapping suitable for merging as layer four.

    Raises:
        InvalidConfigError: If a variable names a path that collides with a
            path already claimed by another variable.
    """
    overrides: dict[str, JsonValue] = {}
    claimed: dict[str, str] = {}

    # Sorted by the *resolved* path, not by the raw variable name. Section names
    # are case-insensitive, so sorting by the raw name lets HARDPOINT__A__B__C
    # be processed before hardpoint__a__b, at which point the shorter variable
    # overwrites the section the longer one built -- silently, and in the
    # direction that loses configuration rather than reporting a conflict.
    # Sorting by the lowercased path makes a prefix always come first, which is
    # what the conflict detection below relies on.
    entries: list[tuple[tuple[str, ...], str]] = []
    for name in environ:
        if not name.upper().startswith(f"{ENV_PREFIX}{ENV_SEPARATOR}"):
            continue
        remainder = name[len(ENV_PREFIX) + len(ENV_SEPARATOR) :]
        parts = tuple(part.lower() for part in remainder.split(ENV_SEPARATOR) if part)
        if parts:
            entries.append((parts, name))
    entries.sort()

    def conflict(name: str, path: str, at: str) -> InvalidConfigError:
        return InvalidConfigError(
            f"Environment override {name} sets {path!r}, but {at!r} was already "
            f"claimed by {claimed.get(at, 'another variable')}.",
            config_path=path,
            remedy=(
                "Unset one of the two variables. A configuration path cannot be "
                "both a value and a section."
            ),
        )

    for parts, name in entries:
        path = ".".join(parts)
        try:
            value = yaml.safe_load(environ[name])
        except yaml.YAMLError:
            value = environ[name]

        node: dict[str, JsonValue] = overrides
        for index, part in enumerate(parts[:-1]):
            branch = node.get(part)
            if not isinstance(branch, dict):
                if branch is not None:
                    raise conflict(name, path, ".".join(parts[: index + 1]))
                branch = {}
                node[part] = branch
            node = branch

        # A section already stands here, so this variable would replace it with
        # a scalar. Refusing is the whole point: the alternative is losing every
        # setting underneath it without a word.
        if isinstance(node.get(parts[-1]), dict):
            raise conflict(name, path, path)

        node[parts[-1]] = value
        claimed[path] = name

    return overrides


def _interpolate(
    data: Mapping[str, JsonValue],
    environ: Mapping[str, str],
    secret_paths: set[str],
    prefix: str = "",
) -> dict[str, JsonValue]:
    """Resolve every ``${env:...}`` reference, recording which paths held one.

    Raises:
        MissingEnvironmentVariableError: For a reference with no value and no
            default, naming the variable and the file and line it appeared on.
    """
    result: dict[str, JsonValue] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result[str(key)] = _interpolate_value(value, environ, secret_paths, path)
    return result


def _interpolate_value(
    value: JsonValue, environ: Mapping[str, str], secret_paths: set[str], path: str
) -> JsonValue:
    """Interpolate any value, however nested -- including mappings inside lists.

    Guards are configured as a list of mappings, and a reference inside one
    would otherwise reach validation as the literal text ``${env:...}``.
    """
    if isinstance(value, dict):
        return _interpolate(value, environ, secret_paths, path)
    if isinstance(value, list):
        # Redaction walks mappings, not list positions, so a secret anywhere
        # inside a list marks the whole list: `guards.output` is redacted as a
        # unit rather than leaking a key from its third element.
        inside: set[str] = set()
        items = [_interpolate_value(item, environ, inside, path) for item in value]
        if inside:
            secret_paths.add(path)
        return items
    if isinstance(value, str):
        return _interpolate_scalar(value, environ, secret_paths, path)
    return value


def _interpolate_scalar(
    value: str,
    environ: Mapping[str, str],
    secret_paths: set[str],
    path: str,
) -> str:
    """Resolve references inside one string, marking the path secret if warranted.

    A path is marked secret when a reference actually consumed a value from the
    environment. A reference that fell back to its literal default does **not**
    mark the path: that default is written in a tracked YAML file, so it is
    visible to anyone who can read the repository and redacting it would hide
    nothing while making ``config show`` less useful. ARCHITECTURE.md §19 makes
    the same assumption in the other direction, by having ``doctor`` fail when a
    literal-looking API key appears in tracked YAML.
    """
    if not _ENV_REFERENCE.search(value):
        # A plain `str`: the position was only needed for this function's
        # errors, and the private subclass must not leak into dumps.
        return str(value)

    where = value.where() if isinstance(value, _LocatedStr) else path
    consumed_environment = False

    def substitute(match: re.Match[str]) -> str:
        nonlocal consumed_environment
        variable, fallback = match.group(1), match.group(2)
        if variable in environ:
            consumed_environment = True
            return environ[variable]
        if fallback is not None:
            return fallback
        raise MissingEnvironmentVariableError(
            f"{where} references the environment variable {variable}, which is not set.",
            variable=variable,
            config_path=path,
            remedy=(
                f"Set {variable} in the environment, or give the reference a default:\n"
                f"    ${{env:{variable}:-some-default}}\n"
                f"For local development, add {variable} to your .env file."
            ),
        )

    resolved = _ENV_REFERENCE.sub(substitute, str(value))
    if consumed_environment:
        secret_paths.add(path)
    return resolved


def _suggest(unknown: str, valid: Iterable[str]) -> str:
    """Return a 'did you mean' fragment, or an empty string when nothing is close."""
    candidates = difflib.get_close_matches(unknown, list(valid), n=1, cutoff=0.6)
    return f" Did you mean {candidates[0]!r}?" if candidates else ""


def _validate(data: Mapping[str, JsonValue], sources: str) -> HardpointConfig:
    """Validate the merged mapping, translating pydantic errors into remedied ones.

    Raises:
        InvalidConfigError: With the offending path, and for an unknown key, the
            closest valid key at that level.
    """
    try:
        return HardpointConfig.model_validate(dict(data))
    except ValidationError as exc:
        problems: list[str] = []
        first_path = ""
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            first_path = first_path or location
            if error["type"] == "extra_forbidden":
                valid = _valid_keys_at(error["loc"][:-1])
                problems.append(
                    f"  {location}: unknown configuration key."
                    f"{_suggest(str(error['loc'][-1]), valid)}"
                )
            else:
                problems.append(f"  {location}: {error['msg']}")

        raise InvalidConfigError(
            f"Configuration is invalid ({sources}):\n" + "\n".join(problems),
            config_path=first_path or None,
            remedy=(
                "Correct the keys listed above. `hardpoint config schema` writes a "
                "JSON Schema your editor can use to validate the file in place, and "
                "`hardpoint config show --resolved` prints which layer set each key."
            ),
            cause=exc,
        ) from exc


def _valid_keys_at(location: tuple[int | str, ...]) -> list[str]:
    """Return the field names valid at a location in the config model tree."""
    model: Any = HardpointConfig
    for part in location:
        fields = getattr(model, "model_fields", None)
        if fields is None:
            return []
        field = fields.get(str(part))
        if field is None:
            # A mapping of arbitrary names, such as `indexes`. Descend into its value type.
            args = getattr(model, "__pydantic_generic_metadata__", {}).get("args", ())
            model = args[-1] if args else None
            if model is None:
                return []
            continue
        model = field.annotation
        for candidate in getattr(model, "__args__", ()):
            if hasattr(candidate, "model_fields"):
                model = candidate
                break
    fields = getattr(model, "model_fields", None)
    return sorted(fields) if fields else []


class ResolvedConfig:
    """A validated config model paired with the snapshot it came from.

    Two views of the same resolution. ``config`` is typed and is what code reads
    for behaviour; ``snapshot`` carries origins, secrets and the hash, and is
    what a run manifest and ``config show`` read.

    Args:
        config: The validated root model.
        snapshot: The immutable, hashed snapshot.
    """

    __slots__ = ("config", "snapshot")

    def __init__(self, config: HardpointConfig, snapshot: ConfigSnapshot) -> None:
        self.config = config
        self.snapshot = snapshot

    def __repr__(self) -> str:
        """Render via the snapshot, which is already redacted."""
        return f"ResolvedConfig({self.snapshot!r})"


def resolve(
    *,
    env: str = _DEFAULT_ENV,
    base: Mapping[str, JsonValue] | None = None,
    env_file: Mapping[str, JsonValue] | None = None,
    overrides: Mapping[str, JsonValue] | None = None,
    environ: Mapping[str, str] | None = None,
    source_label: str = "in-memory configuration",
) -> ResolvedConfig:
    """Resolve configuration from already-loaded mappings.

    The file-free half of :func:`load_config`, so that layering can be tested
    without a filesystem and so a generated project can supply its own loading.

    Args:
        env: Environment name, recorded on the snapshot and part of its hash.
        base: Layer two, normally ``config/base.yaml``.
        env_file: Layer three, normally ``config/{env}.yaml``.
        overrides: Layer five, explicit code overrides.
        environ: The environment to read for layer four and for ``${env:...}``.
            Defaults to ``os.environ``, read here rather than at import time.
        source_label: What to name in an error message.

    Returns:
        The validated config and its snapshot.

    Raises:
        InvalidConfigError: On an unknown key, a bad value, or a version
            mismatch.
        MissingEnvironmentVariableError: On an unresolvable ``${env:...}``.
    """
    environment = os.environ if environ is None else environ

    origins: dict[str, Layer] = {}
    merged: dict[str, JsonValue] = {}
    for layer, contribution in (
        (Layer.BASE_FILE, base),
        (Layer.ENV_FILE, env_file),
        (Layer.ENV_VARS, parse_env_overrides(environment)),
        (Layer.OVERRIDES, overrides),
    ):
        if contribution:
            merged = _deep_merge(merged, contribution, layer, origins)

    secret_paths: set[str] = set()
    resolved = _interpolate(merged, environment, secret_paths)

    _check_version(resolved, source_label)
    config = _validate(resolved, source_label)

    snapshot = ConfigSnapshot(
        env=env,
        data=resolved,
        origins=origins,
        secret_paths=frozenset(secret_paths),
    )
    return ResolvedConfig(config=config, snapshot=snapshot)


def _check_version(data: Mapping[str, JsonValue], sources: str) -> None:
    """Reject a config written against a different schema version."""
    version = data.get("version", CONFIG_VERSION)
    if version == CONFIG_VERSION:
        return
    raise InvalidConfigError(
        f"Configuration version {version!r} is not supported by this release "
        f"({sources}); this release reads version {CONFIG_VERSION}.",
        config_path="version",
        remedy=(
            f"Set `version: {CONFIG_VERSION}` at the top of the configuration and "
            "check CHANGELOG.md for the shape changes between the two versions."
        ),
    )


def load_config(
    config_dir: str | Path = "config",
    *,
    env: str | None = None,
    overrides: Mapping[str, JsonValue] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Load and resolve configuration from a directory of YAML files.

    Reads ``base.yaml`` and ``{env}.yaml`` from ``config_dir``. Both are
    optional: a project may configure entirely through environment variables,
    and a missing environment overlay is normal rather than an error.

    Args:
        config_dir: Directory holding the YAML layers.
        env: Environment name. Defaults to ``HARDPOINT_ENV`` from the
            environment, then to ``"dev"``.
        overrides: Layer five, explicit code overrides.
        environ: The environment to read. Defaults to ``os.environ``.

    Returns:
        The validated config and its snapshot.

    Raises:
        InvalidConfigError: On invalid YAML, an unknown key, or a bad value.
        MissingEnvironmentVariableError: On an unresolvable ``${env:...}``.
    """
    environment = os.environ if environ is None else environ
    resolved_env = env or environment.get("HARDPOINT_ENV") or _DEFAULT_ENV

    directory = Path(config_dir)
    base_path = directory / "base.yaml"
    env_path = directory / f"{resolved_env}.yaml"

    base = _load_yaml_file(base_path) if base_path.is_file() else None
    env_file = _load_yaml_file(env_path) if env_path.is_file() else None

    read = [str(path) for path in (base_path, env_path) if path.is_file()]
    label = ", ".join(read) if read else f"no config files found in {directory}"

    return resolve(
        env=resolved_env,
        base=base,
        env_file=env_file,
        overrides=overrides,
        environ=environment,
        source_label=label,
    )
