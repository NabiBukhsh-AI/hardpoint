"""Five-layer configuration resolution. **[LOCKED]** INSTRUCTIONS.md §5.7.

The M0 Definition of Done asks for a layering test covering all five layers and
secret redaction. Both are here, plus the error-quality requirements that make
configuration debuggable rather than a guessing game: a missing environment
variable names the file and line, and an unknown key names the closest valid
one.

Nothing here touches the real environment or the real filesystem outside
``tmp_path``. ``resolve`` and ``load_config`` both take ``environ`` explicitly
so a test never has to mutate ``os.environ``.
"""

from __future__ import annotations

import inspect
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from hardpoint.core.config import (
    CONFIG_VERSION,
    REDACTED,
    ComponentSpec,
    ConfigSnapshot,
    GuardSpec,
    HardpointConfig,
    Layer,
    load_config,
    parse_env_overrides,
    resolve,
)
from hardpoint.core.config import schema as schema_module
from hardpoint.core.errors import InvalidConfigError, MissingEnvironmentVariableError

# S105: this is a fake credential whose whole purpose is to be looked for in
# output. A real one would defeat the test.
SECRET = "sk-live-do-not-log-me-0123456789"  # noqa: S105


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The five layers                                                             #
# --------------------------------------------------------------------------- #


def test_all_five_layers_resolve_in_order() -> None:
    """**[LOCKED]** Five layers, last writer wins, and each records its origin.

    One key per layer, so a regression tells you *which* layer stopped working
    rather than only that layering is broken.
    """
    resolved = resolve(
        env="prod",
        base={
            "retrieval": {
                "top_k": 10,
                "score_threshold": 0.1,
                "context": {"token_budget": 1000},
                "no_context_policy": "abstain",
            }
        },
        env_file={"retrieval": {"score_threshold": 0.5, "context": {"token_budget": 2000}}},
        environ={"HARDPOINT__RETRIEVAL__CONTEXT__TOKEN_BUDGET": "3000"},
        overrides={"retrieval": {"no_context_policy": "raise"}},
    )
    config, snapshot = resolved.config, resolved.snapshot

    # Layer 1: a typed default nothing overrode.
    assert config.retrieval.context.citation_style == "numeric"
    assert snapshot.origin("retrieval.context.citation_style") is Layer.DEFAULTS

    # Layer 2: base.yaml, unopposed.
    assert config.retrieval.top_k == 10
    assert snapshot.origin("retrieval.top_k") is Layer.BASE_FILE

    # Layer 3: the environment overlay beats base.
    assert config.retrieval.score_threshold == 0.5
    assert snapshot.origin("retrieval.score_threshold") is Layer.ENV_FILE

    # Layer 4: an environment variable beats both files.
    assert config.retrieval.context.token_budget == 3000
    assert snapshot.origin("retrieval.context.token_budget") is Layer.ENV_VARS

    # Layer 5: an explicit code override beats everything.
    assert config.retrieval.no_context_policy == "raise"
    assert snapshot.origin("retrieval.no_context_policy") is Layer.OVERRIDES


def test_merge_is_deep_for_mappings() -> None:
    """An overlay sets one key of a section without erasing its siblings."""
    resolved = resolve(
        base={"indexes": {"primary": {"type": "qdrant", "collection": "v1", "distance": "cosine"}}},
        env_file={"indexes": {"primary": {"collection": "v3"}}},
        environ={},
    )
    primary = resolved.config.indexes["primary"]
    assert primary.type == "qdrant"
    assert primary.options() == {"collection": "v3", "distance": "cosine"}


def test_switching_a_component_type_replaces_the_block() -> None:
    """An offline overlay swapping a provider must not inherit the old one's options.

    Merged, ``fake_llm`` would receive ``api_key`` -- an unknown option -- and the
    base's ``${env:OPENAI_API_KEY}`` would still demand a key the offline
    environment never needs.
    """
    resolved = resolve(
        base={
            "providers": {
                "llm": {"type": "openai_chat", "model": "m", "api_key": "${env:OPENAI_API_KEY}"}
            }
        },
        env_file={"providers": {"llm": {"type": "fake_llm"}}},
        environ={},
    )
    llm = resolved.config.providers.llm
    assert llm is not None
    assert llm.type == "fake_llm"
    assert llm.options() == {}
    assert "providers.llm.api_key" not in resolved.snapshot.paths()
    assert resolved.snapshot.origin("providers.llm.type") is Layer.ENV_FILE


def test_keeping_a_component_type_still_merges() -> None:
    resolved = resolve(
        base={"providers": {"llm": {"type": "openai_chat", "model": "a", "base_url": "u"}}},
        env_file={"providers": {"llm": {"type": "openai_chat", "model": "b"}}},
        environ={},
    )
    llm = resolved.config.providers.llm
    assert llm is not None
    assert llm.options() == {"model": "b", "base_url": "u"}


def test_lists_replace_wholesale_rather_than_merging() -> None:
    """A production overlay must be able to *shorten* a list, not only extend it.

    Element-wise list merging would make it impossible to turn a guard off in
    one environment, which is the main thing an overlay exists to do.
    """
    resolved = resolve(
        base={"guards": {"output": [{"type": "schema"}, {"type": "groundedness"}]}},
        env_file={"guards": {"output": [{"type": "schema"}]}},
        environ={},
    )
    assert [guard.type for guard in resolved.config.guards.output] == ["schema"]


def test_resolution_with_no_layers_at_all_yields_defaults() -> None:
    """A project may configure entirely through code. That is not an error."""
    resolved = resolve(environ={})
    assert resolved.config == HardpointConfig()
    assert resolved.config.version == CONFIG_VERSION
    assert resolved.snapshot.data == {}


# --------------------------------------------------------------------------- #
# Environment overrides                                                       #
# --------------------------------------------------------------------------- #


def test_env_overrides_use_double_underscore_and_keep_single_ones() -> None:
    """``HARDPOINT__RETRIEVAL__TOP_K`` is ``retrieval.top_k``, not ``retrieval.top.k``."""
    assert parse_env_overrides({"HARDPOINT__RETRIEVAL__TOP_K": "40"}) == {
        "retrieval": {"top_k": 40}
    }


def test_env_override_section_names_are_case_insensitive() -> None:
    assert parse_env_overrides({"hardpoint__Retrieval__TOP_K": "5"}) == {"retrieval": {"top_k": 5}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("40", 40),
        ("0.5", 0.5),
        ("true", True),
        ("false", False),
        ("null", None),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("[1, 2]", [1, 2]),
    ],
)
def test_env_override_values_parse_as_yaml_scalars(raw: str, expected: Any) -> None:
    """So that ``TOP_K=40`` is an integer and ``DISCOVER=true`` is a boolean."""
    assert parse_env_overrides({"HARDPOINT__SECTION__KEY": raw}) == {"section": {"key": expected}}


def test_unrelated_environment_variables_are_ignored() -> None:
    assert parse_env_overrides({"PATH": "/usr/bin", "HARDPOINTX__A": "1", "HARDPOINT": "x"}) == {}


def test_env_override_conflicting_with_a_scalar_path_is_an_error() -> None:
    """A path cannot be both a value and a section; silently picking one would hide a typo."""
    with pytest.raises(InvalidConfigError) as exc_info:
        parse_env_overrides(
            {"HARDPOINT__A__B": "1", "HARDPOINT__A__B__C": "2"},
        )
    assert "HARDPOINT__A__B" in str(exc_info.value)
    assert exc_info.value.remedy


# --------------------------------------------------------------------------- #
# ${env:...} interpolation                                                    #
# --------------------------------------------------------------------------- #


def test_env_reference_is_resolved() -> None:
    resolved = resolve(
        base={"indexes": {"p": {"type": "qdrant", "url": "${env:QDRANT_URL}"}}},
        environ={"QDRANT_URL": "https://q:6333"},
    )
    assert resolved.snapshot.reveal("indexes.p.url") == "https://q:6333"


def test_env_reference_default_is_used_when_the_variable_is_unset() -> None:
    resolved = resolve(
        base={"providers": {"llm": {"type": "x", "model": "${env:LLM_MODEL:-gpt-4o-mini}"}}},
        environ={},
    )
    assert resolved.snapshot.get("providers.llm.model") == "gpt-4o-mini"


def test_env_reference_can_be_embedded_in_a_larger_string() -> None:
    """A DSN with a password in the middle is the case that matters here."""
    resolved = resolve(
        base={"indexes": {"p": {"type": "pg", "dsn": "postgres://u:${env:PW}@host/db"}}},
        environ={"PW": SECRET},
    )
    assert resolved.snapshot.reveal("indexes.p.dsn") == f"postgres://u:{SECRET}@host/db"
    assert resolved.snapshot.get("indexes.p.dsn") == REDACTED


def test_env_references_resolve_inside_lists() -> None:
    resolved = resolve(
        base={"observability": {"redact": ["${env:FIELD:-messages.content}"]}},
        environ={},
    )
    assert resolved.config.observability.redact == ("messages.content",)


def test_missing_environment_variable_names_the_file_and_line(tmp_path: Path) -> None:
    """ "Missing QDRANT_URL" is a shrug; naming the file and line is a fix."""
    write(tmp_path, "base.yaml", "version: 1\n")
    write(
        tmp_path,
        "prod.yaml",
        "indexes:\n  primary:\n    type: qdrant\n    url: ${env:QDRANT_URL}\n",
    )

    with pytest.raises(MissingEnvironmentVariableError) as exc_info:
        load_config(tmp_path, env="prod", environ={})

    error = exc_info.value
    assert error.variable == "QDRANT_URL"
    rendered = str(error)
    assert "prod.yaml" in rendered
    assert "line 4" in rendered
    assert error.config_path == "indexes.primary.url"
    assert error.remedy is not None
    assert "${env:QDRANT_URL:-" in error.remedy


# --------------------------------------------------------------------------- #
# Secret redaction  [LOCKED]                                                  #
# --------------------------------------------------------------------------- #


def resolved_with_secret() -> Any:
    return resolve(
        env="prod",
        base={
            "indexes": {"primary": {"type": "qdrant", "api_key": "${env:QDRANT_API_KEY}"}},
            "providers": {"llm": {"type": "openai_chat", "model": "gpt-4o-mini"}},
        },
        environ={"QDRANT_API_KEY": SECRET},
    )


def test_the_secret_appears_in_no_dump_repr_or_serialisation() -> None:
    """**[LOCKED]** Dump a config containing a secret; the value appears nowhere.

    Every route by which a value could plausibly escape is checked, because a
    single unredacted path is enough to put an API key in a log aggregator.
    """
    snapshot = resolved_with_secret().snapshot

    surfaces = {
        "to_dict": json.dumps(snapshot.to_dict()),
        "redacted": json.dumps(dict(snapshot.redacted)),
        "repr": repr(snapshot),
        "str": str(snapshot),
        "f-string": f"{snapshot}",
        "get": str(snapshot.get("indexes.primary.api_key")),
        "pickle": repr(pickle.dumps(snapshot)),
        "resolved-repr": repr(resolved_with_secret()),
    }
    leaked = [name for name, text in surfaces.items() if SECRET in text]
    assert not leaked, f"the secret leaked through: {leaked}"


def test_the_redacted_value_is_an_obvious_placeholder() -> None:
    snapshot = resolved_with_secret().snapshot
    assert snapshot.to_dict()["indexes"]["primary"]["api_key"] == REDACTED  # type: ignore[index,call-overload]


def test_reveal_is_the_only_way_out() -> None:
    """A factory needs the real key. Naming the method makes every escape greppable."""
    snapshot = resolved_with_secret().snapshot
    assert snapshot.reveal("indexes.primary.api_key") == SECRET
    assert snapshot.is_secret("indexes.primary.api_key") is True


def test_non_secret_values_are_not_redacted() -> None:
    """Over-redaction would make ``config show`` useless, so it is not blanket."""
    snapshot = resolved_with_secret().snapshot
    assert snapshot.get("providers.llm.model") == "gpt-4o-mini"
    assert snapshot.is_secret("providers.llm.model") is False


def test_a_literal_default_is_not_treated_as_a_secret() -> None:
    """It is written in a tracked YAML file, so redacting it hides nothing."""
    snapshot = resolve(
        base={"providers": {"llm": {"type": "x", "model": "${env:LLM_MODEL:-gpt-4o-mini}"}}},
        environ={},
    ).snapshot
    assert snapshot.is_secret("providers.llm.model") is False
    assert snapshot.get("providers.llm.model") == "gpt-4o-mini"


def test_a_consumed_environment_value_is_a_secret_even_with_a_default_present() -> None:
    snapshot = resolve(
        base={"providers": {"llm": {"type": "x", "model": "${env:LLM_MODEL:-fallback}"}}},
        environ={"LLM_MODEL": "gpt-4o"},
    ).snapshot
    assert snapshot.is_secret("providers.llm.model") is True


# --------------------------------------------------------------------------- #
# Snapshot hashing                                                            #
# --------------------------------------------------------------------------- #


def test_hash_is_stable_across_equivalent_resolutions() -> None:
    """The same effective config must hash identically however it was assembled."""
    from_files = resolve(env="prod", base={"retrieval": {"top_k": 7}}, environ={})
    from_overrides = resolve(env="prod", overrides={"retrieval": {"top_k": 7}}, environ={})
    assert from_files.snapshot.hash == from_overrides.snapshot.hash


def test_hash_changes_when_a_value_changes() -> None:
    a = resolve(base={"retrieval": {"top_k": 7}}, environ={}).snapshot
    b = resolve(base={"retrieval": {"top_k": 8}}, environ={}).snapshot
    assert a.hash != b.hash


def test_hash_changes_with_the_environment_name() -> None:
    """Staging and production must not share a config hash in a run manifest."""
    staging = resolve(env="staging", base={"retrieval": {"top_k": 7}}, environ={}).snapshot
    production = resolve(env="prod", base={"retrieval": {"top_k": 7}}, environ={}).snapshot
    assert staging.hash != production.hash


def test_rotating_a_secret_does_not_change_the_hash() -> None:
    """A documented consequence of hashing the redacted content, and the right one.

    A key rotation changes no behaviour, so it must not invalidate an eval
    baseline or make two run manifests incomparable.
    """
    base = {"indexes": {"p": {"type": "qdrant", "api_key": "${env:KEY}"}}}
    before = resolve(base=base, environ={"KEY": "old-key"}).snapshot
    after = resolve(base=base, environ={"KEY": "new-key-entirely"}).snapshot
    assert before.hash == after.hash


def test_hash_changes_when_a_field_becomes_secret_bearing() -> None:
    """Which paths hold secrets is part of the identity of a configuration."""
    literal = resolve(base={"indexes": {"p": {"type": "q", "url": "http://h"}}}, environ={})
    from_env = resolve(
        base={"indexes": {"p": {"type": "q", "url": "${env:U}"}}},
        environ={"U": "http://h"},
    )
    assert literal.snapshot.hash != from_env.snapshot.hash


def test_snapshot_is_immutable() -> None:
    snapshot = resolve(environ={}).snapshot
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        snapshot.env = "other"  # type: ignore[misc]


def test_snapshot_reports_paths_and_absent_lookups() -> None:
    snapshot = resolve(base={"retrieval": {"top_k": 7}}, environ={}).snapshot
    assert snapshot.paths() == ["retrieval.top_k"]
    assert snapshot.get("nope.nothing", default="fallback") == "fallback"
    assert snapshot.reveal("retrieval.top_k.deeper", default=None) is None
    assert snapshot.origin("never.set") is Layer.DEFAULTS


def test_snapshot_can_be_built_directly_for_tests() -> None:
    """``testing`` needs to build one without a loader; that must stay easy."""
    snapshot = ConfigSnapshot(env="test", data={"a": {"b": 1}})
    assert snapshot.hash
    assert snapshot.get("a.b") == 1


# --------------------------------------------------------------------------- #
# Validation and error quality                                                #
# --------------------------------------------------------------------------- #


def test_unknown_key_is_an_error_naming_the_closest_valid_key() -> None:
    """**Never silently ignore an unknown config key** (INSTRUCTIONS.md §13.7)."""
    with pytest.raises(InvalidConfigError) as exc_info:
        resolve(base={"retrieval": {"top_kk": 5}}, environ={})

    rendered = str(exc_info.value)
    assert "retrieval.top_kk" in rendered
    assert "unknown configuration key" in rendered
    assert "'top_k'" in rendered, "the suggestion is the point of the error"


def test_unknown_top_level_section_is_an_error_with_a_suggestion() -> None:
    with pytest.raises(InvalidConfigError) as exc_info:
        resolve(base={"retreival": {}}, environ={})
    assert "'retrieval'" in str(exc_info.value)


def test_unknown_key_with_no_close_match_still_errors() -> None:
    with pytest.raises(InvalidConfigError, match="unknown configuration key"):
        resolve(base={"zzzzzz": 1}, environ={})


def test_invalid_value_is_reported_with_its_path() -> None:
    with pytest.raises(InvalidConfigError) as exc_info:
        resolve(base={"retrieval": {"top_k": 0}}, environ={})
    assert "retrieval.top_k" in str(exc_info.value)


def test_closed_enums_reject_a_plausible_typo() -> None:
    """``no_context_policy: abstein`` must fail, not silently answer anyway."""
    with pytest.raises(InvalidConfigError):
        resolve(base={"retrieval": {"no_context_policy": "abstein"}}, environ={})


def test_retry_on_is_a_closed_set() -> None:
    """A misspelled kind would quietly disable retries for that error class."""
    with pytest.raises(InvalidConfigError):
        resolve(
            base={
                "providers": {
                    "llm": {"type": "x", "policies": {"retry": {"on": ["rate_limitted"]}}}
                }
            },
            environ={},
        )


def test_wrong_config_version_is_rejected_with_a_remedy() -> None:
    with pytest.raises(InvalidConfigError) as exc_info:
        resolve(base={"version": 99}, environ={})
    assert "version" in str(exc_info.value)
    assert exc_info.value.remedy is not None
    assert "CHANGELOG" in exc_info.value.remedy


def test_error_remedy_points_at_the_tools_that_would_have_caught_it() -> None:
    with pytest.raises(InvalidConfigError) as exc_info:
        resolve(base={"retrieval": {"top_kk": 5}}, environ={})
    remedy = exc_info.value.remedy or ""
    assert "config schema" in remedy
    assert "config show" in remedy


# --------------------------------------------------------------------------- #
# File loading                                                                #
# --------------------------------------------------------------------------- #


def test_env_references_resolve_inside_mappings_inside_lists() -> None:
    """Guards are a list of mappings; a reference in one must still resolve."""
    resolved = resolve(
        base={"guards": {"output": [{"type": "moderation", "api_key": "${env:MOD_KEY}"}]}},
        environ={"MOD_KEY": SECRET},
    )
    (guard,) = resolved.config.guards.output
    assert guard.options() == {"api_key": SECRET}
    assert SECRET not in json.dumps(resolved.snapshot.to_dict())


def test_loaded_values_are_plain_strings(tmp_path: Path) -> None:
    """The loader's position-tracking subclass must not leak into dumps."""
    import yaml

    write(tmp_path, "base.yaml", "project: {name: demo}\n")
    data = load_config(tmp_path, env="dev", environ={}).snapshot.to_dict()
    assert "name: demo" in yaml.safe_dump(data)


def test_yaml_booleans_follow_yaml_1_2(tmp_path: Path) -> None:
    """``on:`` is a key, not ``True``, as in ARCHITECTURE.md §15.2's own example.

    PyYAML defaults to YAML 1.1, where on/off/yes/no are booleans. The retry
    block the architecture documents would then fail validation with "keys
    should be strings".
    """
    write(
        tmp_path,
        "base.yaml",
        "providers:\n  llm:\n    type: x\n    region: no\n"
        "    policies:\n      retry: {max_attempts: 4, on: [rate_limited]}\n"
        "plugins: {discover: false}\n",
    )
    resolved = load_config(tmp_path, env="dev", environ={})
    llm = resolved.config.providers.llm
    assert llm is not None
    assert llm.policies.retry.on == ("rate_limited",)
    assert llm.options() == {"region": "no"}
    assert resolved.config.plugins.discover is False


def test_load_config_reads_base_and_environment_overlay(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "version: 1\nretrieval:\n  top_k: 10\n")
    write(tmp_path, "prod.yaml", "retrieval:\n  top_k: 40\n")

    resolved = load_config(tmp_path, env="prod", environ={})
    assert resolved.config.retrieval.top_k == 40
    assert resolved.snapshot.env == "prod"


def test_missing_config_files_are_not_an_error(tmp_path: Path) -> None:
    """A project may configure entirely through environment variables."""
    resolved = load_config(tmp_path, env="prod", environ={})
    assert resolved.config == HardpointConfig()


def test_missing_environment_overlay_is_normal(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "retrieval:\n  top_k: 3\n")
    assert load_config(tmp_path, env="staging", environ={}).config.retrieval.top_k == 3


def test_environment_name_comes_from_hardpoint_env(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "retrieval:\n  top_k: 1\n")
    write(tmp_path, "ci.yaml", "retrieval:\n  top_k: 2\n")
    resolved = load_config(tmp_path, environ={"HARDPOINT_ENV": "ci"})
    assert resolved.config.retrieval.top_k == 2
    assert resolved.snapshot.env == "ci"


def test_environment_defaults_to_dev(tmp_path: Path) -> None:
    assert load_config(tmp_path, environ={}).snapshot.env == "dev"


def test_empty_yaml_file_is_treated_as_empty(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "# nothing but a comment\n")
    assert load_config(tmp_path, environ={}).config == HardpointConfig()


def test_malformed_yaml_names_the_file_and_suggests_the_usual_cause(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "retrieval:\n  top_k: 1\n   bad_indent: 2\n")
    with pytest.raises(InvalidConfigError) as exc_info:
        load_config(tmp_path, environ={})
    assert "base.yaml" in str(exc_info.value)
    assert exc_info.value.remedy is not None
    assert "indentation" in exc_info.value.remedy


def test_yaml_whose_top_level_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "base.yaml", "- one\n- two\n")
    with pytest.raises(InvalidConfigError, match="mapping at the top level"):
        load_config(tmp_path, environ={})


def test_yaml_loading_does_not_construct_arbitrary_objects(tmp_path: Path) -> None:
    """The positional loader must stay a SafeLoader subclass.

    A config file is often written by a different team than the one running it,
    so an unsafe loader would be a remote code execution path.
    """
    write(tmp_path, "base.yaml", "retrieval: !!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(InvalidConfigError, match="not valid YAML"):
        load_config(tmp_path, environ={})


# --------------------------------------------------------------------------- #
# Schema shape                                                                #
# --------------------------------------------------------------------------- #


def config_models() -> list[type[BaseModel]]:
    return [
        obj
        for obj in vars(schema_module).values()
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


def test_only_component_blocks_may_accept_unknown_keys() -> None:
    """Every other config model forbids extras, so a typo is an error.

    ``ComponentSpec`` and ``GuardSpec`` allow them because the valid set belongs
    to the selected component; the registry enforces it against that
    component's own ``extra="forbid"`` model.
    """
    permitted = {"ComponentSpec", "GuardSpec"}
    offenders = [
        cls.__name__
        for cls in config_models()
        if cls.model_config.get("extra") != "forbid" and cls.__name__ not in permitted
    ]
    assert not offenders, f"these config models allow unknown keys: {offenders}"


def test_every_config_model_is_frozen() -> None:
    not_frozen = [cls.__name__ for cls in config_models() if not cls.model_config.get("frozen")]
    assert not not_frozen


def test_component_spec_separates_options_from_reserved_keys() -> None:
    spec = ComponentSpec(type="qdrant", url="http://h", collection="docs")  # type: ignore[call-arg]
    assert spec.options() == {"url": "http://h", "collection": "docs"}
    assert spec.policies.retry.max_attempts == 3


def test_guard_spec_separates_options_and_defaults_to_flag() -> None:
    """The injection guard flags rather than blocks, so nobody builds on a false
    sense of prevention (ARCHITECTURE.md §30).
    """
    guard = GuardSpec(type="groundedness", min_supported_ratio=0.7)  # type: ignore[call-arg]
    assert guard.action == "flag"
    assert guard.options() == {"min_supported_ratio": 0.7}


def test_plugin_discovery_is_off_by_default() -> None:
    """Import-time entry-point scanning was rejected outright (ARCHITECTURE.md §6.3)."""
    assert HardpointConfig().plugins.discover is False


def test_budgets_default_to_unbounded() -> None:
    """A default ceiling would silently truncate a legitimate workload."""
    request = HardpointConfig().budgets.request
    assert request.max_cost_usd is None
    assert request.deadline_s is None


def test_reranking_degrades_rather_than_fails_by_default() -> None:
    """Availability-preserving degradation for optional quality stages."""
    assert HardpointConfig().retrieval.rerank.on_failure == "skip"


def test_no_context_policy_defaults_to_abstain() -> None:
    assert HardpointConfig().retrieval.no_context_policy == "abstain"


def test_json_schema_can_be_generated() -> None:
    """``hardpoint config schema`` depends on this, and editors depend on that."""
    generated = HardpointConfig.model_json_schema()
    assert generated["title"] == "HardpointConfig"
    assert "retrieval" in generated["properties"]


# --------------------------------------------------------------------------- #
# Regressions                                                                 #
# --------------------------------------------------------------------------- #


def test_get_on_a_parent_path_does_not_leak_a_nested_secret() -> None:
    """Regression: ``get`` checked whether *this* path was secret.

    That left ``get("indexes.primary")`` returning the section as a raw dict
    with a live API key inside it -- a leak through the accessor documented as
    the safe one, reachable from any code that reads a whole config section.
    """
    snapshot = resolved_with_secret().snapshot

    section = snapshot.get("indexes")
    assert SECRET not in repr(section)
    assert section == {"primary": {"type": "qdrant", "api_key": REDACTED}}

    nested = snapshot.get("indexes.primary")
    assert SECRET not in repr(nested)
    assert nested == {"type": "qdrant", "api_key": REDACTED}


def test_get_on_the_root_of_a_config_is_redacted() -> None:
    """The widest possible read must be redacted too."""
    snapshot = resolved_with_secret().snapshot
    for path in ("indexes", "indexes.primary", "indexes.primary.api_key"):
        assert SECRET not in repr(snapshot.get(path)), path


def test_reveal_still_returns_whole_sections_unredacted() -> None:
    """The escape hatch must keep working, or a factory cannot build a client."""
    snapshot = resolved_with_secret().snapshot
    assert snapshot.reveal("indexes.primary")["api_key"] == SECRET  # type: ignore[index,call-overload]


def test_a_pickled_snapshot_round_trips_with_its_hash_intact() -> None:
    """Regression: ``__getstate__`` omitted ``hash``, so a revived snapshot
    reported an empty string for it.

    Worse than raising: a run manifest would have recorded ``config_hash=""``
    and nobody would have noticed until a regression could not be bisected.
    """
    snapshot = resolved_with_secret().snapshot
    revived = pickle.loads(pickle.dumps(snapshot))  # noqa: S301

    assert revived.hash == snapshot.hash
    assert revived.env == snapshot.env
    assert revived.secret_paths == snapshot.secret_paths
    assert revived.get("indexes.primary.api_key") == REDACTED
    assert revived.reveal("indexes.primary.api_key") == REDACTED, (
        "a pickled configuration has left the process; it must not carry secrets"
    )


def test_mixed_case_env_overrides_cannot_silently_replace_a_section() -> None:
    """Regression: entries were sorted by raw variable name.

    Section names are case-insensitive, so ``HARDPOINT__A__B__C`` sorted before
    ``hardpoint__a__b``; the shorter variable then overwrote the section the
    longer one had built, losing configuration without a word.
    """
    with pytest.raises(InvalidConfigError) as exc_info:
        parse_env_overrides({"HARDPOINT__A__B__C": "2", "hardpoint__A__B": "1"})

    rendered = str(exc_info.value)
    assert "hardpoint__A__B" in rendered
    assert "HARDPOINT__A__B__C" in rendered


def test_env_override_conflict_is_detected_in_either_declaration_order() -> None:
    """The conflict is symmetric, so detection must be too."""
    for environ in (
        {"HARDPOINT__A__B": "1", "HARDPOINT__A__B__C": "2"},
        {"HARDPOINT__A__B__C": "2", "HARDPOINT__A__B": "1"},
        {"hardpoint__a__b": "1", "HARDPOINT__A__B__C": "2"},
    ):
        with pytest.raises(InvalidConfigError):
            parse_env_overrides(environ)


def test_case_differences_alone_are_still_accepted() -> None:
    """Case-insensitivity is the documented behaviour; only conflicts are errors."""
    assert parse_env_overrides({"hardpoint__RETRIEVAL__top_k": "9"}) == {"retrieval": {"top_k": 9}}


def test_replacing_a_section_with_a_scalar_purges_stale_origins() -> None:
    """Regression: ``config show`` would annotate keys that no longer exist.

    An overlay that replaces a whole section leaves the previous layer's origin
    entries pointing at paths the resolved config does not contain.
    """
    resolved = resolve(
        base={"retrieval": {"rerank": {"enabled": True, "top_k": 4}}},
        env_file={"retrieval": {"rerank": {"enabled": False}}},
        environ={},
    )
    live = set(resolved.snapshot.paths())
    recorded = set(resolved.snapshot.origins)
    assert recorded <= live, f"origins describe paths that do not exist: {recorded - live}"
