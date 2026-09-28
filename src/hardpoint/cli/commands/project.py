"""What every command needs: the project's environment and configuration.

The CLI runs from the project directory -- ``--project-dir`` changes to it
first -- so every relative path in configuration means the same thing to the
CLI, the service and the eval runner.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hardpoint.core.config.loader import ResolvedConfig, load_config

__all__ = ["CliState", "load_project_config", "project_environ", "read_dotenv"]


@dataclass(frozen=True)
class CliState:
    """Options given before the command name.

    Args:
        env: Configuration environment, overriding ``HARDPOINT_ENV``.
        config_dir: Directory holding ``base.yaml`` and the overlays.
    """

    env: str | None = None
    config_dir: str = "config"


def read_dotenv(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file, if there is one.

    Deliberately small: comments, blank lines, an optional ``export`` prefix
    and matching quotes. Not a shell. Values are never logged; they reach the
    configuration only through ``${env:...}``, which marks them secret.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":  # noqa: PLR2004
            value = value[1:-1]
        values[key.strip()] = value
    return values


def project_environ() -> dict[str, str]:
    """The process environment over the project's ``.env``.

    The real environment wins, so a CI secret or a shell export overrides a
    developer's local file rather than the other way round.
    """
    return {**read_dotenv(Path(".env")), **os.environ}


def load_project_config(
    state: CliState, *, environ: Mapping[str, str] | None = None
) -> ResolvedConfig:
    """Load the project's configuration with the ``.env`` applied."""
    environment = dict(environ) if environ is not None else project_environ()
    return load_config(state.config_dir, env=state.env, environ=environment)
