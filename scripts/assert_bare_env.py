"""Assert that the running interpreter's environment is genuinely bare.

Reads the extras declared in ``pyproject.toml`` and fails if any distribution
belonging to an extra is installed. Without this check the bare-environment
import job would silently pass in an environment that happens to have provider
SDKs available, which is precisely the situation it exists to rule out.

Run it *with* the bare interpreter::

    .venv-bare/bin/python scripts/assert_bare_env.py

Standard library only, so it needs nothing installed beyond the package itself.
"""

from __future__ import annotations

import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# PEP 508: the distribution name runs up to the first version specifier,
# extras bracket, marker semicolon or whitespace.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def distribution_name(requirement: str) -> str | None:
    """Extract the distribution name from a PEP 508 requirement string."""
    match = _NAME.match(requirement.strip())
    return match.group(0) if match else None


def optional_distributions() -> dict[str, list[str]]:
    """Map each declared extra to the distribution names it pulls in."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"].get("optional-dependencies", {})
    result: dict[str, list[str]] = {}
    for extra, requirements in extras.items():
        names = [n for n in (distribution_name(r) for r in requirements) if n]
        result[extra] = names
    return result


def main() -> int:
    """Fail if any extra's distribution is importable in this environment."""
    installed: list[str] = []
    checked = 0

    for extra, names in sorted(optional_distributions().items()):
        for name in names:
            checked += 1
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            installed.append(f"{name}=={version}  (extra: {extra})")

    try:
        hardpoint_version = metadata.version("hardpoint")
    except metadata.PackageNotFoundError:
        sys.stderr.write("hardpoint is not installed in this environment.\n")
        return 2

    sys.stdout.write(
        f"hardpoint=={hardpoint_version} on {sys.version.split()[0]}\n"
        f"checked {checked} optional distributions across the declared extras\n"
    )

    if installed:
        sys.stderr.write(
            "\nThis environment is not bare. The following optional distributions "
            "are installed, so a module-import-time dependency on them would not "
            "be detected:\n"
        )
        for line in installed:
            sys.stderr.write(f"  - {line}\n")
        return 1

    sys.stdout.write("environment is bare: no optional distribution installed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
