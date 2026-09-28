"""Generate-and-run tests: every template must produce a project that works.

ARCHITECTURE.md §30 names the risk: the generator's output diverges from the
library as both evolve. So each template is generated into a temporary
directory and its *own* test suite is run there, in a separate interpreter, the
way its owner would run it. The CI ``template-smoke`` job goes further and pip
installs the generated project into a clean environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import hardpoint
from hardpoint.cli.commands.init import TEMPLATES, write_project


@pytest.mark.parametrize("template", TEMPLATES)
def test_the_generated_project_passes_its_own_tests(template: str, tmp_path: Path) -> None:
    project = tmp_path / "generated"
    write_project(project, template)

    environment = {k: v for k, v in os.environ.items() if not k.startswith("HARDPOINT")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("template", TEMPLATES)
def test_the_generated_project_is_installable(template: str, tmp_path: Path) -> None:
    """The metadata a pip install reads: a name, and this release as a floor."""
    project = tmp_path / "support-bot"
    write_project(project, template)

    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "support-bot"
    (requirement,) = [d for d in metadata["project"]["dependencies"] if d.startswith("hardpoint")]
    assert requirement.endswith(f">={hardpoint.__version__}")
    for package in metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]:
        assert (project / package / "__init__.py").is_file()
