"""``hardpoint init``: generate a project from a template.

Templates are plain files under ``hardpoint/templates/<name>/``, shipped in the
wheel. Rendering is two substitutions and a rename, not a template language:

- ``{{ project_name }}`` and ``{{ hardpoint_version }}`` in file contents;
- a leading ``dot_`` in a file name becomes ``.`` (``dot_gitignore`` ->
  ``.gitignore``), because dotfiles inside a package are easy to lose in a
  build and awkward to see in a tree.

**The generator never rewrites user files** (ARCHITECTURE.md §24). Generating
into a non-empty directory is refused; ``--diff`` shows what a fresh template
would change instead, and the user decides.
"""

import difflib
from collections.abc import Iterator
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import hardpoint
from hardpoint.core.errors import ConfigError

__all__ = ["TEMPLATES", "diff_project", "render_template_files", "write_project"]

TEMPLATES = ("rag-minimal", "rag-service")
"""Every template this release ships."""

_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".jsonl", ".json", ".example", ""}


def _template_root(name: str) -> Traversable:
    if name not in TEMPLATES:
        raise ConfigError(
            f"No template named {name!r}.",
            remedy=f"Available templates: {', '.join(TEMPLATES)}.",
        )
    return files("hardpoint.templates").joinpath(name)


def _walk(node: Traversable, prefix: str = "") -> Iterator[tuple[str, Traversable]]:
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        if child.name == "__pycache__":
            continue
        relative = f"{prefix}{child.name}"
        if child.is_dir():
            yield from _walk(child, f"{relative}/")
        else:
            yield relative, child


def _target_name(relative: str) -> str:
    return "/".join(
        f".{part.removeprefix('dot_')}" if part.startswith("dot_") else part
        for part in relative.split("/")
    )


def render_template_files(template: str, project_name: str) -> dict[str, bytes]:
    """Render a template in memory.

    Args:
        template: The template's name.
        project_name: Substituted for ``{{ project_name }}``.

    Returns:
        Relative path to file content, for every file the project will hold.

    Raises:
        ConfigError: For an unknown template.
    """
    rendered: dict[str, bytes] = {}
    for relative, source in _walk(_template_root(template)):
        data = source.read_bytes()
        if Path(relative).suffix in _TEXT_SUFFIXES:
            # LF everywhere, however the template was checked out: a generated
            # project must not depend on the machine that built the wheel.
            text = data.decode("utf-8").replace("\r\n", "\n")
            text = text.replace("{{ project_name }}", project_name)
            text = text.replace("{{ hardpoint_version }}", hardpoint.__version__)
            data = text.encode("utf-8")
        rendered[_target_name(relative)] = data
    return rendered


def write_project(destination: Path, template: str) -> list[Path]:
    """Generate a project into a new or empty directory.

    Raises:
        ConfigError: If the directory exists and is not empty. The generator
            never overwrites a user's files.
    """
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(
            f"{destination} already exists and is not empty.",
            remedy=(
                f"Choose a new directory, or run `hardpoint init {destination} --template "
                f"{template} --diff` to see what a fresh template would change."
            ),
        )
    written: list[Path] = []
    for relative, data in render_template_files(template, destination.name).items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(path)
    return written


def diff_project(destination: Path, template: str) -> str:
    """Show how an existing project differs from a fresh template. Writes nothing."""
    chunks: list[str] = []
    for relative, data in render_template_files(template, destination.name).items():
        current = destination / relative
        fresh = data.decode("utf-8", errors="replace").splitlines(keepends=True)
        existing = (
            current.read_bytes()
            .decode("utf-8", errors="replace")
            .replace("\r\n", "\n")
            .splitlines(keepends=True)
            if current.is_file()
            else []
        )
        chunks.extend(
            difflib.unified_diff(existing, fresh, f"project/{relative}", f"template/{relative}")
        )
    return "".join(chunks)
