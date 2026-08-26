"""Bare-environment import guarantees.

INSTRUCTIONS.md §3 **[LOCKED]**: no module may import an optional third-party
dependency at module import time, and importing every ``hardpoint.*`` module in
an environment with only the base dependencies must raise no ``ImportError``.

ARCHITECTURE.md §5.3: importing ``hardpoint`` starts no clients, reads no
environment, opens no sockets and registers no plugins.

The CI ``bare-environment`` job proves the first property the strong way, by
installing the package with no extras at all. These tests prove it in the
ordinary development environment as well, and additionally catch the case an
empty environment cannot catch: a module-import-time dependency that happens to
be satisfied because someone installed an extra locally.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import textwrap

import pytest

# Import names, not distribution names: what would appear in sys.modules if a
# module reached for one of these at import time. Derived from the extras
# declared in pyproject.toml (ARCHITECTURE.md §16.2).
OPTIONAL_TOP_LEVEL_MODULES = frozenset(
    {
        "anthropic",
        "boto3",
        "bs4",
        "chromadb",
        "cohere",
        "docx",
        "fastapi",
        "google",
        "ollama",
        "openai",
        "opensearchpy",
        "opentelemetry",
        "pgvector",
        "pinecone",
        "pypdf",
        "qdrant_client",
        "redis",
        "uvicorn",
        "weaviate",
    }
)


def all_module_names() -> list[str]:
    """Every importable module name under the ``hardpoint`` package."""
    package = importlib.import_module("hardpoint")
    names = ["hardpoint"]
    names.extend(info.name for info in pkgutil.walk_packages(package.__path__, prefix="hardpoint."))
    return sorted(names)


@pytest.mark.parametrize("module_name", all_module_names())
def test_module_imports_without_optional_dependencies(module_name: str) -> None:
    """Every hardpoint module imports cleanly with base dependencies only."""
    importlib.import_module(module_name)


def test_no_optional_dependency_is_imported_at_module_import_time() -> None:
    """Importing the whole package must not pull an optional SDK into sys.modules.

    Run in a subprocess so the result does not depend on what other tests, or the
    test runner itself, have already imported.
    """
    program = textwrap.dedent(
        f"""
        import importlib, json, pkgutil, sys

        optional = {sorted(OPTIONAL_TOP_LEVEL_MODULES)!r}
        before = set(sys.modules)

        package = importlib.import_module("hardpoint")
        for info in pkgutil.walk_packages(package.__path__, prefix="hardpoint."):
            importlib.import_module(info.name)

        leaked = sorted(
            name for name in optional
            if name in sys.modules and name not in before
        )
        print(json.dumps(leaked))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    leaked = result.stdout.strip().splitlines()[-1]
    assert leaked == "[]", (
        "These optional third-party modules were imported at hardpoint module "
        f"import time, violating INSTRUCTIONS.md 3: {leaked}"
    )


def test_importing_hardpoint_has_no_side_effects() -> None:
    """Importing the package opens no sockets and reads no environment.

    ARCHITECTURE.md §5.3. Run in a subprocess with ``socket`` and ``os.environ``
    instrumented *before* ``hardpoint`` is imported, so nothing the test session
    already did can mask a violation.

    Environment reads are attributed to their immediate caller. CPython's
    ``site`` module and pydantic's own ``PYDANTIC_DISABLE_PLUGINS`` kill-switch
    both read the environment during any import, and neither is hardpoint
    reading configuration. Attribution keeps the assertion about the rule that
    was actually written down: a ``hardpoint`` module must not reach for an API
    key, a region, or an endpoint at import time. Any socket at all is a
    violation regardless of who opened it, so sockets are not attributed.
    """
    program = textwrap.dedent(
        """
        import json, os, socket, sys
        from pathlib import Path

        violations = []

        real_connect = socket.socket.connect
        def guarded_connect(self, address):
            violations.append(f"socket.connect({address!r})")
            return real_connect(self, address)
        socket.socket.connect = guarded_connect

        real_create = socket.create_connection
        def guarded_create(address, *a, **k):
            violations.append(f"socket.create_connection({address!r})")
            return real_create(address, *a, **k)
        socket.create_connection = guarded_create

        import hardpoint
        package_dir = str(Path(hardpoint.__file__).resolve().parent)

        def caller_is_hardpoint():
            frame = sys._getframe(2)
            filename = frame.f_code.co_filename
            try:
                resolved = str(Path(filename).resolve())
            except (OSError, ValueError):
                return False
            return resolved.startswith(package_dir)

        class WatchedEnviron(dict):
            def __getitem__(self, key):
                if caller_is_hardpoint():
                    violations.append(f"os.environ[{key!r}]")
                return super().__getitem__(key)
            def get(self, key, default=None):
                if caller_is_hardpoint():
                    violations.append(f"os.environ.get({key!r})")
                return super().get(key, default)
        os.environ = WatchedEnviron(os.environ)

        import importlib, pkgutil
        for info in pkgutil.walk_packages(hardpoint.__path__, prefix="hardpoint."):
            importlib.import_module(info.name)

        print(json.dumps(violations))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    violations = result.stdout.strip().splitlines()[-1]
    assert violations == "[]", (
        f"Importing hardpoint had side effects, violating ARCHITECTURE.md 5.3: {violations}"
    )


def test_the_side_effect_detector_actually_detects() -> None:
    """The attribution in the previous test must not make it unfalsifiable.

    A test that can only pass is worse than no test. This plants a module inside
    the installed package that reads the environment at import time, and asserts
    the detector reports it.
    """
    program = textwrap.dedent(
        """
        import json, os, sys
        from pathlib import Path

        import hardpoint
        package_dir = str(Path(hardpoint.__file__).resolve().parent)
        planted = Path(package_dir) / "_side_effect_probe.py"
        planted.write_text("import os\\nos.environ.get('HARDPOINT_PROBE')\\n")

        violations = []

        def caller_is_hardpoint():
            frame = sys._getframe(2)
            try:
                resolved = str(Path(frame.f_code.co_filename).resolve())
            except (OSError, ValueError):
                return False
            return resolved.startswith(package_dir)

        class WatchedEnviron(dict):
            def get(self, key, default=None):
                if caller_is_hardpoint():
                    violations.append(f"os.environ.get({key!r})")
                return super().get(key, default)
        os.environ = WatchedEnviron(os.environ)

        try:
            import importlib
            importlib.import_module("hardpoint._side_effect_probe")
        finally:
            planted.unlink()

        print(json.dumps(violations))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = result.stdout.strip().splitlines()[-1]
    assert "HARDPOINT_PROBE" in reported, (
        "The side-effect detector failed to notice a hardpoint module reading the "
        f"environment at import time, so it cannot be trusted. Got: {reported}"
    )
