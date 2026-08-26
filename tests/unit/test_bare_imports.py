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

    Asserted in a subprocess with ``socket`` and ``os.environ`` instrumented
    before ``hardpoint`` is imported, so nothing the test session already did can
    mask a violation.
    """
    program = textwrap.dedent(
        """
        import builtins, json, os, socket, sys

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

        class WatchedEnviron(dict):
            def __getitem__(self, key):
                violations.append(f"os.environ[{key!r}]")
                return super().__getitem__(key)
            def get(self, key, default=None):
                violations.append(f"os.environ.get({key!r})")
                return super().get(key, default)
        os.environ = WatchedEnviron(os.environ)

        import importlib, pkgutil
        package = importlib.import_module("hardpoint")
        for info in pkgutil.walk_packages(package.__path__, prefix="hardpoint."):
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
