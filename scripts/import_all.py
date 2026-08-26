"""Import every ``hardpoint`` module and report failures.

This is the executable half of the bare-environment guarantee in
INSTRUCTIONS.md §3: *no module may import an optional third-party dependency at
module import time.* Run with an interpreter whose environment has the base
dependencies and nothing else; any module that reaches for an optional SDK at
import time raises here.

Uses only the standard library so it can run inside a bare virtual environment
that has no test tooling installed.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback


def iter_module_names() -> list[str]:
    """Return every importable module name under the ``hardpoint`` package."""
    package = importlib.import_module("hardpoint")
    names = ["hardpoint"]
    names.extend(info.name for info in pkgutil.walk_packages(package.__path__, prefix="hardpoint."))
    return sorted(names)


def main() -> int:
    """Import everything, printing one line per module. Returns an exit code."""
    failures: list[tuple[str, BaseException]] = []
    names = iter_module_names()

    for name in names:
        try:
            importlib.import_module(name)
        except BaseException as exc:  # a reporting tool must survive every module
            failures.append((name, exc))
            sys.stdout.write(f"FAIL  {name}\n")
        else:
            sys.stdout.write(f"ok    {name}\n")

    sys.stdout.write(f"\n{len(names) - len(failures)}/{len(names)} modules imported\n")

    if failures:
        sys.stderr.write("\nModules that could not be imported with base dependencies only:\n")
        for name, exc in failures:
            sys.stderr.write(f"\n--- {name} ---\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
