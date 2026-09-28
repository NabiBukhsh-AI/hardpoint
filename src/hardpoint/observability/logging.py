"""Structured logging that carries ``run_id`` and ``trace_id`` (INSTRUCTIONS.md §12.4).

The library logs through ``logging.getLogger("hardpoint.<module>")`` and nothing
else. It never calls ``logging.basicConfig``, never adds a handler and never
touches the root logger (INSTRUCTIONS.md §13.13): how records are formatted and
where they go is the application's decision.

What the library supplies is the *content*: :func:`run_logger` returns a logger
adapter that stamps every record with the run and trace ids, and
:class:`JsonFormatter` is a formatter a generated project installs to write one
JSON object per line with those ids on every record.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any

__all__ = ["JsonFormatter", "RunLogger", "run_logger"]

_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


class RunLogger(logging.LoggerAdapter[logging.Logger]):
    """A logger adapter adding ``run_id`` and ``trace_id`` to every record."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Merge the run's ids into the record's ``extra``."""
        extra: dict[str, Any] = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


def run_logger(name: str, *, run_id: str, trace_id: str | None = None) -> RunLogger:
    """Return a module logger that stamps records with a run's identifiers.

    Args:
        name: The module, for example ``"hardpoint.runtime.pipeline"``.
        run_id: The run.
        trace_id: The trace, when tracing is on.
    """
    return RunLogger(logging.getLogger(name), {"run_id": run_id, "trace_id": trace_id})


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with any ``extra`` fields -- ids included.

    For the generated project to install on its own handler::

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(handler)

    Args:
        static: Fields added to every record, for example the service name.
    """

    def __init__(self, static: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.static = dict(static or {})

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as JSON."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self.static,
        }
        payload.update({key: value for key, value in vars(record).items() if key not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)
