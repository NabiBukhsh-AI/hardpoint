"""The default tracer: one that does nothing, correctly.

ARCHITECTURE.md §21 and ADR-010. The ``Tracer`` port lives in ``core.ports``;
this module supplies the implementation used when no tracing backend is
configured.

**The default is a no-op tracer, never ``None``.** That is the whole reason this
exists: if a step had to write ``if ctx.tracer is not None`` before opening a
span, half of them eventually would not, and tracing would be absent exactly
where somebody forgot. A tracer that discards everything costs one object
allocation per span and removes the question.

The span taxonomy, attribute redaction and the OpenTelemetry adapter are M2
(INSTRUCTIONS.md §7). What is here is the minimum that lets a ``RunContext`` be
constructed outside a test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Literal

from hardpoint.core.types import JsonValue

__all__ = ["NoOpSpan", "NoOpTracer"]


class NoOpSpan:
    """A span that discards everything written to it.

    Every method is a no-op rather than raising, because a step must be able to
    set attributes and record events unconditionally.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Discard an attribute."""

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Discard an event."""

    def record_exception(self, exc: BaseException) -> None:
        """Discard an exception."""

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Discard a status."""

    @property
    def trace_id(self) -> str | None:
        """Return ``None``: there is no trace to correlate against."""
        return None

    def __repr__(self) -> str:
        """Render the span name."""
        return f"NoOpSpan({self.name!r})"


class NoOpTracer:
    """A ``Tracer`` that records nothing.

    The default on every ``RunContext`` that was not given a real tracer.
    Stateless and safe to share between runs and between threads.
    """

    __slots__ = ()

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[NoOpSpan]:
        """Open a span that discards everything."""

        @asynccontextmanager
        async def scope() -> AsyncIterator[NoOpSpan]:
            yield NoOpSpan(name)

        return scope()

    def __repr__(self) -> str:
        """Render the class name; there is no state to show."""
        return "NoOpTracer()"
