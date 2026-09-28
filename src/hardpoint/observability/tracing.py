"""Tracers: the no-op default, a collecting tracer, a console tracer, and redaction.

ARCHITECTURE.md §21 and ADR-010. The ``Tracer`` port lives in ``core.ports``.

**The default is a no-op tracer, never ``None``.** If a step had to write
``if ctx.tracer is not None`` before opening a span, half of them eventually
would not, and tracing would be absent exactly where somebody forgot.

## The span taxonomy

:data:`SPAN_NAMES` is the whole vocabulary. Attributes follow the OpenTelemetry
GenAI semantic conventions where they exist (``gen_ai.*``) and a documented
``hardpoint.*`` namespace where they do not.

## Redaction

Traces can hold user questions and document text. :class:`RedactingTracer`
replaces the value of any attribute whose key is, or sits under, a configured
path -- ``messages.content`` redacts ``messages.content`` and
``messages.content.0`` -- before the attribute reaches the backend. It wraps
any tracer, so redaction is applied once, here, rather than in every exporter.

## Nesting across async boundaries

Parentage is tracked with a :class:`~contextvars.ContextVar`, which anyio and
asyncio copy into each task they start. Two retrievers running concurrently
under one pipeline span therefore both nest under that span, rather than under
whichever sibling happened to open last.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from typing import Final, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field

from hardpoint.core.ports import Span, Tracer
from hardpoint.core.types import JsonValue

__all__ = [
    "REDACTED_VALUE",
    "SPAN_NAMES",
    "CollectedSpan",
    "CollectingTracer",
    "CollectingTracerConfig",
    "ConsoleTracer",
    "ConsoleTracerConfig",
    "NoOpSpan",
    "NoOpTracer",
    "NoOpTracerConfig",
    "RedactingTracer",
    "build_collecting",
    "build_console",
    "build_noop",
    "render_tree",
]

SPAN_NAMES: Final = frozenset(
    {
        "hardpoint.pipeline",
        "hardpoint.step",
        "hardpoint.llm",
        "hardpoint.embed",
        "hardpoint.index.query",
        "hardpoint.index.upsert",
        "hardpoint.rerank",
        "hardpoint.guard",
        "hardpoint.loop.iteration",
        "hardpoint.tool",
        "hardpoint.retry",
        "hardpoint.fallback",
    }
)
"""The span taxonomy of ARCHITECTURE.md §21, plus the two policy spans."""

REDACTED_VALUE: Final = "[redacted]"


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


# --------------------------------------------------------------------------- #
# Collecting                                                                  #
# --------------------------------------------------------------------------- #


class CollectedSpan:
    """A finished (or open) span held in memory, with its children."""

    __slots__ = (
        "attributes",
        "children",
        "ended_at",
        "events",
        "exceptions",
        "name",
        "parent",
        "started_at",
        "status",
        "trace",
    )

    def __init__(
        self, name: str, attributes: dict[str, JsonValue], trace: str, parent: CollectedSpan | None
    ) -> None:
        self.name = name
        self.attributes = attributes
        self.trace = trace
        self.parent = parent
        self.children: list[CollectedSpan] = []
        self.events: list[tuple[str, dict[str, JsonValue]]] = []
        self.exceptions: list[str] = []
        self.status: tuple[str, str] = ("unset", "")
        self.started_at = time.perf_counter()
        self.ended_at: float | None = None

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Attach an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Record a point-in-time event."""
        self.events.append((name, dict(attributes or {})))

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception by type and message."""
        self.exceptions.append(f"{type(exc).__name__}: {exc}")

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Mark the outcome."""
        self.status = (status, description)

    @property
    def trace_id(self) -> str | None:
        """The trace this span belongs to."""
        return self.trace

    @property
    def duration_ms(self) -> float | None:
        """Wall time, once the span has ended."""
        return None if self.ended_at is None else (self.ended_at - self.started_at) * 1000.0

    def walk(self) -> list[CollectedSpan]:
        """This span and every descendant, depth first."""
        found = [self]
        for child in self.children:
            found.extend(child.walk())
        return found

    def __repr__(self) -> str:
        """Render the name and child count."""
        return f"CollectedSpan({self.name!r}, children={len(self.children)})"


class CollectingTracer:
    """A ``Tracer`` that keeps every span in memory, nested correctly.

    Production-safe, unlike ``testing.RecordingTracer``: bounded by
    ``max_traces``, so a long-running service using it for ``/debug`` output
    cannot grow without limit. What ``hardpoint ask --explain`` and the service
    tests read traces from.

    Args:
        max_traces: How many root spans to keep. Oldest are dropped first.
    """

    def __init__(self, *, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self.roots: list[CollectedSpan] = []
        self._current: ContextVar[CollectedSpan | None] = ContextVar(
            f"hardpoint_collecting_span_{id(self)}", default=None
        )

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[CollectedSpan]:
        """Open a span nested under the current one in this task."""

        @asynccontextmanager
        async def scope() -> AsyncIterator[CollectedSpan]:
            parent = self._current.get()
            trace = parent.trace if parent else uuid.uuid4().hex
            span = CollectedSpan(name, dict(attrs), trace, parent)
            if parent is None:
                self.roots.append(span)
                del self.roots[: max(0, len(self.roots) - self.max_traces)]
            else:
                parent.children.append(span)
            token = self._current.set(span)
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                if span.status[0] == "unset":
                    span.set_status("error", type(exc).__name__)
                raise
            finally:
                span.ended_at = time.perf_counter()
                self._current.reset(token)

        return scope()

    def spans(self) -> list[CollectedSpan]:
        """Every span of every kept trace, depth first."""
        return [span for root in self.roots for span in root.walk()]

    def find(self, name: str) -> list[CollectedSpan]:
        """Every kept span with a given name."""
        return [span for span in self.spans() if span.name == name]

    def __repr__(self) -> str:
        """Render how many traces are held."""
        return f"CollectingTracer(traces={len(self.roots)})"


class ConsoleTracer(CollectingTracer):
    """Prints each finished trace as an indented tree.

    ARCHITECTURE.md §27.1's "tracing to console" for a laptop prototype. Writes
    when a root span ends, so concurrent requests print whole trees rather than
    interleaved lines.

    Args:
        stream: Where to write. Defaults to standard error, so traces never mix
            with a command's output.
        max_traces: Passed to :class:`CollectingTracer`.
    """

    def __init__(self, stream: TextIO | None = None, *, max_traces: int = 100) -> None:
        super().__init__(max_traces=max_traces)
        self._stream = stream

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[CollectedSpan]:
        """Open a span; print its tree when it is a root and it ends."""
        inner = super().span(name, **attrs)

        @asynccontextmanager
        async def scope() -> AsyncIterator[CollectedSpan]:
            async with inner as span:
                yield span
            if span.parent is None:
                (self._stream or sys.stderr).write(render_tree(span) + "\n")

        return scope()


def render_tree(root: CollectedSpan) -> str:
    """Render a trace as an indented tree with durations and key attributes."""
    lines: list[str] = []

    def visit(span: CollectedSpan, depth: int) -> None:
        duration = span.duration_ms
        label = span.attributes.get("step.name") or span.attributes.get("gen_ai.request.model")
        status = "" if span.status[0] in ("ok", "unset") else f"  [{span.status[0]}]"
        timing = f"{duration:8.1f} ms" if duration is not None else "    open   "
        lines.append(f"{timing}  {'  ' * depth}{span.name}{f' {label}' if label else ''}{status}")
        for child in span.children:
            visit(child, depth + 1)

    visit(root, 0)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Redaction                                                                   #
# --------------------------------------------------------------------------- #


def _redacts(paths: Sequence[str]) -> Callable[[str], bool]:
    prefixes = tuple(f"{path}." for path in paths)
    exact = frozenset(paths)
    return lambda key: key in exact or key.startswith(prefixes)


class _RedactingSpan:
    """Wraps a span, replacing redacted attributes before they are written."""

    __slots__ = ("_inner", "_redacts")

    def __init__(self, inner: Span, redacts: Callable[[str], bool]) -> None:
        self._inner = inner
        self._redacts = redacts

    def _clean(self, attributes: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return {k: REDACTED_VALUE if self._redacts(k) else v for k, v in attributes.items()}

    def set_attribute(self, key: str, value: JsonValue) -> None:
        """Write the attribute, or the redaction marker in its place."""
        self._inner.set_attribute(key, REDACTED_VALUE if self._redacts(key) else value)

    def add_event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        """Write the event with its attributes redacted."""
        self._inner.add_event(name, self._clean(attributes or {}))

    def record_exception(self, exc: BaseException) -> None:
        """Pass the exception through. Errors carry no secrets by construction."""
        self._inner.record_exception(exc)

    def set_status(self, status: Literal["ok", "error"], description: str = "") -> None:
        """Pass the status through."""
        self._inner.set_status(status, description)

    @property
    def trace_id(self) -> str | None:
        """The inner span's trace."""
        return self._inner.trace_id


class RedactingTracer:
    """Applies ``observability.redact`` paths to every attribute on every span.

    Args:
        inner: The tracer that receives the redacted spans.
        paths: Attribute keys to redact, and everything under them.
    """

    def __init__(self, inner: Tracer, paths: Sequence[str]) -> None:
        self.inner = inner
        self.paths = tuple(paths)
        self._redacts = _redacts(self.paths)

    def span(self, name: str, **attrs: JsonValue) -> AbstractAsyncContextManager[Span]:
        """Open a span on the inner tracer, redacting as it goes."""
        clean = {k: REDACTED_VALUE if self._redacts(k) else v for k, v in attrs.items()}
        inner = self.inner.span(name, **clean)

        @asynccontextmanager
        async def scope() -> AsyncIterator[Span]:
            async with inner as span:
                yield _RedactingSpan(span, self._redacts)

        return scope()

    def __repr__(self) -> str:
        """Render the wrapped tracer and the paths."""
        return f"RedactingTracer({self.inner!r}, paths={list(self.paths)})"


# --------------------------------------------------------------------------- #
# Registry entries                                                            #
# --------------------------------------------------------------------------- #


class NoOpTracerConfig(BaseModel):
    """``observability.tracer: {type: noop}``. The default."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConsoleTracerConfig(BaseModel):
    """``observability.tracer: {type: console}``: trace trees on standard error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_traces: int = Field(default=100, ge=1)


class CollectingTracerConfig(BaseModel):
    """``observability.tracer: {type: collecting}``: the last traces, kept in memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_traces: int = Field(default=1000, ge=1)


def build_collecting(config: CollectingTracerConfig) -> CollectingTracer:
    """Registry factory for ``type: collecting``."""
    return CollectingTracer(max_traces=config.max_traces)


def build_noop(config: NoOpTracerConfig) -> NoOpTracer:
    """Registry factory for ``type: noop``."""
    return NoOpTracer()


def build_console(config: ConsoleTracerConfig) -> ConsoleTracer:
    """Registry factory for ``type: console``."""
    return ConsoleTracer(max_traces=config.max_traces)
