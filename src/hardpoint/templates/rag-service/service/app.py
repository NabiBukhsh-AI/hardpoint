"""The HTTP service. This file is yours: change the routes, the payloads, the auth.

Routes (ARCHITECTURE.md §25):

- ``POST /query``          -> the whole ``Answer`` as JSON.
- ``POST /query/stream``   -> server-sent events: ``delta`` events with text as it
  is generated, then one ``done`` event with citations, usage and degradations.
- ``GET /healthz``         -> liveness: the process is up.
- ``GET /readyz``          -> readiness: every index is reachable and the
  embedding provider accepts our credentials.
- ``GET /metrics``         -> Prometheus text, when ``observability.metrics`` is
  ``memory``.

Every request is answered by the same pipeline factory ``hardpoint ask`` and the
eval suite use (``pipelines/rag.py``), through ``answer_query`` -- so the service
behaves exactly as what was evaluated.

Shutdown is graceful: uvicorn stops accepting connections, lets in-flight
requests finish within ``--timeout-graceful-shutdown``, and only then does the
lifespan below close the providers' connection pools and the index.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from hardpoint.core.config import load_config
from hardpoint.core.errors import (
    BudgetExceeded,
    ConfigError,
    GuardViolation,
    HardpointError,
    InvalidRequestError,
)
from hardpoint.core.models import Answer
from hardpoint.observability.metrics import InMemoryMetricSink
from hardpoint.runtime import answer_query, build_resources, load_pipeline, new_run_id

LOGGER = logging.getLogger("service")
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
READINESS_TTL_S = 30.0


class Query(BaseModel):
    """What a client sends."""

    question: str = Field(min_length=1, max_length=4000)


def request_id(request: Request) -> str:
    """The caller's ``x-request-id`` when it is safe to log, otherwise a fresh one.

    It becomes the run id, so a support ticket quoting it finds the trace.
    """
    supplied = request.headers.get("x-request-id", "")
    return supplied if REQUEST_ID.match(supplied) else new_run_id()


def payload(answer: Answer) -> dict[str, Any]:
    """The JSON a client receives: everything but the (large) context bundle."""
    return answer.model_dump(mode="json", exclude={"context"})


def failure(exc: HardpointError, run_id: str) -> JSONResponse:
    """Map a library error to a status code, without leaking internals."""
    if isinstance(exc, GuardViolation | InvalidRequestError):
        status = 422
    elif isinstance(exc, BudgetExceeded) or (exc.retryable and not isinstance(exc, ConfigError)):
        status = 503
    else:
        status = 500
    LOGGER.warning("request failed: %s", exc.message, extra={"run_id": run_id, "code": exc.code})
    body = {"error": exc.code, "message": exc.message, "run_id": run_id}
    return JSONResponse(body, status_code=status, headers={"x-request-id": run_id})


def create_app(
    config_dir: str = "config", *, env: str | None = None, overrides: dict[str, Any] | None = None
) -> FastAPI:
    """Build the app. ``hardpoint serve`` calls this; so do the tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = await build_resources(load_config(config_dir, env=env, overrides=overrides))
        app.state.resources = resources
        app.state.pipeline = load_pipeline(resources)
        app.state.ready_at = 0.0
        try:
            yield
        finally:
            await resources.aclose()

    app = FastAPI(title="{{ project_name }}", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        resources = request.app.state.resources
        problems: list[str] = []
        for name, index in resources.indexes.items():
            try:
                await index.describe()
            except HardpointError as exc:
                problems.append(f"index {name}: {exc.message}")
        # One tiny embedding proves the credentials, cached so a probe every
        # few seconds costs nothing after the first.
        if not problems and time.monotonic() - request.app.state.ready_at > READINESS_TTL_S:
            try:
                await resources.embedder.embed(["ready"], "query", resources.run_context())
                request.app.state.ready_at = time.monotonic()
            except HardpointError as exc:
                problems.append(f"embeddings: {exc.message}")
        if problems:
            return JSONResponse({"status": "unavailable", "problems": problems}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/metrics")
    async def metrics(request: Request) -> PlainTextResponse:
        sink = request.app.state.resources.metrics
        if not isinstance(sink, InMemoryMetricSink):
            return PlainTextResponse("metrics are exported elsewhere\n", status_code=404)
        return PlainTextResponse(sink.render(), media_type="text/plain; version=0.0.4")

    @app.post("/query")
    async def query(body: Query, request: Request) -> JSONResponse:
        run_id = request_id(request)
        state = request.app.state
        try:
            answer = await answer_query(state.pipeline, body.question, state.resources, run_id=run_id)
        except HardpointError as exc:
            return failure(exc, run_id)
        return JSONResponse(payload(answer), headers={"x-request-id": run_id})

    @app.post("/query/stream")
    async def query_stream(body: Query, request: Request) -> StreamingResponse:
        run_id = request_id(request)
        state = request.app.state
        send, receive = anyio.create_memory_object_stream[tuple[str, Any]](max_buffer_size=64)

        async def produce() -> None:
            async with send:

                async def on_delta(text: str) -> None:
                    await send.send(("delta", {"text": text}))

                try:
                    answer = await answer_query(
                        state.pipeline, body.question, state.resources, run_id=run_id, on_delta=on_delta
                    )
                except HardpointError as exc:
                    await send.send(("error", {"error": exc.code, "message": exc.message}))
                    return
                await send.send(("done", payload(answer)))

        async def events() -> AsyncIterator[str]:
            # If the client disconnects, the response is cancelled, which cancels
            # this task group and with it the pipeline run and its provider call.
            async with anyio.create_task_group() as group:
                group.start_soon(produce)
                async with receive:
                    async for kind, data in receive:
                        yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"x-request-id": run_id, "cache-control": "no-cache"},
        )

    return app
