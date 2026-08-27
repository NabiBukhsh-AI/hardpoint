"""A loopback OpenAI-compatible server, so adapters get real transport coverage.

The alternative to this is mocking ``httpx``, which tests that the adapter calls
a mock the way the test expects rather than that it speaks the protocol. Real
sockets, real JSON, real SSE framing and real status codes catch a different and
more useful class of bug: a wrong path, a header the server rejects, a streaming
chunk that never gets parsed.

It binds to ``127.0.0.1``, which the suite's network guard permits precisely so
that this is possible without any traffic leaving the machine
(``tests/conftest.py``).

Deliberately not a general OpenAI emulator. It serves the two endpoints the M1
adapters use, returns deterministic bodies, and can be told to return an error
status so the taxonomy mapping is exercised end to end.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import struct
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _vector(text: str, dimensions: int) -> list[float]:
    """A deterministic unit vector, so ordering assertions are checkable."""
    values: list[float] = []
    block = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{text}|{block}".encode()).digest()
        for offset in range(0, len(digest), 4):
            if len(values) >= dimensions:
                break
            (raw,) = struct.unpack(">I", digest[offset : offset + 4])
            values.append((raw / 2**31) - 1.0)
        block += 1
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class ServerState:
    """What the server should do next, and what it has seen."""

    def __init__(self) -> None:
        self.status = 200
        self.error_body: dict[str, Any] = {"error": {"message": "simulated failure"}}
        self.headers: dict[str, str] = {}
        self.omit_usage = False
        self.shuffle_embeddings = False
        self.dimensions = 8
        self.reply = "hello"
        self.requests: list[dict[str, Any]] = []
        self.auth_headers: list[str | None] = []


class _Handler(BaseHTTPRequestHandler):
    state: ServerState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr logging; a test run is not a web log."""

    def _read(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        parsed: dict[str, Any] = json.loads(raw or b"{}")
        return parsed

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        for key, value in self.state.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self, chunks: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        state = self.state
        body = self._read()
        state.requests.append({"path": self.path, "body": body})
        state.auth_headers.append(self.headers.get("authorization"))

        if state.status != 200:
            self._send(state.status, state.error_body)
            return

        if self.path.endswith("/chat/completions"):
            self._chat(body)
        elif self.path.endswith("/embeddings"):
            self._embeddings(body)
        else:
            self._send(404, {"error": {"message": f"no such path {self.path}"}})

    def _chat(self, body: dict[str, Any]) -> None:
        state = self.state
        usage = None if state.omit_usage else {"prompt_tokens": 11, "completion_tokens": 7}

        if body.get("stream"):
            words = state.reply.split(" ")
            chunks: list[dict[str, Any]] = [
                {"choices": [{"delta": {"content": word + (" " if i < len(words) - 1 else "")}}]}
                for i, word in enumerate(words)
            ]
            chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            if usage is not None:
                chunks.append({"choices": [], "usage": usage})
            self._send_stream(chunks)
            return

        response: dict[str, Any] = {
            "id": "chatcmpl-fake",
            "model": body.get("model", "fake"),
            "choices": [
                {"message": {"role": "assistant", "content": state.reply}, "finish_reason": "stop"}
            ],
        }
        if usage is not None:
            response["usage"] = usage
        self._send(200, response)

    def _embeddings(self, body: dict[str, Any]) -> None:
        state = self.state
        texts = body.get("input") or []
        if isinstance(texts, str):
            texts = [texts]

        items = [
            {"index": index, "embedding": _vector(text, state.dimensions)}
            for index, text in enumerate(texts)
        ]
        if state.shuffle_embeddings:
            # Reversed, with the `index` field left correct. An adapter that
            # trusts arrival order transposes every vector; one that sorts on
            # `index` is unaffected.
            items = list(reversed(items))

        response: dict[str, Any] = {"model": body.get("model", "fake"), "data": items}
        if not state.omit_usage:
            response["usage"] = {"prompt_tokens": sum(len(t) for t in texts)}
        self._send(200, response)


@contextlib.contextmanager
def fake_openai_server() -> Iterator[tuple[str, ServerState]]:
    """Run the server on an ephemeral loopback port for the duration of a test.

    Yields:
        ``(base_url, state)``. Mutate ``state`` to change what the next request
        returns.
    """
    state = ServerState()
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
