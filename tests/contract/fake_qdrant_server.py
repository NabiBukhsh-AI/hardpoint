"""A loopback Qdrant, enough to exercise the adapter's transport.

Not a Qdrant implementation: it stores points in a dict, ignores the vectors for
ranking beyond a trivial dot product, and applies no filters at all. Filter
*translation* is tested directly as a pure function, which is where the real
risk lives; what this server exists to check is that the adapter sends the right
method to the right path with the right body, and maps the response.

Binds to 127.0.0.1, which the suite's network guard permits (tests/conftest.py).
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class QdrantState:
    """What the fake holds, and what it has been asked."""

    def __init__(self) -> None:
        self.status = 200
        self.collections: dict[str, dict[str, Any]] = {}
        self.points: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.api_keys: list[str | None] = []


class _Handler(BaseHTTPRequestHandler):
    state: QdrantState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr logging."""

    def _read(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        parsed: dict[str, Any] = json.loads(raw)
        return parsed

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, method: str) -> dict[str, Any]:
        body = self._read()
        path = self.path.split("?")[0]
        self.state.requests.append({"method": method, "path": path, "body": body})
        self.state.api_keys.append(self.headers.get("api-key"))
        return body

    def do_GET(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        self._record("GET")
        if self.state.status != 200:
            self._send(self.state.status, {"status": {"error": "simulated"}})
            return

        name = self.path.split("?")[0].removeprefix("/collections/")
        collection = self.state.collections.get(name)
        if collection is None:
            self._send(404, {"status": {"error": "Not found"}})
            return

        self._send(
            200,
            {
                "result": {
                    "config": {"params": {"vectors": collection}},
                    "points_count": len(self.state.points),
                }
            },
        )

    def do_PUT(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        body = self._record("PUT")
        if self.state.status != 200:
            self._send(self.state.status, {"status": {"error": "simulated"}})
            return

        path = self.path.split("?")[0]
        if path.endswith("/points"):
            for point in body.get("points", []):
                self.state.points[str(point["id"])] = point
            self._send(200, {"result": {"status": "completed"}})
            return

        name = path.removeprefix("/collections/")
        self.state.collections[name] = body.get("vectors", {})
        self._send(200, {"result": True})

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatches on this exact name
        body = self._record("POST")
        if self.state.status != 200:
            self._send(self.state.status, {"status": {"error": "simulated"}})
            return

        path = self.path.split("?")[0]
        if path.endswith("/points/delete"):
            for identifier in body.get("points", []):
                self.state.points.pop(str(identifier), None)
            self._send(200, {"result": {"status": "completed"}})
            return

        if path.endswith("/points/search"):
            query = body.get("vector") or []
            results = [
                {
                    "id": point["id"],
                    "score": sum(
                        a * b for a, b in zip(query, point.get("vector", []), strict=False)
                    ),
                    "payload": point.get("payload", {}),
                }
                for point in self.state.points.values()
            ]
            results.sort(key=lambda item: -item["score"])
            self._send(200, {"result": results[: body.get("limit", 10)]})
            return

        self._send(404, {"status": {"error": f"no such path {path}"}})


@contextlib.contextmanager
def fake_qdrant_server() -> Iterator[tuple[str, QdrantState]]:
    """Run the fake on an ephemeral loopback port for the duration of a test."""
    state = QdrantState()
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
