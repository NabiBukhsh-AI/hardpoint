"""The network guard in conftest.py must actually block, and must opt out correctly.

INSTRUCTIONS.md §12.2 **[LOCKED]** requires a CI fixture that fails any test
making an outbound socket connection unless it is marked ``integration``. A
guard nobody tests is a guard that silently stops working after a refactor, so
these tests exercise it directly.

No connection is ever attempted: the guard raises before any network activity,
and the opt-out branch is driven with a stub request object rather than by
actually reaching a remote host.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from tests import conftest
from tests.conftest import NetworkAccessDeniedError, is_local

UNROUTABLE = ("203.0.113.1", 9)  # TEST-NET-3, RFC 5737, discard port


def test_socket_connect_is_blocked() -> None:
    """socket.socket.connect raises instead of reaching the network."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessDeniedError) as exc_info,
    ):
        sock.connect(UNROUTABLE)
    assert "203.0.113.1" in str(exc_info.value)
    assert "INSTRUCTIONS.md 12.2" in str(exc_info.value)


def test_socket_connect_ex_is_blocked() -> None:
    """The connect_ex variant is blocked too; it would otherwise be an escape hatch."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessDeniedError),
    ):
        sock.connect_ex(UNROUTABLE)


def test_create_connection_is_blocked() -> None:
    """The module-level helper most HTTP clients reach for is blocked."""
    with pytest.raises(NetworkAccessDeniedError):
        socket.create_connection(UNROUTABLE, timeout=0.001)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (("127.0.0.1", 8080), True),
        (("127.9.9.9", 1), True),
        (("::1", 8080), True),
        (("localhost", 80), True),
        (("LOCALHOST", 80), True),
        ("/tmp/socket", True),
        (b"/tmp/socket", True),
        (("203.0.113.1", 443), False),
        (("api.openai.com", 443), False),
        (("8.8.8.8", 53), False),
        ((), False),
        (None, False),
        ((12345, 80), False),
    ],
)
def test_loopback_is_local_and_everything_else_is_not(address: object, expected: bool) -> None:
    """The carve-out must be exactly loopback, and unrecognised forms fail closed.

    Loopback is permitted because asyncio's own event loop opens a loopback
    socketpair on Windows, and because a local fake server is a legitimate way to
    test transport. Anything the classifier cannot understand is treated as
    external.
    """
    assert is_local(address) is expected


def test_loopback_connections_are_permitted() -> None:
    """A local server must be reachable, or the async event loop cannot start.

    Uses a real listening socket on an ephemeral port, so this exercises the
    guard's allow path rather than only asserting on the classifier.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(("127.0.0.1", port))  # must not raise
            assert client.getpeername()[1] == port


def test_guard_is_installed_and_the_originals_are_saved() -> None:
    """While a guarded test runs, the live functions are not the saved originals."""
    assert socket.create_connection is not conftest._REAL_CREATE_CONNECTION
    assert callable(conftest._REAL_CREATE_CONNECTION)
    assert callable(conftest._REAL_SOCKET_CONNECT)
    assert callable(conftest._REAL_SOCKET_CONNECT_EX)


class _StubNode:
    """Minimal stand-in for a pytest item, carrying only the marker lookup."""

    def __init__(self, *, integration: bool) -> None:
        self._integration = integration

    def get_closest_marker(self, name: str) -> object | None:
        if name == "integration" and self._integration:
            return object()
        return None


class _StubRequest:
    """Minimal stand-in for a FixtureRequest."""

    def __init__(self, *, integration: bool) -> None:
        self.node = _StubNode(integration=integration)


def _drive_fixture(*, integration: bool) -> Any:
    """Run the guard fixture's generator against a stub request and yield inside it."""
    fixture_function = conftest._block_network.__wrapped__
    generator = fixture_function(_StubRequest(integration=integration))
    next(generator)
    return generator


def test_fixture_opts_out_for_integration_marked_tests() -> None:
    """An integration-marked test keeps the real socket API.

    The guard is currently installed for *this* test, so the assertion is that
    the fixture, when driven with an integration-marked request, leaves the live
    attributes exactly as it found them rather than replacing them again.
    """
    before_connect = socket.socket.connect
    before_create = socket.create_connection

    generator = _drive_fixture(integration=True)
    try:
        assert socket.socket.connect is before_connect
        assert socket.create_connection is before_create
    finally:
        with pytest.raises(StopIteration):
            next(generator)


def test_fixture_restores_the_originals_on_teardown() -> None:
    """Tearing the fixture down puts the real functions back.

    Without this the guard would leak past the test session and break anything
    that legitimately opens a socket afterwards, such as an integration run.
    """
    generator = _drive_fixture(integration=False)
    assert socket.create_connection is not conftest._REAL_CREATE_CONNECTION

    with pytest.raises(StopIteration):
        next(generator)

    assert socket.create_connection is conftest._REAL_CREATE_CONNECTION
    assert socket.socket.connect is conftest._REAL_SOCKET_CONNECT
    assert socket.socket.connect_ex is conftest._REAL_SOCKET_CONNECT_EX
