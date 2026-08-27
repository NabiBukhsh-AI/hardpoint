"""Shared test configuration.

Contains the network guard required by INSTRUCTIONS.md §12.2 **[LOCKED]**:
unit tests never touch the network, and a test that opens an outbound socket
fails unless it is marked ``integration``.

## Why loopback is allowed

The guard blocks connections to *external* hosts. It deliberately permits
loopback, for two reasons.

Blocking loopback does not catch a single real violation, because no provider
lives at 127.0.0.1. What it does catch is the asyncio event loop: on Windows,
``ProactorEventLoop`` builds its internal self-pipe with ``socket.socketpair()``,
which is implemented as a loopback connect. Blocking that makes every async test
fail with a message about network access, which is both wrong and extremely
confusing.

The second reason is that a local fake HTTP server is a legitimate way to test
an adapter's transport without reaching a provider, and forbidding it would push
those tests toward mocking the HTTP client instead, which tests less.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest

_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection

_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", ""})


class NetworkAccessDeniedError(RuntimeError):
    """Raised when a non-integration test attempts an outbound connection."""


def is_local(address: Any) -> bool:
    """Return whether an address is loopback, a Unix socket, or otherwise local.

    Anything this cannot classify is treated as *not* local, so an unrecognised
    address form fails closed.
    """
    if isinstance(address, str | bytes):
        return True  # AF_UNIX path, which reaches nothing off this machine

    if not isinstance(address, tuple) or not address:
        return False

    host = address[0]
    if not isinstance(host, str):
        return False
    if host.lower() in _LOCAL_HOSTNAMES:
        return True

    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _denied(address: Any) -> NetworkAccessDeniedError:
    return NetworkAccessDeniedError(
        f"This test attempted to connect to {address!r}.\n"
        "Unit and contract tests must not touch the network "
        "(INSTRUCTIONS.md 12.2). Use a fake from hardpoint.testing, or mark the "
        "test with @pytest.mark.integration if a real system is genuinely required."
    )


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Block outbound socket connections for every test that is not integration."""
    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not is_local(address):
            raise _denied(address)
        _REAL_SOCKET_CONNECT(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not is_local(address):
            raise _denied(address)
        return _REAL_SOCKET_CONNECT_EX(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not is_local(address):
            raise _denied(address)
        return _REAL_CREATE_CONNECTION(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = _REAL_SOCKET_CONNECT  # type: ignore[method-assign]
        socket.socket.connect_ex = _REAL_SOCKET_CONNECT_EX  # type: ignore[method-assign]
        socket.create_connection = _REAL_CREATE_CONNECTION  # type: ignore[assignment]


@pytest.fixture
def anyio_backend() -> str:
    """Run every ``@pytest.mark.anyio`` test on asyncio only.

    The library is written against anyio so that it *can* run on trio, but the
    test suite pins one backend: running every async test twice doubles the
    suite for no signal, since nothing here is backend-specific. A backend
    matrix belongs in a dedicated job if it is ever wanted.
    """
    return "asyncio"
