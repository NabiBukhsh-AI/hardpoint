"""Shared test configuration.

Contains the network guard required by INSTRUCTIONS.md §12.2 **[LOCKED]**:
unit tests never touch the network, and a test that opens an outbound socket
fails unless it is marked ``integration``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


class NetworkAccessDeniedError(RuntimeError):
    """Raised when a non-integration test attempts an outbound connection."""


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
        raise _denied(address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        raise _denied(address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        raise _denied(address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = _REAL_SOCKET_CONNECT  # type: ignore[method-assign]
        socket.socket.connect_ex = _REAL_SOCKET_CONNECT_EX  # type: ignore[method-assign]
        socket.create_connection = _REAL_CREATE_CONNECTION  # type: ignore[assignment]
