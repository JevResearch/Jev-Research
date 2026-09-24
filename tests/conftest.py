"""Shared fixtures. Offline discipline: any attempt to open a network socket
fails the test immediately (DESIGN.md §11: no network calls in tests)."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FAKE_KEY = "apikey_deadbeefdeadbeefdeadbeefdeadbeef"
REAL_SOCKET = socket.socket  # captured before any fixture patches it


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test reaches for the network."""

    class Forbidden(socket.socket):
        def __init__(self, *args, **kwargs):
            raise AssertionError("tests must not open network sockets")

    monkeypatch.setattr(socket, "socket", Forbidden)
    yield


@pytest.fixture
def run_root(tmp_path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    return FAKE_KEY