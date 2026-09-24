"""HTTP transports: real (httpx), recording, and offline replay.

The transport layer is deliberately thin and returns *bytes* plus honest timing:
it never parses, never retries and never invents measurements.  `first_byte_ms`
is None for any transport that cannot observe it, rather than being copied from
total latency.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol

from .redact import Redactor
from .schema import SystemOneRequest


@dataclass
class TransportResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    total_ms: float
    first_byte_ms: float | None = None
    cold_connection: bool | None = None
    error: str | None = None  # transport-level failure class, if any


class Transport(Protocol):
    name: str

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse: ...

    def close(self) -> None: ...


class HttpxTransport:
    """Real network transport. Disabled unless the caller opted in to live calls."""

    name = "httpx"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        redactor: Redactor | None = None,
        clock: Callable[[], float] = time.monotonic,
        http_transport: Any | None = None,
    ) -> None:
        import httpx  # imported lazily so offline tooling works without the wheel

        if not api_key:
            raise ValueError("no API key in environment; refusing to construct a live transport")
        self._httpx = httpx
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.redactor = redactor or Redactor()
        self.clock = clock
        # `http_transport` allows injecting an httpx transport (e.g.
        # httpx.MockTransport) for OFFLINE capture of the exact serialized wire
        # body. When None the client uses real connections; live gates are
        # enforced upstream of this class.
        client_kwargs: dict[str, Any] = dict(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
        if http_transport is not None:
            client_kwargs["transport"] = http_transport
        self._http_transport = http_transport
        self._client = httpx.Client(**client_kwargs)
        self._requests_since_reset = 0

    def reset_pool(self) -> None:
        """Tear down connections so the next attempt is genuinely cold."""
        self._client.close()
        client_kwargs: dict[str, Any] = dict(
            base_url=self.base_url,
            headers=dict(self._client.headers),
            timeout=self.timeout_seconds,
        )
        if self._http_transport is not None:
            client_kwargs["transport"] = self._http_transport
        self._client = self._httpx.Client(**client_kwargs)
        self._requests_since_reset = 0

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cold = self._requests_since_reset == 0
        self._requests_since_reset += 1
        started = self.clock()
        first_byte_ms: float | None = None
        try:
            with self._client.stream("POST", path, content=body) as response:
                chunks: list[bytes] = []
                for chunk in response.iter_bytes():
                    if first_byte_ms is None:
                        first_byte_ms = (self.clock() - started) * 1000.0
                    chunks.append(chunk)
                content = b"".join(chunks)
                total_ms = (self.clock() - started) * 1000.0
                return TransportResponse(
                    status_code=response.status_code,
                    headers=self.redactor.headers(dict(response.headers)),
                    body=content,
                    total_ms=total_ms,
                    first_byte_ms=first_byte_ms,
                    cold_connection=cold,
                )
        except self._httpx.TimeoutException as exc:
            return TransportResponse(
                status_code=0,
                headers={},
                body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                cold_connection=cold,
                error=f"timeout:{type(exc).__name__}",
            )
        except self._httpx.RequestError as exc:
            return TransportResponse(
                status_code=0,
                headers={},
                body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                cold_connection=cold,
                error=f"transport_error:{type(exc).__name__}",
            )

    def close(self) -> None:
        self._client.close()


class ScriptedTransport:
    """Deterministic offline transport that replays a queued list of responses.

    Each entry is either a `TransportResponse` or a dict describing one.  Used by
    tests to script 401/422/429/529/truncated/malformed behaviour.
    """

    name = "scripted"

    def __init__(self, script: list[Any], clock: Callable[[], float] = time.monotonic) -> None:
        self.script = list(script)
        self.clock = clock
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        self.calls.append({"path": path, "payload": payload})
        if not self.script:
            raise AssertionError("ScriptedTransport exhausted: transport must never invent responses")
        entry = self.script.pop(0)
        if isinstance(entry, TransportResponse):
            return entry
        return TransportResponse(
            status_code=entry.get("status_code", 200),
            headers=entry.get("headers", {}),
            body=entry["body"] if isinstance(entry.get("body"), bytes) else json.dumps(entry.get("body", {})).encode(),
            total_ms=entry.get("total_ms", 1.0),
            first_byte_ms=entry.get("first_byte_ms"),
            cold_connection=entry.get("cold_connection"),
            error=entry.get("error"),
        )

    def close(self) -> None:
        return


class RecordingTransport:
    """Wrap any transport and store request/response pairs keyed by request hash."""

    name = "recording"

    def __init__(self, inner: Transport, directory: Path | str, redactor: Redactor | None = None) -> None:
        self.inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or Redactor()

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        response = self.inner.post(path, payload)
        key = sha256(json.dumps({"path": path, "payload": payload}, sort_keys=True).encode()).hexdigest()
        record = {
            "path": path,
            "payload": self.redactor.obj(payload),
            "status_code": response.status_code,
            "headers": self.redactor.headers(response.headers),
            "body_b64": _b64(response.body),
            "total_ms": response.total_ms,
            "first_byte_ms": response.first_byte_ms,
            "cold_connection": response.cold_connection,
            "error": response.error,
        }
        _write_once(self.directory / f"{key}.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return response

    def close(self) -> None:
        self.inner.close()


class ReplayTransport:
    """Serve previously recorded responses; a miss is an error, never a network call."""

    name = "replay"

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"no recording directory at {self.directory}")
        self._cursor: dict[str, int] = {}

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        key = sha256(json.dumps({"path": path, "payload": payload}, sort_keys=True).encode()).hexdigest()
        # Retries share a payload, so replay them in recorded order via the
        # enumerated suffixes written by RecordingTransport.
        index = self._cursor.get(key, 0)
        suffix = "" if index == 0 else f".{index + 1}"
        file = self.directory / f"{key}.json{suffix}"
        if not file.exists() and index > 0:
            # Identical retry responses were deduplicated on write; reproduce the
            # base recording rather than pretending the attempt never happened.
            file = self.directory / f"{key}.json"
        if not file.exists():
            raise KeyError(
                f"no recorded response for request {key} attempt {index + 1}; replay must not hit the network"
            )
        self._cursor[key] = index + 1
        record = json.loads(file.read_text())
        return TransportResponse(
            status_code=record["status_code"],
            headers=record.get("headers", {}),
            body=_unb64(record["body_b64"]),
            total_ms=record["total_ms"],
            first_byte_ms=record.get("first_byte_ms"),
            cold_connection=record.get("cold_connection"),
            error=record.get("error"),
        )

    def close(self) -> None:
        return


def request_key(request: SystemOneRequest) -> str:
    return request.request_sha256()


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


def _unb64(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def _write_once(path: Path, text: str) -> None:
    """Write deterministically; retries of an identical payload are stored side by side.

    A retry sends a byte-identical payload, so the key is not unique.  Identical
    repeats are ignored and differing ones get an enumerated suffix rather than
    silently overwriting the earlier observation.
    """
    for suffix in ("", ".2", ".3", ".4", ".5"):
        target = Path(str(path) + suffix)
        if target.exists():
            if target.read_text() == text:
                return
            continue
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    raise RuntimeError(f"too many differing recordings for {path}")  # pragma: no cover
