"""Local-only web UI for Talk sessions.

Security posture (DESIGN.md §11, IMPLEMENTATION.md M3):
* loopback binding only — binding to any other interface is refused at startup;
* Origin/Host checks: only localhost origins may call the API, so arbitrary
  websites cannot spend the key through localhost;
* per-session bearer token issued at creation; the API key never leaves the
  server process and is never present in client assets or responses;
* one in-flight request per session (409 otherwise).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .guards import BudgetLedger, BudgetCaps
from .talk import DecoderConfig, TalkLimits, TalkSession

STATIC_DIR = Path(__file__).parent / "static"
_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


class SessionRegistry:
    def __init__(self) -> None:
        self.sessions: dict[str, TalkSession] = {}
        self.tokens: dict[str, str] = {}  # token -> session_id
        self.locks: dict[str, asyncio.Lock] = {}

    def create(self, session: TalkSession) -> str:
        token = uuid.uuid4().hex
        self.sessions[session.session_id] = session
        self.tokens[token] = session.session_id
        self.locks[session.session_id] = asyncio.Lock()
        return token

    def authorize(self, token: str | None, session_id: str) -> bool:
        return token is not None and self.tokens.get(token) == session_id


def create_app(
    provider_factory: Callable[[], Any],
    *,
    limits: TalkLimits | None = None,
    decoder: DecoderConfig | None = None,
    model: str = "jev-1.13.0",
    budget: BudgetLedger | None = None,
    require_loopback: bool = True,
) -> "FastAPI":  # noqa: F821
    limits = limits or TalkLimits()
    decoder = decoder or DecoderConfig()
    registry = SessionRegistry()

    app = FastAPI(title="Jev Observatory — Talk (local)")

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":")[0]
        if require_loopback and host not in _ALLOWED_HOSTS:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin is not None:
            origin_host = origin.split("//", 1)[-1].split(":")[0]
            if origin_host not in _ALLOWED_HOSTS:
                return JSONResponse({"error": "cross-origin requests are not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.post("/api/session")
    async def create_session(request: Request) -> JSONResponse:
        body = await request.json()
        user_prompt = str(body.get("user_prompt", "")).strip()
        if not user_prompt:
            raise HTTPException(status_code=422, detail="user_prompt is required")
        previous = body.get("previous_turns") or []
        if not isinstance(previous, list):
            raise HTTPException(status_code=422, detail="previous_turns must be a list")
        session_id = uuid.uuid4().hex[:12]
        session = TalkSession(
            session_id=session_id,
            provider=provider_factory(),
            user_prompt=user_prompt,
            previous_turns=[{"role": str(t.get("role", "user")), "text": str(t.get("text", ""))}
                            for t in previous],
            limits=TalkLimits(
                max_chars=min(int(body.get("max_chars", limits.max_chars)), limits.max_chars),
                max_seconds=limits.max_seconds,
                max_provider_requests=(budget.caps.max_requests if budget is not None else None),
            ),
            decoder=decoder,
            model=model,
            hook=budget,
        )
        token = registry.create(session)
        return JSONResponse({"session_id": session_id, "token": token,
                             "limits": {"max_chars": session.limits.max_chars,
                                        "max_seconds": session.limits.max_seconds}})

    def _session_and_token(request: Request, session_id: str, token_param: str | None) -> TalkSession:
        token = request.headers.get("x-session-token") or token_param
        if not registry.authorize(token, session_id):
            raise HTTPException(status_code=401, detail="invalid session token")
        return registry.sessions[session_id]

    @app.post("/api/session/{session_id}/step")
    async def step(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        event = await asyncio.to_thread(session.step)
        return JSONResponse(event)

    @app.post("/api/session/{session_id}/run")
    async def run(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        lock = registry.locks[session_id]
        if lock.locked():
            raise HTTPException(status_code=409, detail="a run is already in flight for this session")
        async with lock:
            events = await asyncio.to_thread(session.run)
        return JSONResponse({"events": events, "stop_reason": session.stop_reason,
                             "prefix": session.prefix})

    @app.post("/api/session/{session_id}/cancel")
    async def cancel(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        session.cancel()
        return JSONResponse({"cancelled": True})

    @app.post("/api/session/{session_id}/backtrack")
    async def backtrack(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        body = await request.json()
        try:
            session.backtrack(str(body.get("node_id", "")))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return JSONResponse({"current_node": session.current_node_id, "prefix": session.prefix})

    @app.post("/api/session/{session_id}/edit_prefix")
    async def edit_prefix(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        body = await request.json()
        node_id = session.edit_prefix(str(body.get("text", "")))
        return JSONResponse({"node_id": node_id, "prefix": session.prefix})

    @app.get("/api/session/{session_id}/state")
    async def state(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        trace = session.export_trace()
        return JSONResponse({
            "session_id": session_id,
            "prefix": trace["prefix"],
            "stop_reason": trace["stop_reason"],
            "events": session.events,
            "n_nodes": trace["n_nodes"],
            "elapsed_seconds": round(session._elapsed(), 3),
        })

    @app.get("/api/session/{session_id}/events")
    async def events(session_id: str, request: Request, token: str | None = None,
                     stream: int = 0):
        session = _session_and_token(request, session_id, token)
        if not stream:
            return JSONResponse({"events": session.events})

        async def generate():
            sent = 0
            idle = 0
            while True:
                current = session.events
                while sent < len(current):
                    yield f"data: {json.dumps(current[sent])}\n\n"
                    sent += 1
                if session.stop_reason is not None and sent >= len(current):
                    yield "event: done\ndata: {}\n\n"
                    return
                idle += 1
                if idle > 150:  # ~30s without new events: close the stream
                    return
                await asyncio.sleep(0.2)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/session/{session_id}/trace")
    async def trace(session_id: str, request: Request, token: str | None = None):
        session = _session_and_token(request, session_id, token)
        return JSONResponse(session.export_trace())

    return app


def assert_loopback(host: str, port: int) -> None:
    """Refuse non-loopback binding before any server starts."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"Talk UI must bind to loopback (127.0.0.1), got {host!r}")
    if not 0 < port < 65536:
        raise ValueError(f"invalid port {port}")
