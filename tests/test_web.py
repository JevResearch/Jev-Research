"""Web layer tests: loopback/origin guards, session auth, limits, SSE."""

import json

import pytest
from fastapi.testclient import TestClient

from jev_observatory.guards import BudgetCaps, BudgetLedger
from jev_observatory.talk import DecoderConfig, ScriptedCharProvider, TalkLimits
from jev_observatory.web import assert_loopback, create_app
from conftest import REAL_SOCKET as conftest_real_socket


@pytest.fixture(autouse=True)
def _loopback_only_sockets(monkeypatch):
    """Web tests need TestClient, which constructs sockets internally.

    Construction is allowed; any actual connect() to a non-loopback address
    fails, so no network egress is possible while still guarding the tests.
    """
    class Guarded(conftest_real_socket):
        def connect(self, address):
            host = address[0] if isinstance(address, tuple) else str(address)
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise AssertionError(f"tests must not connect to {host!r}")
            return super().connect(address)

        def connect_ex(self, address):
            raise AssertionError("tests must not connect_ex")

    import socket

    monkeypatch.setattr(socket, "socket", Guarded)
    yield


@pytest.fixture
def client():
    app = create_app(lambda: ScriptedCharProvider("Hello from the fixture."),
                     limits=TalkLimits(max_chars=64, max_seconds=30.0))
    return TestClient(app, base_url="http://127.0.0.1"), app


def _create(client, **over):
    body = {"user_prompt": "greet", **over}
    res = client.post("/api/session", json=body)
    assert res.status_code == 200
    return res.json()


def test_full_session_over_http(client):
    http, _ = client
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    run = http.post(f"/api/session/{data['session_id']}/run", json={}, headers=headers)
    assert run.status_code == 200
    body = run.json()
    assert body["prefix"] == "Hello from the fixture."
    assert body["stop_reason"] == "end_token"
    steps = [e for e in body["events"] if e["type"] == "step"]
    assert steps[0]["top_next"][0]["p"] == 1.0


def test_invalid_token_rejected(client):
    http, _ = client
    data = _create(http)
    res = http.post(f"/api/session/{data['session_id']}/step", json={},
                    headers={"X-Session-Token": "wrong"})
    assert res.status_code == 401


def test_cross_origin_forbidden(client):
    http, _ = client
    data = _create(http)
    res = http.post(f"/api/session/{data['session_id']}/step", json={},
                    headers={"X-Session-Token": data["token"],
                             "Origin": "https://evil.example"})
    assert res.status_code == 403


def test_forbidden_host_rejected(client):
    http, _ = client
    data = _create(http)
    res = http.post(f"/api/session/{data['session_id']}/step", json={},
                    headers={"X-Session-Token": data["token"], "Host": "tracker.example"})
    assert res.status_code == 403


def test_local_origin_allowed(client):
    http, _ = client
    data = _create(http)
    res = http.post(f"/api/session/{data['session_id']}/step", json={},
                    headers={"X-Session-Token": data["token"],
                             "Origin": "http://127.0.0.1:8765"})
    assert res.status_code == 200


def test_empty_prompt_rejected(client):
    http, _ = client
    res = http.post("/api/session", json={"user_prompt": "  "})
    assert res.status_code == 422


def test_no_api_key_in_assets_or_responses(client):
    http, _ = client
    html = http.get("/").text
    assert "apikey" not in html.lower()
    assert "TYPESAFE_API_KEY" not in html
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    trace = http.get(f"/api/session/{data['session_id']}/trace", headers=headers)
    assert "apikey" not in trace.text.lower()
    # a foreign session id cannot read another session's trace
    res = http.get(f"/api/session/{data['session_id']}/trace",
                   headers={"X-Session-Token": "forged"})
    assert res.status_code == 401


def test_session_limits_are_clamped(client):
    http, _ = client
    data = _create(http, max_chars=10_000)  # above the server cap of 64
    assert data["limits"]["max_chars"] == 64


def test_backtrack_via_http(client):
    http, _ = client
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    http.post(f"/api/session/{data['session_id']}/run", json={}, headers=headers)
    state = http.get(f"/api/session/{data['session_id']}/state", headers=headers).json()
    trace = http.get(f"/api/session/{data['session_id']}/trace", headers=headers).json()
    first_step = next(n for n in trace["nodes"] if n["char"] == "H")
    res = http.post(f"/api/session/{data['session_id']}/backtrack",
                    json={"node_id": first_step["node_id"]}, headers=headers)
    assert res.status_code == 200
    assert res.json()["prefix"] == "H"
    # original branch still in the trace
    trace2 = http.get(f"/api/session/{data['session_id']}/trace", headers=headers).json()
    assert trace2["n_nodes"] >= trace["n_nodes"]


def test_edit_prefix_via_http_marks_user_edited(client):
    http, _ = client
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    res = http.post(f"/api/session/{data['session_id']}/edit_prefix",
                    json={"text": "Hand-typed: "}, headers=headers)
    assert res.status_code == 200 and res.json()["prefix"] == "Hand-typed: "
    trace = http.get(f"/api/session/{data['session_id']}/trace", headers=headers).json()
    assert any(n["user_edited"] for n in trace["nodes"])


def test_sse_stream_emits_events_and_closes(client):
    http, _ = client
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    http.post(f"/api/session/{data['session_id']}/run", json={}, headers=headers)
    with http.stream("GET", f"/api/session/{data['session_id']}/events?stream=1&token={data['token']}",
                     headers=headers) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(chunk for chunk in response.iter_text())
    assert 'data: {"type"' in text
    assert "event: done" in text


def test_budget_cap_shared_with_cli_ledger():
    """The Talk server honours the same BudgetLedger used by the CLI."""
    ledger = BudgetLedger(BudgetCaps(max_requests=3))
    app = create_app(lambda: ScriptedCharProvider("A" * 30),
                     limits=TalkLimits(max_chars=64), budget=ledger)
    http = TestClient(app, base_url="http://127.0.0.1")
    data = _create(http)
    headers = {"X-Session-Token": data["token"]}
    run = http.post(f"/api/session/{data['session_id']}/run", json={}, headers=headers)
    assert run.status_code == 200
    body = run.json()
    # the shared cap stops spend via either layer (controller count or ledger)
    assert body["stop_reason"] in {"budget_exceeded", "max_provider_requests"}
    assert len(body["prefix"]) == 3


def test_assert_loopback_refuses_other_interfaces():
    assert_loopback("127.0.0.1", 8765)
    assert_loopback("localhost", 8765)
    with pytest.raises(ValueError, match="loopback"):
        assert_loopback("0.0.0.0", 8765)
    with pytest.raises(ValueError, match="port"):
        assert_loopback("127.0.0.1", 99_999)