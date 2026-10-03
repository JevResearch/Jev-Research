"""Offline tests for the portable knowledge-replay CLI (scripts/benchmark/
replay_knowledge.py).  All HTTP is httpx.MockTransport / injected fakes; the
shared conftest bans sockets outright and no test may make a paid call.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

import replay_knowledge as rk  # noqa: E402
from jev_observatory.transport import TransportResponse  # noqa: E402

FAKE = "apikey_deadbeefdeadbeefdeadbeefdeadbeef"
QPATH = ROOT / "data_report/followup_20261003/knowledge-questions.json"
KPATH = ROOT / "data_report/followup_20261003/knowledge-answer-key.json"


@pytest.fixture(autouse=True)
def _live_env(monkeypatch, tmp_path):
    monkeypatch.setenv("JEVO_ALLOW_LIVE", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)


def _native_body(choice="o0", probs=None, model="jev-1.13.0"):
    return json.dumps({
        "model": model,
        "answers": {"q0": {"choice": choice,
                           "probabilities": probs or {"o0": 0.9, "o1": 0.1}}},
        "usage": {"input_tokens": 120, "output_tokens": 1},
    }).encode()


def _mock(handler):
    return httpx.MockTransport(handler)


def _rows(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------- model payloads
def test_native_payload_carries_no_gold_or_metadata(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_native_body())

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--provider", "jev", "--limit", "2", "--output", str(out)],
                http_transport=_mock(handler))
    assert rc == 0 and len(seen) == 2
    payload = seen[0]
    assert set(payload) == {"state", "model", "questions"}
    assert payload["state"] == "This question is about public events."
    assert payload["model"] == "jev-1.13.0"
    q = payload["questions"]["q0"]
    assert set(q) == {"type", "instructions", "criteria"}
    assert q["instructions"] == ("Which team won the 2024 World Series?")
    assert list(q["criteria"]) == [f"o{i}" for i in range(6)]
    text = json.dumps(payload)
    for banned in ('"gold"', '"answer"', '"kind"', '"source_id"',
                   '"rotation"', '"item_id"', '"date"', "q01"):
        assert banned not in text
    # second call is the SAME item's second variant, same question text
    assert seen[1]["questions"]["q0"]["instructions"] == q["instructions"]


class _FakeOR:
    def __init__(self, body, seen):
        self.body, self.seen, self.closed = body, seen, False

    def post(self, path, payload):
        self.seen.append((path, payload))
        return TransportResponse(status_code=200, headers={}, body=self.body,
                                 total_ms=5.0)

    def close(self):
        self.closed = True


def test_openrouter_default_model_and_limit4(tmp_path):
    seen = []
    body = json.dumps({
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "choices": [{"finish_reason": "stop", "message": {"content": "o0"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1}}).encode()
    or_t = _FakeOR(body, seen)
    out = tmp_path / "out.jsonl"
    rc = rk.run(["--provider", "openrouter", "--limit", "4",
                 "--output", str(out)], or_transport=or_t)
    assert rc == 0 and len(seen) == 4
    paths = {p for p, _ in seen}
    assert paths == {"/chat/completions"}
    for _, payload in seen:
        assert payload["model"] == "qwen/qwen3.5-9b"
        assert set(payload) == {"model", "messages", "max_tokens"}
        assert payload["max_tokens"] == 512   # documented default, no sampler
    assert or_t.closed
    assert len(_rows(out)) == 4


def test_openrouter_model_override(tmp_path):
    seen = []
    body = json.dumps({
        "model": "custom/m", "choices": [{"finish_reason": "stop",
                                          "message": {"content": "o0"}}],
        "usage": {}}).encode()
    or_t = _FakeOR(body, seen)
    out = tmp_path / "out.jsonl"
    rc = rk.run(["--provider", "openrouter", "--model", "custom/m",
                 "--limit", "1", "--output", str(out)], or_transport=or_t)
    assert rc == 0 and seen[0][1]["model"] == "custom/m"
    assert _rows(out)[0]["request_model"] == "custom/m"


# ------------------------------------------------- variants, order, grading
def test_both_variants_order_and_grader_key_joins(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_native_body(choice="o0"))

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--provider", "jev", "--answer-key", str(KPATH),
                 "--output", str(out)], http_transport=_mock(handler))
    assert rc == 0
    rows = _rows(out)
    assert len(rows) == 48 and len(seen) == 48
    assert [r["rotation"] for r in rows] == [0, 3] * 24
    assert [r["item_id"] for r in rows[:2]] == ["q01", "q01"]
    # answer-key join is per item AND rotation (q01: rot0 -> o0, rot3 -> o3)
    assert rows[0]["grade_expected_key"] == "o0"
    assert rows[1]["grade_expected_key"] == "o3"
    assert rows[0]["grade_strict_hit"] is True and rows[1]["grade_strict_hit"] is False
    assert rows[1]["grade_recovered_hit"] is False
    # variant options arrive in export order (rotation 3 starts with "Toronto")
    crit3 = seen[1]["questions"]["q0"]["criteria"]
    assert crit3["o0"] == "Toronto Blue Jays"
    assert crit3["o3"] == "Los Angeles Dodgers"
    assert rows[1]["choice"] == "o0" and rows[1]["choice_text"] == "Toronto Blue Jays"


# ------------------------------------------------------------- gating safety
def test_missing_live_gate_prevents_http(tmp_path, monkeypatch):
    monkeypatch.delenv("JEVO_ALLOW_LIVE", raising=False)
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(200, content=_native_body())

    out = tmp_path / "out.jsonl"
    assert rk.run(["--limit", "2", "--output", str(out)],
                  http_transport=_mock(handler)) == 2
    assert not seen and not out.exists()


def test_missing_key_prevents_http(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(200, content=_native_body())

    out = tmp_path / "out.jsonl"
    assert rk.run(["--limit", "2", "--output", str(out)],
                  http_transport=_mock(handler)) == 2
    assert not seen and not out.exists()


def test_refuses_existing_output(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text("prior study rows stay untouched\n", encoding="utf-8")

    def handler(request):
        raise AssertionError("no HTTP on refused output")

    rc = rk.run(["--output", str(out)], http_transport=_mock(handler))
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "prior study rows stay untouched\n"


def test_auth_error_stops_run(tmp_path):
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(401, content=b'{"error": "unauthorized"}')

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--answer-key", str(KPATH), "--output", str(out)],
                http_transport=_mock(handler))
    assert rc == 2 and len(seen) == 1          # never 48 bad records
    assert len(_rows(out)) == 1


# ------------------------------------------------------------- recordkeeping
def test_full_raw_retention_key_redacted(tmp_path):
    payload_echo = ("long model reply " * 40) + f" key={FAKE} tail"

    def handler(request):
        return httpx.Response(200, content=json.dumps({
            "model": "jev-1.13.0",
            "answers": {"q0": {"choice": "o0", "probabilities": {"o0": 1.0}}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "echo": payload_echo}).encode())

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--limit", "1", "--output", str(out)],
                http_transport=_mock(handler))
    assert rc == 0
    row = _rows(out)[0]
    assert len(row["response_sanitized"]) > 400      # full body, no excerpt
    assert "long model reply" in row["response_sanitized"]
    blob = out.read_text(encoding="utf-8")
    assert FAKE not in blob and "[REDACTED]" in blob


def test_native_choice_validated_never_argmax(tmp_path):
    def handler(request):
        return httpx.Response(200, content=_native_body(
            choice="o9", probs={"o0": 0.01, "o1": 0.98}))

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--limit", "1", "--output", str(out)],
                http_transport=_mock(handler))
    assert rc == 1                                  # explicit safety exit
    row = _rows(out)[0]
    assert row["choice"] is None                    # probabilities do not override
    assert row["contract_error"] == "choice_not_allowed"
    assert row["probabilities"] == {"o0": 0.01, "o1": 0.98}


def test_transport_failure_is_not_a_knowledge_miss(tmp_path):
    def handler(request):
        raise httpx.ConnectError("mock offline")

    out = tmp_path / "out.jsonl"
    rc = rk.run(["--answer-key", str(KPATH), "--output", str(out)],
                http_transport=_mock(handler))
    assert rc == 1
    row = _rows(out)[0]
    assert row["grade_strict_hit"] is None and row["grade_recovered_hit"] is None
    assert row["contract_error"].startswith("transport:")


def test_truncated_reply_flagged_not_claimed(tmp_path):
    or_t = _FakeOR(json.dumps({
        "model": "m", "choices": [{"finish_reason": "length",
                                   "message": {"content": "o0"}}],
        "usage": {}}).encode(), [])
    out = tmp_path / "out.jsonl"
    rc = rk.run(["--provider", "openrouter", "--limit", "1",
                 "--output", str(out)], or_transport=or_t)
    assert rc == 0
    row = _rows(out)[0]
    assert row["truncated"] is True and row["finish_reason"] == "length"


# ------------------------------------------------------------------ recovery
def test_recovery_exact_and_final_line():
    keys = ["o0", "o1", "o2", "o3"]
    criteria = {"o0": "Los Angeles Dodgers", "o1": "New York Yankees",
                "o2": "Boston Red Sox", "o3": "Toronto Blue Jays"}
    assert rk.recover_reply("o2", keys, criteria) == ("o2", "exact")
    assert rk.recover_reply("thinking...\no1", keys, criteria) == \
        ("o1", "final_line_key")
    # a clean key+text echo recovers the key via 2.1's isolated-key stage
    assert rk.recover_reply("o3. Toronto Blue Jays", keys, criteria)[0] == "o3"
    # no permissive first-character guessing
    assert rk.recover_reply("o", keys, criteria)[0] is None
    assert rk.recover_reply("T", keys, criteria)[0] is None


def test_safe_echo_contract_and_ladder_fallthrough():
    # option text mentions another allowed key: 2.1 is ambiguous -> echo runs
    keys = ["o0", "o1"]
    criteria = {"o0": "the o1 marker", "o1": "plain text"}
    assert rk.recover_reply("o0. the o1 marker", keys, criteria) == \
        ("o0", "echo_exact_text")
    # key/text disagreement is rejected, no gold access
    assert rk.echo_recover("o0. plain text", keys, criteria) == \
        (None, "key_text_disagree")
    assert rk.echo_recover("o0. the o1 marker extra", keys, criteria)[0] is None
    assert rk.echo_recover("o9. the o1 marker", keys, criteria)[0] is None
    assert rk.echo_recover("o0", keys, criteria)[0] is None


# --------------------------------------------------------------- module shape
def test_module_imports_without_tokenizer_stack():
    source = (ROOT / "scripts/benchmark/replay_knowledge.py").read_text(
        encoding="utf-8")
    for banned in ("tiktoken", "tokenizers", "transformers", "torch",
                   "followup_battery", "run_followup"):
        assert banned not in source
    assert rk.DEFAULT_QUESTIONS.exists()


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        rk.run(["--help"])
    assert exc.value.code == 0
    assert "--answer-key" in capsys.readouterr().out
