"""OpenRouter baseline adapter tests (offline: ScriptedTransport only).

Covers: deterministic wire payload (explicit model, bounded output budget,
recorded reasoning effort), strict choice parsing (exact key or counted
failure, never repaired), usage + model-returned recording, raw persistence,
one-attempt no-retry policy, and the dry-run one-call-per-model smoke path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_observatory.openrouter import (
    OpenRouterError,
    OpenRouterProvider,
    WireConfig,
    build_chat_payload,
    extract_choice_content,
    parse_choice_answer,
)
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest
from jev_observatory.transport import ScriptedTransport, TransportResponse

WIRE = WireConfig(model="vendor/model-x", reasoning_effort="low", max_output_tokens=64)

REQUEST = SystemOneRequest(
    state="Which option holds?",
    model="jev-1.13.0",
    questions={"q": ChoiceQuestion(
        type="choice",
        instructions="Answer the multiple-choice question. Reply with only the option key.",
        criteria={"A": "first", "B": "second", "C": "third"},
    )},
)


def _fixture_body(content="B", model="vendor/model-x", usage=None) -> dict:
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 42, "completion_tokens": 3,
                           "total_tokens": 45},
    }


def test_wire_payload_is_explicit_and_bounded():
    payload = build_chat_payload(REQUEST, WIRE)
    assert payload["model"] == "vendor/model-x"
    assert payload["max_tokens"] == 64
    assert payload["reasoning_effort"] == "low"
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    # SAME state + options as the Jev serialization; option order never sorted
    assert "Which option holds?" in content
    for line, key in (("A. first", "A"), ("B. second", "B"), ("C. third", "C")):
        assert line in content
    assert payload.keys() >= {"model", "messages", "max_tokens"}
    # the exact same item produces a byte-identical payload
    assert json.dumps(build_chat_payload(REQUEST, WIRE)) == json.dumps(payload)


def test_strict_parsing_exact_key_or_counted_failure():
    assert parse_choice_answer("B", ["A", "B", "C"]) == ("B", None)
    assert parse_choice_answer("  C \n", ["A", "B", "C"]) == ("C", None)
    # prose, lowercase, wrong key: all strict failures, never repaired
    assert parse_choice_answer("The answer is B", ["A", "B", "C"])[1] == \
        "strict_format_not_exact_key"
    assert parse_choice_answer("b", ["A", "B", "C"])[1] == "strict_format_not_exact_key"
    assert parse_choice_answer("D", ["A", "B", "C"])[1] == "strict_format_not_exact_key"
    assert parse_choice_answer("", ["A", "B", "C"])[1] == "empty_answer"
    assert parse_choice_answer(None, ["A", "B", "C"])[1] == "empty_answer"


def test_extract_choice_content_structural_errors():
    assert extract_choice_content(_fixture_body()) == ("B", None)
    assert extract_choice_content({"choices": []})[1] == "missing_choices"
    body = _fixture_body()
    body["choices"][0]["message"]["content"] = None
    assert extract_choice_content(body)[1] == "empty_answer"


def test_provider_ok_path_records_usage_model_and_raw(tmp_path):
    transport = ScriptedTransport([{"status_code": 200,
                                    "body": _fixture_body(model="vendor/ACTUAL-x")}])
    provider = OpenRouterProvider(transport, wire=WIRE)
    outcome = provider.ask(REQUEST, "native:item1", store=None)
    assert outcome.status == "ok"
    assert outcome.n_retries == 0 and len(outcome.attempts) == 1
    assert outcome.usage_input_tokens == 42
    assert outcome.usage_output_tokens == 3
    # model requested vs returned are DISTINCT recorded facts
    assert outcome.attempts[0].model_requested == "vendor/model-x"
    assert outcome.validated.returned_model == "vendor/ACTUAL-x"
    assert outcome.validated.answers["q"].values["choice"] == "B"


def test_strict_format_failure_counted_never_retried(tmp_path):
    transport = ScriptedTransport([{"status_code": 200,
                                    "body": _fixture_body(content="I think it is B.")}])
    provider = OpenRouterProvider(transport, wire=WIRE)
    outcome = provider.ask(REQUEST, "native:item1")
    assert outcome.status == "contract_invalid"
    assert len(outcome.attempts) == 1  # no retry-on-wrong, ever
    assert outcome.validated.codes == ["strict_format_not_exact_key"]
    assert outcome.validated.usable is False


def test_http_error_and_timeout_are_recorded_truth():
    provider = OpenRouterProvider(
        ScriptedTransport([{"status_code": 429, "body": {"error": "rate limited"}}]),
        wire=WIRE)
    outcome = provider.ask(REQUEST, "native:item1")
    assert outcome.status == "http_error"
    assert outcome.attempts[0].http_status == 429
    provider = OpenRouterProvider(
        ScriptedTransport([{"error": "timeout:ReadTimeout"}]), wire=WIRE)
    outcome = provider.ask(REQUEST, "native:item1")
    assert outcome.status == "timeout"
    assert outcome.attempts[0].uncertain is True


def test_raw_request_and_response_persisted(tmp_path):
    from jev_observatory.ledger import RunStore
    from jev_observatory.redact import Redactor

    store = RunStore(tmp_path, "run-smoke", redactor=Redactor())
    provider = OpenRouterProvider(ScriptedTransport([{"status_code": 200,
                                                      "body": _fixture_body()}]),
                                  wire=WIRE)
    provider.ask(REQUEST, "native:item1", store=store)
    raw = sorted((tmp_path / "run-smoke" / "raw").iterdir())
    assert any(p.name.endswith("request.json") for p in raw)
    response = next(p for p in raw if p.name.endswith("response.json"))
    assert json.loads(response.read_text())["model"] == "vendor/model-x"
    request = next(p for p in raw if p.name.endswith("request.json"))
    assert json.loads(request.read_text())["max_tokens"] == 64


def test_cancellation_before_dispatch():
    from jev_observatory.guards import Cancellation

    cancellation = Cancellation()
    cancellation.cancel()
    provider = OpenRouterProvider(ScriptedTransport([]), wire=WIRE)
    outcome = provider.ask(REQUEST, "native:item1", cancellation=cancellation)
    assert outcome.status == "cancelled"
    assert outcome.attempts == []


def test_multi_question_requests_are_refused():
    request = SystemOneRequest(
        state="s", model="m",
        questions={
            "q1": ChoiceQuestion(type="choice", instructions="i",
                                 criteria={"A": "1", "B": "2"}),
            "q2": ChoiceQuestion(type="choice", instructions="i",
                                 criteria={"A": "3", "B": "4"}),
        })
    with pytest.raises(OpenRouterError, match="exactly one question"):
        build_chat_payload(request, WIRE)


def test_smoke_dry_run_script(tmp_path, capsys):
    """The one-call-per-model smoke path runs fully offline on fixtures."""
    import subprocess
    import sys

    root = tmp_path / "runs_benchmark"
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark/smoke_openrouter.py",
         "--dry-run", "--root", str(root), "--models", "vendor/model-x,vendor/model-y"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert completed.returncode == 0, completed.stderr
    reports = sorted((root / "smoke").glob("openrouter_smoke_*.json"))
    assert len(reports) == 1
    doc = json.loads(reports[0].read_text())
    assert doc["mode"] == "dry-run-fixtures"
    assert [m["model_requested"] for m in doc["models"]] == \
        ["vendor/model-x", "vendor/model-y"]
    for model in doc["models"]:
        assert model["chat_call"]["status"] == "ok"
        assert model["chat_call"]["choice"] == "B"
        assert model["chat_call"]["usage"] == {"input_tokens": 120, "output_tokens": 2}


def test_smoke_dry_run_counts_strict_format_failures(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "runs_benchmark"
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark/smoke_openrouter.py",
         "--dry-run", "--root", str(root), "--models", "vendor/model-x",
         "--fail-fixture"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert completed.returncode == 0, completed.stderr
    doc = json.loads(next((root / "smoke").glob("openrouter_smoke_*.json")).read_text())
    call = doc["models"][0]["chat_call"]
    assert call["strict_format_failure"] is True
    assert call["status"] == "contract_invalid"
    assert "strict_format_not_exact_key" in call["violations"]


def test_live_transport_refuses_without_opt_in(monkeypatch):
    from jev_observatory.openrouter import OpenRouterHttpTransport

    monkeypatch.delenv("JEVO_ALLOW_LIVE", raising=False)
    with pytest.raises(OpenRouterError, match="JEVO_ALLOW_LIVE"):
        OpenRouterHttpTransport("fake-key-for-constructor-guard")
