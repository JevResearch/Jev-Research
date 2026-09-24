"""Provider/transport tests: statuses, retries, timing mode, uncertainty, mock."""

import json

import pytest

from jev_observatory.providers import JevProvider, MockProvider, RetryPolicy, NullHook
from jev_observatory.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from jev_observatory.transport import ReplayTransport, ScriptedTransport, TransportResponse


def _request():
    return SystemOneRequest(
        state="ticket text",
        questions={"dep": ChoiceQuestion(instructions="team?", criteria={"billing": "money", "tech": "bugs"})},
    )


def _ok_body(input_tokens=42):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "dep": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.4, "tech": 0.6}, "confidence": 0.2}
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 7},
    }


def _response(body, status=200, **kwargs):
    return TransportResponse(status_code=status, headers=kwargs.pop("headers", {}),
                             body=json.dumps(body).encode(), total_ms=kwargs.pop("total_ms", 1.0), **kwargs)


def _ok_response(input_tokens=42):
    return _response(_ok_body(input_tokens))


# --------------------------------------------------------------------- statuses
@pytest.mark.parametrize("status", [401, 422])
def test_no_retry_on_auth_and_validation_errors(status):
    """401/422 are definitive: exactly one attempt, no hidden retry."""
    provider = JevProvider(ScriptedTransport([_response({}, status=status)]),
                           retry_policy=RetryPolicy.operational(5))
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "http_error"
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].http_status == status


@pytest.mark.parametrize("status", [429, 529])
def test_retry_on_transient_statuses(status):
    script = [TransportResponse(status_code=status, headers={}, body=b"", total_ms=1.0),
              TransportResponse(status_code=status, headers={}, body=b"", total_ms=1.0),
              TransportResponse(status_code=200, headers={}, body=json.dumps(_ok_body()).encode(), total_ms=1.0)]
    provider = JevProvider(ScriptedTransport(script), retry_policy=RetryPolicy.operational(3))
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "ok"
    assert outcome.n_retries == 2
    assert len(outcome.attempts) == 3
    assert outcome.attempts[-1].http_status == 200
    assert outcome.attempts[0].outcome == "http_error"


def test_retry_after_header_recorded_and_honoured():
    recorded = []
    provider = JevProvider(
        ScriptedTransport([TransportResponse(status_code=429, headers={"Retry-After": "7"}, body=b"", total_ms=1.0),
                           TransportResponse(status_code=200, headers={}, body=json.dumps(_ok_body()).encode(), total_ms=1.0)]),
        retry_policy=RetryPolicy.operational(2),
        sleeper=lambda s: recorded.append(s),
    )
    outcome = provider.ask(_request(), "lid")
    assert recorded == [7.0]
    assert outcome.attempts[0].retry_after_seconds == 7.0


def test_transport_failure_retry_then_give_up():
    provider = JevProvider(ScriptedTransport([{"error": "transport_error:ConnectError"}] * 4),
                           retry_policy=RetryPolicy.operational(3))
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "transport_error"
    assert len(outcome.attempts) == 3  # bounded
    assert all(a.usage_input_tokens is None for a in outcome.attempts)


def test_timeout_is_uncertain_and_non_terminal():
    provider = JevProvider(ScriptedTransport([{"error": "timeout:ReadTimeout", "total_ms": 30000.0}]),
                           retry_policy=RetryPolicy.none())
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "timeout"
    assert outcome.attempts[0].uncertain is True
    assert not outcome.terminal  # may have been processed and billed remotely


def test_malformed_and_truncated_bodies():
    provider = JevProvider(ScriptedTransport([{"status_code": 200, "body": b"{not json"}]),
                           retry_policy=RetryPolicy.none())
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "malformed_json"
    assert outcome.attempts[0].outcome == "malformed_json"

    provider2 = JevProvider(ScriptedTransport([{"status_code": 200, "body": b""}]), retry_policy=RetryPolicy.none())
    outcome2 = provider2.ask(_request(), "lid2")
    assert outcome2.status == "malformed_json"
    assert outcome2.attempts[0].error == "empty_body"


def test_timing_mode_has_zero_hidden_retries():
    provider = JevProvider(ScriptedTransport([{"error": "transport_error:ConnectError"}]),
                           retry_policy=RetryPolicy.none())
    outcome = provider.ask(_request(), "lid")
    assert len(outcome.attempts) == 1
    assert outcome.n_retries == 0
    assert outcome.attempts[0].uncertain is False


def test_contract_invalid_is_terminal_but_flagged():
    provider = MockProvider(faults={"lid": {"body": {
        "model": "m",
        "answers": {"dep": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.6, "tech": 0.4}, "confidence": 0.2}},
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }}})
    outcome = provider.ask(_request(), "lid")
    assert outcome.status == "contract_invalid"
    assert outcome.terminal
    assert "choice_not_argmax" in outcome.violation_codes()


# ------------------------------------------------------------- usage accounting
def test_usage_recorded_per_attempt():
    from jev_observatory.guards import BudgetCaps, BudgetLedger

    provider = JevProvider(ScriptedTransport([{"status_code": 200, "body": _ok_body(input_tokens=123)}]),
                           retry_policy=RetryPolicy.none())
    ledger = BudgetLedger(BudgetCaps())

    class Hook:
        def before(self, estimated):
            return ledger.reserve(estimated)

        def after(self, handle, reported):
            ledger.reconcile(handle, reported)

    outcome = provider.ask(_request(), "lid", hook=Hook())
    assert outcome.usage_input_tokens == 123
    assert outcome.usage_output_tokens == 7
    assert ledger.billed_tokens == 123
    assert abs(ledger.billed_cost - 123 * 0.042 / 1_000_000) < 1e-12


def test_error_usage_stays_unknown_not_zero():
    provider = JevProvider(ScriptedTransport([{"status_code": 401, "body": b'{"error": "bad key"}'}]),
                           retry_policy=RetryPolicy.none())
    outcome = provider.ask(_request(), "lid")
    assert outcome.usage_input_tokens is None
    assert outcome.usage_output_tokens is None


def _with_ledger(provider):
    from jev_observatory.guards import BudgetCaps, BudgetLedger

    ledger = BudgetLedger(BudgetCaps())

    class Hook:
        def __init__(self, ledger):
            self.ledger = ledger

        def before(self, estimated):
            return self.ledger.reserve(estimated)

        def after(self, handle, reported):
            self.ledger.reconcile(handle, reported)

    provider._test_hook = Hook(ledger)  # available for explicit wiring in tests
    return provider, ledger


# ------------------------------------------------------------------- raw stores
def test_raw_request_and_response_persisted(tmp_path):
    from jev_observatory.ledger import RunStore
    from jev_observatory.redact import Redactor

    store = RunStore(tmp_path / "runs", "r1", redactor=Redactor())
    provider = MockProvider()
    outcome = provider.ask(_request(), "native:item-1", store=store)
    attempt_id = outcome.attempts[0].attempt_id
    raw_request = store.directory / "raw" / f"{attempt_id}.request.json"
    raw_response = store.directory / "raw" / f"{attempt_id}.response.json"
    assert raw_request.exists() and "ticket text" in raw_request.read_text()
    assert raw_response.exists() and "choice" in raw_response.read_text()


def test_mock_is_deterministic():
    a = MockProvider(seed=1)._respond(_request())
    b = MockProvider(seed=1)._respond(_request())
    assert a == b
    c = MockProvider(seed=2)._respond(_request())
    assert a["answers"]["dep"] != c["answers"]["dep"] or a["usage"] != c["usage"]


def test_mock_respects_option_map_for_255_options():
    request = SystemOneRequest(
        state="s",
        questions={"big": ChoiceQuestion(instructions="pick", criteria={f"o{i:03d}": f"d{i}" for i in range(255)})},
    )
    body = MockProvider()._respond(request)
    assert set(body["answers"]["big"]["probabilities"]) == set(request.option_maps()["big"]["options"])
    assert abs(sum(body["answers"]["big"]["probabilities"].values()) - 1.0) < 1e-5
    best = max(body["answers"]["big"]["probabilities"], key=body["answers"]["big"]["probabilities"].get)
    assert body["answers"]["big"]["choice"] == best


# ------------------------------------------------------------------ replay
def test_record_then_replay_is_byte_identical(tmp_path):
    from jev_observatory.transport import RecordingTransport

    mock_body = MockProvider(seed=3)._respond(_request())
    recorded_dir = tmp_path / "recordings"
    scripted = ScriptedTransport([{"status_code": 200, "body": mock_body},
                                  {"status_code": 200, "body": mock_body}])  # identical retry
    recording = RecordingTransport(scripted, recorded_dir)
    payload = _request().to_payload()
    first = recording_post(recording, payload)
    again = recording_post(recording, payload)
    assert first.status_code == again.status_code == 200

    replay = ReplayTransport(recorded_dir)
    replayed = replay.post("/v1/systemone", payload)
    replayed_again = replay.post("/v1/systemone", payload)  # replay honours attempt order
    assert json.loads(replayed.body) == mock_body
    assert json.loads(replayed_again.body) == mock_body


def test_replay_miss_raises_never_hits_network(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(KeyError):
        ReplayTransport(empty).post("/v1/systemone", {"state": "x", "model": "m", "questions": {}})


def recording_post(transport, payload):
    return transport.post("/v1/systemone", payload)


def test_scripted_transport_exhaustion_is_assertion():
    transport = ScriptedTransport([])
    with pytest.raises(AssertionError):
        transport.post("/x", {})