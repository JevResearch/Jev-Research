"""Validation tests: every violation is recorded, nothing is repaired."""

import json
import math

from jev_observatory.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from jev_observatory.validation import validate_response


def _request(**questions):
    return SystemOneRequest(state="state", questions=questions)


def _choice_request():
    return _request(c=ChoiceQuestion(instructions="team?", criteria={"billing": "money", "tech": "bugs"}))


def _codes(validated):
    return [v.code for v in validated.violations]


def test_clean_response_has_no_violations():
    request = _request(
        c=ChoiceQuestion(instructions="team?", criteria={"billing": "money", "tech": "bugs"}),
        n=NoulQuestion(instructions="urgent?"),
    )
    raw = {
        "model": "jev-1.13.0",
        "answers": {
            "c": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.4, "tech": 0.6}, "confidence": 0.2},
            "n": {"type": "noul", "noul": 0.92},
        },
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    validated = validate_response(raw, request)
    assert validated.contract_valid and validated.usable
    assert validated.violations == []
    assert validated.answers["c"].values["choice"] == "tech"
    assert validated.answers["n"].values["p_yes"] == 0.92


def test_raw_answer_preserved_exactly():
    request = _request(n=NoulQuestion(instructions="q"))
    raw = {"model": "m", "answers": {"n": {"type": "noul", "noul": 0.5, "extra": "kept"}}}
    validated = validate_response(raw, request)
    assert validated.answers["n"].raw is raw["answers"]["n"]  # same object, unmodified
    assert validated.answers["n"].raw["extra"] == "kept"


def test_missing_and_extra_option_keys():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 1.0}}}
    validated = validate_response(raw, request)
    assert "missing_option_key" in validated.codes
    # Extra key case
    raw2 = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.4, "c": 0.1}, "confidence": 0.1}}}
    validated2 = validate_response(raw2, request)
    assert "extra_option_key" in validated2.codes
    assert not validated2.usable


def test_wrong_types_recorded_not_repaired():
    request = _request(n=NoulQuestion(instructions="q"))
    raw = {"answers": {"n": {"type": "noul", "noul": "high"}}}  # string, not number
    validated = validate_response(raw, request)
    assert "noul_out_of_range" in validated.codes
    assert validated.answers["n"].values == {}  # nothing was coerced
    assert validated.answers["n"].raw["noul"] == "high"


def test_nonfinite_probability_detected():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    body = '{"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": NaN, "b": 0.5}, "confidence": 0.5}}}'
    raw = json.loads(body)  # stdlib parser accepts NaN literals
    validated = validate_response(raw, request)
    assert "probability_nonfinite" in validated.codes
    assert not validated.usable


def test_negative_and_above_one_probabilities():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": -0.2, "b": 1.2}, "confidence": 0.9}}}
    validated = validate_response(raw, request)
    assert "probability_negative" in validated.codes
    assert "probability_above_one" in validated.codes


def test_probability_sum_out_of_tolerance():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.3}, "confidence": 0.2}}}
    validated = validate_response(raw, request)
    assert "prob_sum_out_of_tolerance" in validated.codes
    deviation_warning = [v for v in validated.violations if v.code == "prob_sum_out_of_tolerance"]
    assert deviation_warning[0].severity == "error"  # 0.1 off is beyond the hard tolerance


def test_mild_probability_sum_deviation_is_warning():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.52, "b": 0.49}, "confidence": 0.04}}}
    validated = validate_response(raw, request)
    entry = [v for v in validated.violations if v.code == "prob_sum_out_of_tolerance"]
    assert entry and entry[0].severity == "warning"
    assert validated.usable  # recorded, values still usable, never repaired
    assert validated.answers["c"].values["probabilities"] == {"a": 0.52, "b": 0.49}


def test_argmax_tie_is_warning_and_choice_must_be_argmax():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    tie = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.0}}}
    validated = validate_response(tie, request)
    assert "argmax_tie" in validated.codes
    assert validated.usable
    bad = {"answers": {"c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.8}}}
    validated2 = validate_response(bad := bad, request) if False else validate_response(bad, request)
    assert "choice_not_argmax" in validated2.codes and not validated2.usable


def test_noul_out_of_range():
    request = _request(n=NoulQuestion(instructions="q"))
    validated = validate_response({"answers": {"n": {"type": "noul", "noul": 1.5}}}, request)
    assert "noul_out_of_range" in validated.codes


def test_noul_with_confidence_is_a_documented_discrepancy():
    request = _request(n=NoulQuestion(instructions="q"))
    validated = validate_response({"answers": {"n": {"type": "noul", "noul": 0.9, "confidence": 0.8}}}, request)
    assert "unexpected_confidence" in validated.codes
    assert validated.usable


def test_choice_without_confidence_is_error():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.0}}}}
    validated = validate_response(raw, request)
    assert "missing_confidence" in validated.codes
    assert not validated.usable


def test_score_expectation_mismatch():
    request = _request(s=ScoreQuestion(instructions="?", criteria=["low", "high"]))
    raw = {"answers": {"s": {"type": "score", "score": 0.9, "legend": {"0": "low", "1": "high"},
                             "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.0}}}
    validated = validate_response(raw, request)
    codes = validated.answer_violations("s")
    mismatch = [v for v in codes if v.code == "score_expectation_mismatch"]
    assert mismatch and mismatch[0].severity == "error"  # delta 0.4 > 0.05 hard tolerance
    assert validated.answers["s"].values["score_expectation_delta"] == 0.4


def test_score_expectation_small_delta_is_warning():
    request = _request(s=ScoreQuestion(instructions="?", criteria=["low", "mid", "high"]))
    raw = {"answers": {"s": {"type": "score", "score": 1.02, "legend": {"0": "low", "1": "mid", "2": "high"},
                             "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": 1.0}}}
    validated = validate_response(raw, request)
    entries = [v for v in validated.violations if v.code == "score_expectation_mismatch"]
    assert entries and entries[0].severity == "warning"
    assert validated.usable


def test_legend_mismatch_is_warning():
    request = _request(s=ScoreQuestion(instructions="?", criteria=["low", "high"]))
    raw = {"answers": {"s": {"type": "score", "score": 0.0, "legend": {"0": "LOW!", "1": "high"},
                             "probabilities": {"0": 1.0, "1": 0.0}, "confidence": 1.0}}}
    validated = validate_response(raw, request)
    assert "legend_text_mismatch" in validated.codes
    assert validated.usable


def test_missing_answer_and_extra_answer():
    request = _request(a=NoulQuestion(instructions="q1"), b=NoulQuestion(instructions="q2"))
    raw = {"answers": {"a": {"type": "noul", "noul": 0.5}, "z": {"type": "noul", "noul": 0.1}}}
    validated = validate_response(raw, request)
    assert "missing_answer" in validated.codes
    assert "extra_answer" in validated.codes
    assert not validated.usable


def test_answer_type_mismatch():
    request = _request(n=NoulQuestion(instructions="q"))
    raw = {"answers": {"n": {"type": "choice", "noul": 0.5}}}
    validated = validate_response(raw, request)
    assert "answer_type_mismatch" in validated.codes


def test_unknown_option_in_choice():
    request = _request(c=ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"}))
    raw = {"answers": {"c": {"type": "choice", "choice": "surprise", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.0}}}
    validated = validate_response(raw, request)
    assert "unknown_option" in validated.codes


def test_model_drift_is_warning():
    request = _request(n=NoulQuestion(instructions="q"))
    raw = {"model": "jev-1.14.0", "answers": {"n": {"type": "noul", "noul": 0.5}}}
    validated = validate_response(raw, request)
    assert "model_drift" in validated.codes
    drift = [v for v in validated.violations if v.code == "model_drift"]
    assert drift[0].severity == "warning"
    assert validated.usable


def test_transport_error_produces_no_response_violation():
    request = _request(n=NoulQuestion(instructions="q"))
    validated = validate_response(None, request, transport_error="timeout:ReadTimeout")
    assert validated.codes == ["no_response"]
    assert not validated.usable
    assert validated.raw is None


def test_non_object_body():
    request = _request(n=NoulQuestion(instructions="q"))
    validated = validate_response([1, 2, 3], request)
    assert "response_not_object" in validated.codes


def test_usage_wrong_type_flagged():
    request = _request(n=NoulQuestion(instructions="q"))
    raw = {"answers": {"n": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": "ten"}}
    validated = validate_response(raw, request)
    assert "usage_wrong_type" in validated.codes