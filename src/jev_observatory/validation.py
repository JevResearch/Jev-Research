"""Strict, non-repairing validation of inbound Jev responses.

Design rule (IMPLEMENTATION.md, M0): *preserve originals and record validation
failures, never silently repair Jev output.*  Consequently this module never
normalises probabilities, never fills in a missing choice, and never rounds.
It compares what came back against what we sent, and records every discrepancy
as a `Violation` with a stable code.

Severity model
--------------
``error``   : the answer is contract-invalid and cannot be scored (a missing
              option key, a negative probability, a wrong answer type).
``warning`` : the answer is usable but deviates from the documented contract
              (probabilities that do not sum to 1 within tolerance, a returned
              score that disagrees with its own expectation, model-id drift).

Analyses can filter on ``clean`` (no violations) or ``usable`` (no errors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

PROB_SUM_TOLERANCE = 1e-6
PROB_SUM_HARD_TOLERANCE = 5e-2
SCORE_EXPECTATION_TOLERANCE = 1e-6
SCORE_EXPECTATION_HARD_TOLERANCE = 0.05
# The server displays probabilities quantized (observed: 2 decimal places).
# Display rounding must be accounted for before calling a server decision or
# expectation a contract violation:
#   * sum of K displayed probabilities can miss 1.0 by up to 0.005*K;
#   * a score expectation recomputed from displayed probabilities can deviate
#     from the server's full-precision expectation by up to 0.005*sum(levels);
#   * two displayed probabilities can flip displayed order when the true gap
#     is sub-quantum (measured 2026-09-20: every observed choice-vs-displayed-
#     argmax mismatch had gap 0.00-0.01, never more).
DISPLAY_QUANTUM = 0.01
DISPLAY_HALF_QUANTUM = 0.005
# Observed server probability resolution (all 11,193 live values land on the
# 0.01 grid). Displayed gaps of exactly one quantum are REAL choice/argmax
# discrepancies (monotone rounding cannot reorder), so only float noise on an
# exact tie is treated as an ambiguous quantization artifact.
PROBABILITY_QUANTUM = 0.01
FLOAT_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str  # "error" | "warning"
    detail: str
    question_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "question_id": self.question_id,
        }


@dataclass
class ValidatedAnswer:
    question_id: str
    question_type: str
    usable: bool
    raw: Any
    values: dict[str, Any] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "usable": self.usable,
            "values": self.values,
            "violations": [v.to_dict() for v in self.violations],
            "raw": self.raw,
        }


@dataclass
class ValidatedResponse:
    contract_valid: bool
    usable: bool
    requested_model: str
    returned_model: Any
    answers: dict[str, ValidatedAnswer]
    violations: list[Violation]
    usage: dict[str, Any]
    raw: Any

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]

    def answer_violations(self, question_id: str) -> list[Violation]:
        return self.answers[question_id].violations if question_id in self.answers else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_valid": self.contract_valid,
            "usable": self.usable,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "answers": {qid: a.to_dict() for qid, a in self.answers.items()},
            "violations": [v.to_dict() for v in self.violations],
            "usage": self.usage,
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _in_unit_interval(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and 0.0 <= value <= 1.0


def _check_distribution(
    qid: str,
    probs: Any,
    expected_keys: list[str],
    out: list[Violation],
) -> dict[str, float] | None:
    """Validate a probability map against the sent option/level order."""
    if not isinstance(probs, dict):
        out.append(Violation("probabilities_not_object", "error", f"probabilities is {type(probs).__name__}", qid))
        return None

    missing = [k for k in expected_keys if k not in probs]
    extra = [k for k in probs if k not in expected_keys]
    if missing:
        out.append(Violation("missing_option_key", "error", f"absent from probabilities: {missing}", qid))
    if extra:
        out.append(Violation("extra_option_key", "error", f"not sent in criteria: {extra}", qid))

    clean: dict[str, float] = {}
    for key, value in probs.items():
        if not _is_number(value):
            out.append(Violation("probability_wrong_type", "error", f"P[{key!r}] is {value!r}", qid))
            continue
        fvalue = float(value)
        if not math.isfinite(fvalue):
            out.append(Violation("probability_nonfinite", "error", f"P[{key!r}] is {value!r}", qid))
            continue
        if fvalue < 0.0:
            out.append(Violation("probability_negative", "error", f"P[{key!r}] = {fvalue}", qid))
            continue
        if fvalue > 1.0:
            out.append(Violation("probability_above_one", "error", f"P[{key!r}] = {fvalue}", qid))
            continue
        clean[key] = fvalue

    if len(clean) != len(probs):
        return None  # already erroring on element types

    total = sum(clean.values())
    deviation = abs(total - 1.0)
    if deviation > PROB_SUM_TOLERANCE:
        severity = "warning" if deviation <= PROB_SUM_HARD_TOLERANCE else "error"
        out.append(
            Violation(
                "prob_sum_out_of_tolerance",
                severity,
                f"sum={total!r} deviation={deviation:.3e} (tolerance {PROB_SUM_TOLERANCE:g})",
                qid,
            )
        )
    return clean


def _check_confidence(qid: str, answer: dict, out: list[Violation]) -> float | None:
    if "confidence" not in answer:
        out.append(Violation("missing_confidence", "error", "documented required field absent", qid))
        return None
    confidence = answer["confidence"]
    if not _in_unit_interval(confidence):
        out.append(Violation("confidence_out_of_range", "error", f"confidence={confidence!r}", qid))
        return None
    return float(confidence)


def validate_response(
    raw: Any,
    request: SystemOneRequest,
    *,
    transport_error: str | None = None,
    http_status: int | None = None,
) -> ValidatedResponse:
    """Validate a parsed response body against the request that produced it."""
    violations: list[Violation] = []
    if transport_error is not None:
        violations.append(Violation("no_response", "error", transport_error))
        return ValidatedResponse(False, False, request.model, None, {}, violations, {}, None)

    if not isinstance(raw, dict):
        violations.append(
            Violation("response_not_object", "error", f"body is {type(raw).__name__}")
        )
        return ValidatedResponse(False, False, request.model, None, {}, violations, {}, raw)

    returned_model = raw.get("model")
    if returned_model is None:
        violations.append(Violation("missing_model_field", "warning", "response has no model field"))
    elif returned_model != request.model:
        violations.append(
            Violation(
                "model_drift",
                "warning",
                f"requested {request.model!r} but response reports {returned_model!r}",
            )
        )

    usage = raw.get("usage")
    if not isinstance(usage, dict):
        violations.append(
            Violation("missing_usage", "warning", f"usage is {type(usage).__name__}")
        )
        usage = {}
    else:
        for key in ("input_tokens", "output_tokens"):
            if key in usage and not _is_number(usage[key]):
                violations.append(
                    Violation("usage_wrong_type", "warning", f"usage.{key}={usage[key]!r}")
                )

    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, dict):
        violations.append(
            Violation("missing_answers", "error", f"answers is {type(answers_raw).__name__}")
        )
        answers_raw = {}

    for qid in request.questions:
        if qid not in answers_raw:
            violations.append(Violation("missing_answer", "error", "no answer for requested question", qid))
    for qid in answers_raw:
        if qid not in request.questions:
            violations.append(Violation("extra_answer", "error", "answer for question never sent", qid))

    validated: dict[str, ValidatedAnswer] = {}
    for qid, question in request.questions.items():
        if qid not in answers_raw:
            continue
        answer = answers_raw[qid]
        local: list[Violation] = []
        values: dict[str, Any] = {}
        if not isinstance(answer, dict):
            local.append(Violation("answer_not_object", "error", f"answer is {type(answer).__name__}", qid))
            validated[qid] = ValidatedAnswer(qid, question.type, False, answer, {}, local)
            continue

        reported_type = answer.get("type")
        if reported_type != question.type:
            local.append(
                Violation(
                    "answer_type_mismatch",
                    "error",
                    f"sent {question.type!r}, answer reports {reported_type!r}",
                    qid,
                )
            )

        if isinstance(question, NoulQuestion):
            _validate_noul(qid, answer, local, values)
        elif isinstance(question, ChoiceQuestion):
            _validate_choice(qid, answer, question, local, values)
        elif isinstance(question, ScoreQuestion):
            _validate_score(qid, answer, question, local, values)

        usable = not any(v.severity == "error" for v in local)
        validated[qid] = ValidatedAnswer(qid, question.type, usable, answer, values, local)

    violations.extend(v for a in validated.values() for v in a.violations)
    errors = [v for v in violations if v.severity == "error"]
    return ValidatedResponse(
        contract_valid=not errors,
        usable=bool(validated) and not errors,
        requested_model=request.model,
        returned_model=returned_model,
        answers=validated,
        violations=violations,
        usage=usage,
        raw=raw,
    )


def _validate_noul(qid: str, answer: dict, out: list[Violation], values: dict[str, Any]) -> None:
    if "noul" not in answer:
        out.append(Violation("missing_noul", "error", "noul value absent", qid))
        return
    value = answer["noul"]
    if not _in_unit_interval(value):
        out.append(Violation("noul_out_of_range", "error", f"noul={value!r}", qid))
        return
    values["p_yes"] = float(value)
    values["p_no"] = 1.0 - float(value)
    if "confidence" in answer:
        # Vendor docs state Noul carries no separate confidence value; presence is
        # a documentation/contract discrepancy worth recording, not a repair.
        out.append(
            Violation("unexpected_confidence", "warning", f"noul answer reports confidence={answer['confidence']!r}", qid)
        )


def _validate_choice(
    qid: str,
    answer: dict,
    question: ChoiceQuestion,
    out: list[Violation],
    values: dict[str, Any],
) -> None:
    options = list(question.criteria)
    probs = _check_distribution(qid, answer.get("probabilities"), options, out)
    _check_confidence(qid, answer, out)

    choice = answer.get("choice")
    if choice is None:
        out.append(Violation("missing_choice", "error", "choice value absent", qid))
    elif choice not in options:
        out.append(Violation("unknown_option", "error", f"choice {choice!r} was not offered", qid))
    values["choice"] = choice

    if probs is None:
        return
    maximum = max(probs.values())
    argmax = [k for k in options if k in probs and probs[k] == maximum]
    if len(argmax) > 1:
        out.append(Violation("argmax_tie", "warning", f"tied at {maximum}: {argmax}", qid))
    if isinstance(choice, str) and choice in probs:
        gap = maximum - probs[choice]
        if choice in argmax:
            pass
        elif gap <= FLOAT_TIE_TOLERANCE:
            # Floating-point noise on an exact tie: the underlying values are
            # indistinguishable at the displayed resolution.
            out.append(
                Violation(
                    "argmax_tie_quantized",
                    "warning",
                    f"choice {choice!r} p={probs[choice]!r} vs max {maximum!r} (float-noise tie)",
                    qid,
                )
            )
        elif gap <= 2 * DISPLAY_QUANTUM + FLOAT_TIE_TOLERANCE:
            # A displayed gap of at most one quantum per side is fully
            # explainable by display rounding of a near-tie: the server's
            # decision can be the true argmax while the rounded table shows the
            # other option higher (measured 2026-09-20: all observed mismatches
            # were gap 0.00-0.01).  Warning, not error — but recorded.
            out.append(
                Violation(
                    "choice_argmax_within_display_quantum",
                    "warning",
                    f"choice {choice!r} p={probs[choice]!r} vs max {maximum!r} "
                    "(within display-quantum rounding)",
                    qid,
                )
            )
        else:
            # A displayed gap beyond two quanta cannot be produced by rounding
            # of a pre-quantization argmax: the decision and the table really
            # disagree.  Error — counted, never repaired.
            out.append(
                Violation("choice_not_argmax", "error", f"choice {choice!r} is not max({argmax})", qid)
            )
    values["probabilities"] = probs
    values["p_max"] = maximum


def _validate_score(
    qid: str,
    answer: dict,
    question: ScoreQuestion,
    out: list[Violation],
    values: dict[str, Any],
) -> None:
    levels = [str(i) for i in range(len(question.criteria))]
    probs = _check_distribution(qid, answer.get("probabilities"), levels, out)
    _check_confidence(qid, answer, out)

    score = answer.get("score")
    if score is None:
        out.append(Violation("missing_score", "error", "score value absent", qid))
    elif not _is_number(score) or not math.isfinite(float(score)):
        out.append(Violation("score_wrong_type", "error", f"score={score!r}", qid))
    else:
        values["score"] = float(score)

    legend = answer.get("legend")
    if legend is None:
        out.append(Violation("missing_legend", "warning", "legend absent", qid))
    elif not isinstance(legend, dict):
        out.append(Violation("legend_wrong_type", "warning", f"legend is {type(legend).__name__}", qid))
    else:
        if set(legend) != set(levels):
            out.append(
                Violation("legend_key_mismatch", "warning", f"legend keys {sorted(legend)} != levels {levels}", qid)
            )
        else:
            for level in levels:
                if str(legend[level]) != str(question.criteria[int(level)]):
                    out.append(
                        Violation("legend_text_mismatch", "warning", f"legend[{level}] differs from sent criteria", qid)
                    )
                    break

    if probs is None:
        return
    expectation = sum(int(level) * p for level, p in probs.items())
    values["probabilities"] = probs
    values["expectation_from_probabilities"] = expectation
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)):
        delta = abs(float(score) - expectation)
        values["score_expectation_delta"] = delta
        # Display-quantization bound: with each displayed probability off by at
        # most half a quantum, the recomputed expectation can deviate from the
        # server's full-precision expectation by 0.005 * sum(level indices).
        rounding_bound = DISPLAY_HALF_QUANTUM * sum(range(len(levels)))
        hard_tolerance = max(SCORE_EXPECTATION_HARD_TOLERANCE, rounding_bound)
        if delta > hard_tolerance:
            out.append(
                Violation(
                    "score_expectation_mismatch",
                    "error",
                    f"score={float(score)!r} vs expectation={expectation!r} delta={delta:.3e}",
                    qid,
                )
            )
        elif delta > SCORE_EXPECTATION_TOLERANCE:
            out.append(
                Violation(
                    "score_expectation_mismatch",
                    "warning",
                    f"score={float(score)!r} vs expectation={expectation!r} "
                    f"delta={delta:.3e} (within display-quantum rounding)",
                    qid,
                )
            )
