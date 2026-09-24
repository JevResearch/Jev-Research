"""Typed Jev API contract.

Shapes are taken from https://docs.typesafe.ai/api.md (design-phase capture,
SOURCES.md S3-S6).  This module models *outbound requests* and gives a
deliberately *lenient* inbound parser: inbound JSON is kept raw and validated
separately (see ``jev_observatory.validation``) so that nothing is silently
repaired or normalised.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Contract limits from vendor documentation.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
DEFAULT_MODEL_PIN = "jev-1.13.0"
DOC_STATE_MAX_TOKENS = 32_000
DOC_TOTAL_MAX_TOKENS = 64_000

Instruction = Union[str, dict[str, Any], list[Any]]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoulQuestion(_Strict):
    type: Literal["noul"] = "noul"
    instructions: Instruction
    criteria: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "NoulQuestion":
        if self.criteria is not None and set(self.criteria) - {"true", "false"}:
            raise ValueError("noul criteria keys must be a subset of {true, false}")
        return self


class ChoiceQuestion(_Strict):
    type: Literal["choice"] = "choice"
    instructions: Instruction
    criteria: dict[str, Union[str, None]]

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"choice supports at most {MAX_CHOICE_OPTIONS} options, got {len(self.criteria)}"
            )
        if len(self.criteria) < 2:
            raise ValueError("choice needs at least 2 options")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("duplicate choice option keys")
        return self

    @property
    def options(self) -> list[str]:
        """Ordered option map key -> description (order is significant)."""
        return list(self.criteria)


class ScoreQuestion(_Strict):
    type: Literal["score"] = "score"
    instructions: Instruction
    criteria: list[Union[str, dict[str, Any], list[Any]]]

    @model_validator(mode="after")
    def _check(self) -> "ScoreQuestion":
        levels = len(self.criteria)
        if not MIN_SCORE_LEVELS <= levels <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, got {levels}"
            )
        return self

    @property
    def levels(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(_Strict):
    """Outbound payload for POST /v1/systemone."""

    state: Union[str, dict[str, Any], list[Any]]
    model: str = DEFAULT_MODEL_PIN
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _check(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("questions must be non-empty")
        for key in self.questions:
            if not key.strip():
                raise ValueError("question ids must be non-empty")
        return self

    # ---- derived measurements (never sent to the model) ----
    def canonical_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def request_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def measures(self) -> dict[str, Any]:
        """Question/candidate counts tracked separately (DESIGN.md §2)."""
        counts = {"noul": 0, "choice": 0, "score": 0}
        per_question: dict[str, int | None] = {}
        for qid, question in self.questions.items():
            counts[question.type] += 1
            if isinstance(question, ChoiceQuestion):
                per_question[qid] = len(question.criteria)
            elif isinstance(question, ScoreQuestion):
                per_question[qid] = len(question.criteria)
            else:
                per_question[qid] = None
        return {
            "q": self.n_questions,
            "k_per_question": per_question,
            "c_total_candidates": sum(v for v in per_question.values() if v),
            "primitive_mix": counts,
            "state_kind": type(self.state).__name__,
            "state_chars": len(json.dumps(self.state, ensure_ascii=False)),
        }

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    def option_maps(self) -> dict[str, dict[str, Any]]:
        """Exact option/level maps needed to reconstruct answers later."""
        maps: dict[str, dict[str, Any]] = {}
        for qid, question in self.questions.items():
            if isinstance(question, ChoiceQuestion):
                maps[qid] = {"type": "choice", "options": list(question.criteria)}
            elif isinstance(question, ScoreQuestion):
                maps[qid] = {"type": "score", "levels": [str(i) for i in range(len(question.criteria))]}
            else:
                maps[qid] = {"type": "noul", "keys": ["true", "false"]}
        return maps


def estimate_input_tokens(payload_chars: int) -> int:
    """Conservative pre-dispatch token estimate.

    The server is the only authority on tokenisation.  Reserving one token per
    character is a deliberate over-estimate (real text averages ~4 chars/token),
    so the pre-dispatch budget guard stops *earlier* than the real bill would.
    Recorded estimates are always kept distinct from server-reported usage.
    """
    return int(payload_chars) + 1
