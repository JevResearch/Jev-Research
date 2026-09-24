"""Experiment specs: items -> outbound requests, with gold labels kept separate.

An experiment file is a small JSON document.  Each item carries its own `state`
and `questions` (the *only* fields that become outbound payload) plus an optional
`gold` block used solely by offline analysis.  The structural guarantees are:

* the outbound payload can only contain the keys {state, model, questions};
* question objects can only contain {type, instructions, criteria};
* a leakage guard checks that no answer-identifying gold text appears in the
  state when the item opts into the check (synthetic sets: on by default).

Logical request ids are stable hashes, so reruns and resumes address the same
item deterministically.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

PAYLOAD_TOP_KEYS = {"state", "model", "questions"}
QUESTION_KEYS = {"type", "instructions", "criteria"}
GOLD_KEYS = {"gold", "answer", "label", "expected"}


class SpecError(ValueError):
    pass


@dataclass
class Item:
    item_id: str
    group: str
    condition: str
    cluster: str
    state: Any
    questions: dict[str, Any]
    gold: dict[str, Any]
    leakage_check: bool = True
    index: int = 0

    def to_request(self, model: str) -> SystemOneRequest:
        """Build the outbound payload. Gold never reaches this function's output."""
        questions = {}
        for qid, raw in self.questions.items():
            unknown = set(raw) - QUESTION_KEYS
            if unknown:
                raise SpecError(f"item {self.item_id} question {qid} has unsupported keys {sorted(unknown)}")
            qtype = raw.get("type")
            if qtype == "noul":
                questions[qid] = NoulQuestion(**{k: v for k, v in raw.items() if k in QUESTION_KEYS})
            elif qtype == "choice":
                questions[qid] = ChoiceQuestion(**{k: v for k, v in raw.items() if k in QUESTION_KEYS})
            elif qtype == "score":
                questions[qid] = ScoreQuestion(**{k: v for k, v in raw.items() if k in QUESTION_KEYS})
            else:
                raise SpecError(f"item {self.item_id} question {qid} has unknown type {qtype!r}")
        return SystemOneRequest(state=self.state, model=model, questions=questions)

    def check_leakage(self, request: SystemOneRequest) -> list[str]:
        """Names of gold fields whose *text* appears inside the outbound state.

        Option keys are, of course, legitimately present in choice criteria; the
        guard looks for the gold *value text* and reserved label field names in
        the state, which is where a dataset accidentally leaks its answer.
        """
        if not self.leakage_check:
            return []
        payload_json = json.dumps(request.to_payload(), ensure_ascii=False)
        payload = request.to_payload()
        leaks: list[str] = []
        for key in GOLD_KEYS:
            if key in payload:
                leaks.append(f"reserved_field:{key}")
        for qid, answer in self.gold.items():
            value = answer.get("value") if isinstance(answer, dict) else answer
            if value is None:
                continue
            text = str(value)
            if len(text) >= 4 and text in json.dumps(payload.get("state"), ensure_ascii=False):
                leaks.append(f"state_contains_gold:{qid}")
            if len(text) >= 4 and text in payload_json and qid in payload_json:
                # The same string both identifies the gold and names the question.
                if text not in json.dumps(payload["questions"], ensure_ascii=False):
                    leaks.append(f"gold_in_payload:{qid}")
        return leaks


@dataclass
class LogicalRequest:
    logical_request_id: str
    item: Item
    request: SystemOneRequest
    condition: str = "native"


def load_experiment(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("experiment", "items"):
        if key not in data:
            raise SpecError(f"experiment spec missing required key {key!r}")
    ids = [item.get("id") for item in data["items"]]
    if len(ids) != len(set(ids)):
        raise SpecError("item ids must be unique")
    for item in data["items"]:
        for key in item:
            if key not in {
                "id", "state", "questions", "gold", "group", "condition",
                "cluster", "leakage_check", "index",
            }:
                raise SpecError(f"item {item.get('id')} has unsupported key {key!r}")
    return data


def logical_requests(
    spec: dict[str, Any],
    *,
    order_seed: int | None = None,
    shuffle: bool | None = None,
) -> list[LogicalRequest]:
    """Deterministically expand the spec into logical requests.

    Shuffling honours `spec["shuffle"]` (designed experiments set it False so
    the block layout *is* the dispatch order); the explicit argument overrides.
    """
    if shuffle is None:
        shuffle = bool(spec.get("shuffle", True))
    if order_seed is None:
        order_seed = spec.get("seeds", {}).get("order", 0)
    model = spec.get("model", "jev-1.13.0")
    requests: list[LogicalRequest] = []
    allowed = {"id", "state", "questions", "gold", "group", "condition", "cluster",
               "leakage_check", "index", "arc_meta", "math500_meta", "gpqa_meta",
               "hle_meta"}
    for index, raw in enumerate(spec["items"]):
        unknown = set(raw) - allowed - {"_params"}
        if unknown:
            raise SpecError(f"item {raw.get('id')} has unsupported key(s) {sorted(unknown)}")
        item = Item(
            item_id=str(raw["id"]),
            group=str(raw.get("group") or "default"),
            condition=str(raw.get("condition") or "native"),
            cluster=str(raw.get("cluster") or raw.get("id")),
            state=raw["state"],
            questions=raw["questions"],
            gold=raw.get("gold") or {},
            leakage_check=bool(raw.get("leakage_check", True)),
            index=index,
        )
        request = item.to_request(model)
        for leak in item.check_leakage(request):
            raise SpecError(f"item {item.item_id} leaks gold into outbound payload: {leak}")
        logical_id = f"{item.condition}:{item.item_id}"
        requests.append(LogicalRequest(logical_request_id=logical_id, item=item, request=request,
                                       condition=item.condition))
    if shuffle:
        rng = random.Random(order_seed)
        rng.shuffle(requests)
    return requests


def spec_sha256(spec: dict[str, Any]) -> str:
    return sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def item_hashes(spec: dict[str, Any]) -> dict[str, str]:
    return {
        str(raw["id"]): sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        for raw in spec["items"]
    }