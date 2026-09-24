"""Talk to Jev: explicit external autoregression over a fixed character alphabet.

DESIGN.md §9 governs this module.  The essential honesty points:

* this is NOT native chat: every character is a fresh Choice question over the
  transcript plus the exact prefix we supply;
* the output is path-, prompt-, and decoding-dependent;
* the 98-option alphabet (95 printable ASCII + newline + tab + END) is fixed and
  mapped through visible keys; option keys are never concatenated into output;
* every step records raw distributions; any temperature transform is local,
  labelled, and stored alongside the raw values;
* stopping is conservative: END, character cap, wall clock, repetition guard,
  provider failure, cancellation, or budget denial — never silent truncation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .manifest import utc_now
from .providers import BaseProvider, NullHook
from .schema import SystemOneRequest

# ------------------------------------------------------------------ alphabet
END_TOKEN = "END"

def _build_alphabet() -> tuple[list[str], dict[str, str | None]]:
    """95 printable ASCII (32..126) + newline + tab + END = 98 options."""
    keys: list[str] = []
    mapping: dict[str, str | None] = {}
    for code in range(32, 127):
        key = f"char_{code:03d}"
        keys.append(key)
        mapping[key] = chr(code)
    for code in (10, 9):  # newline, tab
        key = f"char_{code:03d}"
        keys.append(key)
        mapping[key] = chr(code)
    keys.append(END_TOKEN)
    mapping[END_TOKEN] = None
    assert len(keys) <= 255
    return keys, mapping

ALPHABET_KEYS, KEY_TO_CHAR = _build_alphabet()
CHAR_TO_KEY = {char: key for key, char in KEY_TO_CHAR.items() if char is not None}

NEXT_CHAR_INSTRUCTIONS = (
    "Select the next character of the assistant response, conditioned on this "
    "conversation transcript and the exact assistant prefix shown. Choose END "
    "when the response is complete."
)

END_DESCRIPTION = "end the assistant response"


def _char_description(char: str) -> str:
    if char == "\n":
        return "the newline character"
    if char == "\t":
        return "the tab character"
    if char == " ":
        return "the space character"
    return f"the character {char!r} (U+{ord(char):04X})"


def alphabet_criteria() -> dict[str, str]:
    """Choice criteria: visible key -> description. Keys are opaque to the model."""
    return {
        key: (END_DESCRIPTION if key == END_TOKEN else _char_description(KEY_TO_CHAR[key]))
        for key in ALPHABET_KEYS
    }


def build_talk_request(
    *, user_prompt: str, previous_turns: list[dict[str, str]], prefix: str,
    model: str = "jev-1.13.0",
) -> SystemOneRequest:
    """One character-choice request. Prefix and transcript are distinct fields."""
    state = {
        "user_prompt": user_prompt,
        "previous_turns": previous_turns,
        "assistant_prefix": prefix,
    }
    return SystemOneRequest(
        state=state,
        model=model,
        questions={"next_char": {
            "type": "choice",
            "instructions": NEXT_CHAR_INSTRUCTIONS,
            "criteria": alphabet_criteria(),
        }},
    )


# ------------------------------------------------------------------ decoders
@dataclass(frozen=True)
class DecoderConfig:
    mode: str = "greedy"            # greedy | sample
    temperature: float | None = None  # local transform; recorded, never hidden
    seed: int | None = None

    def transform(self, probs: dict[str, float]) -> dict[str, float]:
        if self.mode != "sample" or not self.temperature or self.temperature == 1.0:
            return dict(probs)
        temperature = self.temperature
        transformed = {
            key: (value ** (1.0 / temperature) if value > 0 else 0.0)
            for key, value in probs.items()
        }
        total = sum(transformed.values())
        if total <= 0:
            return dict(probs)
        return {key: value / total for key, value in transformed.items()}


def _greedy_pick(probs: dict[str, float]) -> str:
    """Argmax with the fixed tie rule: first key in canonical alphabet order."""
    best_key = None
    best_value = -1.0
    for key in ALPHABET_KEYS:
        value = probs.get(key)
        if value is not None and value > best_value:
            best_value = value
            best_key = key
    return best_key or END_TOKEN


# ------------------------------------------------------------------ session
@dataclass
class TalkLimits:
    max_chars: int = 256
    max_seconds: float = 120.0
    max_provider_requests: int | None = None

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass
class _Node:
    node_id: str
    parent_id: str | None
    char: str | None
    key: str | None
    probabilities: dict[str, float] | None
    transformed: dict[str, float] | None
    user_edited: bool
    model_returned: str | None = None
    latency_ms: float | None = None
    usage_input_tokens: int | None = None


@dataclass
class TalkSession:
    """One generation session: a tree of character nodes with explicit stops."""

    session_id: str
    provider: BaseProvider
    user_prompt: str
    previous_turns: list[dict[str, str]] = field(default_factory=list)
    limits: TalkLimits = field(default_factory=TalkLimits)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    hook: Any = None
    model: str = "jev-1.13.0"
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.nodes: dict[str, _Node] = {
            "root": _Node("root", None, None, None, None, None, user_edited=False)
        }
        self.current_node_id = "root"
        self.events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.cancelled = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._rng = None
        if self.decoder.mode == "sample":
            self._rng = __import__("random").Random(self.decoder.seed)
        self.hook = self.hook or NullHook()
        self._provider_requests = 0

    # ------------------------------------------------------------- state
    @property
    def prefix(self) -> str:
        chars: list[str] = []
        node_id = self.current_node_id
        while node_id != "root":
            node = self.nodes[node_id]
            if node.char is not None:
                chars.append(node.char)
            node_id = node.parent_id
        return "".join(reversed(chars))

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return self.clock() - self.started_at

    def cancel(self) -> None:
        self.cancelled = True

    # -------------------------------------------------------------- stepping
    def step(self) -> dict[str, Any]:
        """Take one decoding step; returns the event (with stop info if stopping)."""
        if self.stop_reason is not None:
            return {"type": "stop", "reason": self.stop_reason, "already_stopped": True}
        if self.cancelled:
            return self._stop("cancelled")
        if self.started_at is None:
            self.started_at = self.clock()
        if self._elapsed() > self.limits.max_seconds:
            return self._stop("deadline")
        if len(self.prefix) >= self.limits.max_chars:
            return self._stop("max_chars")
        if (
            self.limits.max_provider_requests is not None
            and self._provider_requests >= self.limits.max_provider_requests
        ):
            return self._stop("max_provider_requests")
        if self._repetition_stop(self.prefix):
            return self._stop("repetition")

        request = build_talk_request(
            user_prompt=self.user_prompt,
            previous_turns=self.previous_turns,
            prefix=self.prefix,
            model=self.model,
        )
        self._provider_requests += 1
        try:
            outcome = self.provider.ask(
                request,
                f"talk:{self.session_id}:{len(self.events)}",
                hook=self.hook,
            )
        except Exception as exc:
            from .guards import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                return self._stop("budget_exceeded")
            raise
        if outcome.status != "ok" or outcome.validated is None:
            return self._stop("provider_failure", detail=outcome.status)
        answer = outcome.validated.answers.get("next_char")
        if answer is None or not answer.usable:
            return self._stop("provider_failure", detail="missing or unusable next_char answer")

        raw_probs = answer.values.get("probabilities") or {}
        transformed = self.decoder.transform(raw_probs)
        if self.decoder.mode == "sample" and self._rng is not None:
            chosen_key = self._rng.choices(
                population=list(transformed), weights=list(transformed.values()), k=1
            )[0]
        else:
            chosen_key = _greedy_pick(transformed)

        char = KEY_TO_CHAR.get(chosen_key)
        node = _Node(
            node_id=f"n{len(self.nodes):05d}",
            parent_id=self.current_node_id,
            char=char,
            key=chosen_key,
            probabilities=raw_probs,
            transformed=transformed if transformed != raw_probs else None,
            user_edited=False,
            model_returned=outcome.validated.returned_model,
            latency_ms=outcome.attempts[-1].latency_ms if outcome.attempts else None,
            usage_input_tokens=outcome.usage_input_tokens,
        )
        self.nodes[node.node_id] = node
        self.current_node_id = node.node_id
        event = {
            "type": "step",
            "node_id": node.node_id,
            "key": chosen_key,
            "char": char,
            "top_next": self._top(transformed, 5),
            "model_returned": node.model_returned,
            "latency_ms": node.latency_ms,
        }
        self.events.append(event)
        if char is None:
            return self._stop("end_token")
        return event

    def _stop(self, reason: str, *, detail: str | None = None) -> dict[str, Any]:
        self.stop_reason = reason
        self.finished_at = self.clock()
        event: dict[str, Any] = {"type": "stop", "reason": reason, "prefix_chars": len(self.prefix)}
        if detail:
            event["detail"] = detail
        self.events.append(event)
        return event

    @staticmethod
    def _top(probs: dict[str, float], n: int) -> list[dict[str, Any]]:
        ordered = sorted(probs.items(), key=lambda kv: (-kv[1], ALPHABET_KEYS.index(kv[0])))
        return [{"key": key, "char": KEY_TO_CHAR.get(key), "p": round(value, 6)}
                for key, value in ordered[:n]]

    @staticmethod
    def _repetition_stop(prefix: str) -> bool:
        """Conservative loop guard: a short unit repeated at least 4 times at the tail."""
        for unit_length in (8, 4, 2, 1):
            unit = prefix[-unit_length:]
            if unit and len(unit) * 4 <= len(prefix) and prefix.endswith(unit * 4):
                return True
        return False

    # ------------------------------------------------------------ traversal
    def run(self) -> list[dict[str, Any]]:
        """Decode until a stop condition; cancellation is checked between steps."""
        while self.stop_reason is None:
            self.step()
        return self.events

    def backtrack(self, node_id: str) -> None:
        """Move the cursor to an ancestor node. Branches are never deleted."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node {node_id}")
        # must be an ancestor of the current position
        walker = self.current_node_id
        while walker is not None:
            if walker == node_id:
                self.current_node_id = node_id
                self.stop_reason = None  # a backtrack re-opens a stopped session
                return
            walker = self.nodes[walker].parent_id
        raise ValueError(f"node {node_id} is not an ancestor of the current position")

    def edit_prefix(self, text: str) -> str:
        """User-authored prefix: recorded, labelled, never mixed with model output.

        The text *replaces* the current prefix (the node's parent is the root),
        so model output and user text can never be silently interleaved.
        """
        node = _Node(
            node_id=f"n{len(self.nodes):05d}",
            parent_id="root",
            char=text if text else None,
            key=None,
            probabilities=None,
            transformed=None,
            user_edited=True,
        )
        self.nodes[node.node_id] = node
        self.current_node_id = node.node_id
        self.stop_reason = None
        return node.node_id

    # --------------------------------------------------------------- export
    def export_trace(self) -> dict[str, Any]:
        path: list[str] = []
        walker = self.current_node_id
        while walker != "root":
            path.append(walker)
            walker = self.nodes[walker].parent_id
        return {
            "session_id": self.session_id,
            "recorded_at": utc_now(),
            "user_prompt": self.user_prompt,
            "previous_turns": self.previous_turns,
            "decoder": {"mode": self.decoder.mode, "temperature": self.decoder.temperature,
                        "seed": self.decoder.seed},
            "limits": {"max_chars": self.limits.max_chars, "max_seconds": self.limits.max_seconds},
            "stop_reason": self.stop_reason,
            "n_nodes": len(self.nodes),
            "active_path_nodes": list(reversed(path)),
            "prefix": self.prefix,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "char": node.char,
                    "key": node.key,
                    "user_edited": node.user_edited,
                    "probabilities": node.probabilities,
                    "transformed_probabilities": node.transformed,
                    "model_returned": node.model_returned,
                    "latency_ms": node.latency_ms,
                    "usage_input_tokens": node.usage_input_tokens,
                }
                for node in self.nodes.values()
            ],
            "events": self.events,
        }


class ScriptedCharProvider:
    """Deterministic test fixture: emits a known string one character per step,
    then END. One-hot distributions in canonical alphabet order."""

    name = "scripted-char"

    def __init__(self, text: str) -> None:
        self.text = text
        self._position = 0
        self._requests: list[str] = []

    def ask(self, request: SystemOneRequest, logical_request_id: str, **kwargs) -> Any:
        self._requests.append(logical_request_id)
        from .providers import AttemptRecord, CallOutcome
        from .validation import validate_response

        if self._position < len(self.text):
            char = self.text[self._position]
            key = CHAR_TO_KEY[char]
        else:
            key = END_TOKEN
        self._position += 1
        probs = {k: (1.0 if k == key else 0.0) for k in ALPHABET_KEYS}
        body = {
            "model": request.model,
            "answers": {"next_char": {"type": "choice", "choice": key,
                                       "probabilities": probs, "confidence": 1.0}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        validated = validate_response(body, request, http_status=200)
        record = AttemptRecord(
            attempt_id=f"{logical_request_id}.0", logical_request_id=logical_request_id,
            attempt_index=0, provider=self.name, model_requested=request.model,
            started_at=utc_now(), finished_at=utc_now(), latency_ms=1.0, first_byte_ms=None,
            cold_connection=False, http_status=200, outcome="ok", uncertain=False, error=None,
            usage_input_tokens=10, usage_output_tokens=2, estimated_input_tokens=10,
            response_bytes=len(json.dumps(body)), request_sha256=request.request_sha256(),
        )
        return CallOutcome(
            logical_request_id=logical_request_id, status="ok", terminal=True,
            attempts=[record], validated=validated, usage_input_tokens=10,
            usage_output_tokens=2, total_latency_ms=1.0, n_retries=0,
            request_sha256=request.request_sha256(), note="scripted-char",
        )

    def close(self) -> None:
        return
