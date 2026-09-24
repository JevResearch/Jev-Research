"""Providers: the Jev HTTP client, an offline deterministic mock, and call policy.

Retry policy is explicit and recorded per attempt.  Two modes are strictly
separated (DESIGN.md §7):

* **operational** – bounded retries on transient statuses, full user-visible cost;
* **timing**      – zero retries, so measured latency is never contaminated by a
  hidden second request.  A retry-free attempt that fails is a censored sample.

A timeout is recorded as `uncertain`: the provider may have processed and billed
the request even though we cannot observe the result.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Protocol

from .manifest import utc_now
from .redact import Redactor
from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest, estimate_input_tokens
from .transport import Transport, TransportResponse
from .validation import ValidatedResponse, validate_response

ENDPOINT = "/v1/systemone"
RETRYABLE_STATUSES = frozenset({429, 529})
NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 422})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_on_statuses: frozenset[int] = RETRYABLE_STATUSES
    retry_on_transport_error: bool = True
    retry_on_timeout: bool = False  # remote work may already be billed
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    honor_retry_after: bool = True
    timing_mode: bool = False

    @classmethod
    def operational(cls, max_attempts: int = 3) -> "RetryPolicy":
        return cls(max_attempts=max_attempts, timing_mode=False)

    @classmethod
    def none(cls) -> "RetryPolicy":
        """Zero hidden retries: the only policy admissible for latency measurement."""
        return cls(max_attempts=1, retry_on_transport_error=False, timing_mode=True)

    def backoff(self, attempt_index: int, retry_after: float | None = None) -> float:
        if self.honor_retry_after and retry_after is not None:
            return max(0.0, min(retry_after, self.backoff_max_seconds))
        return min(self.backoff_max_seconds, self.backoff_base_seconds * (2**attempt_index))


class AttemptHook(Protocol):
    """Runner-supplied accounting hooks, invoked per *attempt* (retries cost too)."""

    def before(self, estimated_input_tokens: int) -> Any: ...

    def after(self, handle: Any, reported_input_tokens: int | None) -> None: ...


class NullHook:
    def before(self, estimated_input_tokens: int) -> Any:
        return None

    def after(self, handle: Any, reported_input_tokens: int | None) -> None:
        return None


@dataclass
class AttemptRecord:
    attempt_id: str
    logical_request_id: str
    attempt_index: int
    provider: str
    model_requested: str
    started_at: str
    finished_at: str
    latency_ms: float | None
    first_byte_ms: float | None
    cold_connection: bool | None
    http_status: int | None
    outcome: str  # ok | http_error | transport_error | timeout | malformed_json | cancelled
    uncertain: bool
    error: str | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    estimated_input_tokens: int
    response_bytes: int | None
    request_sha256: str
    retry_after_seconds: float | None = None
    raw_request: dict[str, str] | None = None
    raw_response: dict[str, str] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CallOutcome:
    logical_request_id: str
    status: str  # ok | contract_invalid | http_error | transport_error | timeout | cancelled | budget_exceeded
    terminal: bool
    attempts: list[AttemptRecord]
    validated: ValidatedResponse | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    total_latency_ms: float
    n_retries: int
    request_sha256: str
    note: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.terminal

    def violation_codes(self) -> list[str]:
        return self.validated.codes if self.validated else []


class BaseProvider(Protocol):
    name: str

    def ask(
        self,
        request: SystemOneRequest,
        logical_request_id: str,
        *,
        hook: AttemptHook | None = None,
        cancellation: Any = None,
        new_attempt_id: Callable[[str, int], str] = lambda lid, i: f"{lid}.{i}",
    ) -> CallOutcome: ...

    def close(self) -> None: ...


class JevProvider:
    """Thin instrumented client for POST /v1/systemone."""

    name = "jev"

    def __init__(
        self,
        transport: Transport,
        *,
        retry_policy: RetryPolicy | None = None,
        redactor: Redactor | None = None,
        sleeper: Callable[[float], None] = lambda seconds: None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.transport = transport
        self.retry_policy = retry_policy or RetryPolicy.none()
        self.redactor = redactor or Redactor()
        self.sleeper = sleeper
        self.clock = clock

    # ------------------------------------------------------------- internals
    def _parse_body(self, response: TransportResponse) -> tuple[Any, str | None]:
        if not response.body:
            return None, "empty_body"
        try:
            return json.loads(response.body.decode("utf-8")), None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return None, f"malformed_json:{type(exc).__name__}"

    @staticmethod
    def _retry_after(response: TransportResponse) -> float | None:
        for key, value in response.headers.items():
            if key.lower() == "retry-after":
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    # -------------------------------------------------------------------- api
    def ask(
        self,
        request: SystemOneRequest,
        logical_request_id: str,
        *,
        hook: AttemptHook | None = None,
        cancellation: Any = None,
        new_attempt_id: Callable[[str, int], str] = lambda lid, i: f"{lid}.{i}",
        store: Any = None,
    ) -> CallOutcome:
        hook = hook or NullHook()
        payload = request.to_payload()
        request_sha = request.request_sha256()
        estimated = estimate_input_tokens(len(json.dumps(payload, ensure_ascii=False)))
        attempts: list[AttemptRecord] = []
        total_latency = 0.0
        status = "transport_error"
        validated: ValidatedResponse | None = None
        usage_in: int | None = None
        usage_out: int | None = None
        policy = self.retry_policy

        for index in range(policy.max_attempts):
            if cancellation is not None and cancellation.cancelled:
                status = "cancelled"
                break
            handle = hook.before(estimated)
            reported_in: int | None = None
            try:
                response = self.transport.post(ENDPOINT, payload)
            except Exception as exc:
                # A raising transport must not leak the reservation and freeze
                # the budget ledger for the rest of the run.
                hook.after(handle, None)
                raise RuntimeError(f"transport raised {type(exc).__name__}") from exc
            attempt_id = new_attempt_id(logical_request_id, index)
            parsed, parse_error = self._parse_body(response)
            http_status = response.status_code or None

            if response.error and "timeout" in response.error:
                outcome, uncertain = "timeout", True
            elif response.error:
                outcome, uncertain = "transport_error", False
            elif http_status and http_status >= 400:
                # Status classification wins over body parsing: a 429 with an
                # empty body is an http_error, not a malformed response.
                outcome, uncertain = "http_error", False
            elif parse_error:
                outcome, uncertain = "malformed_json", False
            else:
                outcome, uncertain = "ok", False

            if outcome == "ok":
                validated = validate_response(parsed, request, http_status=http_status)
                usage = validated.usage or {}
                reported_in = _as_int(usage.get("input_tokens"))
                usage_out = _as_int(usage.get("output_tokens"))
                usage_in = reported_in
                if not validated.contract_valid:
                    status = "contract_invalid"
                else:
                    status = "ok"
            else:
                usage_in = reported_in = None
                status = outcome

            raw_request = None
            raw_response = None
            if store is not None:
                raw_request = store.write_raw(attempt_id, "request", payload)
                raw_response = store.write_raw(
                    attempt_id,
                    "response",
                    parsed if parsed is not None else {"body_excerpt": _excerpt(response.body)},
                )

            record = AttemptRecord(
                attempt_id=attempt_id,
                logical_request_id=logical_request_id,
                attempt_index=index,
                provider=self.name,
                model_requested=request.model,
                started_at=self.clock(),
                finished_at=self.clock(),
                latency_ms=response.total_ms,
                first_byte_ms=response.first_byte_ms,
                cold_connection=response.cold_connection,
                http_status=http_status,
                outcome=outcome,
                uncertain=uncertain,
                error=self.redactor.text(response.error or parse_error or "") or None,
                usage_input_tokens=reported_in,
                usage_output_tokens=usage_out if outcome == "ok" else None,
                estimated_input_tokens=estimated,
                response_bytes=len(response.body) if response.body else 0,
                request_sha256=request_sha,
                retry_after_seconds=self._retry_after(response),
                raw_request=raw_request,
                raw_response=raw_response,
                headers=response.headers,
            )
            attempts.append(record)
            total_latency += response.total_ms
            hook.after(handle, reported_in)

            retryable = (
                (outcome == "ok" and False)
                or (outcome in {"http_error"} and http_status in policy.retry_on_statuses)
                or (outcome == "transport_error" and policy.retry_on_transport_error)
                or (outcome == "malformed_json" and policy.retry_on_transport_error)
            )
            if outcome == "timeout" and policy.retry_on_timeout:
                retryable = True
            if not retryable or index == policy.max_attempts - 1:
                break
            self.sleeper(policy.backoff(index, record.retry_after_seconds))

        return CallOutcome(
            logical_request_id=logical_request_id,
            status=status,
            terminal=status in {"ok", "contract_invalid", "http_error", "transport_error", "cancelled"},
            attempts=attempts,
            validated=validated,
            usage_input_tokens=usage_in,
            usage_output_tokens=usage_out,
            total_latency_ms=total_latency,
            n_retries=max(0, len(attempts) - 1),
            request_sha256=request_sha,
        )

    def close(self) -> None:
        self.transport.close()


class MockProvider:
    """Deterministic offline provider with programmable contract faults.

    It is *not* a model: it produces well-formed or deliberately malformed
    responses so the harness can be tested without the network or the vendor.
    Every artifact it produces is labelled ``mock``.
    """

    name = "mock"

    def __init__(
        self,
        seed: int = 0,
        faults: dict[str, dict[str, Any]] | None = None,
        clock: Callable[[], str] = utc_now,
        default_fault: dict[str, Any] | None = None,
    ) -> None:
        self.seed = seed
        self.faults = faults or {}  # logical_request_id -> {"body": ...} or {"error": "..."}
        self.default_fault = default_fault  # fault for any id not in `faults`
        self.clock = clock
        self.calls: list[str] = []

    def _rng(self, key: str) -> random.Random:
        return random.Random(int(sha256(f"{self.seed}:{key}".encode()).hexdigest()[:12], 16))

    def _respond(self, request: SystemOneRequest) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for qid, question in request.questions.items():
            rng = self._rng(qid)
            if isinstance(question, NoulQuestion):
                answers[qid] = {"type": "noul", "noul": round(rng.random(), 4)}
            elif isinstance(question, ChoiceQuestion):
                keys = list(question.criteria)
                raw = [rng.random() + 0.05 for _ in keys]
                total = sum(raw)
                probs = [r / total for r in raw]
                best = max(range(len(keys)), key=lambda i: probs[i])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {k: round(p, 6) for k, p in zip(keys, probs)},
                    "confidence": _choice_confidence(probs),
                }
            else:
                levels = len(question.criteria)
                raw = [rng.random() + 0.05 for _ in range(levels)]
                total = sum(raw)
                probs = [r / total for r in raw]
                expectation = sum(i * p for i, p in enumerate(probs))
                answers[qid] = {
                    "type": "score",
                    "score": round(expectation, 4),
                    "legend": {str(i): str(question.criteria[i]) for i in range(levels)},
                    "probabilities": {str(i): round(p, 6) for i, p in enumerate(probs)},
                    "confidence": _score_confidence(probs),
                }
        payload_chars = len(json.dumps(request.to_payload()))
        return {
            "model": request.model,
            "answers": answers,
            "usage": {"input_tokens": max(1, payload_chars // 4), "output_tokens": 48},
        }

    def ask(
        self,
        request: SystemOneRequest,
        logical_request_id: str,
        *,
        hook: AttemptHook | None = None,
        cancellation: Any = None,
        new_attempt_id: Callable[[str, int], str] = lambda lid, i: f"{lid}.{i}",
        store: Any = None,
    ) -> CallOutcome:
        hook = hook or NullHook()
        estimated = estimate_input_tokens(len(json.dumps(request.to_payload(), ensure_ascii=False)))
        handle = hook.before(estimated)
        self.calls.append(logical_request_id)  # a budget-denied dispatch never happened
        fault = self.faults.get(logical_request_id)
        if fault is None and self.default_fault is not None:
            fault = self.default_fault
        parsed: Any = None
        outcome = "ok"
        error_text = None
        usage_in = usage_out = None
        validated: ValidatedResponse | None = None
        status = "ok"
        if fault and "error" in fault:
            # Scripted transport-level failure (timeout, 429, ...)
            outcome = fault.get("outcome", "transport_error")
            error_text = fault["error"]
            status = outcome
        elif fault and "body" in fault:
            # Scripted *semantic* fault: a body that arrives and must be validated.
            parsed = fault["body"]
            validated = validate_response(parsed, request, http_status=200)
            usage_in = _as_int((validated.usage or {}).get("input_tokens"))
            usage_out = _as_int((validated.usage or {}).get("output_tokens"))
            status = "ok" if validated.contract_valid else "contract_invalid"
        else:
            parsed = self._respond(request)
            validated = validate_response(parsed, request, http_status=200)
            usage_in = _as_int((validated.usage or {}).get("input_tokens"))
            usage_out = _as_int((validated.usage or {}).get("output_tokens"))
            status = "ok" if validated.contract_valid else "contract_invalid"

        attempt_id = new_attempt_id(logical_request_id, 0)
        raw_request = raw_response = None
        if store is not None:
            raw_request = store.write_raw(attempt_id, "request", request.to_payload())
            raw_response = store.write_raw(attempt_id, "response", parsed if parsed is not None else {"error": error_text})
        attempts = [
            AttemptRecord(
                attempt_id=attempt_id,
                logical_request_id=logical_request_id,
                attempt_index=0,
                provider=self.name,
                model_requested=request.model,
                started_at=self.clock(),
                finished_at=self.clock(),
                latency_ms=1.0,
                first_byte_ms=None,
                cold_connection=False,
                http_status=200 if parsed is not None else None,
                outcome=outcome,
                uncertain=outcome == "timeout",
                error=error_text,
                usage_input_tokens=usage_in,
                usage_output_tokens=usage_out,
                estimated_input_tokens=estimated,
                response_bytes=len(json.dumps(parsed)) if parsed is not None else 0,
                request_sha256=request.request_sha256(),
                raw_request=raw_request,
                raw_response=raw_response,
            )
        ]
        hook.after(handle, usage_in)
        return CallOutcome(
            logical_request_id=logical_request_id,
            status=status,
            terminal=True,
            attempts=attempts,
            validated=validated,
            usage_input_tokens=usage_in,
            usage_output_tokens=usage_out,
            total_latency_ms=1.0,
            n_retries=0,
            request_sha256=request.request_sha256(),
            note="mock",
        )

    def close(self) -> None:
        return


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _excerpt(body: bytes, limit: int = 2000) -> str:
    text = body.decode("utf-8", errors="replace")
    return text[:limit]


def _choice_confidence(probs: list[float]) -> float:
    """Formula published in typesafe-ai/system-one-adapter-python (S13).

    Reproduced only so mock artifacts look structurally plausible; the server is
    authoritative and its value is what gets analysed.
    """
    uniform = 1.0 / len(probs)
    return max(0.0, (max(probs) - uniform) / (1.0 - uniform))


def _score_confidence(probs: list[float]) -> float:
    mode = max(range(len(probs)), key=lambda i: probs[i])
    mad = sum(p * abs(i - mode) for i, p in enumerate(probs))
    centre = (len(probs) - 1) / 2
    uniform_mad = sum(abs(i - centre) for i in range(len(probs))) / len(probs)
    return max(0.0, 1.0 - mad / uniform_mad)
