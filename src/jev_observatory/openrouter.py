"""OpenRouter baseline adapter (parent gate item 4).

Instrumented standard chat-API client for the nine named modern models. Design
rules carried over from the Jev provider:

* ONE attempt per dispatch — no retries, and in particular **no retry on a
  wrong or malformed answer**: a strict-format failure is counted as truth;
* every attempt persists its raw response and usage (prompt/completion tokens,
  reasoning tokens, cost when reported) before anything else happens;
* the returned model string is recorded per attempt (model_requested AND
  model_returned are distinct facts);
* reasoning mode and output budget are explicit, per model, and recorded in the
  outbound payload (``reasoning`` / ``max_tokens``); nothing is defaulted
  silently;
* no fake one-hot probabilities are ever synthesized from text answers — a
  text-answer baseline contributes a choice key and nothing probabilistic;
* credentials come ONLY from the process environment (OPENROUTER_API_KEY);
  live calls additionally require the JEVO_ALLOW_LIVE opt-in;
* the public /models catalog is a separate GET used by preflight to
  distinguish "key present" from "model actually accessible".

Dry-run discipline: the provider accepts any Transport exposing ``post``; tests
use ScriptedTransport. Nothing here touches the network unless a live
transport is explicitly constructed by the caller (scripts do that only under
the opt-in gate).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from .manifest import utc_now
from .providers import AttemptRecord, CallOutcome, NullHook, _as_int
from .redact import Redactor
from .schema import SystemOneRequest, estimate_input_tokens
from .transport import TransportResponse
from .validation import ValidatedResponse, validate_response

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"
API_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "OPENROUTER_BASE_URL"

DEFAULT_MAX_OUTPUT_TOKENS = 256  # bounded direct answers; no unbounded generation
LETTERS = "ABCDEFGHIJ"


class OpenRouterError(RuntimeError):
    pass


# ------------------------------------------------------------------ transport
class OpenRouterHttpTransport:
    """Real network transport for OpenRouter (POST chat + GET models).

    Constructing it requires a non-empty API key from the environment; the
    live opt-in gate (JEVO_ALLOW_LIVE=1) is enforced here too, so a live
    transport cannot exist by accident.
    """

    name = "openrouter-httpx"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        redactor: Redactor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import httpx  # lazy so offline tooling works without the wheel

        if not api_key:
            raise OpenRouterError(
                f"no API key in process environment ({API_KEY_ENV}); refusing"
            )
        if os.environ.get("JEVO_ALLOW_LIVE") != "1":
            raise OpenRouterError(
                "live OpenRouter calls require environment JEVO_ALLOW_LIVE=1"
            )
        self.base_url = (base_url or os.environ.get(BASE_URL_ENV) or OPENROUTER_BASE_URL).rstrip("/")
        self.redactor = redactor or Redactor()
        self.clock = clock
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = self.clock()
        try:
            response = self._client.post(path, content=body)
            return TransportResponse(
                status_code=response.status_code,
                headers=self.redactor.headers(dict(response.headers)),
                body=response.content,
                total_ms=(self.clock() - started) * 1000.0,
                cold_connection=None,
            )
        except self._httpx.TimeoutException as exc:
            return TransportResponse(
                status_code=0, headers={}, body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                error=f"timeout:{type(exc).__name__}",
            )
        except self._httpx.RequestError as exc:
            return TransportResponse(
                status_code=0, headers={}, body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                error=f"transport_error:{type(exc).__name__}",
            )

    def get(self, path: str) -> TransportResponse:
        started = self.clock()
        try:
            response = self._client.get(path)
            return TransportResponse(
                status_code=response.status_code,
                headers=self.redactor.headers(dict(response.headers)),
                body=response.content,
                total_ms=(self.clock() - started) * 1000.0,
                cold_connection=None,
            )
        except self._httpx.TimeoutException as exc:
            return TransportResponse(
                status_code=0, headers={}, body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                error=f"timeout:{type(exc).__name__}",
            )
        except self._httpx.RequestError as exc:
            return TransportResponse(
                status_code=0, headers={}, body=b"",
                total_ms=(self.clock() - started) * 1000.0,
                error=f"transport_error:{type(exc).__name__}",
            )

    def close(self) -> None:
        self._client.close()


# -------------------------------------------------------------- wire payload
@dataclass(frozen=True)
class WireConfig:
    """Explicit per-model dispatch settings (recorded verbatim per attempt)."""

    model: str
    reasoning_effort: str | None = None   # e.g. "low" | "medium" | "high"
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    extra_body: dict[str, Any] | None = None  # recorded; used only if supported

    def to_dict(self) -> dict[str, Any]:
        doc = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.extra_body:
            doc["extra_body"] = dict(self.extra_body)
        return doc


def option_lines(option_texts: list[str]) -> str:
    """Canonical option block: canonical keys in source order, never sorted."""
    return "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(option_texts))


def build_messages(request: SystemOneRequest) -> list[dict[str, str]]:
    """Serialize the SAME item state/options as the Jev run into chat messages.

    Exactly one choice question per request is required (the benchmark suite
    dispatches one question per item); state text, instructions and the
    option block use the identical strings that went into the Jev payload.
    """
    if len(request.questions) != 1:
        raise OpenRouterError(
            "the OpenRouter baseline dispatches exactly one question per request; "
            f"got {len(request.questions)}"
        )
    question = next(iter(request.questions.values()))
    if question.type != "choice":
        raise OpenRouterError(f"unsupported question type {question.type!r}")
    options = "\n".join(f"{key}. {text}" for key, text in question.criteria.items())
    prompt = (
        f"{request.state}\n\n{question.instructions}\n\n"
        f"{options}\n\n"
        f"Reply with only the option key ({', '.join(question.criteria)})."
    )
    return [{"role": "user", "content": prompt}]


def build_chat_payload(request: SystemOneRequest, wire: WireConfig) -> dict[str, Any]:
    """Deterministic chat-completions payload; explicit model, budget, reasoning."""
    payload: dict[str, Any] = {
        "model": wire.model,
        "messages": build_messages(request),
        "max_tokens": int(wire.max_output_tokens),
    }
    if wire.reasoning_effort is not None:
        payload["reasoning_effort"] = wire.reasoning_effort
    if wire.extra_body:
        payload.update(json.loads(json.dumps(wire.extra_body)))
    return payload


# -------------------------------------------------------------------- parser
def parse_choice_answer(text: str | None, option_keys: list[str]) -> tuple[str | None, str | None]:
    """STRICT choice parsing: the whole message must be exactly one option key.

    No normalization, no substring scan, no case folding — anything else is a
    strict-format failure that is COUNTED, never repaired and never retried.
    """
    if text is None:
        return None, "empty_answer"
    candidate = text.strip()
    if not candidate:
        return None, "empty_answer"
    if candidate not in option_keys:
        return None, "strict_format_not_exact_key"
    return candidate, None


def extract_choice_content(body: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract the first choice's message content; structural errors recorded."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "missing_choices"
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None, "missing_message"
    content = message.get("content")
    if content is None:
        # Some models put the answer only in reasoning; that is still a strict
        # failure for a direct-answer protocol, recorded as truth.
        return None, "empty_answer"
    if not isinstance(content, str):
        return None, "content_not_string"
    return content, None


# ------------------------------------------------------------------ provider
class OpenRouterProvider:
    """Instrumented chat-API provider; mirrors the JevProvider call shape."""

    name = "openrouter"

    def __init__(
        self,
        transport: Any,
        *,
        wire: WireConfig,
        redactor: Redactor | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.transport = transport
        self.wire = wire
        self.redactor = redactor or Redactor()
        self.clock = clock

    def ask(
        self,
        request: SystemOneRequest,
        logical_request_id: str,
        *,
        hook: Any | None = None,
        cancellation: Any = None,
        new_attempt_id: Callable[[str, int], str] = lambda lid, i: f"{lid}.{i}",
        store: Any = None,
    ) -> CallOutcome:
        hook = hook or NullHook()
        if cancellation is not None and cancellation.cancelled:
            return CallOutcome(
                logical_request_id=logical_request_id, status="cancelled", terminal=True,
                attempts=[], validated=None, usage_input_tokens=None,
                usage_output_tokens=None, total_latency_ms=0.0, n_retries=0,
                request_sha256=request.request_sha256(),
            )
        payload = build_chat_payload(request, self.wire)
        request_sha = request.request_sha256()
        estimated = estimate_input_tokens(len(json.dumps(payload, ensure_ascii=False)))
        handle = hook.before(estimated)
        started = self.clock()
        response = self.transport.post(CHAT_PATH, payload)
        parsed, parse_error = _parse_body(response)
        http_status = response.status_code or None
        option_keys = []
        for question in request.questions.values():
            option_keys = list(question.criteria)
        usage_in = usage_out = None
        usage_raw: dict[str, Any] | None = None
        validated: ValidatedResponse | None = None
        status: str
        if response.error and "timeout" in response.error:
            status, outcome = "timeout", "timeout"
        elif response.error:
            status, outcome = "transport_error", "transport_error"
        elif http_status and http_status >= 400:
            status, outcome = "http_error", "http_error"
        elif parse_error:
            status, outcome = "malformed_json", "malformed_json"
        else:
            outcome = "ok"
            model_returned = parsed.get("model")
            usage_raw = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
            if usage_raw is not None:
                usage_in = _as_int(usage_raw.get("prompt_tokens"))
                usage_out = _as_int(usage_raw.get("completion_tokens"))
            content, content_error = extract_choice_content(parsed)
            choice = None
            format_error = None
            if content_error is None:
                choice, format_error = parse_choice_answer(content, option_keys)
            failure_code = content_error or format_error
            status = "contract_invalid" if failure_code is not None else "ok"
            qid = next(iter(request.questions))
            from .validation import ValidatedAnswer, Violation

            violations = (
                [Violation(failure_code, "error", "strict direct-answer protocol", qid)]
                if failure_code is not None else []
            )
            validated = ValidatedResponse(
                contract_valid=status == "ok",
                usable=status == "ok",
                requested_model=self.wire.model,
                returned_model=model_returned,
                answers={
                    qid: ValidatedAnswer(
                        question_id=qid, question_type="choice",
                        usable=status == "ok",
                        raw=content,
                        values={"choice": choice} if choice is not None else {},
                        violations=violations,
                    )
                },
                violations=violations,
                usage=usage_raw or {},
                raw=parsed,
            )
        attempt_id = new_attempt_id(logical_request_id, 0)
        finish_reason = None
        if parsed is not None and isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                candidate = choices[0].get("finish_reason")
                finish_reason = str(candidate) if candidate is not None else None
        raw_request = raw_response = None
        if store is not None:
            raw_request = store.write_raw(attempt_id, "request", payload)
            raw_response = store.write_raw(
                attempt_id, "response",
                parsed if parsed is not None else {"body_excerpt": _excerpt(response.body)},
            )
        record = AttemptRecord(
            attempt_id=attempt_id,
            logical_request_id=logical_request_id,
            attempt_index=0,
            provider=self.name,
            model_requested=self.wire.model,
            started_at=self.clock(),
            finished_at=self.clock(),
            latency_ms=response.total_ms,
            first_byte_ms=response.first_byte_ms,
            cold_connection=response.cold_connection,
            http_status=http_status,
            outcome=outcome if status != "contract_invalid" else "ok",
            uncertain=status == "timeout",
            error=self.redactor.text(response.error or parse_error or "") or None,
            usage_input_tokens=usage_in,
            usage_output_tokens=usage_out,
            estimated_input_tokens=estimated,
            response_bytes=len(response.body) if response.body else 0,
            request_sha256=request_sha,
            raw_request=raw_request,
            raw_response=raw_response,
            headers=response.headers,
        )
        record.extra = {  # type: ignore[attr-defined] - wire + usage are dispatch facts
            "wire_config": self.wire.to_dict(),
            "usage_raw": self.redactor.obj(usage_raw) if usage_raw else None,
            "finish_reason": finish_reason,
        }
        hook.after(handle, usage_in)
        return CallOutcome(
            logical_request_id=logical_request_id,
            status=status,
            terminal=status not in {"timeout"},
            attempts=[record],
            validated=validated,
            usage_input_tokens=usage_in,
            usage_output_tokens=usage_out,
            total_latency_ms=response.total_ms,
            n_retries=0,
            request_sha256=request_sha,
            note=None,
        )

    def close(self) -> None:
        self.transport.close()


def get_api_key() -> str | None:
    """Process-environment key only; never copied or stored."""
    return os.environ.get(API_KEY_ENV) or None


def list_models(transport: Any) -> list[dict[str, Any]]:
    """Public /models catalog via GET. Used by preflight ONLY.

    This distinguishes "API key present" from "model id actually exists and is
    routable": an id absent from the catalog is a hard, recorded preflight
    failure, never silently dispatched.
    """
    response = transport.get(MODELS_PATH)
    if response.error:
        raise OpenRouterError(f"models preflight failed: {response.error}")
    if response.status_code != 200:
        raise OpenRouterError(f"models preflight returned HTTP {response.status_code}")
    parsed, error = _parse_body(response)
    if error or not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        raise OpenRouterError("models preflight returned an unexpected body")
    return parsed["data"]


def model_ids(catalog: list[dict[str, Any]]) -> list[str]:
    return [entry.get("id") for entry in catalog if isinstance(entry, dict) and entry.get("id")]


def _parse_body(response: TransportResponse) -> tuple[Any, str | None]:
    if not response.body:
        return None, "empty_body"
    try:
        return json.loads(response.body.decode("utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"malformed_json:{type(exc).__name__}"


def _excerpt(body: bytes, limit: int = 2000) -> str:
    return body.decode("utf-8", errors="replace")[:limit]


