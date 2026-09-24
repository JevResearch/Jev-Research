#!/usr/bin/env python3
"""One-call-per-model OpenRouter baseline smoke (parent gate item 4).

Discovers actual access/settings for the nine named modern models with ONE
bounded chat call per model — never a benchmark run, never a retry.

Modes:
  --dry-run   Fully offline: replayed fixture responses via ScriptedTransport;
              validates the adapter end-to-end (payload shape, strict parsing,
              usage accounting, report writing) with NO network and NO key.
  --live      Requires OPENROUTER_API_KEY and JEVO_ALLOW_LIVE=1 in the
              environment. Preflight GET /models first (public catalog) to
              distinguish "key present" from "model actually accessible",
              then exactly one chat call per model with an explicit output
              budget (and reasoning effort when configured).

The smoke question is a tiny synthetic 4-option item built here — NOT a
benchmark item, never written into any benchmark artifact.

Output: runs_benchmark/smoke/openrouter_smoke_<stamp>.json + printed summary.
No credential value is ever printed or persisted.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jev_observatory.openrouter import (
    API_KEY_ENV,
    DEFAULT_MAX_OUTPUT_TOKENS,
    OpenRouterError,
    WireConfig,
    get_api_key,
    list_models,
    model_ids,
)
from jev_observatory.redact import Redactor
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest
from jev_observatory.transport import ScriptedTransport, TransportResponse

# Nine requested modern models; explicit IDs resolved from the PUBLIC
# OpenRouter /models catalog (catalog snapshot: docs/modern-comparison/).
NINE_MODELS = [
    "openai/gpt-6-astra",
    "openai/gpt-5.6-sol",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.8-max-0902",     # plain qwen3.8-max absent from catalog; snapshot pinned
    "qwen/qwen3.8-flash",
    "deepseek/deepseek-v4-flash-0731",
]

SMOKE_QUESTION = SystemOneRequest(
    state="What is 4 + 5? This is a synthetic access check, not a benchmark item.",
    model="wire-smoke",
    questions={"q": ChoiceQuestion(
        type="choice",
        instructions="Answer the multiple-choice question. Reply with only the option key.",
        criteria={"A": "7", "B": "9", "C": "11", "D": "13"},
    )},
)


def smoke_fixture_response(model: str) -> dict:
    """Well-formed chat completion fixture used by --dry-run."""
    return {
        "id": "smoke-fixture",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "B"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 2,
                  "total_tokens": 122, "cost": 0.000012},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="offline fixture replay; no network, no key needed")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--models", default=",".join(NINE_MODELS))
    parser.add_argument("--effort", default=None,
                        help="reasoning effort for the smoke call (recorded per model)")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--root", default="runs_benchmark")
    parser.add_argument("--fail-fixture", action="store_true",
                        help="dry-run only: fixture returns prose instead of a key, "
                             "to prove strict-format failures are counted")
    args = parser.parse_args()
    if args.dry_run == args.live:
        print("[refused] pass exactly one of --dry-run / --live", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wires = {m: WireConfig(model=m, reasoning_effort=args.effort,
                           max_output_tokens=args.max_output_tokens) for m in models}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.root) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openrouter_smoke_{stamp}.json"

    if args.dry_run:
        redactor = Redactor()
        entries = []
        for model in models:
            body = smoke_fixture_response(model)
            if args.fail_fixture:
                body["choices"][0]["message"]["content"] = (
                    "The answer is B because reasoning..."
                )
            transport = ScriptedTransport([{"status_code": 200, "body": body}])
            report = _smoke_one(transport, wires[model], redactor=redactor,
                               preflight=None)
            report["dry_run"] = True
            entries.append(report)
        doc = {
            "mode": "dry-run-fixtures",
            "recorded_at": stamp,
            "models": entries,
            "note": "fixture replay; proves the adapter, strict parsing and report path; "
                    "no network, no key, no model call",
        }
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"[ok] dry-run smoke report: {out_path}")
        return 0

    # ---------------- live mode (parent-invoked inside screen) ----------------
    key = get_api_key()
    if not key:
        print(f"[refused] no {API_KEY_ENV} in environment (never echoed)", file=sys.stderr)
        return 2
    if os.environ.get("JEVO_ALLOW_LIVE") != "1":
        print("[refused] live smoke requires JEVO_ALLOW_LIVE=1", file=sys.stderr)
        return 2
    from jev_observatory.openrouter import OpenRouterHttpTransport, OpenRouterProvider

    redactor = Redactor.from_environment()
    transport = OpenRouterHttpTransport(key)
    try:
        # Preflight: key present != models accessible. The catalog GET is free.
        try:
            catalog_ids = model_ids(list_models(transport))
            preflight = {"models_endpoint": "ok", "n_models": len(catalog_ids)}
        except OpenRouterError as exc:
            doc = {"mode": "live", "recorded_at": stamp,
                   "preflight": {"models_endpoint": "failed", "error": str(exc)},
                   "models": [], "note": "key present but catalog unreachable; nothing dispatched"}
            out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"[halt] preflight failed: {exc}", file=sys.stderr)
            return 3
        absent = [m for m in models if m not in catalog_ids]
        entries = []
        for model in models:
            if model in absent:
                entries.append({"model": model, "preflight": "model_absent_from_catalog",
                                "chat_call": "not_dispatched"})
                continue
            report = _smoke_one(OpenRouterProvider(transport, wire=wires[model]),
                                wires[model], redactor=redactor, preflight="catalog_ok")
            entries.append(report)
        doc = {
            "mode": "live",
            "recorded_at": stamp,
            "preflight": {**preflight, "models_absent": absent},
            "models": entries,
            "note": "one bounded chat call per model; no retries; strict failures counted",
        }
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        dispatched = [e for e in entries if e.get("chat_call", {}).get("dispatched")]
        print(f"[ok] live smoke report: {out_path} "
              f"({len(dispatched)} dispatched, {len(absent)} absent from catalog)")
        return 0
    finally:
        transport.close()


def _smoke_one(transport_or_provider, wire: WireConfig, *, redactor: Redactor,
               preflight) -> dict:
    """Run exactly one smoke dispatch and summarize it honestly."""
    started = time.monotonic()
    if hasattr(transport_or_provider, "ask"):
        provider = transport_or_provider
    else:
        from jev_observatory.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(transport_or_provider, wire=wire, redactor=redactor)
    outcome = provider.ask(SMOKE_QUESTION, "smoke:1", store=None)
    entry: dict = {
        "model_requested": wire.model,
        "wire_config": wire.to_dict(),
        "preflight": preflight,
        "chat_call": {
            "dispatched": bool(outcome.attempts),
            "status": outcome.status,
            "http_status": outcome.attempts[-1].http_status if outcome.attempts else None,
            "latency_ms": round(outcome.total_latency_ms, 3),
            "wall_ms_since_start": round((time.monotonic() - started) * 1000, 3),
            "usage": {
                "input_tokens": outcome.usage_input_tokens,
                "output_tokens": outcome.usage_output_tokens,
            },
            "model_returned": outcome.validated.returned_model if outcome.validated else None,
            "violations": outcome.violation_codes(),
            "strict_format_failure": outcome.status == "contract_invalid",
            "raw_response_sha256": (outcome.attempts[0].raw_response or {}).get("sha256")
            if outcome.attempts else None,
        },
    }
    if outcome.validated is not None and outcome.validated.answers:
        answer = next(iter(outcome.validated.answers.values()))
        entry["chat_call"]["choice"] = answer.values.get("choice")
    return entry


if __name__ == "__main__":
    sys.exit(main())
