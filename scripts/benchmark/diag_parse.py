#!/usr/bin/env python3
"""Diagnostic probe: why did strict parsing reject so many matched answers?

The matched sweep counted a strict-format failure as wrong (discipline), but
several models show implausible collapses (Qwen3.7 Flash: 98.5% format-fail
on GPQA, 100% valid-only accuracy) - the parser, not the model, is the
suspect. This probe re-asks a small sample per suspicious model with the RAW
response content stored locally (never bundled), plus wire variants:
  v0: as-run (reasoning_effort=low, max_tokens=1024, no temperature)
  v1: temperature=0 pinned
  v2: reasoning_effort omitted
Items: 4 GPQA + 4 MMLU per model per variant = 24 calls/model. Raw text is
truncated to 400 chars in the artifact. No retries, no scoring claims - this
is a parser/wire diagnostic, and its output decides the v2 re-run config.

  JEVO_ALLOW_LIVE=1 OPENROUTER_API_KEY=... \
    python scripts/benchmark/diag_parse.py --models qwen/qwen3.7-flash
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
import time

from jev_observatory.matched import build_baseline_request, logical_id
from jev_observatory.openrouter import (API_KEY_ENV, CHAT_PATH,
                                        OpenRouterHttpTransport,
                                        OpenRouterProvider, WireConfig,
                                        build_chat_payload)
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs_matched_cheap/diag"

DEFAULT_MODELS = [
    "qwen/qwen3.7-flash", "qwen/qwen3.8-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
    "deepseek/deepseek-v4-flash-0731", "google/gemma-3-4b-it",
    "meta-llama/llama-3.1-8b-instruct", "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash", "openai/gpt-oss-120b",
]


def _items(ds_glob_path: str, ds_name: str, n: int):
    import glob
    if ds_name == "gpqa":
        f = sorted(glob.glob(str(ROOT / "runs_benchmark_ext2/bench-gpqa_diamond-*/items.jsonl")))[0]
    else:
        fz = json.loads((ROOT / "runs_matched/freeze/matched_frozen.json").read_text())
        return fz["datasets"]["mmlu"]["items"][:n]
    rows = [json.loads(l) for l in Path(f).read_text().split("\n") if l.strip()]
    return rows[:n]


def ask_raw(key: str, model: str, req: SystemOneRequest, wire: WireConfig,
            extra: dict | None) -> dict:
    transport = OpenRouterHttpTransport(key)
    provider = OpenRouterProvider(transport, wire=wire)
    lid = logical_id(model, "diag", req.questions and next(iter(req.questions)) or "x")
    try:
        payload = build_chat_payload(req, wire)
        if extra:
            payload.update(extra)
        t0 = time.monotonic()
        resp = transport.post(CHAT_PATH, payload)
        ms = round((time.monotonic() - t0) * 1000)
        body = None
        try:
            body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
        except Exception:
            body = {"unparsed": (resp.body or b"")[:400].decode("utf-8", "replace")}
        out = {"http": resp.status_code, "ms": ms}
        if isinstance(body, dict):
            ch = (body.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            out["finish"] = ch.get("finish_reason")
            out["content"] = (msg.get("content") or "")[:400]
            rc = msg.get("reasoning") or msg.get("reasoning_content") or ""
            out["reasoning_excerpt"] = rc[:200] if isinstance(rc, str) else str(rc)[:200]
            out["usage"] = body.get("usage")
            if body.get("error"):
                out["error"] = str(body["error"])[:300]
        else:
            out["body_excerpt"] = str(body)[:300]
        return out
    finally:
        provider.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=4, help="items per dataset")
    a = ap.parse_args()
    if os.environ.get("JEVO_ALLOW_LIVE") != "1" or not os.environ.get(API_KEY_ENV):
        raise SystemExit("[gate] JEVO_ALLOW_LIVE=1 + OPENROUTER_API_KEY required")
    key = os.environ[API_KEY_ENV]
    models = a.models or DEFAULT_MODELS
    gpqa = _items(None, "gpqa", a.n)
    mmlu = _items(None, "mmlu", a.n)
    variants = {
        "v0_asrun": {"wire": {"reasoning_effort": "low", "max_output_tokens": 1024}, "extra": None},
        "v1_temp0": {"wire": {"reasoning_effort": "low", "max_output_tokens": 1024}, "extra": {"temperature": 0}},
        "v2_noeffort": {"wire": {"reasoning_effort": None, "max_output_tokens": 1024}, "extra": {"temperature": 0}},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for model in models:
        for vname, v in variants.items():
            wire = WireConfig(model=model, **v["wire"])
            for ds, items in (("gpqa", gpqa), ("mmlu", mmlu)):
                for it in items:
                    req, _ = build_baseline_request(it, model=model)
                    gold = next(iter((it.get("gold") or {}).values()), {}).get("value")
                    try:
                        r = ask_raw(key, model, req, wire, v["extra"])
                    except Exception as exc:
                        r = {"exception": str(exc)[:300]}
                    rows.append({"model": model, "variant": vname, "dataset": ds,
                                 "item_id": str(it["id"]), "gold": gold, **r})
                    print(f"{model:42s} {vname:12s} {ds:5s} {str(it['id']):>10s} "
                          f"http={r.get('http')} content={r.get('content','')[:60]!r}")
                    time.sleep(0.4)
    out = OUT_DIR / f"diag_parse_{stamp}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[diag] wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
