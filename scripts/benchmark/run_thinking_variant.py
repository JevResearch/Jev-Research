#!/usr/bin/env python3
"""Thinking-ON variant pass for the two Qwen flashes (audit follow-up).

The v2 matched sweep disabled thinking for Qwen3.7/3.8-Flash (their
endpoints starved under the old cap). The audit noted the Pareto accounting
charges models for whatever thinking they burn - so for a few dollars we run
the same items a second time with reasoning_effort=low and an 8,192-token
cap, recorded as separate roster entries ("@think"). Both variants then sit
on the charts: the direct answer and the billed-thinking answer, each with
its own measured cost.

Discipline identical to v2: one attempt, recovery parser, transport-only
retries, key from environment, live gate, $10 hard cap for this pass.

  JEVO_ALLOW_LIVE=1 OPENROUTER_API_KEY=... \
    python scripts/benchmark/run_thinking_variant.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rcm2", _HERE / "run_cheap_matched2.py")
rcm2 = importlib.util.module_from_spec(spec)
sys.modules["rcm2"] = rcm2
spec.loader.exec_module(rcm2)

from jev_observatory.matched import build_baseline_request, logical_id
from jev_observatory.openrouter import (CHAT_PATH, OpenRouterHttpTransport,
                                        WireConfig, build_chat_payload)

MODELS = ["qwen/qwen3.7-flash", "qwen/qwen3.8-flash"]
DATASETS = ["mmlu", "gpqa", "math500_choice", "hle_text_mc"]
CAP = 8192
HARD_CAP_USD = 10.0
TOTAL_WORKERS = 8
PER_MODEL = 3
RETRIES = 3


def tagged(mid: str) -> str:
    return mid + "@think"


def main() -> int:
    key = rcm2._live_gate()
    items_all = rcm2.load_items()
    results_p = rcm2.RUN_ROOT / "results.jsonl"
    done = set()
    for line in results_p.read_text().split("\n"):
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("terminal"):
                done.add(r["logical_request_id"])
    tasks = []
    for mid in MODELS:
        for ds in DATASETS:
            for it in items_all[ds]:
                lid = logical_id(tagged(mid), ds, str(it["id"]))
                if lid not in done:
                    tasks.append((mid, ds, it))
    print(f"[think] {len(tasks):,} calls to dispatch "
          f"({len(done) and 'resume'} )", flush=True)
    if not tasks:
        return 0

    lock = threading.Lock()
    res_f = results_p.open("a")
    local = threading.local()
    spend = [0.0]
    stop = threading.Event()
    counts = {"done": 0, "unrec": 0, "err": 0}

    def transport():
        if not hasattr(local, "tr"):
            local.tr = OpenRouterHttpTransport(key)
        return local.tr

    def work(task):
        mid, ds, it = task
        if stop.is_set():
            return
        tmid = tagged(mid)
        lid = logical_id(tmid, ds, str(it["id"]))
        req, _ = build_baseline_request(it, model=mid)
        gold = rcm2.gold_of(it)
        keys = list(req.questions[next(iter(req.questions))].criteria)
        wire = WireConfig(model=mid, reasoning_effort="low",
                          max_output_tokens=CAP,
                          extra_body={"temperature": 0})
        payload = build_chat_payload(req, wire)
        rec = {"logical_request_id": lid, "model": tmid, "dataset": ds,
               "item_id": str(it["id"]), "gold": gold, "wire": wire.to_dict(),
               "variant": "thinking-on"}
        for attempt in range(RETRIES + 1):
            try:
                resp = transport().post(CHAT_PATH, payload)
                body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
                if resp.status_code in (429, 500, 502, 503, 529) and attempt < RETRIES:
                    time.sleep(min(2 ** attempt, 15) + (5 if resp.status_code == 429 else 0))
                    continue
                content, reasoning = rcm2._extract_content_reasoning(body)
                usage = (body or {}).get("usage") if isinstance(body, dict) else None
                pred_strict = content.strip() if content.strip() in keys else None
                pred, stage = rcm2.recover_choice(content, keys, reasoning)
                rec.update({
                    "status": "ok" if resp.status_code == 200 else "http_error",
                    "http_status": resp.status_code,
                    "pred_strict": pred_strict, "pred_recovered": pred,
                    "recovery_stage": stage,
                    "correct_strict": pred_strict == gold if pred_strict else False,
                    "correct_recovered": pred == gold if pred else False,
                    "usage_in": (usage or {}).get("prompt_tokens"),
                    "usage_out": (usage or {}).get("completion_tokens"),
                    "cost_reported": (usage or {}).get("cost"),
                    "terminal": True, "at": rcm2.utc_now(),
                })
                break
            except Exception as exc:
                if attempt >= RETRIES:
                    rec.update({"status": "exception", "error": str(exc)[:200],
                                "terminal": True, "at": rcm2.utc_now()})
                else:
                    time.sleep(min(2 ** attempt, 15))
        with lock:
            res_f.write(json.dumps(rec) + "\n")
            res_f.flush()
            c = rec.get("cost_reported")
            if isinstance(c, (int, float)):
                spend[0] += float(c)
                if spend[0] >= HARD_CAP_USD:
                    stop.set()
                    print(f"[think] CAP ${HARD_CAP_USD} hit (${spend[0]:.2f})", flush=True)
            counts["done"] += 1
            if rec.get("recovery_stage") == "unrecovered":
                counts["unrec"] += 1
            if rec.get("status") != "ok":
                counts["err"] += 1
            if counts["done"] % 500 == 0:
                print(f"[think] {counts['done']:,}/{len(tasks):,} "
                      f"spend=${spend[0]:.2f} unrec={counts['unrec']} err={counts['err']}",
                      flush=True)

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for _ in as_completed(futs):
                pass
    finally:
        res_f.close()
    el = time.monotonic() - t0
    print(f"[think] {counts['done']:,} calls in {el/60:.1f} min; "
          f"spend=${spend[0]:.2f}; unrec={counts['unrec']}; err={counts['err']}")
    out = rcm2.RUN_ROOT / "live_summary_think.json"
    out.write_text(json.dumps({"finished_at": rcm2.utc_now(),
                               "calls": counts["done"],
                               "spend_usd": round(spend[0], 4),
                               "unrecovered": counts["unrec"],
                               "errors": counts["err"],
                               "wall_seconds": round(el)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
