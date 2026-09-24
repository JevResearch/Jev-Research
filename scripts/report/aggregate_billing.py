#!/usr/bin/env python3
"""Aggregate every Jev call we made from on-disk billing artifacts.

Single source of truth for the report hero's "total our measurements cost"
statistic (previously hardcoded). Walks, with documented dedup rules:

  1. runs_benchmark*/bench-*/derived/score.json  -> attempts_summary (7 stages)
  2. runs_benchmark*/bench-arc_agi2-*/attempts.jsonl and the aborted stage
     -> per-attempt usage (score.json for these stages carries no summary)
  3. runs_archprobe/BILLING.json                 -> 987-call architecture battery
  4. runs_live/BILLING.json                      -> phase-1 probes + the original
     character-level Talk (so talk_live_traces.json is skipped in (5) to avoid
     counting those 14 calls twice)
  5. runs_live/*.json Talk-program traces        -> per-node usage_input_tokens,
     probe files via their explicit total_in_tokens fields; infrastructure and
     aggregate-only files excluded

Dedup rules (also recorded in the output):
  - matrix_summary.json carries no usage (stop/steps/in per session) but the
    per-session matrix_* traces do; summary is excluded regardless.
  - token_talk_comparison.json has a file-level total AND per-node usage; we
    count nodes once and take calls from them.
  - talk_live_traces.json (original char-level) is covered by runs_live
    BILLING's "talk" entry -> skipped.

Usage: python scripts/report/aggregate_billing.py
Writes: data_report/billing_totals.json
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("JEVO_ROOT", str(Path(__file__).resolve().parents[2])))
OUT = ROOT / "data_report" / "billing_totals.json"


def _price() -> float:
    costs = json.loads((ROOT / "data_report" / "costs.json").read_text())
    return costs["prices_usd_per_M"]["Jev"]["input"]


def _attempts_sum(path: Path) -> tuple[int, int]:
    calls = toks = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                at = json.loads(line)
            except json.JSONDecodeError:
                continue
            if at.get("outcome") == "ok":
                calls += 1
                toks += at.get("usage_input_tokens") or 0
    return calls, toks


def _walk_usage(node) -> tuple[int, int]:
    """Count (calls, tokens) from usage_input_tokens occurrences."""
    calls = toks = 0
    if isinstance(node, dict):
        u = node.get("usage_input_tokens")
        if isinstance(u, (int, float)) and u:
            calls += 1
            toks += int(u)
        for v in node.values():
            c, t = _walk_usage(v)
            calls += c
            toks += t
    elif isinstance(node, list):
        for v in node:
            c, t = _walk_usage(v)
            calls += c
            toks += t
    return calls, toks


def collect() -> dict:
    sources = []

    # 1. score.json attempts summaries
    for f in sorted(glob.glob(str(ROOT / "runs_benchmark*/bench-*/derived/score.json"))):
        d = json.loads(Path(f).read_text())
        a = d.get("attempts_summary")
        if not a:
            continue
        sources.append({
            "name": Path(f).parents[1].name,
            "provenance": str(Path(f).relative_to(ROOT)),
            "calls": a["n_attempts"],
            "input_tokens": a["usage_input_tokens_sum"],
            "method": "derived/score.json attempts_summary",
        })

    # 2. arc_agi stages: usage_summary.json (published roll-up) or attempts.jsonl
    for u in sorted(glob.glob(str(ROOT / "runs_benchmark*/bench-arc_agi2_*/derived/usage_summary.json"))):
        d = json.loads(Path(u).read_text())
        sources.append({
            "name": Path(u).parents[1].name,
            "provenance": str(Path(u).relative_to(ROOT)),
            "calls": d["n_attempts_ok"], "input_tokens": d["usage_input_tokens_sum"],
            "method": "derived/usage_summary.json (published roll-up)",
        })
    for f in sorted(glob.glob(str(ROOT / "runs_benchmark*/bench-arc_agi2_*/attempts.jsonl")))             if not any("usage_summary" in x["provenance"] for x in sources) else []:
        calls, toks = _attempts_sum(Path(f))
        sources.append({
            "name": Path(f).parent.name, "provenance": str(Path(f).relative_to(ROOT)),
            "calls": calls, "input_tokens": toks,
            "method": "attempts.jsonl (outcome=ok)",
        })
    # aborted extension stage: published aggregate, else local attempts
    agg_file = ROOT / "data_report/aborted_stage_usage.json"
    if agg_file.exists():
        d = json.loads(agg_file.read_text())
        calls = sum(v["n_attempts_ok"] for v in d["stages"].values())
        toks = sum(v["usage_input_tokens_sum"] for v in d["stages"].values())
        sources.append({
            "name": "aborted1-extension-stage",
            "provenance": str(agg_file.relative_to(ROOT)),
            "calls": calls, "input_tokens": toks,
            "method": "data_report/aborted_stage_usage.json (real billed spend; raw logs unpublished)",
        })

    # 3. architecture battery
    ap = json.loads((ROOT / "runs_archprobe/BILLING.json").read_text())
    sources.append({
        "name": "archprobe-battery", "provenance": "runs_archprobe/BILLING.json",
        "calls": ap["n_calls"], "input_tokens": ap["usage_input_tokens"],
        "method": "BILLING.json",
    })

    # 4. phase-1 runs + original char-level Talk
    lv = json.loads((ROOT / "runs_live/BILLING.json").read_text())
    agg = lv["updated_after_audit"]
    sources.append({
        "name": "runs-live-phase1", "provenance": "runs_live/BILLING.json",
        "calls": agg["total_calls"], "input_tokens": agg["total_input_tokens"],
        "method": "BILLING.json updated_after_audit (includes original 14-call char Talk)",
    })

    # 5. this thread's Talk program (token/word/char-v2+ probes)
    skip = {"BILLING.json", "talk_live_traces.json", "matrix_summary.json",
            "external_scores.json", "heatmap.json", "next_run_ids.json",
            "retained_anchors.json"}
    talk_calls = talk_toks = 0
    n_files = 0
    for f in sorted(glob.glob(str(ROOT / "runs_live/*.json"))):
        p = Path(f)
        if p.name in skip:
            continue
        d = json.loads(p.read_text())
        if "total_in_tokens" in d:           # orderprobe / ensemble probe summaries
            talk_toks += d["total_in_tokens"]
            n_files += 1
            continue
        if isinstance(d, dict) and "grid" in d:  # sweep results: per-run in_tokens
            talk_toks += sum(g.get("in_tokens") or 0 for g in d["grid"])
            n_files += 1
            continue
        c, t = _walk_usage(d)
        talk_calls += c
        talk_toks += t
        n_files += 1
    sources.append({
        "name": "talk-program", "provenance": f"runs_live/*.json ({n_files} artifacts)",
        "calls": talk_calls, "input_tokens": talk_toks,
        "method": "per-node usage_input_tokens; probe summaries via total_in_tokens fields",
    })

    price = _price()
    tot_calls = sum(s["calls"] for s in sources)
    tot_toks = sum(s["input_tokens"] for s in sources)
    return {
        "generated_by": "scripts/report/aggregate_billing.py",
        "price_usd_per_M_input": price,
        "method": ("input tokens are server-reported and billed; output tokens "
                   "are free. Totals read only from on-disk artifacts; dedup "
                   "rules are documented in the script header."),
        "sources": sources,
        "totals": {
            "calls_recorded": tot_calls,
            "input_tokens": tot_toks,
            "usd": round(tot_toks * price / 1e6, 2),
        },
    }


def main() -> None:
    doc = collect()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    t = doc["totals"]
    print(f"[ok] {OUT.relative_to(ROOT)}: {t['calls_recorded']:,} recorded calls, "
          f"{t['input_tokens']:,} input tokens, ${t['usd']}")


if __name__ == "__main__":
    main()
