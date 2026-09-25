#!/usr/bin/env python3
"""Offline logit-lattice forensics (probe P4's offline arm).

Reads every raw probability vector published in this repository and
characterizes the 0.01 display lattice:

  corpora
    archprobe   runs_archprobe/rows.jsonl                     (987 vectors, K=2..255, with choice)
    live_raw    runs_live/2026-09-18-*/raw/*.response.json    (2,410 answers, with type/choice/confidence)
    ensemble    runs_live/*.json -> nodes[*].ensemble_raw[*]  (per-rotation RAW server vectors at K up to 255;
                transformed/combined vectors are deliberately excluded)

  questions
    1. grid compliance and level coverage per K bucket (is the grid exactly 0.01 everywhere?)
    2. sum-to-1 deviations vs the independent round-to-nearest prediction
       sigma = 0.01 * sqrt(K/12): deviations much smaller than that at large K
       mean the display pipeline applies a bounded correction after rounding.
    3. zero/floor structure: does any probability floor exist (no exact 0.00),
       or do tails round to 0.00 as plain nearest-rounding predicts?
    4. tie and decision structure: displayed ties at the max, choice-vs-table
       argmax mismatches, confidence residuals per K bucket.

Writes data_report/lattice_forensics.json (aggregates only; no vector is
republished). Offline; stdlib only.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_report" / "lattice_forensics.json"

QUANTUM = 0.01
EPS = 1e-9

BUCKETS = [(2, 2), (3, 4), (5, 16), (17, 64), (65, 255)]


def bucket_of(k: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return f">{BUCKETS[-1][1]}"


def on_grid(v: float) -> bool:
    return abs(round(v, 2) - v) <= EPS


# --------------------------------------------------------------- corpora
def load_archprobe() -> list[dict]:
    out = []
    with (ROOT / "runs_archprobe" / "rows.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            p = r.get("probabilities")
            if p:
                out.append({"k": len(p), "vals": list(p.values()),
                            "choice_p": p.get(r["choice"]) if r.get("choice") else None,
                            "order": list(p.values())})
    return out


def load_live_raw() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(ROOT / "runs_live/2026-09-18-*/raw/*.response.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        for a in (d.get("answers") or {}).values():
            if not isinstance(a, dict):
                continue
            p = a.get("probabilities")
            if not isinstance(p, dict) or not p:
                continue
            out.append({"k": len(p), "vals": list(p.values()),
                        "type": a.get("type"),
                        "choice_p": p.get(a["choice"]) if a.get("choice") in p else None,
                        "max_p": max(p.values()),
                        "confidence": a.get("confidence")})
    return out


def load_ensemble() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(ROOT / "runs_live" / "*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [d]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for key, v in o.items():
                    if key == "ensemble_raw" and isinstance(v, list):
                        for e in v:
                            if isinstance(e, dict):
                                cand = e.get("probabilities") if (
                                    "probabilities" in e and isinstance(e.get("probabilities"), dict)
                                ) else e
                                if (cand and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                 for x in cand.values())):
                                    out.append({"k": len(cand), "vals": list(cand.values())})
                    else:
                        stack.append(v)
            elif isinstance(o, list):
                stack.extend(o)
    return out


# --------------------------------------------------------------- metrics
def scan(name: str, vectors: list[dict]) -> dict:
    per_bucket: dict[str, dict] = {}
    offgrid = 0
    n_vals = 0
    mism = 0
    mism_max_gap = 0.0
    for v in vectors:
        k = v["k"]
        b = per_bucket.setdefault(bucket_of(k), {
            "n_vectors": 0, "n_values": 0, "distinct_levels": set(),
            "frac_zero_sum": 0.0, "frac_001_sum": 0.0,
            "dev_sum": 0.0, "dev_sq_sum": 0.0, "dev_max_abs": 0.0,
            "dev_neg": 0, "dev_zero": 0, "dev_pos": 0,
            "k_sum": 0, "distinct_per_vector_sum": 0,
        })
        vals = v["vals"]
        b["n_vectors"] += 1
        b["n_values"] += len(vals)
        b["k_sum"] += k
        n_vals += len(vals)
        zeros = sum(1 for x in vals if abs(x) <= EPS)
        ones = sum(1 for x in vals if abs(x - 0.01) <= EPS)
        b["frac_zero_sum"] += zeros / len(vals)
        b["frac_001_sum"] += ones / len(vals)
        b["distinct_levels"].update(round(x, 2) for x in vals)
        b["distinct_per_vector_sum"] += len({round(x, 2) for x in vals})
        for x in vals:
            if not on_grid(x):
                offgrid += 1
        if len(vals) > 1:
            dev = sum(vals) - 1.0
            b["dev_sum"] += dev
            b["dev_sq_sum"] += dev * dev
            b["dev_max_abs"] = max(b["dev_max_abs"], abs(dev))
            if dev < -0.005 + EPS:
                b["dev_neg"] += 1
            elif dev > 0.005 - EPS:
                b["dev_pos"] += 1
            else:
                b["dev_zero"] += 1
        cp = v.get("choice_p")
        if cp is not None and max(vals) - cp > EPS:
            mism += 1
            mism_max_gap = max(mism_max_gap, max(vals) - cp)
    buckets = {}
    for bname, b in sorted(per_bucket.items()):
        n = b["n_vectors"]
        mean_k = b["k_sum"] / n
        dev_sd = math.sqrt(max(b["dev_sq_sum"] / n - (b["dev_sum"] / n) ** 2, 0.0))
        indep_sd = QUANTUM * math.sqrt(mean_k / 12.0)
        buckets[bname] = {
            "n_vectors": n,
            "n_values": b["n_values"],
            "mean_k": round(mean_k, 1),
            "distinct_levels": len(b["distinct_levels"]),
            "mean_distinct_levels_per_vector": round(b["distinct_per_vector_sum"] / n, 1),
            "frac_values_at_0.00": round(b["frac_zero_sum"] / n, 3),
            "frac_values_at_0.01": round(b["frac_001_sum"] / n, 3),
            "sum_dev_mean": round(b["dev_sum"] / n, 5),
            "sum_dev_sd": round(dev_sd, 5),
            "sum_dev_max_abs": round(b["dev_max_abs"], 4),
            "sum_dev_sign_counts": {"le_-0.005": b["dev_neg"], "near_0": b["dev_zero"],
                                    "ge_+0.005": b["dev_pos"]},
            "independent_rounding_sd_predicted": round(indep_sd, 5),
            "observed_over_predicted_sd": round(dev_sd / indep_sd, 3) if indep_sd else None,
        }
    return {
        "n_vectors": len(vectors),
        "n_values": n_vals,
        "n_offgrid": offgrid,
        "n_choice_not_table_argmax": mism,
        "choice_argmax_max_gap": round(mism_max_gap, 4),
        "by_k_bucket": buckets,
    }


def confidence_by_k(vectors: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for v in vectors:
        if v.get("type") != "choice" or v.get("confidence") is None:
            continue
        k = v["k"]
        pred = round((v["max_p"] - 1.0 / k) / (1.0 - 1.0 / k), 2)
        diff = abs(v["confidence"] - pred)
        b = out.setdefault(bucket_of(k), {"n": 0, "max_diff": 0.0, "beyond_2q": 0})
        b["n"] += 1
        b["max_diff"] = max(b["max_diff"], round(diff, 4))
        if diff > 2.01 * QUANTUM:
            b["beyond_2q"] += 1
    return out


def ties(vectors: list[dict]) -> dict:
    """Displayed ties at the max, among vectors that also carry a choice."""
    n_tied = 0
    winner_first = winner_last = winner_other = 0
    for v in vectors:
        if v.get("choice_p") is None:
            continue
        vals = v["vals"]
        mx = max(vals)
        idxs = [i for i, x in enumerate(vals) if abs(x - mx) <= EPS]
        if len(idxs) < 2:
            continue
        n_tied += 1
        # which tied index holds the choice? we only know its probability, so
        # count vectors where the choice probability equals the tied max
        # (it must be one of them) - positional resolution needs the request
        # order, which the archprobe rows preserve via dict order.
        if abs(v["choice_p"] - mx) <= EPS:
            winner_first += 1  # choice is one of the tied maxima
        else:
            winner_other += 1
    return {"n_vectors_with_displayed_tie_at_max": n_tied,
            "choice_among_tied_maxima": winner_first,
            "choice_not_at_displayed_max": winner_other}


def main() -> int:
    arch = load_archprobe()
    live = load_live_raw()
    ens = load_ensemble()
    out = {
        "generated_by": "scripts/report/lattice_forensics.py (offline; published artifacts only)",
        "claim_level": "behavioral evidence; display-precision facts only (ground rule 3)",
        "method_note": ("ensemble corpus = per-rotation RAW server distributions from "
                        "nodes[*].ensemble_raw in published Talk traces; harness-transformed "
                        "or page-combined vectors are excluded. independent-rounding sigma "
                        "= 0.01*sqrt(K/12) assumes per-value round-to-nearest with no "
                        "cross-value correction."),
        "corpora": {
            "archprobe_rows": scan("archprobe", arch),
            "live_raw_responses": scan("live_raw", live),
            "talk_ensemble_raw": scan("ensemble", ens),
        },
        "confidence_residual_by_k": confidence_by_k(live),
        "displayed_ties": ties(arch + live),
        "provenance": [
            "runs_archprobe/rows.jsonl",
            "runs_live/2026-09-18-*/raw/*.response.json",
            "runs_live/*.json (nodes[*].ensemble_raw)",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)}")
    for cname, c in out["corpora"].items():
        print(f"  {cname}: {c['n_vectors']} vectors, {c['n_values']} values, "
              f"offgrid={c['n_offgrid']}, choice!=argmax={c['n_choice_not_table_argmax']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
