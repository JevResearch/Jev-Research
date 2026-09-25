#!/usr/bin/env python3
"""Offline re-derivation of the secondary architecture audits cited by the
report's architecture section and by ARCHITECTURE-ANALYSIS.md.

Everything here reads published on-disk artifacts only (no network, no live
calls) and writes one aggregate JSON:

    python scripts/report/arch_audits.py     ->  data_report/arch_audits.json

Audits
  1. quantization   - 0.01-grid compliance, distinct levels, sum-to-1
                      deviations, choice-vs-table argmax mismatches, over all
                      probability vectors in runs_archprobe/rows.jsonl and the
                      published runs_live/2026-09-18-*/raw/*.response.json.
  2. confidence     - fit of the choice `confidence` field against the
                      chance-corrected top-probability statistic
                          conf ?= round((p_max - 1/K) / (1 - 1/K), 2)
                      computed on the DISPLAYED (quantized) vector, bucketed
                      by |diff| in display quanta; same statistic applied to
                      score-type answers to show it is NOT their formula;
                      noul answers carry no confidence field at all.
  3. marginal_cost  - decomposition of the headcount/optioncount marginal
                      latencies (runs_archprobe/analysis.json) into the
                      prefill cost of the tokens each question/option adds
                      (rows.jsonl usage counts x prefill slope) and a
                      residual, i.e. an upper bound on the read-out cost.
  4. battery_power  - discriminating information (tokens above the empty
                      template) per probe, short tokenizer battery vs the
                      whitespace-free mergerate samples: why the second,
                      template-free table is the decisive tokenizer test.
  5. rotation_pairs - paired native-vs-rotated accuracy on the shared
                      option-rotation audit items (runs_benchmark score.json
                      per_item lists), with exact McNemar.
  6. prefill_shape  - refit of upstream_ms vs input tokens for the prefill
                      family (linear and quadratic), plus the mean residual
                      at the longest probes: bounds the attention-curvature
                      signature without claiming a parameter count.

Behavioral evidence only: every input is an API-visible signal (answers,
probability vectors, usage token counts, server-compute timing headers).
"""

from __future__ import annotations

import glob
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_report" / "arch_audits.json"

QUANTUM = 0.01


def _lstsq(X: list[list[float]], y: list[float]) -> list[float]:
    n, k = len(y), len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    bv = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    M = [A[a][:] + [bv[a]] for a in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[a][k] / M[a][a] for a in range(k)]


def _r2(y: list[float], pred: list[float]) -> float:
    ybar = sum(y) / len(y)
    ss = sum((a - b) ** 2 for a, b in zip(y, pred))
    tot = sum((a - ybar) ** 2 for a in y)
    return 1 - ss / tot if tot else float("nan")


def _rows() -> list[dict]:
    with (ROOT / "runs_archprobe" / "rows.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _live_vectors() -> list[tuple[str, dict]]:
    """(source-relative path, answer-object) for every published raw response."""
    out: list[tuple[str, dict]] = []
    for p in sorted(glob.glob(str(ROOT / "runs_live/2026-09-18-*/raw/*.response.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        for qid, a in (d.get("answers") or {}).items():
            if isinstance(a, dict):
                out.append((str(Path(p).relative_to(ROOT)) + "#" + qid, a))
    return out


def audit_quantization(rows: list[dict], live: list[tuple[str, dict]]) -> dict:
    def scan(vectors: list[dict]) -> dict:
        vals: list[float] = []
        sum_dev = []
        mism = 0
        mism_gap = 0.0
        for v in vectors:
            probs = v["probabilities"]
            vs = list(probs.values())
            vals += vs
            if len(vs) > 1:
                sum_dev.append(sum(vs) - 1.0)
            ch = v.get("choice")
            if ch and ch in probs and vs and max(vs) - probs[ch] > 1e-9:
                mism += 1
                mism_gap = max(mism_gap, max(vs) - probs[ch])
        offgrid = sum(1 for x in vals if abs(round(x, 2) - x) > 1e-9)
        return {
            "n_vectors": len(vectors),
            "n_values": len(vals),
            "n_offgrid": offgrid,
            "n_distinct_levels": len({round(x, 2) for x in vals}),
            "sum_dev_max_abs": round(max((abs(d) for d in sum_dev), default=0.0), 4),
            "n_sum_dev_gt_half_quantum": sum(1 for d in sum_dev if abs(d) > QUANTUM / 2 + 1e-9),
            "n_choice_not_table_argmax": mism,
            "choice_argmax_max_gap": round(mism_gap, 4),
        }

    arch_vectors = [r for r in rows if r.get("probabilities")]
    live_vectors = [a for _, a in live if isinstance(a.get("probabilities"), dict)]
    return {
        "method": "grid = 0.01; a value is off-grid if round(v,2) != v; "
                  "choice-vs-table mismatch = displayed choice probability < displayed max",
        "archprobe_rows": scan(arch_vectors),
        "runs_live_raw": scan(live_vectors),
        "provenance": ["runs_archprobe/rows.jsonl", "runs_live/2026-09-18-*/raw/*.response.json"],
    }


def audit_confidence(live: list[tuple[str, dict]]) -> dict:
    def cc(pmax: float, K: int) -> float:
        return round((pmax - 1.0 / K) / (1.0 - 1.0 / K), 2)

    def fit(answers: list[dict]) -> dict:
        exact = w1 = w2 = beyond = 0
        worst = 0.0
        for a in answers:
            if "confidence" not in a or not isinstance(a.get("probabilities"), dict):
                continue
            probs = a["probabilities"]
            K = len(probs)
            if K < 2:
                continue
            pred = cc(max(probs.values()), K)
            diff = abs(a["confidence"] - pred)
            worst = max(worst, diff)
            if diff < 1e-9:
                exact += 1
            elif diff <= 1.01 * QUANTUM:
                w1 += 1
            elif diff <= 2.01 * QUANTUM:
                w2 += 1
            else:
                beyond += 1
        return {
            "n": exact + w1 + w2 + beyond,
            "exact": exact, "within_1_quantum": w1, "within_2_quanta": w2,
            "beyond_2_quanta": beyond, "max_abs_diff": round(worst, 4),
        }

    choice = [a for _, a in live if a.get("type") == "choice"]
    score = [a for _, a in live if a.get("type") == "score"]
    noul = [a for _, a in live if a.get("type") == "noul"]
    return {
        "formula_tested": "confidence == round((p_max_displayed - 1/K) / (1 - 1/K), 2)",
        "note": "chance-corrected top-probability (kappa-style). Residuals of 1-2 display "
                "quanta are expected when confidence is computed on the pre-quantization "
                "distribution while the table is rounded separately (see quantization audit: "
                "choice/table argmax mismatches at exactly one quantum).",
        "choice": fit(choice),
        "score_same_formula": fit(score),
        "noul_answers": len(noul),
        "noul_with_confidence_field": sum(1 for a in noul if "confidence" in a),
        "provenance": ["runs_live/2026-09-18-*/raw/*.response.json"],
    }


def audit_marginal_cost(rows: list[dict], analysis: dict) -> dict:
    slope = analysis["prefill"]["ms_per_1k_input_tokens"]

    def unit_cost(family: str, unit_field: str) -> dict:
        fam = [r for r in rows if r["family"] == family]
        lo = min(fam, key=lambda r: r[unit_field])
        hi = max(fam, key=lambda r: r[unit_field])
        dn = hi[unit_field] - lo[unit_field]
        din = hi["usage_input_tokens"] - lo["usage_input_tokens"]
        dout = hi["usage_output_tokens"] - lo["usage_output_tokens"]
        per_in = din / dn
        per_out = dout / dn
        predicted_ms = per_in * slope / 1000.0
        measured_ms = (analysis["headcount"]["marginal_ms_per_question"] if family == "headcount"
                       else analysis["optioncount"]["marginal_ms_per_option"])
        return {
            "units": f"{lo[unit_field]}->{hi[unit_field]}",
            "input_tokens_per_unit": round(per_in, 1),
            "output_tokens_per_unit": round(per_out, 1),
            "prefill_predicted_ms_per_unit": round(predicted_ms, 3),
            "measured_marginal_ms_per_unit": measured_ms,
            "residual_ms_per_unit": round(measured_ms - predicted_ms, 3),
        }

    return {
        "method": "marginal latency per question/option (analysis.json) vs the prefill cost "
                  "of the tokens that unit adds to the request (rows.jsonl usage deltas x "
                  "prefill slope). residual = everything else (read-out, serialization, noise).",
        "per_question": unit_cost("headcount", "n_questions"),
        "per_option": unit_cost("optioncount", "n_options_total"),
        "prefill_slope_ms_per_1k": slope,
        "provenance": ["runs_archprobe/rows.jsonl", "runs_archprobe/analysis.json"],
    }


def audit_battery_power(rows: list[dict]) -> dict:
    tk = [r["usage_input_tokens"] for r in rows if r["family"] == "tokenizer"]
    mr = [r for r in rows if r["family"] == "mergerate"]
    space_runs = [r["usage_input_tokens"] for r in mr if r.get("sample") == "space"]
    base = min(space_runs) if space_runs else 316
    mr_delta = [r["usage_input_tokens"] - base for r in mr]
    return {
        "method": "discriminating information per probe = reported tokens above the empty-"
                  "template baseline. The short battery's deltas are a few tokens wide; the "
                  "whitespace-free mergerate samples are one to two orders of magnitude wider.",
        "short_battery_reported_token_range": [min(tk), max(tk)],
        "short_battery_max_delta_over_baseline": max(tk) - base,
        "mergerate_baseline_tokens": base,
        "mergerate_delta_range": [min(mr_delta), max(mr_delta)],
        "provenance": ["runs_archprobe/rows.jsonl", "runs_archprobe/cleanrun_jev.json"],
    }


def audit_rotation_pairs() -> dict:
    mp = glob.glob(str(ROOT / "runs_benchmark/bench-mmlu_full-*/derived/score.json"))[0]
    rp = glob.glob(str(ROOT / "runs_benchmark/bench-option_rotations-*/derived/score.json"))[0]
    m = json.loads(Path(mp).read_text(encoding="utf-8"))
    r = json.loads(Path(rp).read_text(encoding="utf-8"))
    native = {it["item_id"].split(":")[0]: it for it in m["per_item"]}
    pairs = []
    for it in r["per_item"]:
        iid = it["item_id"].split(":")[0]
        if iid in native and it.get("status") == "ok" and native[iid].get("status") == "ok":
            pairs.append((bool(native[iid]["correct"]), bool(it["correct"])))
    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    nat_only = sum(1 for a, b in pairs if a and not b)
    rot_only = sum(1 for a, b in pairs if b and not a)
    neither = n - both - nat_only - rot_only
    disc = nat_only + rot_only
    k = min(nat_only, rot_only)
    p = min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2 ** disc) if disc else 1.0
    return {
        "method": "pair rotation-audit items to their native-order results by base item id "
                  "(rotation ids carry a :permN suffix); exact two-sided McNemar on discordant "
                  "pairs. Wilson CIs in score.json are item-sampling diagnostics.",
        "n_paired": n, "both_correct": both, "native_only": nat_only,
        "rotated_only": rot_only, "neither": neither,
        "native_acc_on_subset": round((both + nat_only) / n, 4),
        "rotated_acc": round((both + rot_only) / n, 4),
        "flip_rate": round(disc / n, 5),
        "mcnemar_exact_p": round(p, 3),
        "provenance": [
            "runs_benchmark/bench-mmlu_full-*/derived/score.json (per_item)",
            "runs_benchmark/bench-option_rotations-*/derived/score.json (per_item)",
        ],
    }


def audit_prefill_shape(rows: list[dict]) -> dict:
    pf = [r for r in rows if r["family"] == "prefill"]
    xs = [r["usage_input_tokens"] for r in pf]
    ys = [r["upstream_ms"] for r in pf]
    cl = _lstsq([[1, x / 1000] for x in xs], ys)
    predl = [cl[0] + cl[1] * x / 1000 for x in xs]
    cq = _lstsq([[1, x / 1000, (x / 1000) ** 2] for x in xs], ys)
    predq = [cq[0] + cq[1] * x / 1000 + cq[2] * (x / 1000) ** 2 for x in xs]
    top = [x for x, y in zip(xs, ys) if x >= 25000]
    top_res = [y - (cl[0] + cl[1] * x / 1000) for x, y in zip(xs, ys) if x >= 25000]
    return {
        "method": "refit of upstream_ms (x-envoy-upstream-service-time) vs server-reported "
                  "input tokens; residual check at the longest prompts bounds any attention "
                  "quadratic. Slopes are scheduling-inclusive marginal costs under shared "
                  "batching - NOT parameter-count evidence.",
        "n_configs": len(pf),
        "token_range": [min(xs), max(xs)],
        "linear_floor_ms": round(cl[0], 2),
        "linear_ms_per_1k": round(cl[1], 3),
        "linear_r2": round(_r2(ys, predl), 3),
        "quadratic_ms_per_1k2": round(cq[2], 4),
        "quadratic_r2": round(_r2(ys, predq), 3),
        "r2_gain_from_quadratic": round(_r2(ys, predq) - _r2(ys, predl), 3),
        "n_top_bin": len(top),
        "top_bin_mean_residual_vs_linear_ms": round(sum(top_res) / len(top_res), 1) if top_res else None,
        "provenance": ["runs_archprobe/rows.jsonl"],
    }


def main() -> int:
    rows = _rows()
    live = _live_vectors()
    analysis = json.loads((ROOT / "runs_archprobe" / "analysis.json").read_text(encoding="utf-8"))
    out = {
        "generated_by": "scripts/report/arch_audits.py (offline; published artifacts only)",
        "claim_level": "behavioral evidence; API-visible signals only",
        "quantization": audit_quantization(rows, live),
        "confidence_formula": audit_confidence(live),
        "marginal_cost": audit_marginal_cost(rows, analysis),
        "battery_power": audit_battery_power(rows),
        "rotation_pairs": audit_rotation_pairs(),
        "prefill_shape": audit_prefill_shape(rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
