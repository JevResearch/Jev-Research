#!/usr/bin/env python3
"""Cost model for the Pareto frontier section.

Jev's costs are MEASURED: we ran every benchmark and billed only input tokens
(output is free), at $0.042/M input. For comparison models the per-benchmark
cost is ESTIMATED from public list prices and per-item token priors, anchored
wherever a measured cost/question exists (OpenRouter GPQA $0.099/question for
GPT-6 Astra; Vals MMLU-Pro $0.028/test for Claude Opus 5). Estimates carry a
documented +/- few-x caveat; the frontier claims below are 3-6 orders of
magnitude, far beyond that error bar. Every prior is written to the output JSON
so the reader can audit it.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_report/costs.json"

# per-M-token list prices, USD (in/out). "price_note" records date/source basis.
PRICES = {
    "GPT-6 Astra":            (10.0, 50.0, "OpenAI list, Sep 2026"),
    "GPT-5.6 Sol":            (4.0, 20.0, "OpenAI promo rate (standard 5/30)"),
    "Claude Fable 5.1":       (10.0, 50.0, "Anthropic list, Sep 2026"),
    "Claude Opus 5":          (5.0, 25.0, "Anthropic list, Sep 2026"),
    "GLM-5.3":                (1.40, 4.40, "z.ai list, Aug 2026"),
    "GLM-5.3 Flash":          (0.15, 0.50, "z.ai list (promos below this exist)"),
    "Qwen3.8 Max":            (1.65, 4.95, "Alibaba Cloud list, Aug 2026"),
    "DeepSeek V4 Flash 0731": (0.44, 1.32, "peak rate; off-peak 0.22/0.66"),
    "GPT-4o":                 (2.50, 10.0, "launch May 2024 (08-06 same)"),
    "Claude 3.7 Sonnet (no thinking)": (3.00, 15.0, "launch Feb 2025"),
    "Jev":                    (0.042, 0.0, "measured billing rate; output free"),
}

# per-item token priors (in/out) for the protocol each external score comes from
PRIORS = {
    "mmlu_pro":    {"in": 750, "out": 300,  "n": 12032,
                    "basis": "Vals 5-shot CoT; question+10 options+few-shot in, short CoT out"},
    "gpqa":        {"in": 350, "out": 1800, "n": 196,
                    "basis": "AA-style reasoning; Astra measured 1.77k out/question on OpenRouter"},
    "arc":         {"in": 420, "out": 150,  "n": 1172,
                    "basis": "25-shot MCQ, short answers"},
    "math500":     {"in": 200, "out": 3000, "n": 500,
                    "basis": "free-form math with reasoning"},
    "hle":         {"in": 400, "out": 6000, "n": 494,
                    "basis": "reasoning on the MC sub-track"},
}

# Jev measured usage (tokens summed from attempts.jsonl of the actual runs)
JEV_MEASURED_IN_TOKENS = {
    "mmlu_pro": 6734956, "gpqa": 105465, "arc": 451947,
    "math500": 115490, "hle": 453718,   # MC sub-track (494 items)
}

# measured anchors used for calibration (USD per item, cost of the whole run at
# the anchor model); other models scale by blend-price ratio at the same priors.
ANCHORS = {
    "gpqa":     {"model": "GPT-6 Astra", "per_item": 0.099,
                 "source": "OpenRouter GPQA leaderboard cost/question"},
    "mmlu_pro": {"model": "Claude Opus 5", "per_item": 0.028,
                 "source": "Vals MMLU-Pro 'cost per test' column"},
}

def _blend(model: str) -> float:
    pi, po, _ = PRICES[model]
    return pi, po

def external_cost(model: str, bench: str) -> dict:
    """Estimate cost to run `bench` for one external model (USD).

    Base estimate = n_items x (in*Pin + out*Pout)/1e6. Where a measured anchor
    exists for that benchmark at a model we have, rescale it by the model's
    token-blend price relative to the anchor model (keeps priors honest).
    """
    p = PRIORS[bench]; pi, po = PRICES[model][:2]
    base = p["n"] * (p["in"] * pi + p["out"] * po) / 1e6
    anchored, anchor_src = base, None
    a = ANCHORS.get(bench)
    if a and model != a["model"]:
        api, apo = PRICES[a["model"]][:2]
        anchor_blend = p["in"] * api + p["out"] * apo
        model_blend = p["in"] * pi + p["out"] * po
        if anchor_blend > 0:
            anchored = a["per_item"] * p["n"] * model_blend / anchor_blend
            anchor_src = f"calibrated on {a['model']} measured cost ({a['source']})"
    return {"usd": round(anchored, 2), "usd_uncalibrated": round(base, 2),
            "method": "anchored estimate" if anchor_src else "prior estimate",
            "anchor": anchor_src}

def main() -> None:
    benches = list(PRIORS)
    out = {"prices_usd_per_M": {m: {"input": pi, "output": po, "note": nt}
                                for m, (pi, po, nt) in PRICES.items()},
           "priors": PRIORS, "anchors": ANCHORS,
           "jev_measured_input_tokens": JEV_MEASURED_IN_TOKENS,
           "caveat": ("External costs are estimates from public list prices and "
                      "per-item token priors; where a measured cost/question "
                      "exists we calibrate to it. Expect a few-x error on "
                      "estimates; the frontier gaps shown are 3-6 orders of "
                      "magnitude. Jev's own costs are measured from billing."),
           "costs_usd": {}}
    for b in benches:
        out["costs_usd"][b] = {"Jev": {
            "usd": round(JEV_MEASURED_IN_TOKENS[b] * 0.042 / 1e6, 4),
            "method": "measured billing (input tokens x $0.042/M; output free)"}}
        for m in PRICES:
            if m == "Jev":
                continue
            out["costs_usd"][b][m] = external_cost(m, b)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for b in benches:
        js = out["costs_usd"][b]["Jev"]["usd"]
        cheap = min(v["usd"] for k, v in out["costs_usd"][b].items() if k != "Jev")
        exp = max(v["usd"] for k, v in out["costs_usd"][b].items() if k != "Jev")
        print(f"{b:10s} Jev ${js:>8.4f} | cheapest ext ${cheap:>9.2f} "
              f"| dearest ${exp:>10.2f} | gap {cheap/js:>6.0f}x - {exp/js:>8.0f}x")

if __name__ == "__main__":
    main()
