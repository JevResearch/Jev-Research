#!/usr/bin/env python3
"""Two-angle model-size estimation for jev-1.13.0 -> data_report/size_estimate.json

ANGLE 1 - prefill throughput. The measured marginal prefill rate (6.053 ms per
1k input tokens, runs_archprobe/analysis.json) is ~165k tokens/s of
incremental server compute. REVISED 2026-09-25: the first pass assumed bf16 on
A100/H100-class hardware (250-500 TFLOPS), which is dated - late-2026 serving
at this price point is quantized (fp8/int4) on H100/H200/B200/TPU-v6/MI325X-
class parts (400-2250 effective TFLOPS). The revision raises the active-
parameter bound ~2-4x and changes the reconciliation. Under standard serving assumptions - a
compute-bound batched prefill (165k tok/s per stream is far above the
memory-bound regime), FLOPs ~= 2 * N_active per token (Kaplan/Chinchilla
convention), and an accelerator delivering MFU x peak - the ACTIVE parameter
count that can sustain that rate is bounded by

    N_active <= MFU * peak_FLOPS * S / (2 * R)

with S = devices sharing the pass and R = 165k tok/s. Multi-tenant batching
(B > 1 concurrent streams sharing weight fetches) only LOWERS the per-stream
bound, so B = 1 is the conservative choice. This is NOT a slope-to-size
conversion forbidden by ground rule 4 - it is an explicit-assumptions bound,
and the assumptions are enumerated with ranges, not hidden.

ANGLE 2 - capability band. Direct-answer MMLU-Pro 82.8 / GPQA 76.5 sit beside
Qwen 3.5 9B (82.5 / 77.6, reasoning protocol) and above Claude 3.7 Sonnet
no-thinking (80.7 / 76.8, direct): a 2025-era ~9B-class instruct band,
4-14B dense-equivalent with the protocol caveat.

The two angles overlap only under specific reconciliations (MoE active-vs-total
split, quantized serving, distillation raising capability-per-active-parameter,
or a soft top of the throughput band). This script records both bands, the
reconciliation readings, and a single converged statement. Speculative by
construction; labeled as such everywhere.

  python scripts/report/size_estimate.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_report" / "size_estimate.json"

# Assumption grids (industry-standard ranges; NOT repo measurements).
# Late-2026 revision: serving at this price point is quantized (fp8 the
# conservative floor, int4/NVFP4-class the aggressive end), and the hardware
# range spans H100/H200-class through B200/TPU-v6/MI325X-class accelerators.
# The bf16-on-A100 grid this script shipped with was dated and biased the
# active-parameter bound low by ~2-10x; it is retained below for continuity.
MFU_RANGE = (0.25, 0.45)              # achieved utilization, large-batch prefill
# BF16 nameplate per device for the plausible late-2026 fleet (the numbers the
# audit supplied): ~400 (smaller parts) to ~2250 (B200-class dense).
BF16_FLEET_TFLOPS_RANGE = (400, 2250)
# Quantization multiplies EFFECTIVE math throughput over bf16 nameplate:
#   fp8 ~2x, int4/NVFP4-class ~4x. Serving at this price point is quantized,
#   so the honest effective peak is the bf16 fleet range scaled by 2-4x.
QUANT_MULT = (2, 4)
PEAK_EFF_TFLOPS_RANGE = (BF16_FLEET_TFLOPS_RANGE[0] * QUANT_MULT[0],
                         BF16_FLEET_TFLOPS_RANGE[1] * QUANT_MULT[1])  # (800, 9000)
SHARDS = (1, 2, 4)                    # devices jointly serving one pass
# Prior (dated) grid, kept for the audit trail:
BF16_LEGACY = {"peak_TFLOPS": (250, 500),
               "note": "bf16 dense, A100/H100-class; what the first pass assumed"}


def _band(R: float, mfu_range, peak_range, shards) -> list:
    lo_mfu, hi_mfu = mfu_range
    lo_pk, hi_pk = peak_range
    n_lo = lo_mfu * lo_pk * 1e12 * min(shards) / (2 * R)
    n_hi = hi_mfu * hi_pk * 1e12 * max(shards) / (2 * R)
    return [round(n_lo / 1e9, 2), round(n_hi / 1e9, 2)]


def band_from_prefill(prefill: dict) -> dict:
    slope_ms = prefill["ms_per_1k_input_tokens"]
    R = 1000.0 / slope_ms * 1000.0            # tokens/s marginal
    band = _band(R, MFU_RANGE, PEAK_EFF_TFLOPS_RANGE, SHARDS)
    legacy = _band(R, MFU_RANGE, BF16_LEGACY["peak_TFLOPS"], SHARDS)
    # central case: fp8 on H100/H200-class (~2x its ~990 bf16 => ~1979 eff),
    # MFU 0.35, S=2
    n_mid = 0.35 * 1979e12 * 2 / (2 * R)
    return {
        "method": "N_active <= MFU * peak_effective * S / (2 * R); R from the measured marginal prefill slope",
        "marginal_prefill_rate_tok_s": round(R),
        "assumptions": {
            "MFU": list(MFU_RANGE),
            "peak_effective_TFLOPS_per_device": list(PEAK_EFF_TFLOPS_RANGE),
            "bf16_nameplate_fleet_TFLOPS": list(BF16_FLEET_TFLOPS_RANGE),
            "quant_multiplier": list(QUANT_MULT),
            "peak_basis": ("bf16 nameplate for the late-2026 fleet is ~400-2250 "
                           "TFLOPS/device (H100/H200 ~990; B200-class ~2250; "
                           "smaller parts ~400); quantized serving multiplies "
                           "effective math throughput ~2x (fp8) to ~4x "
                           "(int4/NVFP4), giving ~800-9000 effective"),
            "shard_count_S": list(SHARDS),
            "flops_per_token": "2 * N_active (attention/overhead excluded; adds <= ~15% at <=29k ctx)",
            "concurrent_streams_B": "1 (conservative: batching with other tenants only lowers the per-stream bound)",
            "regime": "compute-bound batched prefill (165k tok/s marginal is far above memory-bound decode rates)",
        },
        "active_params_B_range": band,
        "central_case_B": round(n_mid / 1e9, 2),
        "legacy_bf16_A100_band_B": legacy,
        "legacy_note": BF16_LEGACY["note"],
        "reading": ("Serving this marginal prefill rate on 1-4 quantized-serving-era "
                    "accelerators bounds the ACTIVE footprint at roughly "
                    f"{band[0]} to {band[1]}B parameters; multi-tenant sharing only "
                    "lowers the bound. Counting quantization correctly (bf16 "
                    "nameplate x2-4) widens the bound several-fold over the first "
                    "revision, which itself was ~2-4x above the dated bf16/A100 "
                    f"grid ({legacy[0]}-{legacy[1]}B)."),
    }


def conventions_note() -> str:
    return ("2N FLOPs/token counts weight matmuls only; attention adds "
            "~2*L*d_model per token per layer, which at <=29k context and typical "
            "small-model widths is <15% of the total - inside the assumption ranges.")


def band_from_capability() -> dict:
    return {
        "nearest_published_neighbors": {
            "Qwen 3.5 9B": {"mmlu_pro": 0.825, "gpqa": 0.776, "protocol": "reasoning (aggregator)"},
            "Claude 3.7 Sonnet (no thinking)": {"mmlu_pro": 0.807, "gpqa": 0.768, "protocol": "direct"},
            "Qwen3.8-27B": {"mmlu_pro": 0.843, "gpqa": 0.822, "protocol": "cot_5shot / reasoning"},
        },
        "jev": {"mmlu_pro": 0.8282, "gpqa": 0.7653, "protocol": "direct, one-shot, no CoT"},
        "band_dense_equivalent_b": [4.0, 14.0],
        "basis": ("Jev's direct-answer MMLU-Pro/GPQA sit at/above the no-thinking "
                  "Claude 3.7 row and beside Qwen 3.5 9B's reasoning-protocol "
                  "scores; a 2025-era ~9B-class instruct model is the nearest "
                  "published analog, with a 4-14B dense-equivalent band to "
                  "absorb protocol mismatch and vintage effects. HLE 21.9% and "
                  "ARC-AGI-2 0/120 exact cap it far below frontier sizes."),
        "caveats": [
            "capability bands are loose: hundreds of models share this band",
            "distillation shifts capability-per-parameter upward (a distilled "
            "1-4B can sit in this band on knowledge MCQs)",
            "MoE models are sized by ACTIVE parameters in this band too",
        ],
    }


def reconcile(pf: dict, cap: dict) -> dict:
    lo, hi = pf["active_params_B_range"]
    clo, chi = cap["band_dense_equivalent_b"]
    joint_lo, joint_hi = max(lo, clo), min(hi, chi)
    overlap = joint_lo <= joint_hi
    return {
        "tension": (f"throughput angle: {lo:.1f}-{hi:.1f}B ACTIVE (quantized-serving "
                    f"assumptions); capability angle: {clo:.0f}-{chi:.0f}B "
                    f"dense-equivalent. Under the revised assumptions the bands "
                    f"{'OVERLAP' if overlap else 'do not overlap'}"
                    + (f" (joint zone ~{joint_lo:.0f}-{joint_hi:.0f}B)." if overlap else ".")),
        "readings": [
            {"reading": "quantized dense (parsimonious, now sufficient)",
             "statement": (f"a dense ~{joint_lo:.0f}-{min(joint_hi,9):.0f}B model served at fp8/int4 "
                           "fits BOTH angles with no further machinery; this is the "
                           "simplest hypothesis consistent with everything measured"),
             "independent_evidence": "the $0.042/M price and 73 ms floor are consistent with small-model single-host serving"},
            {"reading": "MoE (possible; not favored)",
             "statement": ("an MoE with ~15-100B total at ~5-20% activation also "
                           "fits, and would explain the TOP of the capability band "
                           "with less compute per token; the loose corrected "
                           "throughput band removes the capacity-density argument "
                           "that once favored this reading"),
             "moe_total_B_range": [15, 100],
             "activation_pct_range": [5, 20],
             "independent_evidence": "none - the API cannot see expert structure (ARCHITECTURE-ANALYSIS.md sec.5)"},
            {"reading": "distilled small dense",
             "statement": (f"a distilled 2-6B dense model can reach the bottom of the "
                           f"{clo:.0f}-{chi:.0f}B capability band on knowledge MCQs "
                           "(teacher labels transfer knowledge, not reasoning) - and "
                           "Jev's recognition>>production asymmetry is exactly that shape"),
             "independent_evidence": ("weak/indirect: generation tax (83.1% MCQ vs "
                                      "13.6% digit read-out), HLE near floor, brand "
                                      "prior (ARCHITECTURE-ANALYSIS.md sec.6)")},
            {"reading": "dense below the joint zone",
             "statement": ("if serving is more aggressive than assumed (int4, S>4, "
                           "MFU>0.45), active could sit at 1-4B with distillation "
                           "supplying the capability"),
             "independent_evidence": "none; assumption-dependent"},
        ],
        "converged_statement": (
            f"Two-sided best guess after the quantized-serving revision: ACTIVE "
            f"parameters most plausibly ~{joint_lo:.0f}-{min(joint_hi,9):.0f}B (the overlap of "
            f"the {lo:.1f}-{hi:.1f}B throughput bound with the {clo:.0f}-{chi:.0f}B "
            "capability band), i.e. dense-equivalent order 4-9B - a 2025-era "
            "small model. A dense quantized model in this range needs no MoE to "
            "reconcile the angles; an MoE (~15-100B total) remains possible and "
            "unfalsifiable from the API. TOTAL parameters unconstrained either "
            "way. NOT a parameter claim from latency alone: the throughput angle "
            "is an explicit-assumptions bound, not a slope-to-size conversion."),
        "what_would_tighten_it": [
            "vendor disclosure (trivially)",
            "a matched-protocol capability evaluation (removes band looseness)",
            "the staged P2 option-cost probe at large K under load (bounds per-token marginal compute more tightly)",
        ],
    }


def main() -> int:
    A = json.loads((ROOT / "runs_archprobe" / "analysis.json").read_text(encoding="utf-8"))
    pf = band_from_prefill(A["prefill"])
    cap = band_from_capability()
    out = {
        "generated_by": "scripts/report/size_estimate.py (offline; assumptions explicit)",
        "claim_level": "speculative band under stated assumptions; NOT an identification",
        "prefill_angle": pf,
        "capability_angle": cap,
        "conventions_note": conventions_note(),
        "reconciliation": reconcile(pf, cap),
        "provenance": ["runs_archprobe/analysis.json",
                       "docs/modern-comparison/canonical/comparable-scores.json",
                       "runs_benchmark*/bench-*/derived/score.json"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)}")
    print("prefill band (B active):", pf["active_params_B_range"], "central", pf["central_case_B"])
    print("capability band (B dense-equiv):", cap["band_dense_equivalent_b"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
