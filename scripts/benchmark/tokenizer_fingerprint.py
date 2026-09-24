#!/usr/bin/env python3
"""Offline tokenizer fingerprint: which base family does Jev's token COUNT match?

Reads the tokenizer-family rows written by run_arch_probe.py (each row carries
the probe string and the SERVER-REPORTED input_tokens for an otherwise-fixed
request template). The reported count is an affine function of the reference
tokenizer's count of the exact serialized payload we sent (plus a constant
template/chat overhead). For every candidate tokenizer we fit that affine map
by least squares and score the residual spread; the family with the flattest
residuals across the deliberately adversarial probe strings (CJK, emoji/ZWJ,
long digit runs, repeated subwords, brand names) is the fingerprint match.

This is exploratory ancestry evidence, not identification: an OpenAI-family
match, say, is consistent with an OpenAI-derived tokenizer and says nothing
proven about the weights. Gated reference tokenizers are skipped if the
HF_TOKEN is unavailable.

  python scripts/benchmark/tokenizer_fingerprint.py --rows runs_archprobe/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jev_observatory.arch_probe import _choice, DEFAULT_MODEL_PIN

# tiktoken (OpenAI) families that need no download beyond the BPE blob
TIKTOKEN_CANDIDATES = {
    "o200k_base": ("o200k_base", "GPT-4o / o-series"),
    "o200k_harmony": ("o200k_harmony", "GPT-5-style harmony"),
    "cl100k_base": ("cl100k_base", "GPT-3.5/4"),
    "gpt2": ("gpt2", "GPT-2/3 BPE"),
}
# HF tokenizer.json repos (ungated where possible); skipped if unavailable
HF_CANDIDATES = {
    "llama3": ("meta-llama/Llama-3.1-8B", "Llama-3.1 / Hermes"),
    "llama3_mirror": ("NousResearch/Meta-Llama-3-8B", "Llama-3 mirror"),
    "qwen25": ("Qwen/Qwen2.5-7B", "Qwen2.5"),
    "mistral": ("mistralai/Mistral-7B-v0.3", "Mistral Nemo/BPE"),
    "gemma2": ("google/gemma-2-2b", "Gemma 2"),
    "deepseek_v3": ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3"),
    "phi": ("microsoft/Phi-3-medium-4k-instruct", "Phi-3"),
}


def payload_string(probe: str) -> str:
    """The EXACT serialized body the probe sent (server tokenizes this + template)."""
    req = _choice(["yes", "no"], "Is the bracketed text empty?",
                  "Token probe [" + probe + "]")
    return json.dumps(req.to_payload(), ensure_ascii=False)


def _fit_linear(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rss = sum(r * r for r in resid)
    tss = sum((y - my) ** 2 for y in ys)
    r2 = 1 - rss / tss if tss else None
    rmse = (rss / n) ** 0.5
    return {"intercept": round(a, 2), "slope": round(b, 4),
            "r2": round(r2, 4) if r2 is not None else None,
            "rmse_tokens": round(rmse, 3)}


def load_tokenizer_counts():
    encoders: dict[str, callable] = {}
    labels: dict[str, str] = {}
    try:
        import tiktoken
        for name, (enc, label) in TIKTOKEN_CANDIDATES.items():
            try:
                e = tiktoken.get_encoding(enc)
                dis = getattr(e, "_disallowed_special", ())
                encoders[name] = lambda s, e=e: len(e.encode(s, disallowed_special=()))
                labels[name] = label
            except Exception as exc:  # missing blob / unsupported name
                print(f"[skip tiktoken {name}] {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[tiktoken unavailable] {exc}", file=sys.stderr)
    try:
        from tokenizers import Tokenizer
        for name, (repo, label) in HF_CANDIDATES.items():
            try:
                tok = Tokenizer.from_pretrained(
                    repo, token=os.environ.get("HF_TOKEN") or None)
                encoders[name] = lambda s, tok=tok: len(tok.encode(s, add_special_tokens=False).ids)
                labels[name] = label
            except Exception as exc:
                print(f"[skip hf {name}] {type(exc).__name__}: {str(exc)[:80]}",
                      file=sys.stderr)
    except Exception as exc:
        print(f"[tokenizers unavailable] {exc}", file=sys.stderr)
    return encoders, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="runs_archprobe/rows.jsonl")
    parser.add_argument("--out", default="runs_archprobe/tokenizer_fingerprint.json")
    args = parser.parse_args()

    rows = []
    for line in Path(args.rows).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("family") == "tokenizer" and isinstance(r.get("usage_input_tokens"), int):
            probe = r.get("probe")
            if probe is not None:
                rows.append((probe, r["usage_input_tokens"]))
    if len(rows) < 5:
        print(f"[refused] only {len(rows)} tokenizer rows; run the probe first",
              file=sys.stderr)
        return 2
    # average repeats per distinct probe string
    by_probe: dict[str, list[int]] = {}
    for probe, tok in rows:
        by_probe.setdefault(probe, []).append(tok)
    probes = sorted(by_probe, key=lambda p: -statistics_median(by_probe[p]))
    reported = [statistics_median(by_probe[p]) for p in probes]

    encoders, labels = load_tokenizer_counts()
    results = {}
    for name, enc in encoders.items():
        try:
            xs = [enc(payload_string(p)) for p in probes]
            fit = _fit_linear(xs, reported)
        except Exception as exc:
            results[name] = {"error": str(exc)[:120]}
            continue
        if fit:
            fit["label"] = labels.get(name, name)
            results[name] = fit
    ranked = sorted(results.items(),
                    key=lambda kv: kv[1].get("rmse_tokens", float("inf")))
    doc = {
        "claim_type": "exploratory",
        "n_probes": len(probes),
        "method": ("reported input_tokens vs each reference tokenizer's count of "
                   "the exact serialized payload; per-family affine fit; lowest "
                   "RMSE (≈ slope 1, intercept = fixed template overhead) wins"),
        "caveat": ("a tokenizer match is evidence about the preprocessing layer "
                   "only; it does not identify the weights or prove provenance"),
        "ranking": [
            {"family": k, "label": v.get("label"), "rmse_tokens": v.get("rmse_tokens"),
             "r2": v.get("r2"), "slope": v.get("slope"), "intercept": v.get("intercept")}
            for k, v in ranked if "rmse_tokens" in v
        ],
        "per_probe_best_residuals": None,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(json.dumps(doc["ranking"], indent=2))
    return 0


def statistics_median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0)


if __name__ == "__main__":
    sys.exit(main())
