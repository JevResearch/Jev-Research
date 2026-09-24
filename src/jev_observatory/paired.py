"""Reusable paired-encoding comparison (audit follow-up: replaces ad hoc scripts).

Measures two System One encodings of the same records (e.g. BoolQ as Choice vs
Noul) on the *binary yes/no* scale:

* accuracy per condition and the paired difference (exact McNemar);
* binary Brier score of P(yes) against the gold yes/no label per condition, and
  the paired difference with a percentile bootstrap CI (resampling base records);
* P(yes) reliability per condition, calibrated against gold yes/no.

Everything is computed from run artifacts + the run's own items.jsonl; gold
labels never enter outbound payloads. The earlier ad hoc script used P(max)
against string golds and mislabeled the scale — do not resurrect it.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from typing import Any

def _extract(record: dict[str, Any], item: dict[str, Any], question_id: str) -> tuple[bool, float, bool] | None:
    """Return (correct, p_yes, gold_yes) for one encoding of one record."""
    prediction = record.get("predictions", {}).get(question_id)
    if prediction is None:
        return None
    gold_value = (item.get("gold", {}).get(question_id) or {})
    gold = gold_value.get("value") if isinstance(gold_value, dict) else gold_value
    if prediction.get("type") == "choice":
        if not isinstance(gold, str):
            return None
        p_yes = float((prediction.get("probabilities") or {}).get("yes", 0.0))
        gold_yes = gold == "yes"
        correct = prediction.get("choice") == gold
    elif prediction.get("type") == "noul":
        if not isinstance(gold, bool):
            return None
        p_yes = float(prediction.get("noul", 0.5))
        gold_yes = gold
        correct = (p_yes >= 0.5) == gold_yes
    else:
        return None
    return correct, p_yes, gold_yes


def _binary_brier(p_yes: float, gold_yes: bool) -> float:
    return (p_yes - (1.0 if gold_yes else 0.0)) ** 2


def _reliability(pairs: list[tuple[float, bool]], n_bins: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == n_bins - 1 and p == hi)]
        if not members:
            continue
        out.append({
            "bin": b, "n": len(members),
            "mean_p_yes": round(statistics.mean(p for p, _ in members), 6),
            "yes_rate": round(statistics.mean(1.0 if y else 0.0 for _, y in members), 6),
        })
    return out


def _ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    bins = _reliability(pairs, n_bins)
    total = sum(b["n"] for b in bins)
    if total == 0:
        return float("nan")
    return sum(b["n"] * abs(b["mean_p_yes"] - b["yes_rate"]) for b in bins) / total


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b01, b10."""
    n = b01 + b10
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(b01, b10) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def paired_encoding_comparison(
    results: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    *,
    question_id: str = "answer",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two encodings over the *same* base records.

    `results` are result records; `items` maps item_id -> spec entry (gold
    included). Records pair up via `cluster` (the base-record id). Conditions
    are the distinct `condition` prefixes (strip `repeat=` if present is NOT
    needed: paired BoolQ has one request per record per condition).
    """
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for record in results:
        condition = record.get("condition", "").split(";")[0]
        base = record.get("cluster") or record["logical_request_id"]
        extracted = _extract(record, items[record["item_id"]], question_id)
        if extracted is None:
            continue
        by_condition.setdefault(condition, {})[base] = {
            "record": record, "correct": extracted[0], "p_yes": extracted[1], "gold_yes": extracted[2],
        }
    conditions = sorted(by_condition)
    if len(conditions) != 2:
        return {"available": False, "reason": f"expected exactly 2 conditions, got {conditions}"}
    shared = sorted(set(by_condition[conditions[0]]) & set(by_condition[conditions[1]]))
    if not shared:
        return {"available": False, "reason": "no shared base records between conditions"}

    per_condition: dict[str, Any] = {}
    for condition in conditions:
        rows = [by_condition[condition][base] for base in shared]
        per_condition[condition] = {
            "n": len(rows),
            "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
            "binary_brier_p_yes": statistics.mean(_binary_brier(r["p_yes"], r["gold_yes"]) for r in rows),
            "p_yes_ece_10": _ece([(r["p_yes"], r["gold_yes"]) for r in rows]),
            "n_zero_true_probability": sum(1 for r in rows
                                           if r["p_yes"] == 0.0 and r["gold_yes"]
                                           or r["p_yes"] == 1.0 and not r["gold_yes"]),
        }

    # paired differences aligned by base record
    brier_diff = []
    acc_pairs = []
    for base in shared:
        a = by_condition[conditions[0]][base]
        b = by_condition[conditions[1]][base]
        brier_diff.append(_binary_brier(a["p_yes"], a["gold_yes"]) - _binary_brier(b["p_yes"], b["gold_yes"]))
        acc_pairs.append((a["correct"], b["correct"]))

    n = len(shared)
    acc_diff = sum(1 for a, b in acc_pairs if a) / n - sum(1 for a, b in acc_pairs if b) / n
    b01 = sum(1 for a, b in acc_pairs if a and not b)
    b10 = sum(1 for a, b in acc_pairs if not a and b)

    rng = random.Random(seed)
    brier_boot: list[float] = []
    acc_boot: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(range(n), k=n)
        brier_boot.append(statistics.mean(brier_diff[i] for i in sample))
        a_hits = sum(1 for i in sample if acc_pairs[i][0])
        b_hits = sum(1 for i in sample if acc_pairs[i][1])
        acc_boot.append((a_hits - b_hits) / n)
    brier_boot.sort()
    acc_boot.sort()
    lo_index = int(0.025 * n_bootstrap)
    hi_index = min(n_bootstrap - 1, int(0.975 * n_bootstrap))

    return {
        "available": True,
        "conditions": conditions,
        "n_pairs": n,
        "per_condition": per_condition,
        "accuracy_difference": round(acc_diff, 6),
        "accuracy_difference_ci95": {"low": round(acc_boot[lo_index], 6),
                                     "high": round(acc_boot[hi_index], 6)},
        "mcnemar_exact_p": mcnemar_exact(b01, b10),
        "discordant": {"first_right_second_wrong": b01, "first_wrong_second_right": b10},
        "brier_difference_first_minus_second": round(statistics.mean(brier_diff), 6),
        "brier_difference_ci95": {"low": round(brier_boot[lo_index], 6),
                                  "high": round(brier_boot[hi_index], 6)},
        "reliability": {
            conditions[0]: _reliability([(by_condition[conditions[0]][b]["p_yes"],
                                          by_condition[conditions[0]][b]["gold_yes"]) for b in shared]),
            conditions[1]: _reliability([(by_condition[conditions[1]][b]["p_yes"],
                                          by_condition[conditions[1]][b]["gold_yes"]) for b in shared]),
        },
        "claim_type": "confirmatory",
        "note": ("accuracy difference is exact McNemar on paired records; Brier difference uses a "
                 "paired percentile bootstrap over base records"),
    }


def load_results_and_items(run_dir):
    """Convenience loader: (results, items-by-id) from a run directory."""
    results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    items = {entry["id"]: entry for entry in _items(run_dir / "items.jsonl")}
    return results, items


def _items(path):
    import json as _json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield _json.loads(line)