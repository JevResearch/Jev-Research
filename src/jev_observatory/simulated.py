"""Simulated endpoint with a *known* cost model, for validating the analysis.

The simulation implements Transport: latency = base + aL·L + aQ·Q + aC·C + noise,
with optional failures and a drift step.  Because the coefficients are known by
construction, tests can assert that the analysis *recovers* them qualitatively.
The simulation never runs against real runs and is clearly labelled `simulated`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

from .providers import MockProvider
from .schema import SystemOneRequest
from .transport import Transport, TransportResponse


@dataclass
class SimConfig:
    base_ms: float = 60.0
    per_kilochar_ms: float = 8.0    # aL: milliseconds per 1000 state chars
    per_question_ms: float = 3.0    # aQ
    per_candidate_ms: float = 0.05  # aC
    noise_ms: float = 5.0
    cold_penalty_ms: float = 40.0
    failure_rate: float = 0.0
    drift_after_calls: int | None = None
    drift_multiplier: float = 1.0
    seed: int = 0


class SimulatedTransport(Transport):
    name = "simulated"

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.rng = random.Random(self.config.seed)
        self.calls = 0
        self._responder = MockProvider(seed=self.config.seed)

    def reset_pool(self) -> None:
        self.calls = 0  # emulate a fresh connection

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        self.calls += 1
        cfg = self.config
        state_chars = len(json.dumps(payload.get("state", ""), ensure_ascii=False))
        questions = payload.get("questions", {})
        n_questions = len(questions)
        n_candidates = sum(
            len(q.get("criteria", {})) for q in questions.values() if isinstance(q, dict)
        )
        latency = (
            cfg.base_ms
            + cfg.per_kilochar_ms * state_chars / 1000.0
            + cfg.per_question_ms * n_questions
            + cfg.per_candidate_ms * n_candidates
        )
        cold = self.calls == 1
        if cold:
            latency += cfg.cold_penalty_ms
        if cfg.drift_after_calls is not None and self.calls > cfg.drift_after_calls:
            latency *= cfg.drift_multiplier
        latency += self.rng.gauss(0.0, cfg.noise_ms)
        latency = max(1.0, latency)

        if cfg.failure_rate > 0 and self.rng.random() < cfg.failure_rate:
            return TransportResponse(status_code=529, headers={}, body=b"",
                                     total_ms=latency, first_byte_ms=latency * 0.6,
                                     cold_connection=cold, error=None)
        request = SystemOneRequest.model_validate(payload)
        body = json.dumps(self._responder._respond(request)).encode()
        return TransportResponse(status_code=200, headers={}, body=body,
                                 total_ms=latency, first_byte_ms=latency * 0.6,
                                 cold_connection=cold, error=None)

    def close(self) -> None:
        return


# ------------------------------------------------------------------ OLS toolkit
def ols_fit(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares via normal equations + Gaussian elimination (pure Python).

    Small feature counts only; no dependency on numpy.
    """
    n_features = len(X[0])
    n_rows = len(X)
    if n_rows <= n_features:
        raise ValueError("need more rows than features")
    # A = XᵀX, b = Xᵀy
    A = [[sum(X[r][i] * X[r][j] for r in range(n_rows)) for j in range(n_features)]
         for i in range(n_features)]
    b = [sum(X[r][i] * y[r] for r in range(n_rows)) for i in range(n_features)]
    # Gaussian elimination with partial pivoting
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n_features):
        pivot = max(range(col, n_features), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix (collinear features?)")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n_features):
            factor = M[r][col] / M[col][col]
            for c in range(col, n_features + 1):
                M[r][c] -= factor * M[col][c]
    coeffs = [0.0] * n_features
    for i in range(n_features - 1, -1, -1):
        coeffs[i] = (M[i][n_features] - sum(M[i][j] * coeffs[j] for j in range(i + 1, n_features))) / M[i][i]
    return coeffs


def ols_predict(coeffs: list[float], x: list[float]) -> float:
    return sum(c * v for c, v in zip(coeffs, x))


def fit_latency_model(
    records: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit competing latency models on training blocks; validate on held-out.

    Features: intercept, L (state chars), Q (questions), C (candidates).
    Reports coefficients with NO architectural interpretation — see DESIGN.md §7
    (milliseconds do not identify FLOPs, parameters, or training cost).
    """
    usable = [r for r in records
              if r.get("status") in {"ok", "contract_invalid"}
              and r.get("latency_ms_first_attempt") is not None]
    if len(usable) < 40:
        return {"fit_available": False, "reason": f"need >= 40 timing records, got {len(usable)}"}
    rows = []
    for record in usable:
        measures = record.get("measures", {})
        condition = record.get("condition", "")
        rows.append({
            "latency": record["latency_ms_first_attempt"],
            "L": float(measures.get("state_chars", 0)),
            "Q": float(measures.get("q", 0)),
            "C": float(measures.get("c_total_candidates", 0)),
            "cluster": record.get("cluster", record["logical_request_id"]),
            "block": _block_of(condition),
        })
    blocks = sorted({r["block"] for r in rows if r["block"] is not None})
    if len(blocks) >= 2:
        # Designed sweeps: hold out whole *blocks*, preserving the execution
        # design. Holding out random rows leaks block-level serving conditions.
        n_holdout_blocks = max(1, int(round(len(blocks) * holdout_fraction)))
        holdout_blocks = set(blocks[-n_holdout_blocks:])
        holdout = [r for r in rows if r["block"] in holdout_blocks]
        train = [r for r in rows if r["block"] not in holdout_blocks]
        split_method = f"block_holdout (blocks {sorted(holdout_blocks)} held out)"
    else:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n_holdout = max(5, int(len(rows) * holdout_fraction))
        holdout = [rows[i] for i in indices[:n_holdout]]
        train = [rows[i] for i in indices[n_holdout:]]
        split_method = "random_rows (no block metadata available)"

    features = ["intercept", "L_kilo", "Q", "C"]
    def design(row):
        return [1.0, row["L"] / 1000.0, row["Q"], row["C"]]

    try:
        coeffs = ols_fit([design(r) for r in train], [r["latency"] for r in train])
    except ValueError as exc:
        return {"fit_available": False, "reason": str(exc)}
    residuals = [r["latency"] - ols_predict(coeffs, design(r)) for r in train]
    holdout_errors = [abs(r["latency"] - ols_predict(coeffs, design(r))) for r in holdout]
    naive_errors = [abs(r["latency"] - sum(x["latency"] for x in train) / len(train)) for r in holdout]
    sse = sum(e * e for e in residuals)
    sst = sum((r["latency"] - sum(x["latency"] for x in train) / len(train)) ** 2 for r in train)
    return {
        "fit_available": True,
        "features": features,
        "coefficients": dict(zip(features, [round(c, 6) for c in coeffs])),
        "train_r2": round(1.0 - sse / sst, 4) if sst > 0 else None,
        "holdout_mae_ms": round(sum(holdout_errors) / len(holdout_errors), 3),
        "holdout_mae_baseline_ms": round(sum(naive_errors) / len(naive_errors), 3),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "split_method": split_method,
        "interpretation": ("coefficients describe this endpoint's measured latency only; "
                           "they do not identify architecture, parameters, or training cost"),
        "claim_type": "exploratory",
    }


def _block_of(condition: str) -> int | None:
    """Extract the designed block index from a sweep condition string."""
    for part in condition.split(";"):
        if part.startswith("block="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _state_chars_hint(record: dict[str, Any]) -> float:
    """Fallback when request_state_chars was not recorded: estimate from usage."""
    usage = record.get("usage_input_tokens") or 0
    return float(usage)


def detect_drift(sentinel_latencies: list[float], *, threshold_ratio: float = 1.5,
                 min_per_half: int = 3) -> dict[str, Any]:
    """Compare early vs late sentinel medians; a flag, not a verdict.

    Drift may be infrastructure, load, or a model update — the marker only says
    'look here' (DESIGN.md §8: drift is not automatically a weight update).
    """
    n = len(sentinel_latencies)
    if n < 2 * min_per_half:
        return {"flagged": False, "reason": f"need >= {2 * min_per_half} sentinel samples, got {n}"}
    # Arrival order matters: sorting by magnitude would compare the fastest and
    # slowest calls rather than early vs late behaviour.
    half = n // 2
    early = list(sentinel_latencies[:half])
    late = list(sentinel_latencies[half:])
    median_early = _median(early)
    median_late = _median(late)
    ratio = median_late / median_early if median_early > 0 else float("inf")
    return {
        "flagged": ratio >= threshold_ratio,
        "median_early_ms": round(median_early, 3),
        "median_late_ms": round(median_late, 3),
        "ratio": round(ratio, 4),
        "threshold_ratio": threshold_ratio,
        "n_samples": n,
        "note": "infrastructure, load, or model update; not attributable by this test",
    }


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[index]
