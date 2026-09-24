"""Static report generation. Mock/simulated runs are labelled loudly and
repeatedly — a synthetic report must never be mistaken for a measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MOCK_BANNER = (
    "> **SYNTHETIC DATA — NOT MEASURED FROM JEV.**\n"
    "> This run used the deterministic offline mock provider. Nothing in this\n"
    "> report is a measurement of the TypeSafe AI endpoint, and it must never be\n"
    "> presented as one.\n"
)

_LIVE_NOTE = (
    "> **LIVE ENDPOINT OBSERVATIONS.**\n"
    "> Records were produced by the real provider transport. Timing and cost\n"
    "> figures reflect this machine and this account only.\n"
)


def build_report(run_id: str, root: str | None = None) -> str:
    directory = (Path(root) if root else Path("runs")) / run_id
    metrics_path = directory / "derived" / "metrics.json"
    if not metrics_path_exists(metrics_path):
        raise FileNotFoundError(f"run {run_id} has no derived/metrics.json; run analyze first")
    report_data = json.loads(metrics_path.read_text())
    return render(run_id, report_data, directory)


def metrics_path_exists(path: Path) -> bool:
    return path.exists()


def render(run_id: str, report: dict, directory: Path) -> str:
    lines: list[str] = []
    lines.append(f"# Jev Observatory report — `{run_id}`\n")
    if report.get("synthetic"):
        lines.append(MOCK_BANNER)
    else:
        lines.append(_LIVE_NOTE)
    lines.append("")

    summary = _load_summary(directory)
    if summary:
        lines.append("## Run summary\n")
        budget = summary.get("budget") or {}
        lines.append(f"- results recorded: {summary.get('results')}")
        lines.append(f"- attempts recorded: {summary.get('attempts')}")
        if budget:
            lines.append(f"- reported input tokens: {budget.get('reported_input_tokens')}")
            lines.append(f"- reported cost ceiling (USD): {budget.get('cost_ceiling_usd')}")
            denied = budget.get("denied_by_limit") or []
            if denied:
                lines.append(f"- budget denials: {', '.join(denied)}")
        open_uncertain = summary.get("uncertain_open") or []
        lines.append(f"- uncertain attempts still open: {len(open_uncertain)}")
        lines.append("")

    aggregates = report.get("aggregates", {})
    if aggregates.get("choice"):
        _choice_block(lines, aggregates["choice"])
    if aggregates.get("noul"):
        _noul_block(lines, aggregates["noul"])
    if aggregates.get("score"):
        _score_block(lines, aggregates["score"])

    violations = report.get("violation_summary") or {}
    if violations:
        lines.append("## Contract violations recorded\n")
        lines.append("| violation code | rows |")
        lines.append("|---|---|")
        for code, count in sorted(violations.items()):
            lines.append(f"| `{code}` | {count} |")
        lines.append("")

    for group, stats in sorted((report.get("per_group") or {}).items()):
        lines.append(f"### {group}\n")
        lines.append(f"- accuracy {stats.get('accuracy')} (n={stats.get('n')}, "
                     f"Wilson 95% {stats.get('wilson_95')})")
        lines.append("")

    lines.append("## Reproducibility\n")
    lines.append(f"- manifest sha256: `{report.get('manifest_sha256')}`")
    lines.append(f"- analyzed at: {report.get('analyzed_at')}")
    lines.append(f"- provider observed: {report.get('provider_observed')}")
    lines.append(f"- logical results: {report.get('n_logical_results')}")
    lines.append("- machine-readable metrics: `derived/metrics.json`")
    lines.append("- derived tables: `derived/attempts.parquet`, `derived/results.parquet`, "
                 "`derived/prediction_rows.parquet`")
    lines.append("")
    return "\n".join(lines)


def _choice_block(lines: list[str], stats: dict) -> None:
    lines.append("## Choice questions\n")
    acc = stats.get("accuracy", {})
    lines.append(f"- accuracy: {acc.get('accuracy')} (n={acc.get('n')}, "
                 f"Wilson 95% {acc.get('wilson_95')})")
    lines.append(f"- Brier (sum-over-classes, 0..K-1 scale) mean: {stats.get('brier_sum_mean')}")
    exact = stats.get("log_loss_exact_mean")
    clipped = stats.get("log_loss_clipped_1e-12_mean")
    lines.append(f"- log loss (exact, natural log; inf when a true outcome gets p=0): {exact}")
    lines.append(f"- log loss (clipped at ε=1e-12, for plotting only): {clipped}")
    reliability = stats.get("p_max_reliability") or []
    if reliability:
        lines.append("\nReliability (binned on p_max):\n")
        lines.append("| bin | range | n | mean p | empirical |")
        lines.append("|---|---|---|---|---|")
        for b in reliability:
            lines.append(f"| {b['bin']} | {b['low']:.1f}–{b['high']:.1f} | {b['n']} | "
                         f"{b['mean_probability']} | {b['empirical_accuracy']} |")
    lines.append("")


def _noul_block(lines: list[str], stats: dict) -> None:
    lines.append("## Noul questions\n")
    acc = stats.get("accuracy", {})
    lines.append(f"- accuracy at 0.5 threshold: {acc.get('accuracy')} (n={acc.get('n')}, "
                 f"Wilson 95% {acc.get('wilson_95')})")
    lines.append(f"- Brier (binary): {stats.get('brier_binary_mean')}")
    lines.append(f"- log loss (exact): {stats.get('log_loss_exact_mean')}")
    lines.append("")


def _score_block(lines: list[str], stats: dict) -> None:
    lines.append("## Score questions\n")
    acc = stats.get("exact_level_accuracy", {})
    lines.append(f"- exact level accuracy: {acc.get('accuracy')} (n={acc.get('n')})")
    lines.append(f"- mean absolute error (rubric levels): {stats.get('mean_absolute_error')}")
    lines.append(f"- ranked probability score mean: {stats.get('rps_mean')}")
    lines.append("")


def _load_summary(directory: Path) -> dict | None:
    path = directory / "run_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None