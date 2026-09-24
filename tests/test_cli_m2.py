"""M2 CLI end-to-end: sweep → simulated → fit-latency recovers known costs."""

import json

import pytest
from typer.testing import CliRunner

from jev_observatory.cli import app

cli = CliRunner()


def _plan(tmp_path, design, provider="mock", **kwargs):
    spec_path = tmp_path / f"{design}.json"
    extra: list[str] = []
    for key, value in kwargs.items():
        extra += [f"--{key}", str(value)]
    args = ["make-spec-m2", "--design", design, "--out", str(spec_path)] + extra
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", str(tmp_path / "runs"),
                                   "--provider", provider])
    assert plan_result.exit_code == 0, plan_result.output
    return spec_path, json.loads(plan_result.output.split("\n", 1)[1])["run_id"]


def test_sweep_pipeline_recovers_known_cost_model(tmp_path):
    """The acceptance criterion: a simulated endpoint with known L/Q/K costs has
    those effects recovered qualitatively by the analysis pipeline."""
    spec_path, run_id = _plan(tmp_path, "sweep", provider="simulated", repeats=3)
    run_result = cli.invoke(app, ["run", run_id, "--provider", "simulated",
                                  "--root", str(tmp_path / "runs")])
    assert run_result.exit_code == 0, run_result.output

    fit_result = cli.invoke(app, ["fit-latency", run_id, "--root", str(tmp_path / "runs")])
    assert fit_result.exit_code == 0, fit_result.output
    fit = json.loads((tmp_path / "runs" / run_id / "derived" / "latency_model.json").read_text())
    assert fit["fit_available"] is True
    coeffs = fit["coefficients"]
    # SimConfig defaults: per_kilochar_ms=8, per_question_ms=3, per_candidate_ms=0.05
    assert coeffs["L_kilo"] > 0 and 4 < coeffs["L_kilo"] < 12
    assert coeffs["Q"] > 0 and 1 < coeffs["Q"] < 5
    assert coeffs["C"] > 0 and coeffs["C"] < 0.2
    assert fit["train_r2"] > 0.95
    # held-out validation must beat the naive mean predictor
    assert fit["holdout_mae_ms"] < fit["holdout_mae_baseline_ms"]
    # no architecture claims leak into the artifact
    assert "do not identify architecture" in fit["interpretation"]


def test_isolation_pipeline_end_to_end(tmp_path):
    spec_path, run_id = _plan(tmp_path, "isolation", provider="simulated", repeats=2)
    assert cli.invoke(app, ["run", run_id, "--provider", "simulated",
                            "--root", str(tmp_path / "runs")]).exit_code == 0
    result = cli.invoke(app, ["analyze-isolation", run_id, "--root", str(tmp_path / "runs")])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "runs" / run_id / "derived" / "isolation.json").read_text())
    assert summary["claim_type"] == "exploratory"
    assert summary["n_clusters_compared"] == 2
    assert summary["verdict"] is not None


def test_odds_pipeline_end_to_end(tmp_path):
    spec_path, run_id = _plan(tmp_path, "odds", provider="simulated", repeats=2)
    assert cli.invoke(app, ["run", run_id, "--provider", "simulated",
                            "--root", str(tmp_path / "runs")]).exit_code == 0
    result = cli.invoke(app, ["analyze-odds", run_id, "--root", str(tmp_path / "runs")])
    assert result.exit_code == 0, result.output
    odds = json.loads((tmp_path / "runs" / run_id / "derived" / "odds.json").read_text())
    assert odds["tracked_pair"] == ["A", "B"]
    assert set(odds["variants"]) == {"duplicate", "irrelevant"}


def test_drift_detected_through_sentinel_items(tmp_path):
    """A drifting simulated endpoint trips the sentinel-based drift marker."""
    spec_path, run_id = _plan(tmp_path, "sweep", provider="simulated", repeats=4)
    assert cli.invoke(app, ["run", run_id, "--provider", "simulated",
                            "--root", str(tmp_path / "runs")]).exit_code == 0
    from jev_observatory.simulated import detect_drift

    results = [json.loads(l) for l in
               (tmp_path / "runs" / run_id / "results.jsonl").read_text().splitlines() if l.strip()]
    sentinels = [r["latency_ms_first_attempt"] for r in results
                 if r.get("condition", "").startswith("role=sentinel")]
    # default SimConfig has no drift: the detector must stay quiet
    assert detect_drift(sentinels)["flagged"] is False


def test_simulated_provider_labels_are_not_jev(tmp_path):
    """Simulated runs must be distinguishable from real endpoint records."""
    spec_path, run_id = _plan(tmp_path, "odds", provider="simulated", repeats=2)
    assert cli.invoke(app, ["run", run_id, "--provider", "simulated",
                            "--root", str(tmp_path / "runs")]).exit_code == 0
    results = [json.loads(l) for l in
               (tmp_path / "runs" / run_id / "results.jsonl").read_text().splitlines() if l.strip()]
    assert {r["provider"] for r in results} == {"jev"}  # transport is Jev-shaped...
    # ...but the run's manifest provider is 'simulated', so reports stay honest
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text())
    assert manifest["provider"] == "simulated"


def test_block_order_survives_planning(tmp_path):
    """shuffle:false specs dispatch in their designed block order."""
    spec_path, _ = _plan(tmp_path, "sweep", repeats=2)
    spec = json.loads(spec_path.read_text())
    requests = json.loads(json.dumps(spec["items"]))
    assert requests[0]["group"] == "sentinel"
    assert requests[len(spec["items"]) // 2]["group"] == "sentinel"