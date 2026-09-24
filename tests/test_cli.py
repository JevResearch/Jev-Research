"""CLI tests: full offline pipeline plan → run → analyze → report."""

import json

import pytest
from typer.testing import CliRunner

from jev_observatory.cli import app

runner = CliRunner()


def _write_spec(tmp_path, n_items=3, **overrides):
    items = []
    for i in range(n_items):
        items.append({
            "id": f"item-{i}",
            "group": "smoke",
            "cluster": f"c{i % 2}",
            "state": f"Ticket {i}: refund requested for order {1000 + i}.",
            "gold": {"dep": {"value": "billing"}},
            "questions": {"dep": {"type": "choice", "instructions": "Which team?",
                                  "criteria": {"billing": "money", "tech": "bugs"}}},
        })
    spec = {"experiment": "cli-smoke", "model": "jev-1.13.0", "items": items, "seeds": {"order": 3},
            "test_family": "m0-smoke"}
    spec.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_validate_ok(tmp_path):
    result = runner_invoke(runner := CliRunner(), ["validate", str(_write_spec(tmp_path))])
    assert result.exit_code == 0
    assert "pass schema" in result.output


def test_validate_rejects_leakage(tmp_path):
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["items"][0]["state"] = "The right team is billing, clearly."
    spec_path.write_text(json.dumps(spec))
    result = runner_invoke(runner := CliRunner(), ["validate", str(spec_path)])
    assert result.exit_code == 2
    assert "leaks gold" in result.output


def test_validate_rejects_unknown_item_key(tmp_path):
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["items"][0]["prompt_answer"] = "billing"
    spec_path.write_text(json.dumps(spec))
    result = runner_invoke(runner := CliRunner(), ["validate", str(spec_path)])
    assert result.exit_code == 2


def test_plan_writes_manifest_and_estimate(tmp_path):
    root = tmp_path / "runs"
    result = runner_invoke(CliRunner(), ["plan", str(_write_spec(tmp_path)), "--root", str(root)])
    assert result.exit_code == 0
    plan_doc = list(root.glob("*/plan.json"))
    assert plan_doc, "plan.json must exist"
    doc = json.loads(plan_doc[0].read_text())
    assert doc["n_logical_requests"] == 3
    assert doc["estimated_input_tokens_conservative"] > 0


def test_full_offline_pipeline_labels_mock(tmp_path):
    """plan → run(mock) → analyze → report; the report must shout SYNTHETIC."""
    root = str(tmp_path / "runs")
    spec_path = _write_spec(tmp_path)
    cli = CliRunner()
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", root])
    assert plan_result.exit_code == 0, plan_result.output
    run_id = json.loads(plan_result.output.split("\n", 1)[1])["run_id"]

    run_result = cli.invoke(app, ["run", run_id, "--provider", "mock", "--root", root])
    assert run_result.exit_code == 0, run_result.output

    analyze_result = cli.invoke(app, ["analyze", run_id, "--root", root])
    assert analyze_exit_code(analyze_result) == 0

    report_result = cli.invoke(app, ["report", run_id, "--root", root])
    assert report_result.exit_code == 0
    report_text = (tmp_path / "runs" / run_id / "report.md").read_text()
    assert "SYNTHETIC" in report_text
    assert "mock" in report_text.lower()
    # derived tables exist
    derived = tmp_path / "runs" / run_id / "derived"
    assert (derived / "metrics.json").exists()
    assert (derived / "attempts.parquet").exists()
    assert (derived / "results.parquet").exists()
    assert (derived / "prediction_rows.parquet").exists()


def analyze_exit_code(result):
    return result.exit_code


def runner_invoke(cli, args):
    return cli.invoke(app, args)


def test_live_calls_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("JEVO_ALLOW_LIVE", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_deadbeef")
    root = str(tmp_path / "runs")
    cli = CliRunner()
    plan_result = cli.invoke(app, ["plan", str(_write_spec(tmp_path)), "--root", root, "--provider", "jev"])
    run_id = json.loads(plan_result.output.split("\n", 1)[1])["run_id"]
    result = cli.invoke(app, ["run", run_id, "--provider", "jev", "--root", root])  # no --live
    assert result.exit_code == 2
    assert "refusing live calls" in result.output


def test_live_calls_require_environment_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("JEVO_ALLOW_LIVE", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    root = str(tmp_path / "runs")
    cli = CliRunner()
    plan_result = cli.invoke(app, ["plan", str(_write_spec(tmp_path)), "--root", root, "--provider", "jev"])
    run_id = json.loads(plan_result.output.split("\n", 1)[1])["run_id"]
    result = cli.invoke(app, ["run", run_id, "--provider", "jev", "--live", "--root", root])
    assert result.exit_code == 2
    assert "JEVO_ALLOW_LIVE" in result.output


def test_items_mutation_detected(tmp_path):
    """Resuming against mutated items is refused, not silently accepted."""
    root = str(tmp_path / "runs")
    spec_path = _write_spec(tmp_path)
    cli = CliRunner()
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", root])
    run_id = json.loads(plan_result.output.split("\n", 1)[1])["run_id"]
    items_path = tmp_path / "runs" / run_id / "items.jsonl"
    items = [json.loads(l) for l in items_path.read_text().splitlines() if l]
    items[0]["gold"] = {"dep": {"value": "tech"}}
    items_path.write_text("\n".join(json.dumps(i, sort_keys=True) for i in items) + "\n")
    result = cli.invoke(app, ["run", run_id, "--provider", "mock", "--root", root])
    assert result.exit_code == 2
    assert "items_sha256" in result.output