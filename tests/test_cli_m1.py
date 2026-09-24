"""M1 CLI tests: make-spec, stratified analysis, replay provider, heatmap."""

import base64
import json

import pytest
from typer.testing import CliRunner

from jev_observatory.cli import app
from jev_observatory.datasets.generators import build_synthetic_spec
from jev_observatory.experiment import logical_requests
from jev_observatory.providers import MockProvider
from jev_observatory.schema import SystemOneRequest

cli = CliRunner()


def _write_spec(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def _plan_run_id(result_output):
    return json.loads(result_output.split("\n", 1)[1])["run_id"]


def test_make_spec_synthetic_pipeline(tmp_path):
    spec_path = tmp_path / "synth.json"
    result = cli.invoke(app, ["make-spec", "--dataset", "synthetic",
                              "--out", str(spec_path), "--synthetic-n", "6", "--seed", "4"])
    assert result.exit_code == 0, result.output
    spec = json.loads(spec_path.read_text())
    assert len(spec["items"]) == 24  # 4 generators x 6

    root = str(tmp_path / "runs")
    validate_result = cli.invoke(app, ["validate", str(spec_path)])
    assert validate_result.exit_code == 0, validate_result.output
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", root])
    assert plan_result.exit_code == 0, plan_result.output
    run_id = _plan_run_id(plan_result.output)
    run_result = cli.invoke(app, ["run", run_id, "--provider", "mock", "--root", root])
    assert run_result.exit_code == 0, run_result.output
    analyze_result = cli.invoke(app, ["analyze", run_id, "--root", root])
    assert analyze_result.exit_code == 0, analyze_result.output
    metrics = json.loads((tmp_path / "runs" / run_id / "derived" / "metrics.json").read_text())
    # synthetic specs are full generated sets: no stratified estimate expected
    assert metrics["sampling_note"] is None
    assert metrics["synthetic"] is True


def test_mmlu_pilot_produces_stratified_estimate(tmp_path):
    """End-to-end: pilot sampling design flows from manifest into analysis."""
    categories = ["math", "law", "bio", "chem"]
    records = []
    for i in range(40):
        records.append({"question_id": f"q{i}", "question": f"Q{i}?",
                        "choices": ["a", "b", "c", "d"], "answer_index": i % 4,
                        "answer": "ABCD"[i % 4], "category": categories[i % 4], "src": "f"})
    data = tmp_path / "mmlu.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records))
    spec_path = tmp_path / "spec.json"
    result = cli.invoke(app, ["make-spec", "--dataset", "mmlu-pro", "--data-file", str(data),
                              "--out", str(spec_path), "--pilot", "12", "--seed", "2",
                              "--revisions", "fixture-rev-1"])
    assert result.exit_code == 0, result.output
    spec = json.loads(spec_path.read_text())
    assert spec["dataset"]["sampling"]["population_category_sizes"] == {
        "math": 10, "law": 10, "bio": 10, "chem": 10}

    root = str(tmp_path / "runs")
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", root])
    run_id = _plan_run_id(plan_result.output)
    assert cli.invoke(app, ["run", run_id, "--provider", "mock", "--root", root]).exit_code == 0
    assert cli.invoke(app, ["analyze", run_id, "--root", root]).exit_code == 0
    metrics = json.loads((tmp_path / "runs" / run_id / "derived" / "metrics.json").read_text())
    assert "stratified_estimate_choice" in metrics["aggregates"]
    assert metrics["sampling_note"] and "pilot" in metrics["sampling_note"]
    # cluster bootstrap present for choice
    assert "cluster_bootstrap" in metrics["aggregates"]["choice"]


def _write_replay_fixture(tmp_path, spec, mock: MockProvider):
    """Record responses for every logical request, like RecordingTransport would."""
    from hashlib import sha256

    replay_dir = tmp_path / "recordings"
    replay_dir.mkdir()
    for logical in logical_requests(spec):
        request: SystemOneRequest = logical.request
        payload = request.to_payload()
        body = json.dumps(mock._respond(request)).encode()
        key = sha256(json.dumps({"path": "/v1/systemone", "payload": payload},
                                sort_keys=True).encode()).hexdigest()
        record = {
            "path": "/v1/systemone",
            "payload": payload,
            "status_code": 200,
            "headers": {},
            "body_b64": base64.b64encode(body).decode(),
            "total_ms": 1.0,
            "first_byte_ms": None,
            "cold_connection": False,
            "error": None,
        }
        (replay_dir / f"{key}.json").write_text(json.dumps(record))
    return replay_dir


def test_replay_provider_reproduces_mock_run(tmp_path):
    """Recorded responses replayed through the Jev path reproduce a mock run."""
    spec_path = _write_spec(tmp_path, build_synthetic_spec(
        [("relation", __import__("jev_observatory.datasets.generators", fromlist=["gen_relation_lookup"]).gen_relation_lookup)],
        seed=3, n_per_generator=4))
    spec = json.loads(spec_path.read_text())
    replay_dir = _write_replay_fixture(tmp_path, spec, MockProvider(seed=0))

    root = str(tmp_path / "runs")
    plan_result = cli.invoke(app, ["plan", str(spec_path), "--root", root, "--provider", "replay"])
    run_id = _plan_run_id(plan_result.output)

    run_result = cli.invoke(app, ["run", run_id, "--provider", "replay",
                                  "--replay-dir", str(replay_dir), "--root", root])
    assert run_result.exit_code == 0, run_result.output

    # replay refuses an unknown/empty recording dir instead of hitting the network
    missing = cli.invoke(app, ["run", run_id, "--provider", "replay",
                               "--replay-dir", str(tmp_path / "nope"), "--root", root])
    assert missing.exit_code == 2


def test_heatmap_command(tmp_path):
    scores = {"scores": [{
        "source": {"source_url": "https://example.org", "accessed_at": "2026-09-18",
                   "model": "ref-a", "dataset": "mmlu-pro"},
        "comparison_class": "historical_protocol_compatible",
        "metric": "accuracy", "value": 0.7, "n": 100,
        "protocol_notes": "0-shot; chance=0.25",
    }]}
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores))
    out_path = tmp_path / "heatmap.json"
    result = cli.invoke(app, ["import-scores", str(scores_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    grid = json.loads(out_path.read_text())
    assert grid["datasets"] == ["mmlu-pro"]
    cell = grid["rows"][0]["cells"][0]
    assert cell["value"] == 0.7 and cell["n"] == 100

    adjusted = cli.invoke(app, ["import-scores", str(scores_path), "--chance-adjust"])
    assert adjusted.exit_code == 0
    assert "0.6" in adjusted.output  # (0.7-0.25)/0.75


def test_permutation_variants_via_cli(tmp_path):
    """make-spec works with all generators when n covers the policy branches."""
    result = cli.invoke(app, ["make-spec", "--dataset", "synthetic",
                              "--out", str(tmp_path / "plain.json"), "--synthetic-n", "8", "--seed", "6"])
    assert result.exit_code == 0, result.output
    spec = json.loads((tmp_path / "plain.json").read_text())
    # attach permutations through the library path and re-validate
    from jev_observatory.datasets.permutation import attach_permutations

    extended = attach_permutations(spec, seed=6, variants=1)
    spec_path = _write_spec(tmp_path, extended)
    assert cli.invoke(app, ["validate", str(spec_path)]).exit_code == 0
    assert any(i["id"].endswith(":perm1") for i in extended["items"])
