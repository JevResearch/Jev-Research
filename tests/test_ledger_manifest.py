"""Ledger + manifest tests: append-only, immutability, resume semantics."""

import json

import pytest

from jev_observatory.ledger import DuplicateRecord, RunStore
from jev_observatory.manifest import RunManifest, source_fingerprint, utc_now, verify_manifest, write_manifest
from jev_observatory.redact import Redactor


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs", "run-1", redactor=None or __import__("jev_observatory.redact", fromlist=["Redactor"]).Redactor())


def _result(lid, **over):
    record = {"logical_request_id": lid, "terminal": True, "status": "ok", "item_id": lid.split(":")[-1]}
    record.update(over)
    return record


def test_results_first_write_wins(store):
    store.append_result(_result("native:i1"))
    with pytest.raises(DuplicateRecord):
        store.append_result(_result("native:i1", status="error"))
    assert len(store.results()) == 1
    assert store.completed_ids() == {"native:i1"}


def test_attempts_append_only_no_duplicate_ids(store):
    store.append_attempt({"attempt_id": "a1", "logical_request_id": "native:i1"})
    with pytest.raises(DuplicateRecord):
        store.append_attempt({"attempt_id": "a1", "logical_request_id": "native:i1"})
    store.append_attempt({"attempt_id": "a2", "logical_request_id": "native:i1"})  # resume adds new attempt
    assert len(store.attempts()) == 2
    assert len(store.attempts_for("native:i1")) == 2


def test_uncertain_ids_pending_only(store):
    store.append_attempt({"attempt_id": "a1", "logical_request_id": "native:i1", "uncertain": True})
    store.append_attempt({"attempt_id": "a2", "logical_request_id": "native:i2", "uncertain": True})
    store.append_result(_result("native:i2"))
    assert store.uncertain_ids() == {"native:i1"}


def test_reopening_store_sees_prior_state(tmp_path):
    first = RunStore(tmp_path / "runs", "run-x")
    first.append_attempt({"attempt_id": "a1", "logical_request_id": "native:i1"})
    first.append_result(_result("native:i1"))
    reopened = RunStore(tmp_path / "runs", "run-x")
    assert reopened.completed_ids() == {"native:i1"}
    with pytest.raises(DuplicateRecord):
        reopened.append_result(_result("native:i1", status="other"))


def test_raw_write_once_rejects_conflicting_content(tmp_path):
    store = RunStore(tmp_path / "runs", "run-x")
    store.write_raw("a1", "request", {"a": 1})
    with pytest.raises(Exception):
        store.write_raw("a1", "request", {"a": "different"})


def test_identical_raw_writes_are_idempotent(tmp_path):
    store = RunStore(tmp_path / "runs", "run-x")
    ref1 = store.write_raw("a1", "request", {"same": True})
    ref2 = store.write_raw("a1", "request", {"same": True})
    assert ref1["sha256"] == ref2["sha256"]


def test_manifest_immutable_and_hash_verified(tmp_path):
    from jev_observatory.manifest import RunManifest

    manifest = RunManifest(
        run_id="run-x", experiment="e", provider="mock", model_requested="jev-1.13.0",
        created_at=utc_now(), items_sha256="abc", n_items=1,
    )
    write_manifest(tmp_path, manifest)
    # Same content: allowed (idempotent).
    write_manifest(tmp_path, manifest)
    # Different content: refused.
    manifest2 = RunManifest(
        run_id="run-x", experiment="e", provider="jev", model_requested="jev-1.13.0",
        created_at=utc_now(), items_sha256="x", n_items=1,
    )
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path, manifest2)
    # Verify detects tampering.
    loaded = verify_manifest(tmp_path)
    assert loaded.items_sha256 == manifest.items_sha256
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["n_items"] = 999  # tamper
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path)


def test_source_fingerprint_present():
    fp = source_fingerprint()
    assert len(fp["source_sha256"]) == 64
    assert fp["source_files"] > 0


from jev_observatory.manifest import source_fingerprint  # noqa: E402


def test_redaction_in_ledger_and_raw(tmp_path):
    from jev_observatory.redact import Redactor

    # Same shape as a real key, but an obvious fake: proves redaction is
    # content-driven without embedding any real credential in the repo.
    secret = "apikey_0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    redactor = Redactor(secrets=[secret])
    store = RunStore(tmp_path / "runs", "run-x", redactor=redactor)
    store.append_attempt({
        "attempt_id": "a1", "logical_request_id": "l", "headers": {"Authorization": f"Bearer {secret}"},
        "error": f"Connection refused while using {secret}",
    })
    blob = (store.directory / "attempts.jsonl").read_text()
    assert secret not in blob
    assert "[REDACTED]" in blob
    ref = store.write_raw("a1", "request", {"note": f"key={secret}"})
    assert secret not in (store.directory / ref["path"]).read_text()


def test_redactor_scrubs_structural_patterns():
    from jev_observatory.redact import Redactor

    redactor = Redactor()  # no known literals; structural patterns must still catch it
    text = redactor.text("Authorization: Bearer sk-somethinglongish123; other=data")
    assert "sk-somethinglongish" not in text or "sk-something" not in text
    assert "[REDACTED]" in text
    assert redactor.text("apikey_1234567890abcdef") == "[REDACTED]"