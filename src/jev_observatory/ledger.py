"""Append-only run storage: attempt ledger, logical results, raw bodies.

Three invariants make runs resumable and audit-friendly:

1. **Append-only.** Ledger files are only ever opened in append mode; a record
   already written is never rewritten.  Duplicate ids are rejected.
2. **Result first-wins.** A logical request becomes terminal exactly once.  A
   resume can add *new* attempts but cannot replace an observation.
3. **Uncertainty is explicit.** A timeout may have been processed and billed
   remotely, so such attempts are recorded as ``uncertain`` and a resume needs an
   explicit policy before touching them.

Gold labels never enter this module: `results.jsonl` holds predictions and
metadata only, so run artifacts are safe to share without leaking answers.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator

from .redact import Redactor

ATTEMPTS_FILE = "attempts.jsonl"
RESULTS_FILE = "results.jsonl"
RAW_DIR = "raw"


class DuplicateRecord(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalStatus:
    """Enumeration of logical-request outcomes (kept as plain strings on disk)."""

    OK = "ok"
    ERROR = "error"
    UNCERTAIN = "uncertain"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - corruption guard
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc


class RunStore:
    """Filesystem-backed artifacts for a single run id."""

    def __init__(self, root: Path | str, run_id: str, redactor: Redactor | None = None) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.directory = self.root / run_id
        self.attempts_path = self.directory / ATTEMPTS_FILE
        self.results_path = self.directory / RESULTS_FILE
        self.raw_dir = self.directory / RAW_DIR
        self.redactor = redactor or Redactor()
        self._lock = threading.Lock()
        self._seen_attempts: set[str] = set()
        self._seen_results: set[str] = set()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._seen_attempts = {r["attempt_id"] for r in _iter_jsonl(self.attempts_path) if "attempt_id" in r}
        self._seen_results = {
            r["logical_request_id"] for r in _iter_jsonl(self.results_path) if "logical_request_id" in r
        }

    # ------------------------------------------------------------------ raw
    @staticmethod
    def _raw_filename(attempt_id: str, kind: str) -> str:
        """Filename for a raw blob. Attempt ids are logical strings and may
        contain ``/`` (e.g. dataset ids like ``test/algebra/0.json``); a raw
        slash would turn the blob path into a nonexistent nested directory.
        Percent-encode the separator so names stay unique, flat and stable.
        """
        return f"{attempt_id.replace('/', '%2F')}.{kind}.json"

    def write_raw(self, attempt_id: str, kind: str, payload: Any) -> dict[str, str]:
        """Persist a raw request/response blob (redacted) and return ref+hash."""
        if kind not in {"request", "response", "error"}:
            raise ValueError(f"unknown raw kind {kind!r}")
        path = self.raw_dir / self._raw_filename(attempt_id, kind)
        body = json.dumps(self.redactor.obj(payload), sort_keys=False, ensure_ascii=False, indent=2) + "\n"
        encoded = body.encode("utf-8")
        with self._file_lock("raw.lock"):
            if path.exists():
                # Identical repeats (a retry with the same body) are idempotent;
                # a *different* payload for the same attempt id is a bug.
                if path.read_bytes() == encoded:
                    return {"path": str(path.relative_to(self.directory)),
                            "sha256": sha256(encoded).hexdigest(), "bytes": len(encoded)}
                raise FileExistsError(
                    f"raw blob {path.name} already exists with different content"
                )
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        return {"path": str(path.relative_to(self.directory)), "sha256": sha256(encoded).hexdigest(),
                "bytes": len(encoded)}

    # -------------------------------------------------------------- appends
    def append_attempt(self, record: dict[str, Any]) -> dict[str, Any]:
        record = json.loads(self.redactor.text(json.dumps(record, default=str)))
        attempt_id = record.get("attempt_id")
        if not attempt_id:
            raise ValueError("attempt record requires attempt_id")
        with self._lock:
            if attempt_id in self._seen_attempts:
                raise DuplicateRecord(f"attempt {attempt_id} already recorded; the ledger is append-only")
            self._append_line(self.attempts_path, record, "attempts.lock")
            self._seen_attempts.add(attempt_id)
        return record

    def append_result(self, record: dict[str, Any]) -> dict[str, Any]:
        """Terminal logical-request record; first writer wins, later ones fail loudly."""
        record = json.loads(self.redactor.text(json.dumps(record, default=str)))
        logical_id = record.get("logical_request_id")
        if not logical_id:
            raise ValueError("result record requires logical_request_id")
        with self._lock:
            if logical_id in self._seen_results:
                raise DuplicateRecord(
                    f"logical request {logical_id} already terminal; results are never overwritten"
                )
            self._append_line(self.results_path, record, "results.lock")
            self._seen_results.add(logical_id)
        return record

    def _append_line(self, path: Path, record: dict[str, Any], lock_name: str) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._file_lock(lock_name):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    class _Locked:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.handle = None

        def __enter__(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.path, "a+", encoding="utf-8")
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            except OSError as exc:  # pragma: no cover - exotic filesystems
                if exc.errno not in {errno.EACCES, errno.ENOLCK, errno.EBADF}:
                    raise
            return self.handle

        def __exit__(self, *exc_info):
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except OSError:  # pragma: no cover
                pass
            self.handle.close()
            return False

    def _file_lock(self, name: str) -> "RunStore._Locked":
        return RunStore._Locked(self.directory / f".{name}")

    # --------------------------------------------------------------- reads
    def attempts(self) -> list[dict[str, Any]]:
        return list(_iter_jsonl(self.attempts_path))

    def results(self) -> list[dict[str, Any]]:
        return list(_iter_jsonl(self.results_path))

    def completed_ids(self) -> set[str]:
        """Logical requests that are terminal and must not be re-dispatched."""
        return {r["logical_request_id"] for r in self.results() if r.get("terminal") is True}

    def terminal_ids(self) -> set[str]:
        return set(self._seen_results)

    def uncertain_ids(self) -> set[str]:
        """Logical requests with only uncertain attempts and no terminal result.

        These are exactly the requests where the provider may have processed and
        billed work we cannot observe, so a resume needs an explicit decision.
        """
        uncertain: dict[str, int] = {}
        for attempt in self.attempts():
            if attempt.get("uncertain"):
                uncertain[attempt["logical_request_id"]] = uncertain.get(attempt["logical_request_id"], 0) + 1
        return {lid for lid in uncertain if lid not in self._seen_results}

    def attempts_for(self, logical_request_id: str) -> list[dict[str, Any]]:
        return [a for a in self.attempts() if a.get("logical_request_id") == logical_request_id]

    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def derived_dir(self) -> Path:
        path = self.directory / "derived"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_paths(self) -> Iterable[Path]:
        return sorted(p for p in self.directory.rglob("*") if p.is_file())
