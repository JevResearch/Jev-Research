"""Immutable run manifests and source fingerprinting.

A manifest is written exactly once (`O_EXCL`) and its canonical SHA-256 is kept
in a sidecar file.  Resuming a run re-verifies that hash, so a mutated manifest
is detected rather than silently trusted.  No git repository is required: the
fingerprint is a content hash over the package sources plus pinned dependency
versions, with an explicit `dirty_tree: null` marker when git is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


def _git(args: list[str]) -> str | None:
    """Return trimmed stdout, or None if git/repo is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - optional dependency
        return "absent"
    return str(getattr(module, "__version__", getattr(module, "VERSION", "unknown")))


def source_fingerprint() -> dict[str, Any]:
    """Content hash of every package file, so artifact ↔ code is traceable."""
    digest = hashlib.sha256()
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        digest.update(str(path.relative_to(PACKAGE_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    repo = {"revision": None, "dirty": None}
    try:
        head = _git(["rev-parse", "HEAD"])
        if head is not None:
            status = _git(["status", "--porcelain"])
            repo = {"revision": head, "dirty": bool(status)}
    except Exception:  # pragma: no cover - git absent or repo unavailable
        pass
    return {
        "source_sha256": digest.hexdigest(),
        "source_files": len(files),
        "git": repo,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "pydantic": _version("pydantic"),
            "httpx": _version("httpx"),
            "typer": _version("typer"),
            "pyarrow": _version("pyarrow"),
        },
    }


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class RunManifest:
    """Everything needed to reproduce *which* requests a run intended to make."""

    run_id: str
    experiment: str
    provider: str
    model_requested: str
    created_at: str
    items_sha256: str
    n_items: int
    item_hashes: dict[str, Any] = field(default_factory=dict)
    option_maps: dict[str, Any] = field(default_factory=dict)
    prompt_templates: dict[str, str] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    inclusion_rules: dict[str, Any] = field(default_factory=dict)
    shuffle: bool = True  # designed experiments set False: dispatch order is data
    pricing: dict[str, Any] = field(default_factory=dict)
    test_family: str = "unspecified"
    preregistration: str | None = None
    execution_region: str | None = None
    fingerprint: dict[str, Any] = field(default_factory=source_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def write_manifest(store_dir: Path, manifest: RunManifest) -> Path:
    """Write once; refuse to overwrite an existing manifest."""
    path = store_dir / "manifest.json"
    sidecar = store_dir / "MANIFEST.sha256"
    payload = manifest.canonical_json() + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = path.read_text()
        if existing != payload:
            raise FileExistsError(
                f"manifest at {path} already exists with different content; manifests are immutable"
            )
        return path
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w") as handle:
            handle.write(manifest.sha256() + "\n")
    except FileExistsError:
        pass
    return path


def verify_manifest(store_dir: Path) -> RunManifest:
    """Load a manifest and confirm it matches its recorded hash.

    The hash is computed over the manifest *as stored* (the raw parsed JSON),
    not over a dataclass round-trip.  Round-tripping would make any later
    schema evolution (new fields with defaults) look like tampering with
    manifests written before the change.
    """
    path = store_dir / "manifest.json"
    sidecar = store_dir / "MANIFEST.sha256"
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    data = json.loads(path.read_text())
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        actual = canonical_sha256(data)
        if expected != actual:
            raise ValueError(f"manifest hash mismatch: sidecar={expected} recomputed={actual} (file was modified)")
    return RunManifest.from_dict(data)
