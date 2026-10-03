"""Re-render gates must not add undeployable files to the publish manifest."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bundle_render_hygiene", ROOT / "scripts/report/build_site.py")
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)


def test_rerender_disables_bytecode_even_without_parent_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    scripts = tmp_path / "scripts/report"
    scripts.mkdir(parents=True)
    for folder, contents in (("report", "main"), ("counterpoint", "side")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "index.html").write_text(contents)
    (scripts / "build_report.py").write_text("pass\n")
    (scripts / "helper.py").write_text("VALUE = 'side'\n")
    (scripts / "build_counterpoint.py").write_text(
        "from helper import VALUE\n"
        "from pathlib import Path\n"
        "Path('counterpoint/index.html').write_text(VALUE)\n")
    assert SITE.verify_render(tmp_path) is None
    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob("*.pyc"))


def test_final_structural_scan_rejects_rerender_side_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(SITE, "BUNDLE", tmp_path)
    monkeypatch.setattr(SITE, "_cap_memory", lambda: None)
    monkeypatch.setattr(SITE, "build", lambda: None)
    monkeypatch.setattr(SITE, "write_scaffolding", lambda: None)
    monkeypatch.setattr(SITE, "secret_scan", lambda _root: [])
    monkeypatch.setattr(SITE, "licensed_scan", lambda _root: [])

    def unsafe_render(root):
        cache = root / "scripts/report/__pycache__/unexpected.pyc"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"generated after the preflight")
        return None

    monkeypatch.setattr(SITE, "verify_render", unsafe_render)
    with pytest.raises(SystemExit) as error:
        SITE.main()
    assert error.value.code == 1
    assert not (tmp_path / "BUNDLE.sha256").exists()
