"""The standalone report guard must fail the process, not just print FAIL."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/report/verify_report_items.py",
    "scripts/report/chartkit.py",
    "report/index.html",
    "docs/modern-comparison/comparison-graphs.html",
    "docs/modern-comparison/pareto-frontiers.html",
    "docs/modern-comparison/architecture-diagram.html",
    "docs/modern-comparison/architecture-evidence.html",
    "data_report/baselines/v4r1/public_summary.json",
)


def test_guard_exit_status_matches_checks(tmp_path):
    for relative in FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    script = tmp_path / FILES[0]
    good = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "FAILS: 0" in good.stdout

    page = tmp_path / "report/index.html"
    page.write_text(page.read_text() + "\nserver compute\n")
    bad = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, check=False)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "FAILS: 1" in bad.stdout
    assert "FAIL " in bad.stdout
