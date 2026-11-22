"""Smoke tests: the commands run from a clean process and write what they say."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "creditadj", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_cli_smoke(art, tmp_path):
    explain = run("explain", "--row", "3")
    assert explain.returncode == 0, explain.stderr
    assert "Predicted probability of default" in explain.stdout
    assert "APPROVE" in explain.stdout or "DECLINE" in explain.stdout

    reports, web = tmp_path / "reports", tmp_path / "web"
    everything = run("all", "--n-boot", "20", "--reports-dir", str(reports), "--web-dir", str(web))
    assert everything.returncode == 0, everything.stderr
    for name in (
        "metrics.json",
        "roc.png",
        "deciles.png",
        "calibration.png",
        "cutoff_tradeoff.png",
    ):
        assert (reports / name).stat().st_size > 0
    metrics = json.loads((reports / "metrics.json").read_text())
    assert 0.5 < metrics["champion"]["auc"] < 1.0
    assert json.loads((web / "model.json").read_text())["meta"]["n_test"] == 6000
