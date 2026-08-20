"""End-to-end non-execution checks for the v2 verifier CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_preexecution_cli_reports_not_executed_without_writing_artifacts(tmp_path: Path) -> None:
    """The public CLI validates only and leaves its target root unchanged."""
    for relative in (
        "docs/experiments/defense-v1-preregistration.json",
        "docs/experiments/defense-v1-result.json",
        "docs/experiments/defense-v1-run-manifests.json",
        "fixtures/defense/v1/hash-manifest.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    completed = subprocess.run(
        [sys.executable, "scripts/verify_defense_v2_preexecution.py", "--root", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert completed.returncode == 0
    assert '"status":"not_executed"' in completed.stdout
    assert before == after
