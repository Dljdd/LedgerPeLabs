"""Read-only pre-execution verifier tests for Defend v4."""

from __future__ import annotations

import shutil
from pathlib import Path

from apar.evaluation.v4_preexecution import verify_v4_preexecution

ROOT = Path(__file__).resolve().parents[2]


def test_preexecution_is_admissible_in_worktree() -> None:
    report = verify_v4_preexecution(ROOT)
    assert report.admissible
    assert report.codes == ()
    assert report.status == "not_executed"


def test_receipt_presence_fails_preexecution(tmp_path: Path) -> None:
    for relative in (
        "docs/experiments/defense-v1-preregistration.json",
        "docs/experiments/defense-v1-result.json",
        "docs/experiments/defense-v1-run-manifests.json",
        "fixtures/defense/v1/hash-manifest.json",
        "config/defense/competition-v2-preregistration.json",
        "config/defense/competition-v2-profile.json",
        "config/defense/competition-v2-manifests.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    receipt_dir = tmp_path / ".apar/defense-v4"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "execution-receipt.json").write_bytes(b"{}")
    report = verify_v4_preexecution(tmp_path)
    assert not report.admissible
    assert "V4_EXECUTION_RECEIPT_PRESENT" in report.codes
