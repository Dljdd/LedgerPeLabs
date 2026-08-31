from __future__ import annotations

from pathlib import Path

import pytest

from scripts.submission.model import ReleaseError
from scripts.submission.release_gate import ensure_protected_paths_unchanged

from .conftest import run_git


def test_release_gate_rejects_changes_to_protected_evidence(
    tracked_repository: tuple[Path, Path]
) -> None:
    """A release commit must never alter the accepted model/evidence baseline."""
    root, _ = tracked_repository
    (root / "evidence.json").write_text("accepted\n")
    run_git(root, "add", "evidence.json")
    run_git(root, "commit", "-q", "-m", "test: add protected evidence")
    protected_baseline = run_git(root, "rev-parse", "HEAD")
    (root / "evidence.json").write_text("changed\n")
    run_git(root, "add", "evidence.json")
    run_git(root, "commit", "-q", "-m", "test: mutate protected evidence")
    current = run_git(root, "rev-parse", "HEAD")

    ensure_protected_paths_unchanged(root, baseline=current, paths=("evidence.json",))
    with pytest.raises(ReleaseError, match="protected evidence/model paths changed"):
        ensure_protected_paths_unchanged(
            root, baseline=protected_baseline, paths=("evidence.json",)
        )
