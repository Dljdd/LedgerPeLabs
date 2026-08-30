"""Single bounded local gate for the APAR judge release."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.submission.archive import build_archive, verify_archive
from scripts.submission.clean_room import run_clean_room
from scripts.submission.inventory import validate_dependency_inventory
from scripts.submission.model import ReleaseError
from scripts.submission.policy import load_policy

_PROTECTED_PATHS = (
    "config/defense",
    "demo/sentinel-v5",
    "docs/experiments",
    "evidence",
    "fixtures",
    "src",
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=root, check=False, capture_output=True, text=True
    )


def ensure_protected_paths_unchanged(
    root: Path, *, baseline: str, paths: tuple[str, ...]
) -> None:
    """Refuse any committed or staged change to evidence/model authority inputs."""
    completed = _git(root, "diff", "--quiet", baseline, "--", *paths)
    if completed.returncode == 1:
        raise ReleaseError("protected evidence/model paths changed from the baseline")
    if completed.returncode != 0:
        raise ReleaseError(f"protected path diff failed: {completed.stderr.strip()}")


def _ensure_clean(root: Path) -> None:
    completed = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if completed.returncode != 0:
        raise ReleaseError(f"git status failed: {completed.stderr.strip()}")
    if completed.stdout:
        raise ReleaseError("release gate requires a clean committed worktree")


def _run(root: Path, command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise ReleaseError(f"release gate command failed with exit {completed.returncode}")


def run_release_gate(root: Path) -> dict[str, Any]:
    """Run all bounded build, scan, inventory, quality, and clean-room checks."""
    repo_root = root.resolve()
    policy_path = repo_root / "scripts" / "submission" / "submission-policy.json"
    policy = load_policy(policy_path)
    baseline = policy.release.get("evidence_baseline_commit")
    if not isinstance(baseline, str):
        raise ReleaseError("release policy evidence baseline is absent")
    _ensure_clean(repo_root)
    ensure_protected_paths_unchanged(
        repo_root, baseline=baseline, paths=_PROTECTED_PATHS
    )
    validate_dependency_inventory(
        requirements_path=repo_root / "scripts/submission/requirements-judge.txt",
        sbom_path=repo_root / "scripts/submission/dependency-sbom.cdx.json",
        notice_path=repo_root / "docs/submission/THIRD_PARTY_NOTICES.md",
        web_status=policy.web_status,
    )
    python = sys.executable
    _run(
        repo_root,
        [
            python,
            "-m",
            "pytest",
            "tests/submission",
            "tests/demo/test_sentinel_v5_portable.py",
            "-q",
        ],
    )
    _run(repo_root, [python, "-m", "ruff", "check", "scripts/submission", "tests/submission"])
    _run(
        repo_root,
        [python, "-m", "mypy", "--explicit-package-bases", "scripts/submission"],
    )
    with tempfile.TemporaryDirectory(prefix="apar-submission-release-gate-") as temporary:
        temporary_root = Path(temporary)
        first_path = temporary_root / "first.zip"
        second_path = temporary_root / "second.zip"
        first = build_archive(repo_root, policy_path, first_path)
        build_archive(repo_root, policy_path, second_path)
        if first_path.read_bytes() != second_path.read_bytes():
            raise ReleaseError("two release builds produced different archive bytes")
        manifest = verify_archive(first_path)
        clean_room = run_clean_room(first_path, python_executable=python)
    _ensure_clean(repo_root)
    ensure_protected_paths_unchanged(
        repo_root, baseline=baseline, paths=_PROTECTED_PATHS
    )
    fallback = clean_room.get("fallback_trace")
    if not isinstance(fallback, dict):
        raise ReleaseError("clean-room fallback trace is absent")
    return {
        "archive_sha256": first.archive_sha256,
        "deterministic_core_sha256": manifest["deterministic_core_sha256"],
        "fallback_trace_sha256": fallback.get("trace_sha256"),
        "prediction_sha256": fallback.get("prediction_sha256"),
        "replay_verified": clean_room.get("replay_verified"),
        "scenario_count": fallback.get("scenario_count"),
        "schema_version": "apar-submission-release-gate/1",
        "source_commit": first.source_commit,
        "source_tree": first.source_tree,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    try:
        result = run_release_gate(root)
    except ReleaseError as error:
        print(f"release gate failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
