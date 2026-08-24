"""Fail-closed pre-execution audit for the one-time Sentinel v5 development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from apar.v5_independent_verifier import (
    IndependentVerificationError,
    verify_evidence_bytes,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def verify_preexecution(
    *, root: Path, safe_evidence: Path, approved_commit: str
) -> dict[str, object]:
    """Verify immutable inputs without invoking any experiment workload."""
    root = root.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("worktree is not clean")
    head = _git(root, "rev-parse", "HEAD")
    if head != approved_commit or len(approved_commit) != 40:
        raise ValueError("HEAD does not equal the exact approved commit")

    evidence_config_path = root / "config/defense/defense-v5-evidence.json"
    development_config_path = root / "config/defense/defense-v5-development.json"
    evidence_config = json.loads(evidence_config_path.read_bytes())
    development_config = json.loads(development_config_path.read_bytes())
    if (
        evidence_config["safe_development_test_seed"] != 404
        or evidence_config["locked_development_test_seed"] != 2404
        or development_config["seeds"]["development_test"] != 2404
    ):
        raise ValueError("safe/locked seed bindings differ from 404/2404")

    result_path = root / evidence_config["existing_development_result_path"]
    if not result_path.is_file():
        raise ValueError("frozen existing development result is absent")
    result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if result_sha256 != evidence_config["existing_development_result_sha256"]:
        raise ValueError("existing development result bytes changed")

    verification = verify_evidence_bytes(
        safe_evidence.read_bytes(), root=root
    )
    return {
        "verified": True,
        "approved_commit": approved_commit,
        "worktree_clean": True,
        "safe_evidence_verified": verification["verified"],
        "safe_evidence_status": verification["status"],
        "safe_seed_executed": verification["safe_seed"],
        "locked_seed_asserted_only": 2404,
        "existing_result_sha256": result_sha256,
        "evidence_config_sha256": hashlib.sha256(
            evidence_config_path.read_bytes()
        ).hexdigest(),
        "development_config_sha256": hashlib.sha256(
            development_config_path.read_bytes()
        ).hexdigest(),
        "payload_sha256": verification["payload_sha256"],
        "envelope_sha256": verification["envelope_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--safe-evidence", type=Path, required=True)
    parser.add_argument("--approved-commit", required=True)
    args = parser.parse_args()
    try:
        report = verify_preexecution(
            root=args.root,
            safe_evidence=args.safe_evidence,
            approved_commit=args.approved_commit,
        )
    except (IndependentVerificationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"verified": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
