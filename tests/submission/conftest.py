from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def tracked_repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    root.mkdir()
    run_git(root, "init", "-q")
    run_git(root, "config", "user.name", "Dylan Moraes")
    run_git(root, "config", "user.email", "dylanmoraesdljdd@gmail.com")
    (root / "safe.txt").write_text("portable demo\n")
    policy_path = root / "submission-policy.json"
    write_policy(policy_path)
    run_git(root, "add", "safe.txt", "submission-policy.json")
    run_git(root, "commit", "-q", "-m", "test: seed release fixture")
    return root, policy_path


def write_policy(path: Path, **overrides: Any) -> None:
    policy: dict[str, Any] = {
        "schema_version": "apar-submission-policy/1",
        "archive_root": "APAR",
        "max_file_bytes": 4096,
        "max_total_bytes": 8192,
        "allowed_extensions": [".json", ".txt"],
        "extensionless_paths": [],
        "entries": [
            {"archive": "safe.txt", "required": True, "source": "safe.txt"},
            {
                "archive": "release/submission-policy.json",
                "required": True,
                "source": "submission-policy.json",
            },
        ],
        "scan": {"allowed_emails": [], "exemptions": []},
        "web": {"entries": [], "status": "pending"},
        "release": {
            "accepted_demo_arm": "ensemble_with_graph",
            "accepted_stage": "30_arms",
            "build_command": (
                "python -m scripts.submission.cli build --output APAR-submission.zip"
            ),
            "evidence_authority": {
                "accepted_capacity_evidence": False,
                "authoritative": False,
                "demo_only": True,
                "official_chain_complete": False,
                "production_ready": False,
                "real_cardholder_data": False,
                "recovered_metrics_verified": True,
            },
            "evidence_baseline_commit": "e" * 40,
            "first_missing_official_stage": "70_metrics",
            "portable_bundle_manifest_sha256": "5" * 64,
            "runtime": {"python": ">=3.12,<3.13", "supported_os": ["Linux", "macOS"]},
            "source_checkpoint_manifest_sha256": "a" * 64,
        },
    }
    policy.update(overrides)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
