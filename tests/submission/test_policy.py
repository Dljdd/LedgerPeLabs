from __future__ import annotations

from pathlib import Path

import pytest

from scripts.submission.model import ReleaseError
from scripts.submission.policy import load_policy

from .conftest import write_policy


@pytest.mark.parametrize(
    "bad_path",
    ("../escape.txt", "/absolute.txt", "folder/../../escape.txt", "folder\\file.txt"),
)
def test_policy_rejects_unsafe_source_or_archive_paths(tmp_path: Path, bad_path: str) -> None:
    """Path traversal in policy would escape either the repository or archive root."""
    policy_path = tmp_path / "policy.json"
    write_policy(
        policy_path,
        entries=[{"archive": "safe.txt", "required": True, "source": bad_path}],
    )

    with pytest.raises(ReleaseError, match="unsafe relative path"):
        load_policy(policy_path)


def test_policy_rejects_broad_scan_exemptions(tmp_path: Path) -> None:
    """Wildcard exemptions would turn a fail-closed secret scanner into fail-open."""
    policy_path = tmp_path / "policy.json"
    write_policy(
        policy_path,
        scan={
            "allowed_emails": [],
            "exemptions": [
                {"path": "*", "reason": "models are noisy", "rule": "pii_pan"}
            ],
        },
    )

    with pytest.raises(ReleaseError, match="exact path"):
        load_policy(policy_path)
