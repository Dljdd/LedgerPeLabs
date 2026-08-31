from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from scripts.submission.archive import build_archive, verify_archive
from scripts.submission.model import ReleaseError

from .conftest import run_git, write_policy


def test_builder_emits_identical_archives_and_complete_manifest(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """Variable ZIP metadata or omitted file hashes would break reproducible review."""
    root, policy_path = tracked_repository
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = build_archive(root, policy_path, first)
    second_result = build_archive(root, policy_path, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.archive_sha256 == second_result.archive_sha256
    assert first_result.source_commit == run_git(root, "rev-parse", "HEAD")
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert {entry.date_time for entry in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        manifest = json.loads(archive.read("APAR/SUBMISSION_MANIFEST.json"))
    assert manifest["source"]["commit"] == run_git(root, "rev-parse", "HEAD")
    assert manifest["source"]["tree"] == run_git(root, "rev-parse", "HEAD^{tree}")
    assert manifest["files"] == [
        {
            "path": "release/submission-policy.json",
            "sha256": manifest["files"][0]["sha256"],
            "size": len((root / "submission-policy.json").read_bytes()),
        },
        {
            "path": "safe.txt",
            "sha256": "fad3c2a6b874f2723099cdd93baa3fc39263b8d575b28eeb1eccb17dfbe619ce",
            "size": 14,
        },
    ]
    assert manifest["evidence_authority"]["authoritative"] is False
    assert manifest["web"]["status"] == "pending"
    verify_archive(first)


def test_builder_refuses_missing_required_tracked_file(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """Silently dropping a required judge file would produce an incomplete archive."""
    root, policy_path = tracked_repository
    policy = json.loads(policy_path.read_text())
    policy["entries"].append(
        {"archive": "required.txt", "required": True, "source": "required.txt"}
    )
    write_policy(policy_path, **policy)
    run_git(root, "add", "submission-policy.json")

    with pytest.raises(ReleaseError, match="required tracked file is missing"):
        build_archive(root, policy_path, tmp_path / "release.zip")


def test_builder_refuses_tracked_symlink(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """Materializing a tracked symlink could read content outside the repository."""
    root, policy_path = tracked_repository
    os.symlink("safe.txt", root / "link.txt")
    policy = json.loads(policy_path.read_text())
    policy["entries"].append(
        {"archive": "link.txt", "required": True, "source": "link.txt"}
    )
    write_policy(policy_path, **policy)
    run_git(root, "add", "link.txt", "submission-policy.json")

    with pytest.raises(ReleaseError, match="symlink"):
        build_archive(root, policy_path, tmp_path / "release.zip")


def test_builder_refuses_existing_output(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """An overwrite flag could destroy a prior release or canonical result."""
    root, policy_path = tracked_repository
    output = tmp_path / "release.zip"
    output.write_bytes(b"keep me")

    with pytest.raises(ReleaseError, match="already exists"):
        build_archive(root, policy_path, output)
    assert output.read_bytes() == b"keep me"


def test_builder_refuses_policy_that_differs_from_tracked_index(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """An unstaged policy edit could bypass the allowlist bound to the source tree."""
    root, policy_path = tracked_repository
    policy_path.write_text(policy_path.read_text().replace("8192", "8193"))

    with pytest.raises(ReleaseError, match="policy differs from the tracked index"):
        build_archive(root, policy_path, tmp_path / "release.zip")


def test_builder_refuses_output_under_protected_evidence_prefix(
    tracked_repository: tuple[Path, Path]
) -> None:
    """Release tooling must never write a new archive into canonical evidence paths."""
    root, policy_path = tracked_repository
    policy = json.loads(policy_path.read_text())
    policy["release"]["protected_output_prefixes"] = ["evidence", "results"]
    write_policy(policy_path, **policy)
    run_git(root, "add", "submission-policy.json")

    with pytest.raises(ReleaseError, match="protected canonical path"):
        build_archive(root, policy_path, root / "evidence" / "release.zip")


def test_pending_web_integration_fails_honestly(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """Requesting an absent UI must not silently produce a CLI-only archive."""
    root, policy_path = tracked_repository

    with pytest.raises(ReleaseError, match="web artifact integration is pending"):
        build_archive(root, policy_path, tmp_path / "release.zip", include_web=True)


def test_archive_verifier_rejects_duplicate_or_tampered_members(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """A duplicate ZIP member can shadow the hash-bound payload during extraction."""
    root, policy_path = tracked_repository
    output = tmp_path / "release.zip"
    build_archive(root, policy_path, output)
    with (
        zipfile.ZipFile(output, "a") as archive,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        archive.writestr("APAR/safe.txt", b"tampered")

    with pytest.raises(ReleaseError, match="duplicate archive member"):
        verify_archive(output)
