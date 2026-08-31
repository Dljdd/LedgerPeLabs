from __future__ import annotations

from pathlib import Path

from scripts.submission.archive import build_archive
from scripts.submission.clean_room import extract_verified_archive


def test_clean_room_extracts_only_verified_manifest_members(
    tracked_repository: tuple[Path, Path], tmp_path: Path
) -> None:
    """Clean-room replay must consume a fresh extraction, not the source worktree."""
    root, policy = tracked_repository
    archive = tmp_path / "release.zip"
    extraction = tmp_path / "extracted"
    build_archive(root, policy, archive)

    release_root = extract_verified_archive(archive, extraction)

    assert release_root == extraction / "APAR"
    assert (release_root / "safe.txt").read_text() == "portable demo\n"
    assert (release_root / "SUBMISSION_MANIFEST.json").is_file()
