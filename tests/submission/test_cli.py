from __future__ import annotations

from pathlib import Path

from scripts.submission.cli import main


def test_cli_builds_and_verifies_archive(
    tracked_repository: tuple[Path, Path], tmp_path: Path, capsys: object
) -> None:
    """A release command that omits either build or verification is not judge-ready."""
    del capsys
    root, policy = tracked_repository
    output = tmp_path / "release.zip"

    assert main(
        [
            "build",
            "--repo",
            str(root),
            "--policy",
            str(policy),
            "--output",
            str(output),
        ]
    ) == 0
    assert output.is_file()
    assert main(["verify-archive", "--archive", str(output)]) == 0


def test_cli_reports_pending_web_as_failure(
    tracked_repository: tuple[Path, Path], tmp_path: Path, capsys: object
) -> None:
    """CLI exit status must expose that requested UI integration is unavailable."""
    root, policy = tracked_repository
    assert main(
        [
            "build",
            "--repo",
            str(root),
            "--policy",
            str(policy),
            "--output",
            str(tmp_path / "release.zip"),
            "--include-web",
        ]
    ) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "web artifact integration is pending" in captured.err
