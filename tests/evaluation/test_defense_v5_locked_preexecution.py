"""Adversarial contracts for the locked pre-execution freeze."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import verify_defense_v5_locked_preexecution as preexecution

ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize("mutation", ["extra_path", "descendant", "wrong_parent"])
def test_locked_preregistration_chronology_rejects_non_exact_freeze(
    tmp_path: Path, mutation: str
) -> None:
    """Only the sole manifest child of SOURCE may authorize the one-time run."""
    _git(tmp_path, "init", "-q")
    source = tmp_path / "source.txt"
    source.write_text("source\n")
    _git(tmp_path, "add", "source.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: source",
    )
    source_commit = _git(tmp_path, "rev-parse", "HEAD")
    prereg_relative = (
        "config/defense/defense-v5-locked-development-preregistration.json"
    )
    prereg = tmp_path / prereg_relative
    prereg.parent.mkdir(parents=True)
    prereg.write_text("{}\n")
    _git(tmp_path, "add", prereg_relative)
    if mutation == "extra_path":
        (tmp_path / "unexpected.txt").write_text("behavior\n")
        _git(tmp_path, "add", "unexpected.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: preregistration",
    )
    approved = _git(tmp_path, "rev-parse", "HEAD")
    if mutation == "descendant":
        (tmp_path / "descendant.txt").write_text("later\n")
        _git(tmp_path, "add", "descendant.txt")
        _git(
            tmp_path,
            "-c",
            "user.name=Dylan Moraes",
            "-c",
            "user.email=dylanmoraesdljdd@gmail.com",
            "commit",
            "-q",
            "-m",
            "test: descendant",
        )
        approved = _git(tmp_path, "rev-parse", "HEAD")
    if mutation == "wrong_parent":
        source_commit = "0" * 40
    with pytest.raises(ValueError, match="PREREGISTRATION|SOURCE|manifest"):
        preexecution.verify_locked_commit_chronology(
            root=tmp_path,
            approved_commit=approved,
            source_commit=source_commit,
            preregistration_path=prereg_relative,
        )


def test_locked_preregistration_chronology_accepts_exact_manifest_only_child(
    tmp_path: Path,
) -> None:
    """The frozen chronology remains executable once every source input is fixed."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "source.txt").write_text("source\n")
    _git(tmp_path, "add", "source.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: source",
    )
    source_commit = _git(tmp_path, "rev-parse", "HEAD")
    relative = "config/defense/defense-v5-locked-development-preregistration.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"frozen": True}) + "\n")
    _git(tmp_path, "add", relative)
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: preregistration",
    )
    approved = _git(tmp_path, "rev-parse", "HEAD")
    preexecution.verify_locked_commit_chronology(
        root=tmp_path,
        approved_commit=approved,
        source_commit=source_commit,
        preregistration_path=relative,
    )
    preexecution.verify_locked_commit_chronology(
        root=tmp_path,
        approved_commit="HEAD",
        source_commit=source_commit,
        preregistration_path=relative,
    )


def test_locked_manifest_schema_fails_before_any_experiment_boundary(
    tmp_path: Path,
) -> None:
    """A legacy summary or partial preregistration can never reach execution."""
    with pytest.raises(ValueError, match="preregistration schema"):
        preexecution._validate_manifest(
            root=tmp_path,
            document={"schema_version": "apar-sentinel-v5-development-result/1"},
            safe_verification={},
        )


def test_exact_command_has_no_seed_profile_output_or_resume_surface() -> None:
    """The preregistered command names a mode implicitly and cannot be retargeted."""
    command = preexecution._EXACT_COMMAND
    assert "--authorize-exactly-once" in command
    assert "--approved-commit HEAD" in command
    assert "--seed" not in command
    assert "--profile" not in command
    assert "--output" not in command
    assert "--resume" not in command


def test_historical_safe_core_and_rejected_result_remain_frozen() -> None:
    """The new production path must bind, not replace, earlier evidence."""
    freeze = json.loads(
        (ROOT / "config/defense/defense-v5-safe-core-freeze.json").read_bytes()
    )
    assert freeze["approved_deterministic_core_sha256"] == (
        "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
    )
    result = ROOT / "docs/experiments/defense-v5-development-result.json"
    assert hashlib.sha256(result.read_bytes()).hexdigest() == (
        "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185"
    )


@pytest.mark.parametrize(
    "existing_kind", ["file", "malformed", "symlink", "hardlink", "directory"]
)
def test_preexecution_treats_any_attempt_receipt_as_consumed(
    tmp_path: Path, existing_kind: str
) -> None:
    """Preexecution must never inspect-and-repair or replace an attempt marker."""
    target = tmp_path / "attempt.json"
    if existing_kind in {"file", "malformed"}:
        target.write_bytes(b"{}" if existing_kind == "file" else b"partial")
    elif existing_kind == "directory":
        target.mkdir()
    elif existing_kind == "symlink":
        source = tmp_path / "source"
        source.write_text("receipt")
        target.symlink_to(source)
    else:
        source = tmp_path / "source"
        source.write_text("receipt")
        os.link(source, target)
    with pytest.raises(ValueError, match="must be absent"):
        preexecution._assert_absent(target, "locked attempt receipt")


def test_preexecution_absence_check_does_not_create_attempt_receipt(
    tmp_path: Path,
) -> None:
    """Auditing readiness must not itself consume the one-time run."""
    target = tmp_path / "attempt.json"
    preexecution._assert_absent(target, "locked attempt receipt")
    assert not os.path.lexists(target)
