"""Closed, deterministic command contracts for the Defend evidence pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apar.defense.orchestration import (
    CliContractError,
    CompetitionProfile,
    load_competition_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "defense" / "competition-profile.json"


def test_committed_competition_profile_is_exact_and_fixture_cannot_export(
    tmp_path: Path,
) -> None:
    profile = load_competition_profile(PROFILE, competition=True)

    assert profile.campaign_count == 200
    assert profile.partition_campaign_indices["development_test"] == (38, 49)
    assert profile.campaign_seed("card_testing_cnp", 49) == 262049
    assert profile.campaign_start("app_scam_mule", 2).isoformat() == (
        "2026-01-17T00:00:00+00:00"
    )
    assert profile.fixture_only is False

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(CompetitionProfile.fixture().to_json())
    with pytest.raises(CliContractError, match="competition profile"):
        load_competition_profile(fixture_path, competition=True)


def test_competition_profile_rejects_modified_noncanonical_and_extra_documents(
    tmp_path: Path,
) -> None:
    original = json.loads(PROFILE.read_text(encoding="utf-8"))
    cases = []
    changed = dict(original)
    changed["model_seed"] = 260817
    cases.append(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode())
    extra = dict(original)
    extra["fixture_only"] = False
    cases.append(json.dumps(extra, sort_keys=True, separators=(",", ":")).encode())
    cases.append(json.dumps(original, indent=2).encode())

    for index, payload in enumerate(cases):
        candidate = tmp_path / f"profile-{index}.json"
        candidate.write_bytes(payload)
        with pytest.raises(CliContractError, match="competition profile|canonical"):
            load_competition_profile(candidate, competition=True)


def test_hidden_cli_refuses_before_resolving_an_unfrozen_defender(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_defender.py",
            "--phase",
            "hidden",
            "--defender",
            "0" * 64,
            "--development-scorecard",
            "1" * 64,
            "--profile",
            str(PROFILE),
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "frozen defender" in completed.stderr.lower()
    assert str(tmp_path) not in completed.stderr


def test_cli_rejects_undeclared_arguments_without_a_traceback(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_defense_corpus.py",
            "--profile",
            str(PROFILE),
            "--run-manifests",
            "0" * 64,
            "--root",
            str(tmp_path),
            "--output-manifest",
            "corpus.json",
            "--fixture",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "unrecognized arguments" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()
