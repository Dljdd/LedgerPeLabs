"""Sealed Defend v3 protocol boundary and frozen v1/v2 isolation tests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apar.v3_protocol import (
    MAX_CONFIRMATORY_ATTEMPTS,
    PROTOCOL_ID,
    SYNTHETIC_NON_CLAIM,
    V3Protocol,
    V3ProtocolError,
    verify_v1_v2_roots,
)

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_id_is_distinct() -> None:
    assert PROTOCOL_ID == "apar-defend-v3"
    assert PROTOCOL_ID not in ("apar-defend-v1", "apar-defend-v2")


def test_maximum_confirmatory_attempts_is_one() -> None:
    assert MAX_CONFIRMATORY_ATTEMPTS == 1


def test_synthetic_non_claim_is_exact() -> None:
    assert SYNTHETIC_NON_CLAIM == (
        "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
    )


def test_fixture_has_thirteen_named_seed_commitments() -> None:
    protocol = V3Protocol.fixture()
    names = [item.name for item in protocol.seed_commitments]
    assert len(names) == 13
    assert len(set(names)) == 13
    assert "hidden_evaluation" in names
    assert "benign_only_control" in names
    assert "score_permutation_control" in names


def test_fixture_cannot_serialize_as_preregistration() -> None:
    fixture = V3Protocol.fixture()
    with pytest.raises(V3ProtocolError, match="fixture-only"):
        fixture.canonical_bytes()


def test_rejects_v1_or_v2_protocol_identifier() -> None:
    base = V3Protocol.fixture().model_dump(mode="python")
    for bad_id in ("apar-defend-v1", "apar-defend-v2"):
        with pytest.raises(ValueError, match="v1 or v2"):
            V3Protocol.model_validate({**base, "protocol_id": bad_id})


def test_missing_seed_name_is_rejected() -> None:
    base = V3Protocol.fixture().model_dump(mode="python")
    commitments = list(base["seed_commitments"])
    commitments.pop(0)
    with pytest.raises(ValueError, match="thirteen named seeds"):
        V3Protocol.model_validate({**base, "seed_commitments": commitments})


def test_public_protocol_import_does_not_load_evaluator_runtime() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import apar.v3_protocol; "
            "blocked=('apar.evaluation', 'apar.runs', 'apar.redteam'); "
            "assert not any(name == root or name.startswith(root + '.') "
            "for name in sys.modules for root in blocked)",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_v1_and_v2_roots_match_in_worktree() -> None:
    verify_v1_v2_roots(ROOT)


def test_frozen_root_mismatch_fails_closed(tmp_path: Path) -> None:
    import shutil

    for relative in (
        "docs/experiments/defense-v1-preregistration.json",
        "docs/experiments/defense-v1-result.json",
        "docs/experiments/defense-v1-run-manifests.json",
        "fixtures/defense/v1/hash-manifest.json",
        "config/defense/competition-v2-preregistration.json",
        "config/defense/competition-v2-profile.json",
        "config/defense/competition-v2-manifests.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (tmp_path / "docs/experiments/defense-v1-result.json").write_bytes(b"changed")
    with pytest.raises(V3ProtocolError, match="frozen root mismatch"):
        verify_v1_v2_roots(tmp_path)
