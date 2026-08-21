"""G3 smoke proof through real authenticated APAR run artifacts."""

from __future__ import annotations

from pathlib import Path

from apar.defense.orchestration import run_g3_fixture
from apar.evaluation.gates import DefenseArm
from apar.runs.wire import strict_json_loads
from apar.storage.artifacts import ArtifactStore


def test_g3_fixture_consumes_real_authenticated_run_artifacts(tmp_path: Path) -> None:
    result = run_g3_fixture(tmp_path)

    assert result.run_manifests_verified == 4
    assert result.arms == tuple(arm.value for arm in DefenseArm)
    assert result.scorecard_ref.sha256 == result.public_artifacts[
        "defense-scorecard.json"
    ].sha256
    assert result.reduced_fixture_evidence is True
    assert result.competition_evidence is False
    assert result.fixture_control_campaign_count == 1
    assert result.ensemble_mode == "reduced_pooled_only"
    assert result.champion_status == "no_promotion"

    scorecard = strict_json_loads(
        ArtifactStore(tmp_path / "artifacts").read(result.scorecard_ref)
    )
    assert isinstance(scorecard, dict)
    corpus_summary = scorecard["corpus_summary"]
    assert isinstance(corpus_summary, dict)
    assert corpus_summary["synthetic_only"] is True
    assert [row["arm"] for row in scorecard["leaderboard"]] == [
        arm.value for arm in DefenseArm
    ]
    assert "truth" not in scorecard
    assert "predictions" not in scorecard
    assert "hidden_ref" not in scorecard


def test_fixture_rerun_is_content_identical_across_separate_roots(tmp_path: Path) -> None:
    first = run_g3_fixture(tmp_path / "first")
    second = run_g3_fixture(tmp_path / "second")

    assert first.core_artifact_digests == second.core_artifact_digests
    assert first.scorecard_ref.sha256 == second.scorecard_ref.sha256
    assert first.signer_key_id == second.signer_key_id
    assert first.defender_ref.sha256 == second.defender_ref.sha256
    assert first.threshold_set_ref.sha256 == second.threshold_set_ref.sha256
    assert first.evaluation_bundle_ref.sha256 == second.evaluation_bundle_ref.sha256
    assert first.fixture_control_ref.sha256 == second.fixture_control_ref.sha256
