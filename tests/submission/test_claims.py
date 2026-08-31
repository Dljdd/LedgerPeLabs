from pathlib import Path

from scripts.submission.claims import verify_submission_claims


def test_judge_facing_claims_match_machine_evidence() -> None:
    result = verify_submission_claims(Path(__file__).resolve().parents[2])

    assert result["verified"] is True
    assert result["arm"] == "ensemble_with_graph"
    assert result["official_chain_complete"] is False
