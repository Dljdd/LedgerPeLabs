"""Public, non-executing Defend v2 scorecard contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v2_preregistration import ExecutionReceipt
from apar.evaluation.v2_protocol import load_v2_protocol
from apar.evaluation.v2_reporting import (
    DefenseV2GateReport,
    DefenseV2Scorecard,
    V2ArmScorecard,
    V2ReportingContractError,
    load_current_v2_scorecard,
    not_executed_result,
    render_v2_scorecard,
)
from apar.evaluation.v2_selection import V2GateOutcome
from apar.runs import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_not_executed_lists_all_arms_without_hidden_values() -> None:
    """Dropping an unevaluated comparator would conceal its unavailable evidence."""
    card, files = render_v2_scorecard(not_executed_result(), signer=signer())

    assert card.status == "not_executed"
    assert isinstance(card, DefenseV2Scorecard)
    assert [arm.arm for arm in card.arms] == [
        "rules_only",
        "gbdt_only",
        "layered_hybrid",
    ]
    assert "hidden_seed" not in files["defense-v2-scorecard.json"].decode("utf-8")


def test_scorecard_binds_the_committed_protocol_profile_digest() -> None:
    """Public status must identify the exact profile signed by preregistration."""
    protocol = load_v2_protocol(ROOT / "config/defense/competition-v2-profile.json")

    assert not_executed_result().protocol_digest == protocol.profile_sha256


def test_current_scorecard_rejects_a_different_protocol_digest(tmp_path: Path) -> None:
    """A signed card for another protocol cannot become current V2 state."""
    state = tmp_path / ".apar/defense-v2"
    state.mkdir(parents=True)
    arms = tuple(
        V2ArmScorecard(
            arm=arm,
            status="no_promotion",
            gate=DefenseV2GateReport(
                arm=arm,
                outcome=V2GateOutcome(passed=False, codes=("CONTROL_INVALID",)),
            ),
        )
        for arm in ("rules_only", "gbdt_only", "layered_hybrid")
    )
    result = not_executed_result(protocol_digest="a" * 64).model_copy(
        update={"status": "no_promotion", "arms": arms}
    )
    card, _ = render_v2_scorecard(result, signer=signer())
    (state / "defense-v2-scorecard.json").write_bytes(card.to_json())
    receipt = ExecutionReceipt(
        preregistration_id="apar-defend-v2", execution_nonce="receipt-present"
    )
    (state / "execution-receipt.json").write_bytes(
        canonical_json_bytes(receipt.model_dump(mode="json"))
    )
    fallback, _ = render_v2_scorecard(not_executed_result(), signer=signer())

    with pytest.raises(V2ReportingContractError, match="protocol"):
        load_current_v2_scorecard(tmp_path, fallback=fallback)


def test_workload_csv_has_both_denominators() -> None:
    """Collapsing case and transaction workload rates would misstate capacity evidence."""
    _, files = render_v2_scorecard(not_executed_result(), signer=signer())

    assert (
        b"review_cases,reviewed_transactions,review_case_rate,review_transaction_rate"
        in files["defense-v2-workload.csv"].splitlines()[0]
    )


def test_promotion_eligible_rejects_an_arm_with_a_failed_gate() -> None:
    """A promoted scorecard must not suppress a failing comparator gate."""
    result = not_executed_result().model_copy(
        update={
            "status": "promotion_eligible",
            "arms": (
                V2ArmScorecard(
                    arm="rules_only",
                    status="promotion_eligible",
                    gate=DefenseV2GateReport(
                        arm="rules_only",
                        outcome=V2GateOutcome(passed=True),
                    ),
                ),
                V2ArmScorecard(
                    arm="gbdt_only",
                    status="promotion_eligible",
                    gate=DefenseV2GateReport(
                        arm="gbdt_only",
                        outcome=V2GateOutcome(passed=False, codes=("CONTROL_INVALID",)),
                    ),
                ),
                V2ArmScorecard(
                    arm="layered_hybrid",
                    status="promotion_eligible",
                    gate=DefenseV2GateReport(
                        arm="layered_hybrid",
                        outcome=V2GateOutcome(passed=True),
                    ),
                ),
            ),
        }
    )

    with pytest.raises(ValueError, match="failed gate"):
        render_v2_scorecard(result, signer=signer())


def signer() -> RunSigningIdentity:
    return RunSigningIdentity.from_private_bytes(b"r" * 32)
