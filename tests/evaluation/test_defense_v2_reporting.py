"""Public, non-executing Defend v2 scorecard contracts."""

from __future__ import annotations

import pytest

from apar.evaluation.v2_reporting import (
    DefenseV2GateReport,
    DefenseV2Scorecard,
    V2ArmScorecard,
    not_executed_result,
    render_v2_scorecard,
)
from apar.evaluation.v2_selection import V2GateOutcome
from apar.runs import RunSigningIdentity


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
