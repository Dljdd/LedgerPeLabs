"""Fixtures and contract tests for mandatory Defend v2 negative controls."""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.v2_controls import (
    run_benign_only_control,
    run_score_permutation_control,
)


def truth_row(event_id: str, *, fraud: bool) -> EvaluationTruthRow:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    return EvaluationTruthRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id=f"campaign-{event_id}",
        family="card_testing_cnp",
        viewpoint="development",
        is_fraud=fraud,
        label_source="population_truth",
        label_mature_at=now,
        first_settlement_at=None,
        net_settled_value=Decimal("0"),
        lifecycle_event_ids=(event_id,),
    )


def test_benign_control_reports_interventions_without_true_positives() -> None:
    result = run_benign_only_control(
        actions=(Action.CHALLENGE,), truth=(truth_row("row-1", fraud=False),)
    )
    assert result.valid is True
    assert result.intervention_count == 1
    assert result.true_positive_count == 0


def test_qualifying_permuted_scores_invalidates_run() -> None:
    rows = (truth_row("fraud", fraud=True), truth_row("benign", fraud=False))
    result = run_score_permutation_control(
        scores=np.array([1.0, 0.0]), truth=rows, blocks=("same-case", "same-case"), seed=7
    )
    assert (result.valid, result.reason) == (False, "permuted_scores_qualified")


def test_score_permutation_keeps_blocks_intact() -> None:
    rows = tuple(truth_row(f"row-{index}", fraud=index % 2 == 0) for index in range(4))
    result = run_score_permutation_control(
        scores=np.array([0.9, 0.8, 0.2, 0.1]),
        truth=rows,
        blocks=("case-a", "case-a", "case-b", "case-b"),
        seed=7,
    )
    assert result.valid is True


def test_malformed_control_invalidates_the_whole_run() -> None:
    result = run_score_permutation_control(
        scores=np.array([0.5]), truth=(truth_row("row-1", fraud=False),), blocks=(), seed=7
    )
    assert (result.valid, result.reason) == (False, "malformed_permutation_control")
