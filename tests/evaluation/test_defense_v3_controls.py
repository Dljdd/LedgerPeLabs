"""Mandatory negative control tests for Defend v3."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_controls import V2ControlBinding
from apar.evaluation.v3_controls import (
    V3ControlError,
    run_benign_control,
    run_permutation_control,
)


def _signer() -> EvaluatorSigningIdentity:
    return EvaluatorSigningIdentity.from_private_bytes(b"\x01" * 32)


def _binding() -> V2ControlBinding:
    signer = _signer()
    return V2ControlBinding(
        schema_version="1.0.0",
        preregistration_id="apar-defend-v3",
        execution_nonce="a" * 64,
        arm="rules_only",
        candidate_id="candidate-a",
        input_digest="b" * 64,
        evaluator_key_id=signer.key_id,
        evaluator_public_key_base64=signer.public_key_base64,
    )


def _truth_row(event_id: str, *, fraud: bool) -> EvaluationTruthRow:
    now = datetime(2026, 8, 21, tzinfo=UTC)
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
    result = run_benign_control(
        actions=(Action.CHALLENGE,),
        truth=(_truth_row("row-1", fraud=False),),
        signer=_signer(),
        binding=_binding(),
    )
    assert result.valid is True
    assert result.intervention_count == 1
    assert result.true_positive_count == 0


def test_benign_control_rejects_fraud_rows() -> None:
    result = run_benign_control(
        actions=(Action.APPROVE,),
        truth=(_truth_row("row-1", fraud=True),),
        signer=_signer(),
        binding=_binding(),
    )
    assert result.valid is False
    assert result.reason == "malformed_benign_control"


def test_permutation_control_keeps_blocks_intact() -> None:
    rows = tuple(_truth_row(f"row-{index}", fraud=index % 2 == 0) for index in range(4))
    observed: list[tuple[float, ...]] = []
    result = run_permutation_control(
        scores=np.array([0.9, 0.8, 0.2, 0.1]),
        truth=rows,
        blocks=("case-a", "case-a", "case-b", "case-b"),
        seed=3,
        evaluator=lambda scores, truth, blocks: observed.append(tuple(scores)) or False,
        signer=_signer(),
        binding=_binding(),
    )
    assert result.valid is True
    assert observed == [(0.2, 0.1, 0.9, 0.8)]


def test_qualifying_permuted_scores_invalidates_run() -> None:
    rows = (_truth_row("fraud", fraud=True), _truth_row("benign", fraud=False))
    result = run_permutation_control(
        scores=np.array([1.0, 0.0]),
        truth=rows,
        blocks=("same-case", "same-case"),
        seed=7,
        evaluator=lambda scores, truth, blocks: True,
        signer=_signer(),
        binding=_binding(),
    )
    assert (result.valid, result.reason) == (False, "permuted_scores_qualified")


def test_non_callable_evaluator_rejected() -> None:
    rows = (_truth_row("row-1", fraud=False),)
    with pytest.raises(V3ControlError, match="must be callable"):
        run_permutation_control(
            scores=np.array([0.5]),
            truth=rows,
            blocks=("case",),
            seed=7,
            evaluator="not-callable",
            signer=_signer(),
            binding=_binding(),
        )
