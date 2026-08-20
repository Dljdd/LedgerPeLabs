"""Fixtures and contract tests for mandatory Defend v2 negative controls."""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_controls import (
    ControlResult,
    V2ControlBinding,
    V2ControlContext,
    V2ControlError,
    admit_control_result,
    run_benign_only_control,
    run_score_permutation_control,
)
from tests.evaluation.v2_authority import ephemeral_v2_authority

AUTHORITY = ephemeral_v2_authority()
CONTROL_BINDING = V2ControlBinding.from_preregistration(
    AUTHORITY.preregistration,
    arm="rules_only",
    candidate_id="candidate-a",
    input_digest="a" * 64,
)
CONTROL_CONTEXT = V2ControlContext.from_preregistration(
    AUTHORITY.preregistration,
    verified_authority=AUTHORITY.verified_authority,
    arm="rules_only",
    candidate_id="candidate-a",
    input_digest="a" * 64,
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
        actions=(Action.CHALLENGE,),
        truth=(truth_row("row-1", fraud=False),),
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert result.valid is True
    assert result.intervention_count == 1
    assert result.true_positive_count == 0


def test_qualifying_permuted_scores_invalidates_run() -> None:
    rows = (truth_row("fraud", fraud=True), truth_row("benign", fraud=False))
    result = run_score_permutation_control(
        scores=np.array([1.0, 0.0]),
        truth=rows,
        blocks=("same-case", "same-case"),
        seed=7,
        evaluator=lambda scores, truth, blocks: True,
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert (result.valid, result.reason) == (False, "permuted_scores_qualified")


def test_score_permutation_keeps_blocks_intact() -> None:
    rows = tuple(truth_row(f"row-{index}", fraud=index % 2 == 0) for index in range(4))
    observed: list[tuple[float, ...]] = []
    result = run_score_permutation_control(
        scores=np.array([0.9, 0.8, 0.2, 0.1]),
        truth=rows,
        blocks=("case-a", "case-a", "case-b", "case-b"),
        seed=3,
        evaluator=lambda scores, truth, blocks: observed.append(tuple(scores)) or False,
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert result.valid is True
    assert observed == [(0.2, 0.1, 0.9, 0.8)]


def test_invalid_control_is_a_typed_no_promotion_admission() -> None:
    control = run_benign_only_control(
        actions=(Action.CHALLENGE,),
        truth=(truth_row("row-1", fraud=True),),
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    admission = admit_control_result(
        control,
        verified_authority=AUTHORITY.verified_authority,
        expected_context=CONTROL_CONTEXT,
    )
    assert (admission.valid, admission.status, admission.reason) == (
        False,
        "no_promotion",
        "malformed_benign_control",
    )


def test_malformed_control_invalidates_the_whole_run() -> None:
    result = run_score_permutation_control(
        scores=np.array([0.5]),
        truth=(truth_row("row-1", fraud=False),),
        blocks=(),
        seed=7,
        evaluator=lambda scores, truth, blocks: False,
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert (result.valid, result.reason) == (False, "malformed_permutation_control")


def test_missing_evaluator_is_invalid() -> None:
    result = run_score_permutation_control(
        scores=np.array([0.5]),
        truth=(truth_row("row-1", fraud=False),),
        blocks=("case",),
        seed=7,
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert (result.valid, result.reason) == (False, "evaluator_missing")


def test_evaluator_exception_is_fail_closed() -> None:
    def explode(scores, truth, blocks):
        raise RuntimeError("gate unavailable")

    result = run_score_permutation_control(
        scores=np.array([0.5]),
        truth=(truth_row("row-1", fraud=False),),
        blocks=("case",),
        seed=7,
        evaluator=explode,
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert (result.valid, result.reason) == (False, "evaluator_failed")


def test_forged_valid_control_cannot_be_admitted() -> None:
    forged = ControlResult.model_construct(valid=True, kind="benign_only", reason=None)
    admission = admit_control_result(
        forged,
        verified_authority=AUTHORITY.verified_authority,
        expected_context=CONTROL_CONTEXT,
    )
    assert (admission.valid, admission.status) == (False, "no_promotion")


def test_benign_action_property_exception_is_fail_closed() -> None:
    class BrokenAction:
        @property
        def action(self):
            raise RuntimeError("action unavailable")

    result = run_benign_only_control(
        actions=(BrokenAction(),),
        truth=(truth_row("row-1", fraud=False),),
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    assert (result.valid, result.reason) == (False, "malformed_benign_control")


def test_invalid_attestation_cannot_be_model_copied_to_valid() -> None:
    """Changing signed invalid evidence cannot create a valid control admission."""
    invalid = run_benign_only_control(
        actions=(Action.APPROVE,),
        truth=(truth_row("fraud", fraud=True),),
        signer=evaluator_signer(),
        binding=CONTROL_BINDING,
    )
    tampered = invalid.model_copy(update={"valid": True, "reason": None})

    admission = admit_control_result(
        tampered,
        verified_authority=AUTHORITY.verified_authority,
        expected_context=CONTROL_CONTEXT,
    )

    assert (admission.valid, admission.status, admission.reason) == (
        False,
        "no_promotion",
        "invalid_control_attestation",
    )


def test_external_signer_cannot_mint_control_attestation() -> None:
    """A self-declared fresh key is not the committed evaluator authority."""
    outsider = ephemeral_v2_authority().evaluator

    with pytest.raises(V2ControlError, match="bound evaluator"):
        run_benign_only_control(
            actions=(Action.APPROVE,),
            truth=(truth_row("benign", fraud=False),),
            signer=outsider,
            binding=CONTROL_BINDING,
        )


def evaluator_signer() -> EvaluatorSigningIdentity:
    return AUTHORITY.evaluator


def test_control_attestation_cannot_replay_across_execution_bindings() -> None:
    """Valid evidence cannot cross candidate, arm, input, preregistration, or nonce."""
    result = run_benign_only_control(
        actions=(Action.APPROVE,),
        truth=(truth_row("benign", fraud=False),),
        signer=AUTHORITY.evaluator,
        binding=CONTROL_BINDING,
    )
    other_authority = ephemeral_v2_authority()
    alternatives = (
        CONTROL_CONTEXT.model_copy(update={"candidate_id": "candidate-b"}),
        CONTROL_CONTEXT.model_copy(update={"arm": "gbdt_only"}),
        CONTROL_CONTEXT.model_copy(update={"input_digest": "b" * 64}),
        V2ControlContext.from_preregistration(
            other_authority.preregistration,
            verified_authority=other_authority.verified_authority,
            arm="rules_only",
            candidate_id="candidate-a",
            input_digest="a" * 64,
        ),
    )

    admissions = tuple(
        admit_control_result(
            result,
            verified_authority=AUTHORITY.verified_authority,
            expected_context=alternative,
        )
        for alternative in alternatives
    )

    assert tuple((item.valid, item.reason) for item in admissions) == (
        (False, "control_binding_mismatch"),
    ) * len(alternatives)
