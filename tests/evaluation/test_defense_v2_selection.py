"""Fixture-only constrained threshold-selection contracts for Defend v2."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.v2_controls import (
    V2ControlBinding,
    V2ControlContext,
    V2ControlError,
    run_benign_only_control,
    run_score_permutation_control,
)
from apar.evaluation.v2_preregistration import V2Preregistration, sign_v2_preregistration
from apar.evaluation.v2_protocol import V2Protocol
from apar.evaluation.v2_selection import (
    ArmThresholdCandidate,
    BootstrapMetricContribution,
    BoundedMetric,
    ControlValidity,
    V2BootstrapBlock,
    V2MetricSet,
    bootstrap_v2_metrics,
    evaluate_v2_gates,
    select_v2_thresholds,
)
from tests.evaluation.v2_authority import EphemeralV2Authority, ephemeral_v2_authority

AUTHORITY = ephemeral_v2_authority()
def test_outsider_self_signed_copy_cannot_supply_selection_trust() -> None:
    """Copied trusted fields signed by an outsider are not sealed control authority."""
    outsider, copied = copied_outsider_preregistration()
    candidate_id = "outsider-copy"
    binding = V2ControlBinding.from_preregistration(
        copied,
        arm="layered_hybrid",
        candidate_id=candidate_id,
        input_digest="b" * 64,
    )
    benign = run_benign_only_control(
        actions=(Action.APPROVE,),
        truth=(_control_truth("benign-copy", fraud=False),),
        signer=outsider.evaluator,
        binding=binding,
    )
    permutation = run_score_permutation_control(
        scores=np.array([0.8, 0.2]),
        truth=(
            _control_truth("fraud-copy", fraud=True),
            _control_truth("benign-copy-2", fraud=False),
        ),
        blocks=("case-a", "case-b"),
        seed=5,
        evaluator=lambda scores, truth, blocks: False,
        signer=outsider.evaluator,
        binding=binding,
    )
    evidence = candidate(candidate_id).model_copy(
        update={
            "control": ControlValidity.attest(
                benign_only=benign,
                score_permutation=permutation,
            )
        }
    )

    outcome = evaluate_v2_gates(
        evidence,
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context(candidate_id),
    )

    assert "CONTROL_INVALID" in outcome.codes


def test_outsider_self_signed_copy_cannot_create_trusted_context() -> None:
    """Context construction compares a presented preregistration to the sealed root."""
    _, copied = copied_outsider_preregistration()

    with pytest.raises(V2ControlError, match="sealed preregistration"):
        V2ControlContext.from_preregistration(
            copied,
            sealed_preregistration=AUTHORITY.preregistration,
            arm="layered_hybrid",
            candidate_id="outsider-context",
            input_digest="b" * 64,
        )


@pytest.mark.parametrize(
    "arm,input_digest",
    (("rules_only", "b" * 64), ("layered_hybrid", "c" * 64)),
)
def test_selection_rejects_wrong_independent_arm_or_input(
    arm: str, input_digest: str
) -> None:
    """Expected arm and input come from trusted execution context, not evidence."""
    evidence = candidate("context-mismatch")
    context = V2ControlContext.from_preregistration(
        AUTHORITY.preregistration,
        sealed_preregistration=AUTHORITY.preregistration,
        arm=arm,
        candidate_id=evidence.candidate_id,
        input_digest=input_digest,
    )

    outcome = evaluate_v2_gates(
        evidence,
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=context,
    )

    assert "CONTROL_INVALID" in outcome.codes


def test_high_stratum_review_failure_rejects_candidate() -> None:
    result = select_v2_thresholds(
        (candidate("safe"), candidate("high-review", high_review_rate=0.0101)),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_contexts={
            name: control_context(name) for name in ("safe", "high-review")
        },
    )

    assert result.selected_candidate_id == "safe"
    assert "REVIEW_CASE_BUDGET" in result.gate_outcomes["high-review"].codes


def test_upper_bound_vetoes_maximum_gate() -> None:
    outcome = evaluate_v2_gates(
        metrics_with(challenge_rate=bounded(0.019, lower=0.017, upper=0.021)), protocol()
    )

    assert outcome.passed is False
    assert "CHALLENGE_BUDGET" in outcome.codes


def test_every_family_requires_value_and_alert_bounds() -> None:
    outcome = evaluate_v2_gates(
        candidate("late-alert", family_time=bounded(299.0, lower=270.0, upper=301.0)),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("late-alert"),
    )

    assert outcome.passed is False
    assert "TIME_TO_ALERT" in outcome.codes


def test_invalid_typed_control_vetoes_an_otherwise_safe_candidate() -> None:
    outcome = evaluate_v2_gates(
        candidate("invalid-control", control_valid=False),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("invalid-control"),
    )

    assert outcome.passed is False
    assert outcome.codes == ("CONTROL_INVALID",)


def test_forged_valid_control_cannot_pass_selection() -> None:
    """A caller-constructed validity boolean is not evidence that controls ran."""
    forged = ControlValidity.model_construct(valid=True, reason=None)
    evidence = candidate("forged-control").model_copy(update={"control": forged})

    outcome = evaluate_v2_gates(
        evidence,
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("forged-control"),
    )

    assert outcome.passed is False
    assert "CONTROL_INVALID" in outcome.codes


def test_control_results_cannot_replay_between_candidates() -> None:
    """Selection must compare both controls with the candidate-specific binding."""
    first = candidate("candidate-a")
    replayed = candidate("candidate-b").model_copy(update={"control": first.control})

    outcome = evaluate_v2_gates(
        replayed,
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("candidate-b"),
    )

    assert outcome.passed is False
    assert "CONTROL_INVALID" in outcome.codes


def test_undefined_metric_fails_closed() -> None:
    outcome = evaluate_v2_gates(
        candidate("undefined", ece=BoundedMetric.undefined(numerator=0.0, denominator=0.0)),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("undefined"),
    )

    assert outcome.passed is False
    assert "METRIC_UNDEFINED" in outcome.codes


def test_zero_or_non_2000_bootstrap_replicates_fail_closed() -> None:
    zero = evaluate_v2_gates(
        candidate("zero-bootstrap", ece=bootstrap_bounded(0.05, replicates=0)),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("zero-bootstrap"),
    )
    wrong_count = evaluate_v2_gates(
        candidate("wrong-bootstrap", ece=bootstrap_bounded(0.05, replicates=1_999)),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("wrong-bootstrap"),
    )

    assert "BOOTSTRAP_REPLICATES" in zero.codes
    assert "BOOTSTRAP_REPLICATES" in wrong_count.codes


def test_partially_undefined_bootstrap_distribution_fails_closed() -> None:
    outcome = evaluate_v2_gates(
        candidate(
            "partially-undefined-bootstrap",
            ece=bootstrap_bounded(0.05, replicates=2_000, undefined_replicates=1),
        ),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_context=control_context("partially-undefined-bootstrap"),
    )

    assert outcome.passed is False
    assert "BOOTSTRAP_UNDEFINED" in outcome.codes


def test_tie_break_is_stable_and_lexicographic_after_matched_gates() -> None:
    result = select_v2_thresholds(
        (candidate("later", thresholds=(0.3, 0.8)), candidate("first", thresholds=(0.2, 0.9))),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_contexts={name: control_context(name) for name in ("later", "first")},
    )

    assert result.selected_candidate_id == "first"


def test_no_promotion_when_no_candidate_qualifies() -> None:
    result = select_v2_thresholds(
        (candidate("bad", control_valid=False),),
        protocol(),
        sealed_preregistration=AUTHORITY.preregistration,
        control_contexts={"bad": control_context("bad")},
    )

    assert result.status == "no_promotion"
    assert result.selected_candidate_id is None
    assert result.reason == "no_candidate_satisfies_v2_constraints"


def test_bootstrap_resamples_days_then_case_blocks_for_exactly_2000_replicates() -> None:
    metrics = bootstrap_v2_metrics(
        (
            bootstrap_block(date(2026, 1, 1), "campaign-a", numerator=1.0, denominator=2.0),
            bootstrap_block(date(2026, 1, 1), "campaign-b", numerator=0.0, denominator=2.0),
            bootstrap_block(date(2026, 1, 2), "campaign-c", numerator=2.0, denominator=2.0),
        ),
        seed=19,
    )

    metric = metrics["recall"]
    assert metric.point == 0.5
    assert metric.bootstrap_replicates == 2_000
    assert metric.valid_replicates == 2_000
    assert 0.0 <= metric.lower <= metric.upper <= 1.0


def protocol() -> V2Protocol:
    return V2Protocol.fixture(transaction_count=100)


def control_context(
    candidate_id: str,
    *,
    arm: str = "layered_hybrid",
    input_digest: str = "b" * 64,
) -> V2ControlContext:
    return V2ControlContext.from_preregistration(
        AUTHORITY.preregistration,
        sealed_preregistration=AUTHORITY.preregistration,
        arm=arm,
        candidate_id=candidate_id,
        input_digest=input_digest,
    )


def copied_outsider_preregistration() -> tuple[EphemeralV2Authority, V2Preregistration]:
    outsider = ephemeral_v2_authority()
    payload = AUTHORITY.preregistration.model_dump(
        mode="json",
        exclude={"evaluator_key_id", "evaluator_public_key_base64", "signature_base64"},
    )
    return outsider, sign_v2_preregistration(payload, signer=outsider.evaluator)


def bounded(
    point: float, *, lower: float | None = None, upper: float | None = None
) -> BoundedMetric:
    return BoundedMetric(
        point=point,
        lower=point if lower is None else lower,
        upper=point if upper is None else upper,
        numerator=point * 100.0,
        denominator=100.0,
        bootstrap_replicates=2_000,
        valid_replicates=2_000,
    )


def bootstrap_bounded(
    point: float, *, replicates: int, undefined_replicates: int = 0
) -> BoundedMetric:
    return BoundedMetric(
        point=point,
        lower=point,
        upper=point,
        numerator=point * 100.0,
        denominator=100.0,
        bootstrap_replicates=replicates,
        valid_replicates=replicates - undefined_replicates,
        undefined_replicates=undefined_replicates,
    )


def metrics_with(**updates: BoundedMetric) -> V2MetricSet:
    values = {
        "precision": bounded(0.8),
        "recall": bounded(0.7, lower=0.6),
        "f1": bounded(0.75),
        "pr_auc": bounded(0.8),
        "roc_auc": bounded(0.8),
        "ece": bounded(0.05, upper=0.08),
        "fpr": bounded(0.01),
        "challenge_rate": bounded(0.01, upper=0.015),
        "false_decline_rate": bounded(0.0005, upper=0.0008),
        "review_case_rate": bounded(0.005, upper=0.008),
        "false_interventions_per_10k": bounded(5.0),
        "preventable_settled_value_fraction": bounded(0.7, lower=0.6),
        "escaped_value_fraction": bounded(0.3, upper=0.4),
        "time_to_alert_p95_seconds": bounded(200.0, upper=250.0),
        "p95_decision_latency_ms": bounded(20.0, upper=30.0),
    }
    values.update(updates)
    return V2MetricSet(**values)


def candidate(
    candidate_id: str,
    *,
    thresholds: tuple[float, ...] = (0.2, 0.8),
    high_review_rate: float | None = None,
    family_time: BoundedMetric | None = None,
    control_valid: bool = True,
    ece: BoundedMetric | None = None,
) -> ArmThresholdCandidate:
    metrics = metrics_with(**({"ece": ece} if ece is not None else {}))
    strata = {name: metrics for name in ("low", "medium", "high")}
    if high_review_rate is not None:
        strata["high"] = metrics.model_copy(
            update={"review_case_rate": bounded(high_review_rate, upper=high_review_rate)}
        )
    families = {name: metrics for name in protocol().operating.family_names}
    if family_time is not None:
        families["c"] = metrics.model_copy(update={"time_to_alert_p95_seconds": family_time})
    return ArmThresholdCandidate(
        candidate_id=candidate_id,
        threshold_tuple=thresholds,
        metrics=metrics,
        strata=strata,
        families=families,
        control=control_validity(valid=control_valid, candidate_id=candidate_id),
    )


def control_validity(*, valid: bool, candidate_id: str) -> ControlValidity:
    signer = AUTHORITY.evaluator
    binding = V2ControlBinding.from_preregistration(
        AUTHORITY.preregistration,
        arm="layered_hybrid",
        candidate_id=candidate_id,
        input_digest="b" * 64,
    )
    benign = run_benign_only_control(
        actions=(Action.APPROVE,),
        truth=(_control_truth("benign", fraud=False),),
        signer=signer,
        binding=binding,
    )
    permutation = run_score_permutation_control(
        scores=np.array([0.8, 0.2]),
        truth=(
            _control_truth("fraud", fraud=True),
            _control_truth("permuted-benign", fraud=False),
        ),
        blocks=("case", "case"),
        seed=7,
        evaluator=lambda scores, truth, blocks: False,
        signer=signer,
        binding=binding,
    )
    if not valid:
        benign = run_benign_only_control(
            actions=(Action.APPROVE,),
            truth=(_control_truth("fraud-control", fraud=True),),
            signer=signer,
            binding=binding,
        )
    return ControlValidity.attest(
        benign_only=benign,
        score_permutation=permutation,
    )


def _control_truth(event_id: str, *, fraud: bool) -> EvaluationTruthRow:
    timestamp = datetime(2026, 8, 20, tzinfo=UTC)
    return EvaluationTruthRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id=f"campaign-{event_id}",
        family="card_testing_cnp",
        viewpoint="development",
        is_fraud=fraud,
        label_source="population_truth",
        label_mature_at=timestamp,
        first_settlement_at=None,
        net_settled_value=Decimal("0"),
        lifecycle_event_ids=(event_id,),
    )


def bootstrap_block(
    day: date, block_id: str, *, numerator: float, denominator: float
) -> V2BootstrapBlock:
    return V2BootstrapBlock(
        day=day,
        block_id=block_id,
        metrics={
            "recall": BootstrapMetricContribution(
                kind="ratio", numerator=numerator, denominator=denominator
            )
        },
    )
