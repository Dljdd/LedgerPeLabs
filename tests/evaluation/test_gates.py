"""Hard promotion blockers and truthful champion-selection oracles."""

from __future__ import annotations

import copy
import json
from decimal import Decimal, getcontext

import pytest

from apar.evaluation.gates import (
    AssuranceEvidence,
    ChampionStatus,
    EvaluationDescriptor,
    EvaluationKind,
    GateConfig,
    PromotionMetrics,
    ReplayResult,
    SlicePerformance,
    evaluate_promotion_gates,
)
from apar.evaluation.regimes import RegimeKind
from apar.evaluation.replay import DefenseArm
from apar.evaluation.splits import EntityCohort
from apar.runs.wire import canonical_json_bytes

FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
DECISION_IDS = tuple(f"event-{index:04d}" for index in range(1000))


def _sha(character: str) -> str:
    return character * 64


def _metrics(
    *,
    value: str,
    review_cases: int,
    family_recall: float | None = 0.80,
    slice_recall: float | None = 0.78,
    ece: float = 0.05,
    p95: float = 20.0,
) -> PromotionMetrics:
    slices = tuple(
        SlicePerformance(kind="family", value=family, recall=family_recall)
        for family in FAMILIES
    ) + (
        SlicePerformance(kind="rail", value="card", recall=slice_recall),
    )
    return PromotionMetrics(
        row_count=1000,
        recall=0.80,
        ece=ece,
        p95_latency_ms=p95,
        preventable_settled_value=Decimal(value),
        value_escaped=Decimal("20.00"),
        review_case_count=review_cases,
        challenge_rate=0.01,
        false_decline_rate=0.0005,
        review_case_rate=0.005,
        slice_performance=slices,
    )


def _descriptors() -> tuple[EvaluationDescriptor, ...]:
    return (
        EvaluationDescriptor(kind=EvaluationKind.CHRONOLOGICAL, value="development"),
        EvaluationDescriptor(kind=EvaluationKind.HIDDEN, value="hidden"),
        *(
            EvaluationDescriptor(kind=EvaluationKind.HELD_FAMILY, value=family)
            for family in FAMILIES
        ),
        *(
            EvaluationDescriptor(kind=EvaluationKind.REGIME, value=regime.value)
            for regime in RegimeKind
        ),
        *(
            EvaluationDescriptor(kind=EvaluationKind.COLD_ENTITY, value=cohort.value)
            for cohort in EntityCohort
        ),
    )


def _results(
    *,
    rule_value: str = "80.00",
    gbdt_value: str = "82.00",
    hybrid_value: str = "83.00",
    rule_cases: int = 20,
    gbdt_cases: int = 18,
    hybrid_cases: int = 17,
) -> tuple[ReplayResult, ...]:
    arm_values = {
        DefenseArm.RULES_ONLY: (rule_value, rule_cases),
        DefenseArm.GBDT_ONLY: (gbdt_value, gbdt_cases),
        DefenseArm.LAYERED_HYBRID: (hybrid_value, hybrid_cases),
    }
    rows: list[ReplayResult] = []
    for descriptor in _descriptors():
        for arm in DefenseArm:
            value, cases = arm_values[arm]
            rows.append(
                ReplayResult.create(
                    arm=arm,
                    evaluation=descriptor,
                    decision_event_ids=DECISION_IDS,
                    decision_rows_digest=_sha("1"),
                    common_integrity_digest=_sha("2"),
                    action_digest=_sha("3"),
                    score_digest=_sha("4"),
                    threshold_report_digest=_sha("5"),
                    threshold_set_digest=_sha("6"),
                    bundle_manifest_digest=_sha("7"),
                    case_callback_digest=_sha("8"),
                    metric_report_digest=_sha("9"),
                    metrics=_metrics(value=value, review_cases=cases),
                    assurance=AssuranceEvidence.passing(),
                )
            )
    return tuple(rows)


def _replace(
    results: tuple[ReplayResult, ...],
    *,
    arm: DefenseArm | None = None,
    kind: EvaluationKind | None = None,
    transform: object,
) -> tuple[ReplayResult, ...]:
    changed: list[ReplayResult] = []
    for row in results:
        if (arm is None or row.arm is arm) and (kind is None or row.evaluation.kind is kind):
            assert callable(transform)
            changed.append(transform(row))
        else:
            changed.append(row)
    return tuple(changed)


def test_hybrid_promotes_only_on_exact_value_improvement_over_both_comparators() -> None:
    decision = evaluate_promotion_gates(_results(), GateConfig.competition())

    assert decision.status is ChampionStatus.PROMOTED
    assert decision.champion is DefenseArm.LAYERED_HYBRID
    assert decision.failed_gate_codes == ()


def test_within_one_cent_requires_strictly_lower_workload() -> None:
    retained = evaluate_promotion_gates(
        _results(hybrid_value="82.00", hybrid_cases=18), GateConfig.competition()
    )
    promoted = evaluate_promotion_gates(
        _results(hybrid_value="81.99", hybrid_cases=17), GateConfig.competition()
    )

    assert retained.status is ChampionStatus.RETAINED
    assert retained.champion is DefenseArm.GBDT_ONLY
    assert promoted.status is ChampionStatus.PROMOTED
    assert promoted.champion is DefenseArm.LAYERED_HYBRID


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("leakage_passed", False, "FEATURE_LEAKAGE"),
        ("parity_passed", False, "FEATURE_PARITY"),
        ("artifact_signature_valid", False, "ARTIFACT_SIGNATURE"),
        ("rollback_available", False, "ROLLBACK_MISSING"),
        ("hidden_access_clean", False, "HIDDEN_ACCESS"),
        ("campaign_family_ownership_valid", False, "CAMPAIGN_FAMILY_OWNERSHIP"),
    ),
)
def test_each_assurance_failure_is_a_non_averageable_hard_blocker(
    field: str, value: bool, code: str
) -> None:
    def weaken(row: ReplayResult) -> ReplayResult:
        assurance = row.assurance.model_copy(update={field: value})
        return row.rebuild(assurance=assurance)

    changed = _replace(
        _results(), arm=DefenseArm.LAYERED_HYBRID, transform=weaken
    )
    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.champion is not DefenseArm.LAYERED_HYBRID
    assert code in decision.failed_gate_codes


@pytest.mark.parametrize(
    ("metrics", "code"),
    (
        (_metrics(value="83.00", review_cases=17, ece=0.1000001), "CALIBRATION_ECE"),
        (_metrics(value="83.00", review_cases=17, p95=50.0001), "P95_LATENCY"),
    ),
)
def test_numeric_hard_limits_are_inclusive_and_cannot_be_averaged(
    metrics: PromotionMetrics, code: str
) -> None:
    changed = _replace(
        _results(),
        arm=DefenseArm.LAYERED_HYBRID,
        transform=lambda row: row.rebuild(metrics=metrics),
    )
    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.champion is not DefenseArm.LAYERED_HYBRID
    assert code in decision.failed_gate_codes


def test_one_strategic_family_below_floor_vetoes_all_promotion() -> None:
    weak = _metrics(value="83.00", review_cases=17, family_recall=0.49)
    changed = _replace(
        _results(),
        kind=EvaluationKind.HELD_FAMILY,
        transform=lambda row: row.rebuild(metrics=weak),
    )

    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert "PER_FAMILY_RECALL" in decision.failed_gate_codes


def test_hybrid_slice_recall_regression_over_five_points_vetoes_hybrid() -> None:
    weak = _metrics(value="83.00", review_cases=17, slice_recall=0.729)
    changed = _replace(
        _results(),
        arm=DefenseArm.LAYERED_HYBRID,
        kind=EvaluationKind.CHRONOLOGICAL,
        transform=lambda row: row.rebuild(metrics=weak),
    )

    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.champion is DefenseArm.GBDT_ONLY
    assert "SLICE_RECALL_REGRESSION" in decision.failed_gate_codes


def test_budget_breach_and_missing_required_evaluation_are_hard_failures() -> None:
    breached = _metrics(value="83.00", review_cases=17).model_copy(
        update={"challenge_rate": 0.0200001}
    )
    changed = _replace(
        _results(),
        arm=DefenseArm.LAYERED_HYBRID,
        transform=lambda row: row.rebuild(metrics=breached),
    )
    missing_hidden = tuple(
        row for row in _results() if row.evaluation.kind is not EvaluationKind.HIDDEN
    )

    budget_decision = evaluate_promotion_gates(changed, GateConfig.competition())
    coverage_decision = evaluate_promotion_gates(missing_hidden, GateConfig.competition())

    assert "OPERATING_BUDGET" in budget_decision.failed_gate_codes
    assert coverage_decision.status is ChampionStatus.NO_PROMOTION
    assert "EVALUATION_COVERAGE" in coverage_decision.failed_gate_codes


def test_missing_chronological_view_returns_negative_result_instead_of_crashing() -> None:
    missing = tuple(
        row for row in _results() if row.evaluation.kind is not EvaluationKind.CHRONOLOGICAL
    )

    decision = evaluate_promotion_gates(missing, GateConfig.competition())

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert "EVALUATION_COVERAGE" in decision.failed_gate_codes


def test_zero_row_nonapplicable_family_slices_do_not_create_false_family_failure() -> None:
    undefined = _metrics(
        value="83.00",
        review_cases=17,
        family_recall=None,
        slice_recall=None,
    )
    changed = _replace(
        _results(),
        kind=EvaluationKind.COLD_ENTITY,
        transform=lambda row: row.rebuild(metrics=undefined),
    )

    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.status is ChampionStatus.PROMOTED
    assert "PER_FAMILY_RECALL" not in decision.failed_gate_codes
    assert "SLICE_RECALL_REGRESSION" not in decision.failed_gate_codes


def test_no_promotion_remains_a_valid_negative_result_when_all_arms_fail() -> None:
    changed = _replace(
        _results(),
        transform=lambda row: row.rebuild(
            assurance=row.assurance.model_copy(update={"leakage_passed": False})
        ),
    )
    decision = evaluate_promotion_gates(changed, GateConfig.competition())

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert decision.decision_digest


def test_gate_results_are_order_invariant_and_canonical_roundtrip_rejects_tamper() -> None:
    results = _results()
    forward = evaluate_promotion_gates(results, GateConfig.competition())
    reverse = evaluate_promotion_gates(tuple(reversed(results)), GateConfig.competition())
    loaded = type(forward).from_json(forward.to_json())

    assert reverse == forward
    assert loaded == forward
    document = json.loads(forward.to_json())
    document["champion"] = DefenseArm.RULES_ONLY.value
    with pytest.raises(ValueError):
        type(forward).from_json(canonical_json_bytes(document))


def test_decimal_champion_selection_is_independent_of_ambient_context() -> None:
    original = copy.copy(getcontext())
    try:
        baseline = evaluate_promotion_gates(_results(), GateConfig.competition()).to_json()
        getcontext().prec = 3
        getcontext().rounding = "ROUND_UP"
        changed = evaluate_promotion_gates(_results(), GateConfig.competition()).to_json()
    finally:
        getcontext().prec = original.prec
        getcontext().rounding = original.rounding

    assert changed == baseline
