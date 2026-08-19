"""Hard promotion blockers and truthful champion-selection oracles."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from decimal import Decimal, getcontext

import pytest

from apar.contracts.events import Rail
from apar.evaluation.gates import (
    AssuranceEvidence,
    ChampionStatus,
    EvaluationDescriptor,
    EvaluationKind,
    EvaluationLineage,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    GateConfig,
    GateContractError,
    HiddenPublicProof,
    PromotionMetrics,
    RateEvidence,
    ReplayResult,
    SlicePerformance,
    VerifiedPromotionEnvelope,
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
EVALUATOR_SIGNER = EvaluatorSigningIdentity.from_private_bytes(b"e" * 32)
EVALUATOR_VERIFIER = EvaluatorReplayVerifier.from_signer(EVALUATOR_SIGNER)


def _sha(character: str) -> str:
    return character * 64


def _row_digest(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(ids))).hexdigest()


def _lineage(descriptor: EvaluationDescriptor) -> EvaluationLineage:
    regime = descriptor.kind is EvaluationKind.REGIME
    held = descriptor.kind is EvaluationKind.HELD_FAMILY
    return EvaluationLineage.create(
        descriptor=descriptor,
        decision_rows_digest=_row_digest(DECISION_IDS),
        decision_content_digest=_sha("1"),
        split_digest=_sha("2"),
        cohort_mapping_digest=_sha("3"),
        training_population_digest=_sha("4"),
        bundle_manifest_digest=_sha("7"),
        defender_top_ref_digest=_sha("7"),
        regime_parent_digest=_sha("c") if regime else None,
        regime_output_digest=_sha("d") if regime else None,
        regime_parameters_digest=_sha("e") if regime else None,
        regime_truth_unchanged=True if regime else None,
        held_family=descriptor.value if held else None,
        training_exclusion_verified=held,
    )


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
    ) + tuple(
        SlicePerformance(kind="rail", value=rail.value, recall=slice_recall)
        for rail in Rail
    ) + tuple(
        SlicePerformance(kind="regime", value=regime, recall=slice_recall)
        for regime in ("baseline", *(item.value for item in RegimeKind))
    ) + tuple(
        SlicePerformance(kind="entity_cohort", value=cohort.value, recall=slice_recall)
        for cohort in EntityCohort
    )
    slices = tuple(sorted(slices, key=lambda item: (item.kind, item.value)))
    return PromotionMetrics(
        row_count=1000,
        recall=0.80,
        ece=ece,
        p95_latency_ms=p95,
        preventable_settled_value=Decimal(value),
        value_escaped=Decimal("20.00"),
        review_case_count=review_cases,
        challenge_rate=0.01,
        false_decline=RateEvidence(
            numerator=0,
            denominator=1_000,
            value=0.0,
            defined=True,
        ),
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
                    evaluation_lineage=_lineage(descriptor),
                    decision_event_ids=DECISION_IDS,
                    decision_rows_digest=_row_digest(DECISION_IDS),
                    common_integrity_digest=_sha("2"),
                    action_digest=_sha("3"),
                    score_digest=_sha("4"),
                    threshold_report_digest=_sha("5"),
                    threshold_set_digest=_sha("6"),
                    bundle_manifest_digest=_sha("7"),
                    case_callback_digest=_sha("8"),
                    evaluation_context_digest=_sha("a"),
                    hidden_public_proof_id=(
                        "hpf_" + "b" * 32
                        if descriptor.kind is EvaluationKind.HIDDEN
                        else None
                    ),
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


def _promotion_envelope(
    results: tuple[ReplayResult, ...],
    *,
    signer: EvaluatorSigningIdentity = EVALUATOR_SIGNER,
) -> VerifiedPromotionEnvelope:
    descriptor_keys = sorted(
        {(row.evaluation.kind, row.evaluation.value) for row in results},
        key=lambda item: (tuple(EvaluationKind).index(item[0]), item[1]),
    )
    components = tuple(
        signer.sign_batch(
            tuple(
                row
                for row in results
                if (row.evaluation.kind, row.evaluation.value) == key
            )
        )
        for key in descriptor_keys
    )
    hidden_batches = tuple(
        batch
        for batch in components
        if batch.results[0].evaluation.kind is EvaluationKind.HIDDEN
    )
    proofs: tuple[HiddenPublicProof, ...] = ()
    if hidden_batches:
        hidden_batch = hidden_batches[0]
        hidden_row = hidden_batch.results[0]
        proofs = (
            HiddenPublicProof.create(
                proof_id="hpf_" + "b" * 32,
                replay_batch_digest=hidden_batch.batch_digest,
                decision_bindings_digest=_sha("9"),
                bundle_manifest_digest=hidden_row.bundle_manifest_digest,
                defender_top_ref_digest=(
                    hidden_row.evaluation_lineage.defender_top_ref_digest
                ),
                worker_manifest_digest=_sha("c"),
                evaluator_context_token=hidden_row.evaluation_context_digest,
                cohort_mapping_token=(
                    hidden_row.evaluation_lineage.cohort_mapping_digest
                ),
                issued_at="2026-08-19T00:00:00Z",
                signer=signer,
            ),
        )
    envelope = VerifiedPromotionEnvelope.create(
        component_batches=components,
        hidden_proofs=proofs,
        signer=signer,
    )
    return envelope


def _evaluate(results: tuple[ReplayResult, ...]):
    return evaluate_promotion_gates(
        _promotion_envelope(results),
        GateConfig.competition(),
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=EVALUATOR_VERIFIER,
    )


def _lineage_update(
    lineage: EvaluationLineage, **updates: object
) -> EvaluationLineage:
    fields = lineage.model_dump(mode="python", exclude={"lineage_digest"})
    fields["descriptor"] = lineage.descriptor
    fields.update(updates)
    return EvaluationLineage.create(**fields)


def test_hybrid_promotes_only_on_exact_value_improvement_over_both_comparators() -> None:
    decision = _evaluate(_results())

    assert decision.status is ChampionStatus.PROMOTED
    assert decision.champion is DefenseArm.LAYERED_HYBRID
    assert decision.failed_gate_codes == ()


def test_gate_rejects_raw_caller_authored_replay_matrix() -> None:
    """Self-digested public results are not trusted promotion provenance."""
    with pytest.raises(GateContractError, match="verified|signed|evaluator"):
        evaluate_promotion_gates(_results(), GateConfig.competition())


def test_evaluator_signer_and_verifier_are_immutable_and_pinned() -> None:
    signer = EvaluatorSigningIdentity.from_private_bytes(b"x" * 32)
    verifier = EvaluatorReplayVerifier.from_signer(signer)
    original_key_id = verifier.key_id
    replacement = EvaluatorSigningIdentity.from_private_bytes(b"y" * 32)

    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(signer)
        with pytest.raises(TypeError):
            operation(verifier)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(signer, "key_id", replacement.key_id)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(verifier, "key_id", replacement.key_id)
    with pytest.raises(TypeError, match="reinitialized"):
        type(signer).__init__(signer)
    with pytest.raises(TypeError, match="reinitialized"):
        type(verifier).__init__(
            verifier,
            signer_key_id=replacement.key_id,
            public_key_base64=replacement.public_key_base64,
        )

    assert verifier.key_id == original_key_id
    replacement_envelope = _promotion_envelope(_results(), signer=replacement)
    with pytest.raises(GateContractError, match="verified|signature"):
        evaluate_promotion_gates(
            replacement_envelope,
            GateConfig.competition(),
            evaluator_verifier=verifier,
            hidden_proof_verifier=verifier,
        )


def test_promotion_envelope_rejects_arbitrary_signed_hidden_proof_identity() -> None:
    envelope = _promotion_envelope(_results())
    hidden_batch = next(
        item
        for item in envelope.component_batches
        if item.results[0].evaluation.kind is EvaluationKind.HIDDEN
    )
    row = hidden_batch.results[0]
    substituted = HiddenPublicProof.create(
        proof_id="hpf_" + "c" * 32,
        replay_batch_digest=hidden_batch.batch_digest,
        decision_bindings_digest=_sha("9"),
        bundle_manifest_digest=row.bundle_manifest_digest,
        defender_top_ref_digest=row.evaluation_lineage.defender_top_ref_digest,
        worker_manifest_digest=_sha("c"),
        evaluator_context_token=row.evaluation_context_digest,
        cohort_mapping_token=row.evaluation_lineage.cohort_mapping_digest,
        issued_at="2026-08-19T00:00:00Z",
        signer=EVALUATOR_SIGNER,
    )

    with pytest.raises(GateContractError, match="hidden proof"):
        VerifiedPromotionEnvelope.create(
            component_batches=envelope.component_batches,
            hidden_proofs=(substituted,),
            signer=EVALUATOR_SIGNER,
        )


def test_promotion_envelope_roundtrip_and_constructor_tamper_fail_closed() -> None:
    envelope = _promotion_envelope(_results())

    assert VerifiedPromotionEnvelope.from_json(envelope.to_json()) == envelope
    document = json.loads(envelope.to_json())
    document["hidden_proofs"][0]["proof_id"] = "hpf_" + "c" * 32
    with pytest.raises(GateContractError, match="digest|proof"):
        VerifiedPromotionEnvelope.from_json(canonical_json_bytes(document))
    forged = VerifiedPromotionEnvelope.model_construct(
        **envelope.model_dump(mode="python", exclude={"envelope_digest"}),
        envelope_digest=_sha("0"),
    )
    with pytest.raises((GateContractError, ValueError), match="digest|verified"):
        forged.to_json()


def test_within_one_cent_requires_strictly_lower_workload() -> None:
    retained = _evaluate(_results(hybrid_value="82.00", hybrid_cases=18))
    promoted = _evaluate(_results(hybrid_value="81.99", hybrid_cases=17))

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
        updates: dict[str, object] = {"assurance": assurance}
        if field == "hidden_access_clean" and row.evaluation.kind is EvaluationKind.HIDDEN:
            updates["hidden_public_proof_id"] = None
        return row.rebuild(**updates)

    changed = _replace(
        _results(), arm=DefenseArm.LAYERED_HYBRID, transform=weaken
    )
    if field == "hidden_access_clean":
        with pytest.raises(GateContractError, match="hidden proof"):
            _evaluate(changed)
    else:
        decision = _evaluate(changed)
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
    decision = _evaluate(changed)

    assert decision.champion is not DefenseArm.LAYERED_HYBRID
    assert code in decision.failed_gate_codes


def test_one_strategic_family_below_floor_vetoes_all_promotion() -> None:
    weak = _metrics(value="83.00", review_cases=17, family_recall=0.49)
    changed = _replace(
        _results(),
        kind=EvaluationKind.HELD_FAMILY,
        transform=lambda row: row.rebuild(metrics=weak),
    )

    decision = _evaluate(changed)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert "PER_FAMILY_RECALL" in decision.failed_gate_codes


def test_strategic_family_floor_applies_inside_each_regime() -> None:
    weak = _metrics(value="83.00", review_cases=17, family_recall=0.49)
    changed = _replace(
        _results(),
        arm=DefenseArm.LAYERED_HYBRID,
        kind=EvaluationKind.REGIME,
        transform=lambda row: row.rebuild(metrics=weak),
    )

    decision = _evaluate(changed)

    assert decision.champion is DefenseArm.GBDT_ONLY
    assert "PER_FAMILY_RECALL" in decision.failed_gate_codes


def test_hybrid_slice_recall_regression_over_five_points_vetoes_hybrid() -> None:
    weak = _metrics(value="83.00", review_cases=17, slice_recall=0.729)
    changed = _replace(
        _results(),
        arm=DefenseArm.LAYERED_HYBRID,
        kind=EvaluationKind.CHRONOLOGICAL,
        transform=lambda row: row.rebuild(metrics=weak),
    )

    decision = _evaluate(changed)

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

    budget_decision = _evaluate(changed)
    coverage_decision = _evaluate(missing_hidden)

    assert "OPERATING_BUDGET" in budget_decision.failed_gate_codes
    assert coverage_decision.status is ChampionStatus.NO_PROMOTION
    assert "EVALUATION_COVERAGE" in coverage_decision.failed_gate_codes


def test_missing_chronological_view_returns_negative_result_instead_of_crashing() -> None:
    missing = tuple(
        row for row in _results() if row.evaluation.kind is not EvaluationKind.CHRONOLOGICAL
    )

    decision = _evaluate(missing)

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

    decision = _evaluate(changed)

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
    decision = _evaluate(changed)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert decision.decision_digest


def test_gate_results_are_order_invariant_and_canonical_roundtrip_rejects_tamper() -> None:
    results = _results()
    forward = _evaluate(results)
    reverse = _evaluate(tuple(reversed(results)))
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
        baseline = _evaluate(_results()).to_json()
        getcontext().prec = 3
        getcontext().rounding = "ROUND_UP"
        changed = _evaluate(_results()).to_json()
    finally:
        getcontext().prec = original.prec
        getcontext().rounding = original.rounding

    assert changed == baseline


def test_gate_rejects_reversed_decision_rows_in_one_arm() -> None:
    changed = _replace(
        _results(),
        arm=DefenseArm.GBDT_ONLY,
        kind=EvaluationKind.CHRONOLOGICAL,
        transform=lambda row: row.rebuild(
            decision_event_ids=tuple(reversed(row.decision_event_ids)),
            decision_rows_digest=_row_digest(
                tuple(reversed(row.decision_event_ids))
            ),
            evaluation_lineage=_lineage_update(
                row.evaluation_lineage,
                decision_rows_digest=_row_digest(
                    tuple(reversed(row.decision_event_ids))
                ),
            ),
        ),
    )

    decision = _evaluate(changed)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert "EVALUATION_LINEAGE" in decision.failed_gate_codes


@pytest.mark.parametrize(
    "updates",
    (
        {"common_integrity_digest": _sha("c")},
        {"bundle_manifest_digest": _sha("d")},
        {"threshold_set_digest": _sha("e")},
        {"case_callback_digest": _sha("f")},
        {"evaluation_context_digest": _sha("0")},
        {"threshold_report_digest": _sha("a")},
    ),
)
def test_gate_rejects_each_cross_arm_lineage_substitution(
    updates: dict[str, object],
) -> None:
    def substitute(row: ReplayResult) -> ReplayResult:
        if "bundle_manifest_digest" not in updates:
            return row.rebuild(**updates)
        digest = str(updates["bundle_manifest_digest"])
        return row.rebuild(
            **updates,
            evaluation_lineage=_lineage_update(
                row.evaluation_lineage,
                bundle_manifest_digest=digest,
                defender_top_ref_digest=digest,
            ),
        )

    changed = _replace(
        _results(),
        arm=DefenseArm.GBDT_ONLY,
        kind=EvaluationKind.CHRONOLOGICAL,
        transform=substitute,
    )

    decision = _evaluate(changed)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert "EVALUATION_LINEAGE" in decision.failed_gate_codes


def test_gate_rejects_hidden_public_proof_substitution_in_one_arm() -> None:
    changed = _replace(
        _results(),
        arm=DefenseArm.GBDT_ONLY,
        kind=EvaluationKind.HIDDEN,
        transform=lambda row: row.rebuild(
            hidden_public_proof_id="hpf_" + "c" * 32
        ),
    )

    with pytest.raises(GateContractError, match="hidden proof"):
        _evaluate(changed)


def test_promotion_metrics_reject_omitted_closed_slice_vocabulary() -> None:
    metrics = _metrics(value="80.00", review_cases=20)
    incomplete = tuple(
        item
        for item in metrics.slice_performance
        if not (item.kind == "rail" and item.value == Rail.AGENTIC.value)
    )

    with pytest.raises(ValueError, match="slice vocabulary"):
        PromotionMetrics.model_validate(
            {**metrics.model_dump(mode="python"), "slice_performance": incomplete}
        )


def test_missing_one_comparator_arm_for_one_descriptor_is_global_no_promotion() -> None:
    """Champion logic must never continue on a partial arm-by-descriptor matrix."""
    incomplete = tuple(
        row
        for row in _results()
        if not (
            row.arm is DefenseArm.GBDT_ONLY
            and row.evaluation.kind is EvaluationKind.REGIME
            and row.evaluation.value is RegimeKind.COLD_ID_REMAP.value
        )
    )

    decision = _evaluate(incomplete)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert decision.champion is None
    assert "EVALUATION_COVERAGE" in decision.failed_gate_codes


def test_undefined_overall_false_decline_rate_is_explicit_coverage_failure() -> None:
    """A zero legitimate denominator must not be serialized as a measured zero rate."""
    baseline = _metrics(value="83.00", review_cases=17)
    document = baseline.model_dump(mode="python")
    document["false_decline"] = RateEvidence(
        numerator=0,
        denominator=0,
        value=None,
        defined=False,
    )
    undefined = PromotionMetrics.model_validate(document)
    changed = _replace(
        _results(),
        transform=lambda row: row.rebuild(metrics=undefined),
    )

    decision = _evaluate(changed)

    assert decision.status is ChampionStatus.NO_PROMOTION
    assert "FALSE_DECLINE_COVERAGE" in decision.failed_gate_codes
