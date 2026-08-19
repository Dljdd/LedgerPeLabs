"""Identical-row three-arm replay and lineage integration tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.defense.contracts import PolicyThresholds
from apar.defense.policy import ActionPolicy
from apar.defense.rules import DefenseReason, RuleEngine
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluationDescriptor, EvaluationKind
from apar.evaluation.metrics import LatencySample, SliceAssignment, SliceManifest
from apar.evaluation.replay import (
    DefenseArm,
    ModelFailure,
    ReplayCaseCounterBinding,
    ReplayContractError,
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
    ReplayThresholdSet,
    bind_replay_case_counter,
    replay_defense_arms,
)
from apar.evaluation.splits import EntityCohort
from apar.runs.wire import canonical_json_bytes
from tests.defense.test_bundle import BundleFixture

pytest_plugins = ("tests.defense.test_bundle",)

AS_OF = datetime(2026, 9, 30, tzinfo=UTC)
FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


def _truth(fixture: BundleFixture, *, flip: bool = False) -> tuple[EvaluationTruthRow, ...]:
    rows = []
    for index, event in enumerate(fixture.reload_matrix.events):
        rows.append(
            EvaluationTruthRow(
                event_id=event.event_id,
                payment_id=event.payment_id,
                campaign_id=f"replay-campaign-{index}",
                family=FAMILIES[index],
                viewpoint="development",
                is_fraud=(index % 2 == 0) ^ flip,
                label_source="population_truth",
                label_mature_at=event.decision_at or event.event_time,
                first_settlement_at=None,
                net_settled_value=Decimal("0.00"),
                lifecycle_event_ids=(event.event_id,),
            )
        )
    return tuple(rows)


def _latencies(fixture: BundleFixture) -> tuple[ReplayLatencySamples, ...]:
    return tuple(
        ReplayLatencySamples(
            arm=arm,
            samples=tuple(
                LatencySample(
                    event_id=event.event_id,
                    feature_ms=1.0,
                    rules_ms=1.0,
                    model_ms=0.0 if arm is DefenseArm.RULES_ONLY else 2.0,
                    calibration_policy_ms=1.0,
                    end_to_end_ms=3.0 if arm is DefenseArm.RULES_ONLY else 5.0,
                )
                for event in fixture.reload_matrix.events
            ),
        )
        for arm in DefenseArm
    )


def _context(fixture: BundleFixture, *, flip_truth: bool = False) -> ReplayEvaluationContext:
    return ReplayEvaluationContext(
        evaluation=EvaluationDescriptor(
            kind=EvaluationKind.CHRONOLOGICAL, value="development"
        ),
        truth=_truth(fixture, flip=flip_truth),
        observations=fixture.reload_matrix.events,
        as_of=AS_OF,
        slice_assignments=tuple(
            SliceAssignment(
                event_id=event.event_id,
                regime="baseline",
                entity_cohorts=(EntityCohort.COLD_ACTOR,),
            )
            for event in fixture.reload_matrix.events
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=_latencies(fixture),
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=True,
            parity_passed=True,
            leakage_evidence_digest="a" * 64,
            parity_evidence_digest="b" * 64,
        ),
    )


def _loaded(fixture: BundleFixture):
    manifest, ref = fixture.publisher.freeze(**fixture.kwargs)
    return fixture.publisher.load(ref), manifest


def _replay(fixture: BundleFixture, *, flip_truth: bool = False, model_failure=None):
    loaded, _ = _loaded(fixture)
    case_binding = bind_replay_case_counter(
        fixture.reload_matrix.events,
        tuple(row.event_id for row in fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = ReplayThresholdSet.from_reports(
        loaded,
        case_binding,
        {arm: loaded.threshold_report for arm in DefenseArm},
    )
    return replay_defense_arms(
        matrix=fixture.reload_matrix,
        defender=loaded,
        thresholds=thresholds,
        case_counter=case_binding,
        evaluation=_context(fixture, flip_truth=flip_truth),
        model_failure=model_failure,
    )


def test_all_arms_score_the_exact_same_ordered_decision_rows(
    bundle_fixture: BundleFixture,
) -> None:
    results = _replay(bundle_fixture)

    assert {result.arm for result in results} == set(DefenseArm)
    assert len({result.decision_event_ids for result in results}) == 1
    assert len({result.decision_rows_digest for result in results}) == 1
    assert len({result.common_integrity_digest for result in results}) == 1
    assert len({result.case_callback_digest for result in results}) == 1


def test_truth_changes_metrics_but_cannot_change_scores_or_actions(
    bundle_fixture: BundleFixture,
) -> None:
    before = _replay(bundle_fixture)
    after = _replay(bundle_fixture, flip_truth=True)

    assert tuple(row.score_digest for row in before) == tuple(
        row.score_digest for row in after
    )
    assert tuple(row.action_digest for row in before) == tuple(
        row.action_digest for row in after
    )
    assert tuple(row.metric_report_digest for row in before) != tuple(
        row.metric_report_digest for row in after
    )


def test_model_error_fails_gbdt_only_and_uses_declared_rules_fallback_only_for_hybrid(
    bundle_fixture: BundleFixture,
) -> None:
    results = _replay(
        bundle_fixture,
        model_failure=ModelFailure(
            reason=DefenseReason.MODEL_UNAVAILABLE,
            failed_component_version="catboost-1.2.10:model-v1",
        ),
    )
    by_arm = {row.arm: row for row in results}

    assert by_arm[DefenseArm.RULES_ONLY].failure is None
    assert by_arm[DefenseArm.RULES_ONLY].fallback_count == 0
    assert by_arm[DefenseArm.GBDT_ONLY].failure is not None
    assert by_arm[DefenseArm.GBDT_ONLY].fallback_count == 0
    assert by_arm[DefenseArm.LAYERED_HYBRID].failure is None
    assert by_arm[DefenseArm.LAYERED_HYBRID].fallback_count == len(
        bundle_fixture.reload_matrix.rows
    )


def test_exact_nonmandatory_rule_score_one_cannot_defeat_disabled_threshold() -> None:
    from tests.defense.test_policy import event, vector

    feature_vector = vector(actor_count_1m=40.0)
    rule_result = RuleEngine.default().evaluate(event(), feature_vector)
    assert rule_result.score == 1.0

    decision = ActionPolicy.default().choose(
        event(),
        rule_result,
        calibrated_score=0.0,
        thresholds=PolicyThresholds(challenge=1.0, decline=1.0),
        vector=feature_vector,
    )

    assert decision.action is Action.APPROVE
    assert decision.score < 1.0
    assert decision.rule_score == 1.0


def test_threshold_bundle_callback_and_row_lineage_mismatches_fail_closed(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    event_ids = tuple(row.event_id for row in bundle_fixture.reload_matrix.rows)
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events, event_ids, as_of=AS_OF
    )
    thresholds = ReplayThresholdSet.from_reports(
        loaded, binding, {arm: loaded.threshold_report for arm in DefenseArm}
    )
    other_binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(reversed(event_ids)),
        as_of=AS_OF,
    )

    with pytest.raises(ReplayContractError, match="case callback lineage"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            thresholds=thresholds,
            case_counter=other_binding,
            evaluation=_context(bundle_fixture),
        )
    with pytest.raises(ReplayContractError, match="ordered decision rows"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix.model_copy(
                update={"rows": tuple(reversed(bundle_fixture.reload_matrix.rows))}
            ),
            defender=loaded,
            thresholds=thresholds,
            case_counter=binding,
            evaluation=_context(bundle_fixture),
        )


def test_threshold_set_binds_signed_selection_rows_and_rejects_cross_arm_lineage(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    report_document = json.loads(loaded.threshold_report.to_json())
    report_document["input_labels_digest"] = "e" * 64
    report_document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in report_document.items()
                if key != "report_digest"
            }
        )
    ).hexdigest()
    mismatched_report = type(loaded.threshold_report).from_json(
        canonical_json_bytes(report_document)
    )

    with pytest.raises(ReplayContractError, match="selection lineage"):
        ReplayThresholdSet.from_reports(
            loaded,
            binding,
            {
                DefenseArm.RULES_ONLY: mismatched_report,
                DefenseArm.GBDT_ONLY: loaded.threshold_report,
                DefenseArm.LAYERED_HYBRID: loaded.threshold_report,
            },
        )

    threshold_set = ReplayThresholdSet.from_reports(
        loaded,
        binding,
        {arm: loaded.threshold_report for arm in DefenseArm},
    )
    assert (
        threshold_set.selection_row_ids_digest
        == loaded.threshold_binding.row_ids_digest
    )
    assert ReplayThresholdSet.from_json(threshold_set.to_json()) == threshold_set

    threshold_document = json.loads(threshold_set.to_json())
    threshold_document["selection_row_ids_digest"] = "f" * 64
    threshold_document["threshold_set_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in threshold_document.items()
                if key != "threshold_set_digest"
            }
        )
    ).hexdigest()
    substituted = ReplayThresholdSet.from_json(
        canonical_json_bytes(threshold_document)
    )
    with pytest.raises(ReplayContractError, match="threshold selection lineage"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            thresholds=substituted,
            case_counter=binding,
            evaluation=_context(bundle_fixture),
        )


def test_campaign_with_two_fraud_family_owners_cannot_publish_lofo_evidence(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = ReplayThresholdSet.from_reports(
        loaded, binding, {arm: loaded.threshold_report for arm in DefenseArm}
    )
    context = _context(bundle_fixture)
    truth = list(context.truth)
    truth[0] = truth[0].model_copy(
        update={"campaign_id": "mixed", "is_fraud": True}
    )
    truth[1] = truth[1].model_copy(
        update={"campaign_id": "mixed", "is_fraud": True}
    )

    with pytest.raises(ReplayContractError, match="one fraud family"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            thresholds=thresholds,
            case_counter=binding,
            evaluation=context.model_copy(update={"truth": tuple(truth)}),
        )


def test_mandatory_integrity_decision_is_identical_and_prior_to_all_arm_thresholds(
    bundle_fixture: BundleFixture,
) -> None:
    first = bundle_fixture.reload_matrix.events[0].model_copy(
        update={
            "rail": Rail.AGENTIC,
            "integrity_status": "fail",
            "integrity_reason": "receipt_failed",
        }
    )
    events = (first, *bundle_fixture.reload_matrix.events[1:])
    matrix = bundle_fixture.reload_matrix.model_copy(update={"events": events})
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        events, tuple(row.event_id for row in matrix.rows), as_of=AS_OF
    )
    thresholds = ReplayThresholdSet.from_reports(
        loaded, binding, {arm: loaded.threshold_report for arm in DefenseArm}
    )
    context = _context(bundle_fixture).model_copy(update={"observations": events})

    results = replay_defense_arms(
        matrix=matrix,
        defender=loaded,
        thresholds=thresholds,
        case_counter=binding,
        evaluation=context,
    )

    assert len({row.common_integrity_digest for row in results}) == 1
    assert all(row.mandatory_decline_count == 1 for row in results)


def test_replay_result_canonical_roundtrip_rejects_unchecksummed_tamper(
    bundle_fixture: BundleFixture,
) -> None:
    result = _replay(bundle_fixture)[0]
    payload = result.to_json()

    assert type(result).from_json(payload) == result
    document = json.loads(payload)
    document["metrics"]["review_case_count"] += 1
    with pytest.raises(ValueError):
        type(result).from_json(canonical_json_bytes(document))


def test_case_binding_constructor_and_reinitialization_cannot_replace_production_callback(
    bundle_fixture: BundleFixture,
) -> None:
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    with pytest.raises((TypeError, ReplayContractError)):
        ReplayCaseCounterBinding(
            counter=object(),
            event_ids=(),
            rows_digest="0" * 64,
            as_of=AS_OF,
            callback_digest="0" * 64,
        )
    with pytest.raises((TypeError, ReplayContractError)):
        binding.__init__(
            counter=object(),
            event_ids=(),
            rows_digest="0" * 64,
            as_of=AS_OF,
            callback_digest="0" * 64,
        )
