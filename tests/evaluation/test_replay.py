"""Identical-row three-arm replay and lineage integration tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.defense.contracts import PolicyThresholds
from apar.defense.policy import ActionPolicy
from apar.defense.rules import DefenseReason, RuleEngine
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.defender_attestation import DefenderBundleVerifier
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
from apar.evaluation_hidden.defense_authority import (
    HIDDEN_CONTEXT_MEDIA_TYPE,
    HiddenEvaluationAuthority,
)
from apar.runs.wire import canonical_json_bytes
from tests.defense.test_bundle import BundleFixture

pytest_plugins = ("tests.defense.test_bundle",)

AS_OF = datetime(2026, 9, 30, tzinfo=UTC)
ISSUED_AT = datetime(2026, 8, 19, 12, tzinfo=UTC)
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


def _attestation(fixture: BundleFixture):
    _, ref = fixture.publisher.freeze(**fixture.kwargs)
    verifier = DefenderBundleVerifier(
        fixture.store,
        signer_key_id=fixture.signer.key_id,
        public_key_base64=fixture.signer.public_key_base64,
    )
    return verifier.attest(ref)


def _selection_evidence(
    fixture: BundleFixture, loaded: object
) -> tuple[np.ndarray, np.ndarray]:
    matrix = loaded.threshold_matrix
    labels = np.asarray(
        [int(fixture.split.row_is_fraud[row.event_id]) for row in matrix.rows],
        dtype=np.int64,
    )
    values = np.asarray(
        [float(fixture.split.row_net_settled_values[row.event_id]) for row in matrix.rows],
        dtype=np.float64,
    )
    return labels, values


def _thresholds(
    fixture: BundleFixture,
    loaded: object,
    binding: ReplayCaseCounterBinding,
) -> ReplayThresholdSet:
    labels, values = _selection_evidence(fixture, loaded)
    return ReplayThresholdSet.from_selection(
        loaded, binding, labels=labels, values=values
    )


def _selection_binding(loaded: object) -> ReplayCaseCounterBinding:
    return bind_replay_case_counter(
        loaded.threshold_matrix.events,
        tuple(row.event_id for row in loaded.threshold_matrix.rows),
        as_of=AS_OF,
    )


def _replay(fixture: BundleFixture, *, flip_truth: bool = False, model_failure=None):
    loaded, _ = _loaded(fixture)
    case_binding = bind_replay_case_counter(
        fixture.reload_matrix.events,
        tuple(row.event_id for row in fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(fixture, loaded, _selection_binding(loaded))
    return replay_defense_arms(
        matrix=fixture.reload_matrix,
        defender=loaded,
        defender_attestation=_attestation(fixture),
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
    thresholds = _thresholds(
        bundle_fixture, loaded, _selection_binding(loaded)
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
            defender_attestation=_attestation(bundle_fixture),
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
            defender_attestation=_attestation(bundle_fixture),
            thresholds=thresholds,
            case_counter=binding,
            evaluation=_context(bundle_fixture),
        )


def test_threshold_set_binds_signed_selection_rows_and_rejects_cross_arm_lineage(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = _selection_binding(loaded)
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

    with pytest.raises(ReplayContractError, match="rederived selection evidence"):
        ReplayThresholdSet.from_reports(
            loaded,
            binding,
            {
                DefenseArm.RULES_ONLY: mismatched_report,
                DefenseArm.GBDT_ONLY: loaded.threshold_report,
                DefenseArm.LAYERED_HYBRID: loaded.threshold_report,
            },
        )

    threshold_set = _thresholds(bundle_fixture, loaded, binding)
    assert (
        threshold_set.selection_row_ids_digest
        == loaded.threshold_binding.row_ids_digest
    )
    labels, values = _selection_evidence(bundle_fixture, loaded)
    assert (
        ReplayThresholdSet.from_json(
            threshold_set.to_json(),
            defender=loaded,
            case_counter=binding,
            labels=labels,
            values=values,
        )
        == threshold_set
    )

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
    with pytest.raises(ReplayContractError, match="rederived selection evidence"):
        ReplayThresholdSet.from_json(
            canonical_json_bytes(threshold_document),
            defender=loaded,
            case_counter=binding,
            labels=labels,
            values=values,
        )


def test_forged_rules_threshold_report_with_recomputed_digest_is_rejected(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        loaded.threshold_matrix.events,
        tuple(row.event_id for row in loaded.threshold_matrix.rows),
        as_of=AS_OF,
    )
    document = json.loads(loaded.threshold_report.to_json())
    document["thresholds"] = {"challenge": 0.25, "decline": 0.75}
    document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in document.items() if key != "report_digest"}
        )
    ).hexdigest()
    forged = type(loaded.threshold_report).from_json(canonical_json_bytes(document))

    with pytest.raises(ReplayContractError, match="rederived selection evidence"):
        ReplayThresholdSet.from_reports(
            loaded,
            binding,
            {
                DefenseArm.RULES_ONLY: forged,
                DefenseArm.GBDT_ONLY: loaded.threshold_report,
                DefenseArm.LAYERED_HYBRID: loaded.threshold_report,
            },
        )

    legitimate = _thresholds(bundle_fixture, loaded, binding)
    threshold_document = json.loads(legitimate.to_json())
    threshold_document["reports"][0]["report"] = document
    threshold_document["threshold_set_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in threshold_document.items()
                if key != "threshold_set_digest"
            }
        )
    ).hexdigest()
    threshold_document["reports"] = tuple(threshold_document["reports"])
    forged_set = ReplayThresholdSet.model_validate(threshold_document)
    replay_binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )
    with pytest.raises(ReplayContractError, match="exact rederived selection evidence"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            defender_attestation=_attestation(bundle_fixture),
            thresholds=forged_set,
            case_counter=replay_binding,
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
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    context = _context(bundle_fixture)
    truth = list(context.truth)
    truth[0] = truth[0].model_copy(
        update={"campaign_id": "mixed", "is_fraud": True}
    )
    truth[1] = truth[1].model_copy(
        update={"campaign_id": "mixed", "is_fraud": True}
    )

    with pytest.raises(ReplayContractError, match="one family owner"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            defender_attestation=_attestation(bundle_fixture),
            thresholds=thresholds,
            case_counter=binding,
            evaluation=context.model_copy(update={"truth": tuple(truth)}),
        )


def test_campaign_family_owner_includes_benign_rows(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    context = _context(bundle_fixture)
    truth = list(context.truth)
    truth[0] = truth[0].model_copy(
        update={"campaign_id": "mixed-benign", "is_fraud": True}
    )
    truth[1] = truth[1].model_copy(
        update={"campaign_id": "mixed-benign", "is_fraud": False}
    )

    with pytest.raises(ReplayContractError, match="one family owner"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            defender_attestation=_attestation(bundle_fixture),
            thresholds=thresholds,
            case_counter=binding,
            evaluation=context.model_copy(update={"truth": tuple(truth)}),
        )


def test_hidden_descriptor_rejects_arbitrary_digest_without_authority_receipt(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    binding = bind_replay_case_counter(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    ordinary = _context(bundle_fixture)
    relabelled = ReplayEvaluationContext(
        **{
            **ordinary.model_dump(mode="python"),
            "evaluation": EvaluationDescriptor(
                kind=EvaluationKind.HIDDEN, value="hidden"
            ),
        }
    )

    with pytest.raises(ReplayContractError, match="sealed hidden release receipt"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            defender_attestation=_attestation(bundle_fixture),
            thresholds=thresholds,
            case_counter=binding,
            evaluation=relabelled,
        )


def test_hidden_truth_resolves_only_after_all_three_arm_decisions_are_frozen(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apar.evaluation.replay as replay_module

    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    poison_ref = bundle_fixture.store.put_bytes(
        b'{"poison":true}', HIDDEN_CONTEXT_MEDIA_TYPE
    )
    request = authority.prepare_release(
        capability, poison_ref, released_at=ISSUED_AT
    )
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    completed_arms: list[DefenseArm] = []
    original = replay_module._arm_decisions

    def record_frozen_arm(**kwargs):
        completed_arms.append(kwargs["arm"])
        return original(**kwargs)

    monkeypatch.setattr(replay_module, "_arm_decisions", record_frozen_arm)

    with pytest.raises(ReplayContractError, match="evaluation context"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            defender_attestation=attestation,
            thresholds=thresholds,
            case_counter=binding,
            hidden_release=request,
        )

    assert tuple(completed_arms) == tuple(DefenseArm)


def test_hidden_replay_seals_exact_receipt_without_public_truth_or_reference(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    ordinary = _context(bundle_fixture)
    hidden_context = ReplayEvaluationContext(
        **{
            **ordinary.model_dump(mode="python"),
            "evaluation": EvaluationDescriptor(
                kind=EvaluationKind.HIDDEN, value="hidden"
            ),
            "truth": tuple(
                row.model_copy(
                    update={"viewpoint": "hidden", "label_source": "hidden_truth"}
                )
                for row in ordinary.truth
            ),
        }
    )
    restricted_ref = bundle_fixture.store.put_bytes(
        hidden_context.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    request = authority.prepare_release(
        capability, restricted_ref, released_at=ISSUED_AT
    )
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )

    results = replay_defense_arms(
        matrix=loaded.reload_matrix,
        defender=loaded,
        defender_attestation=attestation,
        thresholds=_thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        case_counter=binding,
        hidden_release=request,
    )

    receipts = {row.hidden_release_receipt_digest for row in results}
    assert len(receipts) == 1
    assert None not in receipts
    assert all(row.assurance.hidden_access_clean for row in results)
    public = b"".join(row.to_json() for row in results)
    assert restricted_ref.sha256.encode() not in public
    assert b"hidden_truth" not in public


def test_hidden_receipt_rejects_defender_bundle_substitution(
    bundle_fixture: BundleFixture,
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    replay_attestation = _attestation(bundle_fixture)
    substituted_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "32345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 19, 11, tzinfo=UTC),
    }
    _, substituted_ref = bundle_fixture.publisher.freeze(**substituted_kwargs)
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    substituted_attestation = verifier.attest(substituted_ref)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    capability = authority.freeze_and_issue(
        substituted_attestation, issued_at=ISSUED_AT
    )
    ordinary = _context(bundle_fixture)
    hidden_context = ordinary.model_copy(
        update={
            "evaluation": EvaluationDescriptor(
                kind=EvaluationKind.HIDDEN, value="hidden"
            ),
            "truth": tuple(
                row.model_copy(
                    update={"viewpoint": "hidden", "label_source": "hidden_truth"}
                )
                for row in ordinary.truth
            ),
        }
    )
    hidden_ref = bundle_fixture.store.put_bytes(
        hidden_context.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    request = authority.prepare_release(
        capability, hidden_ref, released_at=ISSUED_AT
    )
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )

    with pytest.raises(ReplayContractError, match="receipt failed verification"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            defender_attestation=replay_attestation,
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=binding,
            hidden_release=request,
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
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    context = _context(bundle_fixture).model_copy(update={"observations": events})

    results = replay_defense_arms(
        matrix=matrix,
        defender=loaded,
        defender_attestation=_attestation(bundle_fixture),
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
    with pytest.raises((TypeError, AttributeError)):
        object.__setattr__(binding, "_counter", lambda actions: 0)
