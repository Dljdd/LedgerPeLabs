"""Identical-row three-arm replay and lineage integration tests."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.defense.contracts import PolicyThresholds
from apar.defense.policy import ActionPolicy
from apar.defense.rules import DefenseReason, RuleEngine
from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow, FrozenCorpus
from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.gates import (
    EvaluationDescriptor,
    EvaluationKind,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
)
from apar.evaluation.metrics import LatencySample, SliceAssignment, SliceManifest
from apar.evaluation.regimes import RegimeSpec, derive_regime, frozen_corpus_digest
from apar.evaluation.replay import (
    DefenseArm,
    ModelFailure,
    ReplayCaseCounterBinding,
    ReplayContractError,
    ReplayCorpusEvidence,
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
    ReplayRegimeEvidence,
    ReplayThresholdSet,
    bind_replay_case_counter,
    replay_defense_arms,
)
from apar.evaluation.splits import EntityCohort, make_evaluation_split
from apar.evaluation_hidden.defense_authority import (
    HIDDEN_CONTEXT_MEDIA_TYPE,
    HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE,
    HiddenEvaluationAuthority,
)
from apar.features.builders import build_feature_matrix
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactRef, ArtifactStore
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
EVALUATOR_SIGNER = EvaluatorSigningIdentity.from_private_bytes(b"e" * 32)
EVALUATOR_VERIFIER = EvaluatorReplayVerifier.from_signer(EVALUATOR_SIGNER)


def _truth(fixture: BundleFixture, *, flip: bool = False) -> tuple[EvaluationTruthRow, ...]:
    rows = []
    for event in fixture.reload_matrix.events:
        rows.append(
            EvaluationTruthRow(
                event_id=event.event_id,
                payment_id=event.payment_id,
                campaign_id=fixture.split.row_campaigns[event.event_id],
                family=fixture.split.row_families[event.event_id],
                viewpoint="development",
                is_fraud=fixture.split.row_is_fraud[event.event_id] ^ flip,
                label_source="population_truth",
                label_mature_at=event.decision_at or event.event_time,
                first_settlement_at=None,
                net_settled_value=fixture.split.row_net_settled_values[event.event_id],
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
                entity_cohorts=fixture.split.entity_cohorts[event.event_id],
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


def _verifier(fixture: BundleFixture) -> DefenderBundleVerifier:
    return DefenderBundleVerifier(
        fixture.store,
        signer_key_id=fixture.signer.key_id,
        public_key_base64=fixture.signer.public_key_base64,
    )


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


def _corpus_evidence(
    fixture: BundleFixture,
    *,
    reload_events: tuple[object, ...] | None = None,
) -> ReplayCorpusEvidence:
    matrices = (
        fixture.kwargs["training_matrix"],
        fixture.kwargs["calibration_fit_matrix"],
        fixture.kwargs["threshold_matrix"],
        fixture.reload_matrix,
    )
    observations = tuple(
        event for matrix in matrices for event in matrix.events  # type: ignore[union-attr]
    )
    if reload_events is not None:
        replacements = {event.event_id: event for event in reload_events}  # type: ignore[attr-defined]
        observations = tuple(replacements.get(event.event_id, event) for event in observations)
    truth = tuple(
        EvaluationTruthRow(
            event_id=event.event_id,
            payment_id=event.payment_id,
            campaign_id=fixture.split.row_campaigns[event.event_id],
            family=fixture.split.row_families[event.event_id],
            viewpoint="development",
            is_fraud=fixture.split.row_is_fraud[event.event_id],
            label_source="population_truth",
            label_mature_at=event.decision_at or event.event_time,
            first_settlement_at=None,
            net_settled_value=fixture.split.row_net_settled_values[event.event_id],
            lifecycle_event_ids=(event.event_id,),
        )
        for event in observations
    )
    corpus = FrozenCorpus(
        observations=observations,
        truth=truth,
        manifest=CorpusManifest(
            profile_id="task12-replay-fixture",
            run_ids=("task12-replay",),
            run_lineage_digests=(hashlib.sha256(b"task12-replay").hexdigest(),),
            observation_count=len(observations),
            truth_count=len(truth),
        ),
    )
    return ReplayCorpusEvidence.create(
        corpus=corpus,
        split=fixture.split,
        signer=EVALUATOR_SIGNER,
    )


def _replay_evidence(fixture: BundleFixture, loaded: object) -> dict[str, object]:
    labels, values = _selection_evidence(fixture, loaded)
    return {
        "defender_verifier": _verifier(fixture),
        "evaluator_signer": EVALUATOR_SIGNER,
        "evaluator_verifier": EVALUATOR_VERIFIER,
        "corpus_evidence": _corpus_evidence(fixture),
        "threshold_labels": labels,
        "threshold_values": values,
        "evaluation_split": fixture.split,
    }


def _hidden_replay_evidence(
    fixture: BundleFixture,
    loaded: object,
    verifier: DefenderBundleVerifier,
) -> dict[str, object]:
    labels, values = _selection_evidence(fixture, loaded)
    return {
        "defender_verifier": verifier,
        "threshold_labels": labels,
        "threshold_values": values,
    }


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
        **_replay_evidence(fixture, loaded),
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


def test_all_label_flip_rejects_signed_split_truth_substitution(
    bundle_fixture: BundleFixture,
) -> None:
    before = _replay(bundle_fixture)
    with pytest.raises(ReplayContractError, match="signed.*truth|truth.*split"):
        _replay(bundle_fixture, flip_truth=True)

    assert tuple(row.arm for row in before) == tuple(DefenseArm)


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
            **_replay_evidence(bundle_fixture, loaded),
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
            **_replay_evidence(bundle_fixture, loaded),
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
            **_replay_evidence(bundle_fixture, loaded),
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

    with pytest.raises(ReplayContractError, match="one family owner|signed split truth"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            **_replay_evidence(bundle_fixture, loaded),
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

    with pytest.raises(ReplayContractError, match="one family owner|signed split truth"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            **_replay_evidence(bundle_fixture, loaded),
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

    with pytest.raises(ReplayContractError, match="sealed authority"):
        replay_defense_arms(
            matrix=bundle_fixture.reload_matrix,
            defender=loaded,
            **_replay_evidence(bundle_fixture, loaded),
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
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    poison_ref = bundle_fixture.store.put_bytes(
        b'{"poison":true}', HIDDEN_CONTEXT_MEDIA_TYPE
    )
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    completed_arms: list[DefenseArm] = []
    restricted_reads: list[str] = []
    persisted_media: list[str] = []
    original = replay_module._arm_decisions
    original_read = type(bundle_fixture.store).read
    original_put = type(bundle_fixture.store).put_bytes

    def record_frozen_arm(**kwargs):
        completed_arms.append(kwargs["arm"])
        return original(**kwargs)

    monkeypatch.setattr(replay_module, "_arm_decisions", record_frozen_arm)

    def record_read(store, ref):
        if ref == poison_ref:
            assert tuple(completed_arms) == tuple(DefenseArm)
            assert any("decision-freeze-receipt" in item for item in persisted_media)
            restricted_reads.append(ref.sha256)
        return original_read(store, ref)

    def record_put(store, payload, media_type):
        persisted_media.append(media_type)
        return original_put(store, payload, media_type)

    monkeypatch.setattr(type(bundle_fixture.store), "read", record_read)
    monkeypatch.setattr(type(bundle_fixture.store), "put_bytes", record_put)

    with pytest.raises(ReplayContractError, match="failed closed"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
            defender_attestation=attestation,
            thresholds=thresholds,
            case_counter=binding,
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=poison_ref,
            hidden_released_at=ISSUED_AT,
            hidden_sealed_at=AS_OF,
        )

    assert tuple(completed_arms) == tuple(DefenseArm)
    assert restricted_reads == []


def test_hidden_metrics_run_only_in_isolated_worker_not_parent_callables(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent monkeypatches cannot observe truth bytes or influence worker metrics."""
    import apar.evaluation.replay as replay_module

    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    ordinary = _context(bundle_fixture)
    hidden = ordinary.model_copy(
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
    restricted_ref = bundle_fixture.store.put_bytes(
        hidden.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    parent_reads: list[str] = []
    original_read = ArtifactStore.read

    def record_parent_read(store, ref):
        if ref == restricted_ref:
            parent_reads.append(ref.sha256)
        return original_read(store, ref)

    def forbidden_parent_evaluator(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent evaluator callable was invoked")

    monkeypatch.setattr(ArtifactStore, "read", record_parent_read)
    monkeypatch.setattr(
        replay_module, "_evaluate_hidden_frozen", forbidden_parent_evaluator
    )

    outcome = replay_defense_arms(
        matrix=loaded.reload_matrix,
        defender=loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        defender_attestation=attestation,
        thresholds=_thresholds(bundle_fixture, loaded, _selection_binding(loaded)),
        case_counter=bind_replay_case_counter(
            loaded.reload_matrix.events,
            tuple(row.event_id for row in loaded.reload_matrix.rows),
            as_of=AS_OF,
        ),
        hidden_authority=authority,
        hidden_capability=capability,
        hidden_ref=restricted_ref,
        hidden_released_at=ISSUED_AT,
        hidden_sealed_at=AS_OF,
    )

    assert len(outcome.results) == len(DefenseArm)
    assert parent_reads == []


def test_hidden_failure_is_irrevocably_consumed_before_restricted_read(
    bundle_fixture: BundleFixture,
) -> None:
    """A malformed worker input cannot be retried with the same capability."""
    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    malformed = bundle_fixture.store.put_bytes(
        b'{"not":"an evaluation context"}', HIDDEN_CONTEXT_MEDIA_TYPE
    )
    common = {
        "matrix": loaded.reload_matrix,
        "defender": loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        "defender_attestation": attestation,
        "thresholds": _thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        "case_counter": bind_replay_case_counter(
            loaded.reload_matrix.events,
            tuple(row.event_id for row in loaded.reload_matrix.rows),
            as_of=AS_OF,
        ),
        "hidden_authority": authority,
        "hidden_capability": capability,
        "hidden_ref": malformed,
        "hidden_released_at": ISSUED_AT,
        "hidden_sealed_at": AS_OF,
    }

    with pytest.raises(ReplayContractError):
        replay_defense_arms(**common)
    with pytest.raises(ReplayContractError, match="consumed"):
        replay_defense_arms(**common)


def test_hidden_worker_requires_the_persisted_decision_freeze_receipt(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash-shaped but absent freeze receipt cannot authorize restricted access."""
    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    ordinary = _context(bundle_fixture)
    hidden = ordinary.model_copy(
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
    restricted_ref = bundle_fixture.store.put_bytes(
        hidden.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    original_put = ArtifactStore.put_bytes

    def omit_freeze_receipt(store, payload, media_type):
        if media_type == HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE:
            digest = hashlib.sha256(payload).hexdigest()
            return ArtifactRef(
                sha256=digest,
                media_type=media_type,
                size_bytes=len(payload),
                relative_path=f"{digest}/payload",
            )
        return original_put(store, payload, media_type)

    monkeypatch.setattr(ArtifactStore, "put_bytes", omit_freeze_receipt)
    common = {
        "matrix": loaded.reload_matrix,
        "defender": loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        "defender_attestation": attestation,
        "thresholds": _thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        "case_counter": bind_replay_case_counter(
            loaded.reload_matrix.events,
            tuple(row.event_id for row in loaded.reload_matrix.rows),
            as_of=AS_OF,
        ),
        "hidden_authority": authority,
        "hidden_capability": capability,
        "hidden_ref": restricted_ref,
        "hidden_released_at": ISSUED_AT,
        "hidden_sealed_at": AS_OF,
    }

    with pytest.raises(ReplayContractError, match="failed closed"):
        replay_defense_arms(**common)
    with pytest.raises(ReplayContractError, match="consumed"):
        replay_defense_arms(**common)


def test_hidden_worker_rejects_truth_blind_freeze_document_substitution(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The signed freeze receipt must cover lineage fields beyond arm decisions."""
    from apar.evaluation_hidden.worker_client import EvaluatorWorkerClient

    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    ordinary = _context(bundle_fixture)
    hidden = ordinary.model_copy(
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
    restricted_ref = bundle_fixture.store.put_bytes(
        hidden.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    original_invoke = EvaluatorWorkerClient.invoke

    def substitute_freeze(self, document, *, timeout_ms=30_000):
        changed = dict(document)
        frozen = dict(changed["frozen"])
        frozen["decision_content_digest"] = "f" * 64
        changed["frozen"] = frozen
        return original_invoke(self, changed, timeout_ms=timeout_ms)

    monkeypatch.setattr(EvaluatorWorkerClient, "invoke", substitute_freeze)

    with pytest.raises(ReplayContractError, match="failed closed"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
            defender_attestation=attestation,
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=bind_replay_case_counter(
                loaded.reload_matrix.events,
                tuple(row.event_id for row in loaded.reload_matrix.rows),
                as_of=AS_OF,
            ),
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=restricted_ref,
            hidden_released_at=ISSUED_AT,
            hidden_sealed_at=AS_OF,
        )


def test_concurrent_hidden_release_is_atomic_and_exactly_one_use(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    ordinary = _context(bundle_fixture)
    hidden = ordinary.model_copy(
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
    restricted_ref = bundle_fixture.store.put_bytes(
        hidden.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    barrier = threading.Barrier(2)
    original_read = ArtifactStore.read

    def synchronize_parent_reads(store, ref):
        if ref == restricted_ref:
            barrier.wait(timeout=5)
        return original_read(store, ref)

    monkeypatch.setattr(ArtifactStore, "read", synchronize_parent_reads)
    common = {
        "matrix": loaded.reload_matrix,
        "defender": loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        "defender_attestation": attestation,
        "thresholds": _thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        "case_counter": bind_replay_case_counter(
            loaded.reload_matrix.events,
            tuple(row.event_id for row in loaded.reload_matrix.rows),
            as_of=AS_OF,
        ),
        "hidden_authority": authority,
        "hidden_capability": capability,
        "hidden_ref": restricted_ref,
        "hidden_released_at": ISSUED_AT,
        "hidden_sealed_at": AS_OF,
    }

    def invoke() -> object:
        try:
            return replay_defense_arms(**common)
        except Exception as error:  # typed outcome asserted below
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: invoke(), range(2)))

    successes = tuple(item for item in outcomes if hasattr(item, "results"))
    failures = tuple(item for item in outcomes if isinstance(item, Exception))
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ReplayContractError)


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
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
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
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )

    outcome = replay_defense_arms(
        matrix=loaded.reload_matrix,
        defender=loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        defender_attestation=attestation,
        thresholds=_thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        case_counter=binding,
        hidden_authority=authority,
        hidden_capability=capability,
        hidden_ref=restricted_ref,
        hidden_released_at=ISSUED_AT,
        hidden_sealed_at=AS_OF,
    )

    assert not hasattr(outcome, "receipt")
    assert hasattr(outcome, "public_proof")
    results = outcome.results
    proof_ids = {row.hidden_public_proof_id for row in results}
    assert proof_ids == {outcome.public_proof.proof_id}
    assert all(row.assurance.hidden_access_clean for row in results)
    public = b"".join(row.to_json() for row in results) + outcome.public_proof.to_json()
    private_context_digest = hashlib.sha256(
        b"apar-hidden-evaluator-context-v1\x00" + hidden_context.to_json()
    ).hexdigest()
    restricted_cohort_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                row.event_id: [item.value for item in row.entity_cohorts]
                for row in hidden_context.slice_assignments
            }
        )
    ).hexdigest()
    assert restricted_ref.sha256.encode() not in public
    assert restricted_ref.relative_path.encode() not in public
    assert b"hidden_truth" not in public
    assert private_context_digest.encode() not in public
    assert restricted_cohort_digest.encode() not in public
    assert outcome.public_proof.evaluator_context_token != private_context_digest
    assert outcome.public_proof.cohort_mapping_token != restricted_cohort_digest
    assert {
        row.evaluation_context_digest for row in results
    } == {outcome.public_proof.evaluator_context_token}
    assert {
        row.evaluation_lineage.cohort_mapping_digest for row in results
    } == {outcome.public_proof.cohort_mapping_token}
    assert outcome.public_proof.proof_id != restricted_ref.sha256
    assert not hasattr(outcome.public_proof, "restricted_ref_digest")
    assert not hasattr(outcome.public_proof, "restricted_artifact_digest")
    assert not hasattr(outcome.public_proof, "canonical_content_digest")

    with pytest.raises(ReplayContractError):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
            defender_attestation=attestation,
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=binding,
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=restricted_ref,
            hidden_released_at=ISSUED_AT,
            hidden_sealed_at=AS_OF,
        )


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
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
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
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )

    with pytest.raises(ReplayContractError, match="exact re-attestation"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
            defender_attestation=replay_attestation,
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=binding,
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=hidden_ref,
            hidden_released_at=ISSUED_AT,
            hidden_sealed_at=AS_OF,
        )


def test_hidden_replay_rejects_wrong_store_and_capability_identity(
    bundle_fixture: BundleFixture, tmp_path: Path
) -> None:
    loaded, _ = _loaded(bundle_fixture)
    attestation = _attestation(bundle_fixture)
    verifier = _verifier(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store, EVALUATOR_SIGNER)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    other_authority = HiddenEvaluationAuthority(
        verifier, bundle_fixture.store, EVALUATOR_SIGNER
    )
    other_capability = other_authority.freeze_and_issue(
        attestation, issued_at=ISSUED_AT
    )
    hidden_context = _context(bundle_fixture).model_copy(
        update={
            "evaluation": EvaluationDescriptor(
                kind=EvaluationKind.HIDDEN, value="hidden"
            )
        }
    )
    wrong_store = ArtifactStore(tmp_path / "wrong-hidden-store")
    wrong_ref = wrong_store.put_bytes(
        hidden_context.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE
    )
    binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )
    common = {
        "matrix": loaded.reload_matrix,
        "defender": loaded,
        **_hidden_replay_evidence(bundle_fixture, loaded, verifier),
        "defender_attestation": attestation,
        "thresholds": _thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        "case_counter": binding,
        "hidden_authority": authority,
        "hidden_ref": wrong_ref,
        "hidden_released_at": ISSUED_AT,
        "hidden_sealed_at": AS_OF,
    }

    with pytest.raises(ReplayContractError):
        replay_defense_arms(**common, hidden_capability=other_capability)
    with pytest.raises(ReplayContractError):
        replay_defense_arms(**common, hidden_capability=capability)


def test_mandatory_integrity_decision_is_identical_and_prior_to_all_arm_thresholds(
    bundle_fixture: BundleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apar.evaluation.replay as replay_module

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
    decisions_by_arm: dict[DefenseArm, tuple[object, ...]] = {}
    policy_calls: list[str | None] = []
    original_evaluate = replay_module._evaluate_frozen_arm
    original_choose = ActionPolicy.choose

    def record_evaluate(**kwargs):
        decisions_by_arm[kwargs["arm"]] = kwargs["decisions"]
        return original_evaluate(**kwargs)

    def record_choose(self, *args, **kwargs):
        policy_calls.append(kwargs.get("score_mode"))
        return original_choose(self, *args, **kwargs)

    monkeypatch.setattr(replay_module, "_evaluate_frozen_arm", record_evaluate)
    monkeypatch.setattr(ActionPolicy, "choose", record_choose)

    replay_evidence = _replay_evidence(bundle_fixture, loaded)
    replay_evidence["corpus_evidence"] = _corpus_evidence(
        bundle_fixture, reload_events=events
    )
    results = replay_defense_arms(
        matrix=matrix,
        defender=loaded,
        **replay_evidence,
        defender_attestation=_attestation(bundle_fixture),
        thresholds=thresholds,
        case_counter=binding,
        evaluation=context,
    )

    assert len({row.common_integrity_digest for row in results}) == 1
    assert all(row.mandatory_decline_count == 1 for row in results)
    assert len({id(decisions_by_arm[arm][0]) for arm in DefenseArm}) == 1
    assert policy_calls.count(None) == 1
    assert len(policy_calls) == 1 + (len(matrix.rows) - 1) * len(DefenseArm)


def test_split_truth_and_multilabel_cohorts_are_rederived_exactly(
    bundle_fixture: BundleFixture,
) -> None:
    """Partial label and cohort substitutions must fail signed lineage validation."""
    context = _context(bundle_fixture)
    truth = list(context.truth)
    truth[0] = truth[0].model_copy(update={"is_fraud": not truth[0].is_fraud})
    truth[1] = truth[1].model_copy(update={"campaign_id": "substituted-campaign"})
    assignments = list(context.slice_assignments)
    assignments[-1] = assignments[-1].model_copy(
        update={"entity_cohorts": (EntityCohort.COLD_ACTOR,)}
    )
    loaded, _ = _loaded(bundle_fixture)

    with pytest.raises(ReplayContractError, match="signed.*truth|truth.*split|cohort"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_replay_evidence(bundle_fixture, loaded),
            defender_attestation=_attestation(bundle_fixture),
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=bind_replay_case_counter(
                loaded.reload_matrix.events,
                tuple(row.event_id for row in loaded.reload_matrix.rows),
                as_of=AS_OF,
            ),
            evaluation=context.model_copy(
                update={
                    "truth": tuple(truth),
                    "slice_assignments": tuple(assignments),
                }
            ),
        )


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
    with pytest.raises((TypeError, ValueError)):
        ReplayCaseCounterBinding(
            counter=object(),
            event_ids=(),
            rows_digest="0" * 64,
            as_of=AS_OF,
            callback_digest="0" * 64,
        )
    with pytest.raises((TypeError, ValueError)):
        binding.__init__(
            counter=object(),
            event_ids=(),
            rows_digest="0" * 64,
            as_of=AS_OF,
            callback_digest="0" * 64,
        )
    object.__setattr__(binding, "_counter", lambda actions: 0)
    rebuilt = binding.reconstruct(
        bundle_fixture.reload_matrix.events,
        tuple(row.event_id for row in bundle_fixture.reload_matrix.rows),
        AS_OF,
    )
    assert type(rebuilt).__name__ == "ReviewCaseCounter"


def test_replay_security_does_not_depend_on_mutable_factory_registries(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting module provenance maps must not revoke valid concrete evidence."""
    import apar.evaluation.replay as replay_module

    loaded, _ = _loaded(bundle_fixture)
    replay_binding = bind_replay_case_counter(
        loaded.reload_matrix.events,
        tuple(row.event_id for row in loaded.reload_matrix.rows),
        as_of=AS_OF,
    )
    thresholds = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    monkeypatch.setattr(replay_module, "_THRESHOLD_SETS", {}, raising=False)
    monkeypatch.setattr(replay_module, "_CASE_BINDINGS", {}, raising=False)

    results = replay_defense_arms(
        matrix=loaded.reload_matrix,
        defender=loaded,
        **_replay_evidence(bundle_fixture, loaded),
        defender_attestation=_attestation(bundle_fixture),
        thresholds=thresholds,
        case_counter=replay_binding,
        evaluation=_context(bundle_fixture),
    )

    assert tuple(row.arm for row in results) == tuple(DefenseArm)


def test_regime_evidence_rederives_upstream_manifest_and_corpus() -> None:
    from tests.evaluation.test_regimes import _corpus

    parent = _corpus()
    spec = RegimeSpec.missing_optional()
    derived, manifest = derive_regime(parent, spec)
    evidence = ReplayRegimeEvidence.create(
        parent_corpus=parent,
        derived_corpus=derived,
        spec=spec,
        manifest=manifest,
    )

    assert evidence.manifest == manifest
    with pytest.raises(ValueError, match="upstream derivation"):
        ReplayRegimeEvidence.create(
            parent_corpus=parent,
            derived_corpus=derived,
            spec=spec,
            manifest=manifest.model_copy(update={"output_corpus_digest": "f" * 64}),
        )


def test_regime_replay_uses_derived_split_and_rejects_cohort_substitution(
    bundle_fixture: BundleFixture,
) -> None:
    """A transformed corpus has its own evaluation split but the parent's training split."""
    parent = _corpus_evidence(bundle_fixture).corpus
    lineage = bundle_fixture.kwargs["lineage"]
    changed_lineage = lineage.model_copy(  # type: ignore[union-attr]
        update={"corpus_digest": frozen_corpus_digest(parent)}
    )
    _, top_ref = bundle_fixture.publisher.freeze(
        **{
            **bundle_fixture.kwargs,
            "lineage": changed_lineage,
            "bundle_id": "42345678-1234-5678-9234-567812345678",
        }
    )
    loaded = bundle_fixture.publisher.load(top_ref)
    verifier = _verifier(bundle_fixture)
    attestation = verifier.attest(top_ref)
    spec = RegimeSpec.missing_optional()
    derived, manifest = derive_regime(parent, spec)
    derived_split = make_evaluation_split(derived, bundle_fixture.split.config)
    evaluation_ids = derived_split.row_ids["development"]
    truth_by_id = {row.event_id: row for row in derived.truth}
    lifecycle_ids = {
        lifecycle_id
        for event_id in evaluation_ids
        for lifecycle_id in truth_by_id[event_id].lifecycle_event_ids
    }
    evaluation_observations = tuple(
        row for row in derived.observations if row.event_id in lifecycle_ids
    )
    all_features = build_feature_matrix(evaluation_observations, loaded.catalog)
    matrix = all_features.model_copy(
        update={
            "rows": tuple(
                row for row in all_features.rows if row.event_id in set(evaluation_ids)
            )
        }
    )
    context = ReplayEvaluationContext(
        evaluation=EvaluationDescriptor(
            kind=EvaluationKind.REGIME,
            value=spec.kind.value,
        ),
        truth=tuple(truth_by_id[event_id] for event_id in evaluation_ids),
        observations=matrix.events,
        as_of=AS_OF,
        slice_assignments=tuple(
            SliceAssignment(
                event_id=event_id,
                regime=spec.kind.value,
                entity_cohorts=derived_split.entity_cohorts[event_id],
            )
            for event_id in evaluation_ids
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=tuple(
            ReplayLatencySamples(
                arm=arm,
                samples=tuple(
                    LatencySample(
                        event_id=event_id,
                        feature_ms=1.0,
                        rules_ms=1.0,
                        model_ms=0.0 if arm is DefenseArm.RULES_ONLY else 2.0,
                        calibration_policy_ms=1.0,
                        end_to_end_ms=(
                            3.0 if arm is DefenseArm.RULES_ONLY else 5.0
                        ),
                    )
                    for event_id in evaluation_ids
                ),
            )
            for arm in DefenseArm
        ),
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=True,
            parity_passed=True,
            leakage_evidence_digest="a" * 64,
            parity_evidence_digest="b" * 64,
        ),
    )
    labels, values = _selection_evidence(bundle_fixture, loaded)
    common = {
        "matrix": matrix,
        "defender": loaded,
        "defender_verifier": verifier,
        "defender_attestation": attestation,
        "thresholds": _thresholds(
            bundle_fixture, loaded, _selection_binding(loaded)
        ),
        "threshold_labels": labels,
        "threshold_values": values,
        "case_counter": bind_replay_case_counter(
            matrix.events,
            evaluation_ids,
            as_of=AS_OF,
        ),
        "evaluation": context,
        "evaluation_split": derived_split,
        "regime_evidence": ReplayRegimeEvidence.create(
            parent_corpus=parent,
            derived_corpus=derived,
            spec=spec,
            manifest=manifest,
        ),
        "corpus_evidence": ReplayCorpusEvidence.create(
            corpus=derived,
            split=derived_split,
            signer=EVALUATOR_SIGNER,
        ),
        "evaluator_signer": EVALUATOR_SIGNER,
        "evaluator_verifier": EVALUATOR_VERIFIER,
    }

    result = replay_defense_arms(**common)
    assert len(result) == len(DefenseArm)

    assignments = list(context.slice_assignments)
    assignments[0] = assignments[0].model_copy(
        update={"entity_cohorts": (EntityCohort.COLD_ACTOR,)}
    )
    changed = {
        **common,
        "evaluation": context.model_copy(
            update={"slice_assignments": tuple(assignments)}
        ),
    }
    with pytest.raises(ReplayContractError, match="cohort"):
        replay_defense_arms(**changed)


def test_ordinary_development_rows_cannot_be_relabelled_as_held_family(
    bundle_fixture: BundleFixture,
) -> None:
    """Descriptor identity must be derived from split and training receipts."""
    loaded, _ = _loaded(bundle_fixture)
    context = _context(bundle_fixture).model_copy(
        update={
            "evaluation": EvaluationDescriptor(
                kind=EvaluationKind.HELD_FAMILY,
                value="agentic_intent_abuse",
            )
        }
    )

    with pytest.raises(ReplayContractError, match="descriptor lineage"):
        replay_defense_arms(
            matrix=loaded.reload_matrix,
            defender=loaded,
            **_replay_evidence(bundle_fixture, loaded),
            defender_attestation=_attestation(bundle_fixture),
            thresholds=_thresholds(
                bundle_fixture, loaded, _selection_binding(loaded)
            ),
            case_counter=bind_replay_case_counter(
                loaded.reload_matrix.events,
                tuple(row.event_id for row in loaded.reload_matrix.rows),
                as_of=AS_OF,
            ),
            evaluation=context,
        )


def test_every_nonmandatory_arm_action_uses_the_production_policy(
    bundle_fixture: BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand-built replay decisions must not bypass ActionPolicy semantics."""
    calls: list[str] = []
    original = ActionPolicy.choose

    def recording_choose(self, *args, **kwargs):
        calls.append(kwargs.get("score_mode", "layered"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ActionPolicy, "choose", recording_choose)
    _replay(bundle_fixture)

    assert len(calls) == len(bundle_fixture.reload_matrix.rows) * len(DefenseArm)
    assert calls == [
        mode
        for mode in ("rules_only", "model_only", "layered")
        for _ in bundle_fixture.reload_matrix.rows
    ]


def test_action_policy_closed_score_modes_preserve_disabled_endpoint() -> None:
    """Rules/model/layered score choice must share the frozen production primitive."""
    from tests.defense.test_policy import event, vector

    feature_vector = vector(actor_count_1m=40.0)
    rule_result = RuleEngine.default().evaluate(event(), feature_vector)
    policy = ActionPolicy.default()
    thresholds = PolicyThresholds(challenge=1.0, decline=1.0)

    decisions = tuple(
        policy.choose(
            event(),
            rule_result,
            calibrated_score=1.0,
            thresholds=thresholds,
            vector=feature_vector,
            score_mode=mode,
        )
        for mode in ("rules_only", "model_only", "layered")
    )

    assert all(item.action is Action.APPROVE for item in decisions)
    assert all(item.score < 1.0 for item in decisions)
