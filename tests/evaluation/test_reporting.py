"""Canonical, signed, privacy-safe judge reporting contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.gates import (
    CandidateRoleEvidence,
    ChampionDecision,
    EvaluationKind,
    EvaluationLineage,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    GateConfig,
    HiddenPublicProof,
    PromotionMetrics,
    RateEvidence,
    SlicePerformance,
    VerifiedPromotionEnvelope,
    evaluate_promotion_gates,
)
from apar.evaluation.metrics import (
    BootstrapDerivationEvidence,
    MetricDerivationEvidence,
    MetricReport,
    campaign_bootstrap,
    compute_metric_report,
)
from apar.evaluation.publication_inputs import (
    PublicationInputError,
    publish_corpus_attestation,
    verify_evaluation_inputs,
)
from apar.evaluation.replay import ReplayThresholdSet
from apar.evaluation.reporting import (
    PUBLIC_ARTIFACT_MEDIA_TYPES,
    SCORECARD_ARTIFACT_NAME,
    DefenseScorecard,
    EvaluationArtifactBundle,
    MetricPublicationEvidence,
    PublicArtifactVerifier,
    ReportingContractError,
    ScorecardPublicationRequest,
    _privacy_scan,
    load_evaluation_bundle,
    publish_scorecard,
)
from apar.runs.wire import canonical_json_bytes
from tests.evaluation.test_gates import (
    EVALUATOR_SIGNER,
    EVALUATOR_VERIFIER,
    _promotion_envelope,
    _results,
)
from tests.evaluation.test_metrics_classification import four_row_inputs
from tests.evaluation.test_replay import _corpus_evidence, _selection_binding, _thresholds

pytest_plugins = ("tests.defense.test_bundle",)

HIDDEN_SIGNER = EvaluatorSigningIdentity.from_private_bytes(b"h" * 32)
HIDDEN_VERIFIER = EvaluatorReplayVerifier.from_signer(HIDDEN_SIGNER)

FORBIDDEN_PUBLIC_TOKENS = (
    "metricderivationevidence",
    "bootstrapderivationevidence",
    "evaluator_input_digest",
    "derivation_evidence_digest",
    "decision_event_ids",
    "event-a",
    "payment-event-a",
    "campaign-event-a",
    "hidden_public_proof_id",
    "hpf_",
    "evaluator_context",
    "cohort_mapping_digest",
    "restricted",
    "relative_path",
    "private_key",
)


def _promotion_metrics(report: MetricReport) -> PromotionMetrics:
    classification = report.classification
    operations = report.operations
    return PromotionMetrics(
        row_count=classification.row_count,
        recall=classification.recall.value,
        ece=report.calibration.ece.value,
        p95_latency_ms=report.engineering.end_to_end_ms.p95.value,
        preventable_settled_value=report.value.preventable_settled_value,
        value_escaped=report.value.value_escaped,
        review_case_count=operations.review_case_count,
        challenge_rate=operations.challenge_count / classification.row_count,
        false_decline=RateEvidence(
            numerator=operations.false_decline_count,
            denominator=classification.legitimate_count,
            value=(
                operations.false_decline_count / classification.legitimate_count
                if classification.legitimate_count
                else None
            ),
            defined=classification.legitimate_count > 0,
        ),
        review_case_rate=operations.review_case_count / classification.row_count,
        slice_performance=tuple(
            sorted(
                (
                    SlicePerformance(
                        kind=item.kind,
                        value=item.value,
                        recall=item.recall.value,
                    )
                    for item in classification.slices
                ),
                key=lambda item: (item.kind, item.value),
            )
        ),
    )


def _request(
    *,
    defender_digest: str = "7" * 64,
    defender_bundle_id: str = "pooled-candidate",
    corpus_digest: str = "0" * 64,
    split_digest: str | None = None,
    evaluator_signer: EvaluatorSigningIdentity = EVALUATOR_SIGNER,
    evaluator_verifier: EvaluatorReplayVerifier = EVALUATOR_VERIFIER,
    hidden_signer: EvaluatorSigningIdentity = HIDDEN_SIGNER,
    hidden_verifier: EvaluatorReplayVerifier = HIDDEN_VERIFIER,
    threshold_set: ReplayThresholdSet,
    latency_multiplier: float = 1.0,
) -> tuple[ScorecardPublicationRequest, ChampionDecision]:
    inputs = four_row_inputs()
    if latency_multiplier != 1.0:
        inputs = inputs.model_copy(
            update={
                "latency_samples": tuple(
                    item.model_copy(
                        update={
                            "feature_ms": item.feature_ms * latency_multiplier,
                            "rules_ms": item.rules_ms * latency_multiplier,
                            "model_ms": item.model_ms * latency_multiplier,
                            "calibration_policy_ms": (
                                item.calibration_policy_ms * latency_multiplier
                            ),
                            "end_to_end_ms": item.end_to_end_ms * latency_multiplier,
                        }
                    )
                    for item in inputs.latency_samples
                )
            }
        )
    report = compute_metric_report(inputs)
    metric_evidence = MetricDerivationEvidence.from_inputs(inputs)
    confidence = campaign_bootstrap(inputs)
    bootstrap_evidence = BootstrapDerivationEvidence.from_inputs(inputs)
    metrics = _promotion_metrics(report)
    decision_ids = tuple(row.event_id for row in inputs.truth)
    decision_rows_digest = hashlib.sha256(canonical_json_bytes(list(decision_ids))).hexdigest()
    rebuilt_results = []
    for row in _results():
        non_held = row.evaluation.kind.value != "held_family"
        lineage_fields = {
            **row.evaluation_lineage.model_dump(
                mode="python", exclude={"lineage_digest", "descriptor"}
            ),
            "descriptor": row.evaluation,
            "decision_rows_digest": decision_rows_digest,
        }
        if split_digest is not None:
            lineage_fields["split_digest"] = split_digest
        lineage_fields["corpus_digest"] = corpus_digest
        role_fields = row.candidate_role.model_dump(mode="python", exclude={"role_digest"})
        role_fields["threshold_set_digest"] = threshold_set.threshold_set_digest
        if non_held:
            lineage_fields["bundle_manifest_digest"] = defender_digest
            lineage_fields["defender_top_ref_digest"] = defender_digest
            role_fields["bundle_manifest_digest"] = defender_digest
            role_fields["defender_top_ref_digest"] = defender_digest
            role_fields["bundle_id"] = defender_bundle_id
        rebuilt_results.append(
            row.rebuild(
                decision_event_ids=decision_ids,
                decision_rows_digest=decision_rows_digest,
                evaluation_lineage=EvaluationLineage.create(**lineage_fields),
                candidate_role=CandidateRoleEvidence.create(**role_fields),
                bundle_manifest_digest=(
                    defender_digest if non_held else row.bundle_manifest_digest
                ),
                metrics=metrics,
                metric_report_digest=report.report_digest,
                threshold_set_digest=threshold_set.threshold_set_digest,
                threshold_report_digest=threshold_set.report_for(row.arm).report_digest,
                case_callback_digest=threshold_set.case_callback_digest,
            ),
        )
    results = tuple(rebuilt_results)
    initial = _promotion_envelope(results, signer=evaluator_signer)
    original_hidden = next(
        batch
        for batch in initial.component_batches
        if batch.results[0].evaluation.kind is EvaluationKind.HIDDEN
    )
    hidden_batch = hidden_signer.sign_batch(original_hidden.results)
    hidden_row = hidden_batch.results[0]
    hidden_proof = HiddenPublicProof.create(
        proof_id=hidden_row.hidden_public_proof_id or "",
        batch_content_digest=hidden_batch.batch_content_digest,
        decision_bindings_digest="9" * 64,
        bundle_manifest_digest=hidden_row.bundle_manifest_digest,
        defender_top_ref_digest=hidden_row.evaluation_lineage.defender_top_ref_digest,
        worker_manifest_digest="c" * 64,
        evaluator_context_token=hidden_row.evaluation_context_digest,
        cohort_mapping_token=hidden_row.evaluation_lineage.cohort_mapping_digest,
        issued_at="2026-08-19T00:00:00Z",
        signer=hidden_signer,
    )
    envelope = VerifiedPromotionEnvelope.create(
        component_batches=tuple(
            hidden_batch if item is original_hidden else item for item in initial.component_batches
        ),
        hidden_proofs=(hidden_proof,),
        signer=evaluator_signer,
        hidden_proof_verifier=hidden_verifier,  # type: ignore[call-arg]
    )
    decision = evaluate_promotion_gates(
        envelope,
        GateConfig.competition(),
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_verifier,
    )
    primary = tuple(
        row for row in envelope.combined_batch.results if row.evaluation.value == "development"
    )
    evidence = tuple(
        MetricPublicationEvidence(
            arm=row.arm,
            result_digest=row.result_digest,
            metric_report=report,
            metric_derivation_evidence=metric_evidence,
            confidence_intervals=confidence,
            bootstrap_derivation_evidence=bootstrap_evidence,
        )
        for row in primary
    )
    request = ScorecardPublicationRequest(
        promotion_envelope=envelope,
        champion_decision=decision,
        metric_evidence=evidence,
        threshold_set=threshold_set,
    )
    assert evaluator_verifier.verify_promotion_envelope(envelope)
    return request, decision


def _authenticated_inputs(bundle_fixture):
    evidence = _corpus_evidence(bundle_fixture)
    kwargs = {
        **bundle_fixture.kwargs,
        "lineage": bundle_fixture.kwargs["lineage"].model_copy(
            update={"corpus_digest": evidence.corpus_digest}
        ),
    }
    _manifest, defender_ref = bundle_fixture.publisher.freeze(**kwargs)
    _attestation, corpus_ref = publish_corpus_attestation(
        evidence,
        artifact_store=bundle_fixture.store,
        signer=EVALUATOR_SIGNER,
    )
    defender_verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    verified = verify_evaluation_inputs(
        corpus_ref=corpus_ref,
        defender_ref=defender_ref,
        artifact_store=bundle_fixture.store,
        evaluator_verifier=EVALUATOR_VERIFIER,
        defender_verifier=defender_verifier,
    )
    loaded = bundle_fixture.publisher.load(defender_ref)
    threshold_set = _thresholds(
        bundle_fixture, loaded, _selection_binding(loaded)
    )
    return verified, corpus_ref, defender_ref, threshold_set


def test_corpus_content_mismatch_with_matching_split_rejects_before_publication(
    bundle_fixture,
) -> None:
    """A shared split digest cannot authenticate a different frozen corpus."""
    _manifest, defender_ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    evidence = _corpus_evidence(bundle_fixture)
    assert evidence.split_digest == bundle_fixture.kwargs["lineage"].split_manifest_digest
    assert evidence.corpus_digest != bundle_fixture.kwargs["lineage"].corpus_digest
    attestation, corpus_ref = publish_corpus_attestation(
        evidence,
        artifact_store=bundle_fixture.store,
        signer=EVALUATOR_SIGNER,
    )
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )

    with pytest.raises(PublicationInputError, match="corpus.*lineage"):
        verify_evaluation_inputs(
            corpus_ref=corpus_ref,
            defender_ref=defender_ref,
            artifact_store=bundle_fixture.store,
            evaluator_verifier=EVALUATOR_VERIFIER,
            defender_verifier=verifier,
        )
    for restricted in (attestation,):
        assert repr(restricted) == "<restricted verified corpus attestation>"
        assert str(restricted) == "<restricted verified corpus attestation>"


@pytest.fixture
def publication(bundle_fixture):
    store = bundle_fixture.store
    signer = bundle_fixture.signer
    verifier = PublicArtifactVerifier.from_signer(signer)
    verified, _corpus_ref, defender_ref, threshold_set = _authenticated_inputs(bundle_fixture)
    request, decision = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        corpus_digest=verified.corpus.corpus_digest,
        split_digest=verified.corpus.split_digest,
        threshold_set=threshold_set,
    )
    scorecard, bundle = publish_scorecard(
        request,
        verified_inputs=verified,
        artifact_store=store,
        signer=signer,
        publication_verifier=verifier,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=HIDDEN_VERIFIER,
    )
    return store, signer, verifier, verified, request, decision, scorecard, bundle


def test_every_displayed_metric_resolves_to_an_allowlisted_signed_artifact(
    publication,
) -> None:
    """Catch untraceable judge metrics or a scorecard self-reference."""
    store, _signer, verifier, _verified, request, _, scorecard, bundle = publication

    assert set(scorecard.public_artifacts) == set(PUBLIC_ARTIFACT_MEDIA_TYPES)
    assert SCORECARD_ARTIFACT_NAME not in scorecard.public_artifacts
    assert set(bundle.public_artifacts) == {
        *PUBLIC_ARTIFACT_MEDIA_TYPES,
        SCORECARD_ARTIFACT_NAME,
    }
    assert scorecard.core_digest == scorecard.compute_core_digest()
    artifact_digests = {item.sha256 for item in scorecard.public_artifacts.values()}
    assert all(row.metric_artifact_sha256 in artifact_digests for row in scorecard.leaderboard)
    assert all(row.metric_artifact_sha256 in artifact_digests for row in scorecard.slice_summaries)
    evidence_by_arm = {item.arm: item for item in request.metric_evidence}
    for row in scorecard.leaderboard:
        report = evidence_by_arm[row.arm].metric_report
        assert row.precision == report.classification.precision.value
        assert row.f1 == report.classification.f1.value
        assert row.false_positive_rate == report.classification.false_positive_rate.value
        assert row.false_intervention_count == report.operations.false_intervention_count
        assert row.time_to_alert_p50_seconds == report.alerts.p50_seconds.value
        assert row.confidence_intervals == evidence_by_arm[row.arm].confidence_intervals.intervals
    assert "synthetic" in scorecard.external_validity_statement.lower()
    assert "not evidence" in scorecard.external_validity_statement.lower()

    loaded = load_evaluation_bundle(bundle.bundle_ref(), artifact_store=store, verifier=verifier)
    assert loaded == bundle
    assert loaded.scorecard(artifact_store=store, verifier=verifier) == scorecard


def test_signed_replay_lineage_binds_the_authenticated_corpus(publication) -> None:
    """The signed replay descriptor must bind content, not only the split."""
    _store, _signer, _verifier, verified, request, *_ = publication
    primary = tuple(
        row
        for row in request.promotion_envelope.combined_batch.results
        if row.evaluation.value == "development"
    )
    assert {row.evaluation_lineage.corpus_digest for row in primary} == {
        verified.corpus.corpus_digest
    }


def test_public_feature_and_threshold_manifests_are_full_safe_projections(publication) -> None:
    """Judge artifacts expose useful schema/operating facts for every feature and arm."""
    store, _signer, _verifier, _verified, _request_value, *_, bundle = publication
    feature_ref = bundle.public_artifacts["feature-manifest.json"]
    threshold_ref = bundle.public_artifacts["thresholds.json"]
    features = json.loads(store.read(feature_ref.as_artifact_ref()))
    thresholds = json.loads(store.read(threshold_ref.as_artifact_ref()))

    assert len(features["features"]) == features["feature_count"] == 48
    assert tuple(row["name"] for row in features["features"]) == tuple(
        features["ordered_feature_names"]
    )
    assert set(features["features"][0]) == {
        "data_quality_behavior",
        "dtype",
        "family",
        "missing_sentinel",
        "name",
        "past_only_state_keys",
        "past_only_window_seconds",
        "provenance_category",
        "rail_applicability",
        "source_path_ids",
    }
    assert [row["arm"] for row in thresholds["arms"]] == [
        "rules_only",
        "gbdt_only",
        "layered_hybrid",
    ]
    assert thresholds["normalized_score_policy_version"] == "clip-open-unit-interval-v1"
    assert all(row["feasible"] is True for row in thresholds["arms"])


def test_privacy_scan_rejects_casefold_and_unicode_equivalent_restricted_ids() -> None:
    """Restricted IDs cannot escape through case or Unicode-equivalent spellings."""
    with pytest.raises(ReportingContractError, match="restricted row identifier"):
        _privacy_scan(b"STRASSE", restricted_identifiers=("Straße".encode(),))
    with pytest.raises(ReportingContractError, match="restricted row identifier"):
        _privacy_scan("Cafe\u0301".encode(), restricted_identifiers=("Café".encode(),))
    with pytest.raises(ReportingContractError, match="restricted row identifier"):
        _privacy_scan(
            b'{"safe":"Caf\\u00e9"}',
            restricted_identifiers=("Café".encode(),),
        )
    with pytest.raises(ReportingContractError, match="restricted row identifier"):
        _privacy_scan(
            b'{"CAF\\u00c9":"safe"}',
            restricted_identifiers=("café".encode(),),
        )


def test_restricted_handles_have_constant_nonrevealing_representations(publication) -> None:
    _store, _signer, _verifier, verified, *_ = publication
    assert repr(verified) == str(verified) == "<restricted verified evaluation inputs>"
    assert repr(verified.corpus) == str(verified.corpus) == (
        "<restricted verified corpus attestation>"
    )
    assert repr(verified.defender) == str(verified.defender) == (
        "<restricted verified defender attestation>"
    )
    for value in (verified, verified.corpus, verified.defender):
        with pytest.raises(TypeError):
            copy.deepcopy(value)
        with pytest.raises(TypeError):
            tuple.__repr__(value)  # type: ignore[arg-type]


def test_publication_is_byte_reproducible_and_excludes_restricted_evidence(
    publication,
) -> None:
    """Catch clocks, paths, row identifiers, or hidden fingerprints in public bytes."""
    store, signer, verifier, verified, request, _, scorecard, bundle = publication
    repeated_scorecard, repeated_bundle = publish_scorecard(
        request,
        verified_inputs=verified,
        artifact_store=store,
        signer=signer,
        publication_verifier=verifier,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=HIDDEN_VERIFIER,
    )

    assert repeated_scorecard.to_json() == scorecard.to_json()
    assert repeated_bundle.to_json() == bundle.to_json()
    hidden_fingerprints = {
        row.result_digest.encode()
        for row in request.promotion_envelope.combined_batch.results
        if row.evaluation.kind.value == "hidden"
    }
    hidden_fingerprints.add(request.champion_decision.decision_digest.encode())
    hidden_fingerprints.add(request.promotion_envelope.envelope_digest.encode())
    hidden_fingerprints.update(
        evidence.metric_report.report_digest.encode() for evidence in request.metric_evidence
    )
    for _name, reference in bundle.public_artifacts.items():
        payload = store.read(reference.as_artifact_ref())
        lowered = payload.lower()
        representation = repr(payload).lower().encode()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            assert token.encode() not in lowered
            assert token.encode() not in representation
        assert hidden_fingerprints.isdisjoint(payload.split(b'"'))
        assert len(payload) <= reference.size_bytes
    assert b"evaluated_result_digests" not in scorecard.to_json()
    assert b"decision_digest" not in scorecard.to_json()
    assert request.promotion_envelope.envelope_digest.encode() not in scorecard.to_json()
    for evidence in request.metric_evidence:
        assert evidence.metric_report.report_digest.encode() not in scorecard.to_json()


def test_latency_observation_changes_do_not_change_reproducible_core(
    bundle_fixture,
) -> None:
    """Catch observational latency values or their digests entering the core identity."""
    store = bundle_fixture.store
    signer = bundle_fixture.signer
    verifier = PublicArtifactVerifier.from_signer(signer)
    verified, _corpus_ref, defender_ref, threshold_set = _authenticated_inputs(bundle_fixture)
    baseline, _ = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        corpus_digest=verified.corpus.corpus_digest,
        split_digest=verified.corpus.split_digest,
        threshold_set=threshold_set,
        latency_multiplier=1.0,
    )
    changed, _ = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        corpus_digest=verified.corpus.corpus_digest,
        split_digest=verified.corpus.split_digest,
        threshold_set=threshold_set,
        latency_multiplier=2.0,
    )
    first, first_bundle = publish_scorecard(
        baseline,
        verified_inputs=verified,
        artifact_store=store,
        signer=signer,
        publication_verifier=verifier,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=HIDDEN_VERIFIER,
    )
    second, second_bundle = publish_scorecard(
        changed,
        verified_inputs=verified,
        artifact_store=store,
        signer=signer,
        publication_verifier=verifier,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=HIDDEN_VERIFIER,
    )

    assert first.evaluation_id == second.evaluation_id
    assert first.core_digest == second.core_digest
    assert first.leaderboard == second.leaderboard
    first_refs = dict(first_bundle.public_artifacts.items())
    second_refs = dict(second_bundle.public_artifacts.items())
    assert first_refs["latency-evidence.json"] != second_refs["latency-evidence.json"]
    assert {
        name: reference
        for name, reference in first_refs.items()
        if name not in {"latency-evidence.json", SCORECARD_ARTIFACT_NAME}
    } == {
        name: reference
        for name, reference in second_refs.items()
        if name not in {"latency-evidence.json", SCORECARD_ARTIFACT_NAME}
    }


def test_scorecard_and_bundle_reject_rechecksummed_tamper_and_artifact_substitution(
    publication,
) -> None:
    """Catch self-digest recomputation or same-shape artifact substitution."""
    store, _signer, verifier, _verified, _, _, scorecard, bundle = publication
    document = json.loads(scorecard.to_json())
    document["external_validity_statement"] = "Synthetic evidence proves live performance."
    document["core_digest"] = DefenseScorecard.digest_core_document(document)
    with pytest.raises((ReportingContractError, ValueError)):
        DefenseScorecard.from_json(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode(),
            artifact_store=store,
            verifier=verifier,
        )

    entries = list(bundle.public_artifacts.entries)
    leaderboard_index = next(
        index for index, item in enumerate(entries) if item.name == "leaderboard.csv"
    )
    entries[leaderboard_index] = replace(
        entries[leaderboard_index],
        sha256=next(item.sha256 for item in entries if item.name == "slice-metrics.csv"),
    )
    forged = EvaluationArtifactBundle.model_construct(
        **{
            **bundle.model_dump(mode="python", exclude={"public_artifacts"}),
            "public_artifacts": bundle.public_artifacts.model_construct(entries=tuple(entries)),
        }
    )
    with pytest.raises(ReportingContractError):
        forged.to_json()


def test_publication_requires_verified_task11_and_task12_evidence(bundle_fixture) -> None:
    """Catch raw replay matrices or caller-rechecksummed aggregate metrics."""
    store = bundle_fixture.store
    signer = bundle_fixture.signer
    verifier = PublicArtifactVerifier.from_signer(signer)
    verified, _corpus_ref, defender_ref, threshold_set = _authenticated_inputs(bundle_fixture)
    request, _ = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        corpus_digest=verified.corpus.corpus_digest,
        split_digest=verified.corpus.split_digest,
        threshold_set=threshold_set,
    )
    report_evidence = request.metric_evidence[0]
    forged_report = MetricReport.model_construct(
        **{
            **report_evidence.metric_report.model_dump(mode="python"),
            "derivation_evidence_digest": "a" * 64,
        }
    )
    forged_metric = MetricPublicationEvidence.model_construct(
        **{
            **report_evidence.model_dump(mode="python", exclude={"metric_report"}),
            "metric_report": forged_report,
        }
    )
    forged_request = ScorecardPublicationRequest.model_construct(
        **{
            **request.model_dump(mode="python", exclude={"metric_evidence"}),
            "metric_evidence": (forged_metric, *request.metric_evidence[1:]),
        }
    )
    with pytest.raises(ReportingContractError):
        publish_scorecard(
            forged_request,
            verified_inputs=verified,
            artifact_store=store,
            signer=signer,
            publication_verifier=verifier,
            evaluator_verifier=EVALUATOR_VERIFIER,
            hidden_proof_verifier=HIDDEN_VERIFIER,
        )

    with pytest.raises(ValidationError):
        request.rebuild(
            latency_environment={
                "schema_version": "1.0.0",
                "hostname": "judge-laptop",
                "path": "/Users/example/private-run-state",
            }
        )

    with pytest.raises(ValidationError):
        request.rebuild(
            feature_manifest={
                "schema_version": "1.0.0",
                "marketing_claim": "caller supplied",
            }
        )

    other_verifier = EvaluatorReplayVerifier.from_signer(
        EvaluatorSigningIdentity.from_private_bytes(b"x" * 32)
    )
    with pytest.raises(ReportingContractError):
        publish_scorecard(
            request,
            verified_inputs=verified,
            artifact_store=store,
            signer=signer,
            publication_verifier=verifier,
            evaluator_verifier=other_verifier,
            hidden_proof_verifier=HIDDEN_VERIFIER,
        )


def test_public_artifact_index_rejects_aliases_and_is_intrinsically_immutable(
    publication,
) -> None:
    """Catch duplicate, Unicode-normalized, casefold, or mutable name maps."""
    _, _, _, _, _, _, scorecard, _ = publication
    with pytest.raises(TypeError):
        scorecard.public_artifacts.entries[0] = scorecard.public_artifacts.entries[0]
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        scorecard.public_artifacts.entries += (scorecard.public_artifacts.entries[0],)
    assert copy.copy(scorecard).to_json() == scorecard.to_json()
    with pytest.raises((ReportingContractError, ValueError)):
        scorecard.public_artifacts.with_entry(
            replace(
                scorecard.public_artifacts.entries[0],
                name=scorecard.public_artifacts.entries[0].name.upper(),
            )
        )
