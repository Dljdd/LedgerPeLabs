"""Canonical, signed, privacy-safe judge reporting contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from apar.evaluation.gates import (
    CandidateRoleEvidence,
    ChampionDecision,
    EvaluationLineage,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    PromotionMetrics,
    RateEvidence,
    SlicePerformance,
)
from apar.evaluation.metrics import (
    BootstrapDerivationEvidence,
    MetricDerivationEvidence,
    MetricReport,
    campaign_bootstrap,
    compute_metric_report,
)
from apar.evaluation.reporting import (
    PUBLIC_ARTIFACT_MEDIA_TYPES,
    SCORECARD_ARTIFACT_NAME,
    DefenseScorecard,
    EvaluationArtifactBundle,
    MetricPublicationEvidence,
    ReportingContractError,
    ScorecardPublicationRequest,
    load_evaluation_bundle,
    publish_scorecard,
)
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactStore
from tests.evaluation.test_gates import (
    EVALUATOR_SIGNER,
    EVALUATOR_VERIFIER,
    _evaluate,
    _promotion_envelope,
    _results,
)
from tests.evaluation.test_metrics_classification import four_row_inputs

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
    corpus_digest: str = "f" * 64,
    defender_digest: str = "7" * 64,
    evaluator_signer: EvaluatorSigningIdentity = EVALUATOR_SIGNER,
    evaluator_verifier: EvaluatorReplayVerifier = EVALUATOR_VERIFIER,
) -> tuple[ScorecardPublicationRequest, ChampionDecision]:
    inputs = four_row_inputs()
    report = compute_metric_report(inputs)
    metric_evidence = MetricDerivationEvidence.from_inputs(inputs)
    confidence = campaign_bootstrap(inputs)
    bootstrap_evidence = BootstrapDerivationEvidence.from_inputs(inputs)
    metrics = _promotion_metrics(report)
    decision_ids = tuple(row.event_id for row in inputs.truth)
    decision_rows_digest = hashlib.sha256(
        canonical_json_bytes(list(decision_ids))
    ).hexdigest()
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
        role_fields = row.candidate_role.model_dump(
            mode="python", exclude={"role_digest"}
        )
        if non_held:
            lineage_fields["bundle_manifest_digest"] = defender_digest
            lineage_fields["defender_top_ref_digest"] = defender_digest
            role_fields["bundle_manifest_digest"] = defender_digest
            role_fields["defender_top_ref_digest"] = defender_digest
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
            ),
        )
    results = tuple(rebuilt_results)
    envelope = _promotion_envelope(results, signer=evaluator_signer)
    decision = _evaluate(results)
    primary = tuple(
        row
        for row in envelope.combined_batch.results
        if row.evaluation.value == "development"
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
        corpus_artifact_digest=corpus_digest,
        defender_artifact_digest=defender_digest,
        promotion_envelope=envelope,
        champion_decision=decision,
        metric_evidence=evidence,
        feature_manifest={
            "schema_version": "1.0.0",
            "feature_count": 48,
            "availability": "strictly past-only",
        },
        thresholds={
            "schema_version": "1.0.0",
            "matched_budgets": True,
            "challenge_rate_max": 0.02,
        },
        latency_environment={
            "schema_version": "1.0.0",
            "python": "3.12.5",
            "platform": "fixture",
        },
    )
    assert evaluator_verifier.verify_promotion_envelope(envelope)
    return request, decision


@pytest.fixture
def publication(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"r" * 32)
    request, decision = _request()
    scorecard, bundle = publish_scorecard(
        request,
        artifact_store=store,
        signer=signer,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=EVALUATOR_VERIFIER,
    )
    return store, signer, request, decision, scorecard, bundle


def test_every_displayed_metric_resolves_to_an_allowlisted_signed_artifact(
    publication,
) -> None:
    """Catch untraceable judge metrics or a scorecard self-reference."""
    store, signer, request, _, scorecard, bundle = publication

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

    loaded = load_evaluation_bundle(
        bundle.bundle_ref(), artifact_store=store, signer=signer
    )
    assert loaded == bundle
    assert loaded.scorecard(artifact_store=store, signer=signer) == scorecard


def test_publication_is_byte_reproducible_and_excludes_restricted_evidence(
    publication,
) -> None:
    """Catch clocks, paths, row identifiers, or hidden fingerprints in public bytes."""
    store, signer, request, _, scorecard, bundle = publication
    repeated_scorecard, repeated_bundle = publish_scorecard(
        request,
        artifact_store=store,
        signer=signer,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=EVALUATOR_VERIFIER,
    )

    assert repeated_scorecard.to_json() == scorecard.to_json()
    assert repeated_bundle.to_json() == bundle.to_json()
    hidden_fingerprints = {
        row.result_digest.encode()
        for row in request.promotion_envelope.combined_batch.results
        if row.evaluation.kind.value == "hidden"
    }
    hidden_fingerprints.add(request.champion_decision.decision_digest.encode())
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


def test_scorecard_and_bundle_reject_rechecksummed_tamper_and_artifact_substitution(
    publication,
) -> None:
    """Catch self-digest recomputation or same-shape artifact substitution."""
    store, signer, _, _, scorecard, bundle = publication
    document = json.loads(scorecard.to_json())
    document["external_validity_statement"] = "Synthetic evidence proves live performance."
    document["core_digest"] = DefenseScorecard.digest_core_document(document)
    with pytest.raises((ReportingContractError, ValueError)):
        DefenseScorecard.from_json(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode(),
            artifact_store=store,
            signer=signer,
        )

    entries = list(bundle.public_artifacts.entries)
    leaderboard_index = next(
        index for index, item in enumerate(entries) if item.name == "leaderboard.csv"
    )
    entries[leaderboard_index] = replace(
        entries[leaderboard_index],
        sha256=next(
            item.sha256 for item in entries if item.name == "slice-metrics.csv"
        ),
    )
    forged = EvaluationArtifactBundle.model_construct(
        **{
            **bundle.model_dump(mode="python", exclude={"public_artifacts"}),
            "public_artifacts": bundle.public_artifacts.model_construct(
                entries=tuple(entries)
            ),
        }
    )
    with pytest.raises(ReportingContractError):
        forged.to_json()


def test_publication_requires_verified_task11_and_task12_evidence(tmp_path: Path) -> None:
    """Catch raw replay matrices or caller-rechecksummed aggregate metrics."""
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"r" * 32)
    request, _ = _request()
    report_evidence = request.metric_evidence[0]
    forged_report = MetricReport.model_construct(
        **{
            **report_evidence.metric_report.model_dump(mode="python"),
            "derivation_evidence_digest": "a" * 64,
        }
    )
    forged_metric = MetricPublicationEvidence.model_construct(
        **{
            **report_evidence.model_dump(
                mode="python", exclude={"metric_report"}
            ),
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
            artifact_store=store,
            signer=signer,
            evaluator_verifier=EVALUATOR_VERIFIER,
            hidden_proof_verifier=EVALUATOR_VERIFIER,
        )

    with pytest.raises(ValidationError):
        request.rebuild(
            latency_environment={
                "schema_version": "1.0.0",
                "hostname": "judge-laptop",
                "path": "/Users/example/private-run-state",
            }
        )

    other_verifier = EvaluatorReplayVerifier.from_signer(
        EvaluatorSigningIdentity.from_private_bytes(b"x" * 32)
    )
    with pytest.raises(ReportingContractError):
        publish_scorecard(
            request,
            artifact_store=store,
            signer=signer,
            evaluator_verifier=other_verifier,
            hidden_proof_verifier=EVALUATOR_VERIFIER,
        )


def test_public_artifact_index_rejects_aliases_and_is_intrinsically_immutable(
    publication,
) -> None:
    """Catch duplicate, Unicode-normalized, casefold, or mutable name maps."""
    _, _, _, _, scorecard, _ = publication
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
