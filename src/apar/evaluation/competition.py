"""Evaluation-owned Task12/Task13 orchestration for the Task14 G3 gate."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pydantic import field_validator, model_validator

from apar.cases import group_cases, simulate_case_queue
from apar.contracts._validation import ExternalContract
from apar.defense.bundle import DefenderBundlePublisher, LoadedDefenderBundle
from apar.defense.contracts import ObservedEvent
from apar.defense.rules import RuleEngine
from apar.evaluation.contracts import (
    CorpusManifest,
    Family,
    FrozenCorpus,
)
from apar.evaluation.defender_attestation import (
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.evaluation.gates import (
    DefenseArm,
    EvaluationDescriptor,
    EvaluationKind,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    GateConfig,
    HiddenPublicProof,
    VerifiedPromotionEnvelope,
    VerifiedReplayBatch,
    evaluate_promotion_gates,
)
from apar.evaluation.metrics import (
    BootstrapDerivationEvidence,
    LatencySample,
    MetricDerivationEvidence,
    MetricReportInputs,
    SliceAssignment,
    SliceManifest,
    campaign_bootstrap,
    compute_metric_report,
)
from apar.evaluation.publication_inputs import (
    CORPUS_ATTESTATION_MEDIA_TYPE,
    CORPUS_EVIDENCE_MEDIA_TYPE,
    load_corpus_attestation,
    publish_corpus_attestation,
    verify_evaluation_inputs,
)
from apar.evaluation.regimes import (
    RegimeKind,
    RegimeSpec,
    derive_regime,
    frozen_corpus_digest,
)
from apar.evaluation.replay import (
    ReplayCaseCounterBinding,
    ReplayCorpusEvidence,
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
    ReplayRegimeEvidence,
    ReplayThresholdSet,
    _freeze_replay_inputs,
    _HiddenReplayInvocation,
    bind_replay_case_counter,
    replay_defense_arms,
    replay_empty_cold_entity,
)
from apar.evaluation.reporting import (
    RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
    DefenseScorecard,
    EvaluationArtifactBundle,
    MetricPublicationEvidence,
    PublicArtifactVerifier,
    PublicChampionDecision,
    ScorecardPublicationRequest,
    load_evaluation_bundle,
    publish_scorecard,
    store_restricted_publication_receipt,
)
from apar.evaluation.splits import (
    EntityCohort,
    EvaluationSplit,
    make_evaluation_split,
    make_leave_one_family_out,
)
from apar.evaluation_hidden.authority_core import HiddenReplayOutcome
from apar.evaluation_hidden.defense_authority import (
    HIDDEN_CONTEXT_MEDIA_TYPE,
    HiddenEvaluationAuthority,
)
from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.parity import assert_online_offline_parity, audit_feature_matrix
from apar.runs import RunSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_EVALUATOR_SEED = hashlib.sha256(b"apar-g3-development-evaluator-v1").digest()
_HIDDEN_SEED = hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest()
_FIXTURE_PROFILE_ID = "development-fixture-v1+signed-control-v1"
_HIDDEN_ENVELOPE_MEDIA_TYPE = "application/vnd.apar.restricted-hidden-context-envelope+json"
_HIDDEN_OBSERVATIONS_MEDIA_TYPE = "application/vnd.apar.hidden-observations+json"
_DEVELOPMENT_COMPLETION_MEDIA_TYPE = "application/vnd.apar.development-completion+json"
_DEVELOPMENT_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.apar.restricted-development-evidence+json"
)
_ENSEMBLE_MEDIA_TYPE = "application/vnd.apar.defender-ensemble+json"
_CORPUS_ENVELOPE_MEDIA_TYPE = "application/vnd.apar.corpus-envelope+json"
_SCORECARD_MEDIA_TYPE = "application/vnd.apar.defense-scorecard+json"
_EVALUATION_BUNDLE_MEDIA_TYPE = "application/vnd.apar.evaluation-artifact-bundle+json"
_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
_DEVELOPMENT_DESCRIPTOR_SCOPE = tuple(
    sorted(
        (
            "chronological:development",
            *(f"cold_entity:{item.value}" for item in EntityCohort),
            *(f"held_family:{family}" for family in _FAMILIES),
            *(f"regime:{item.value}" for item in RegimeKind),
        )
    )
)


def _digest(value: str, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "media_type": ref.media_type,
        "relative_path": ref.relative_path,
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
    }


def _ref_from_document(
    value: object, *, media_type: str | None = None, max_bytes: int = 128_000_000
) -> ArtifactRef:
    if type(value) is not dict or set(value) != {
        "media_type",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("artifact reference fields differ")
    document = cast(dict[str, object], value)
    digest = _digest(cast(str, document["sha256"]), label="artifact reference")
    size = document["size_bytes"]
    actual_media_type = document["media_type"]
    if (
        type(actual_media_type) is not str
        or (media_type is not None and actual_media_type != media_type)
        or document["relative_path"] != f"{digest}/payload"
        or type(size) is not int
        or not 0 < size <= max_bytes
    ):
        raise ValueError("artifact reference is invalid")
    return ArtifactRef(digest, actual_media_type, size, f"{digest}/payload")


class RestrictedHiddenContextEnvelope(ExternalContract):
    """Authority-signed pointer to one independently sealed hidden context."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    phase: Literal["hidden"] = "hidden"
    profile_sha256: str
    development_corpus_digest: str
    observations_ref: dict[str, object]
    observations_sha256: str
    restricted_context_ref: dict[str, object]
    context_sha256: str
    as_of: datetime
    assurance_mode: Literal["production_measured"] = "production_measured"
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator(
        "profile_sha256",
        "development_corpus_digest",
        "context_sha256",
        "observations_sha256",
        "signer_key_id",
    )
    @classmethod
    def digests_are_exact(cls, value: str) -> str:
        return _digest(value, label="hidden context envelope digest")

    @model_validator(mode="after")
    def context_reference_is_exact(self) -> RestrictedHiddenContextEnvelope:
        observations = _ref_from_document(
            self.observations_ref, media_type=_HIDDEN_OBSERVATIONS_MEDIA_TYPE
        )
        restricted = _ref_from_document(
            self.restricted_context_ref, media_type=HIDDEN_CONTEXT_MEDIA_TYPE
        )
        if (
            observations.sha256 != self.observations_sha256
            or restricted.sha256 != self.context_sha256
            or observations.sha256 == restricted.sha256
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() != timedelta(0)
        ):
            raise ValueError("hidden context digest differs from its reference")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


class DevelopmentCompletionReceipt(ExternalContract):
    """Signed development-only release prerequisite for hidden evaluation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    phase: Literal["development"] = "development"
    ensemble_ref: dict[str, object]
    profile_sha256: str
    corpus_envelope_ref: dict[str, object]
    run_ledger_sha256: str
    scorecard_ref: dict[str, object]
    evaluation_bundle_ref: dict[str, object]
    development_evidence_ref: dict[str, object]
    restricted_publication_receipt_ref: dict[str, object]
    promotion_envelope_digest: str
    descriptor_scope: tuple[str, ...]
    hidden_included: Literal[False] = False
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator(
        "profile_sha256",
        "run_ledger_sha256",
        "promotion_envelope_digest",
        "signer_key_id",
    )
    @classmethod
    def receipt_digests_are_exact(cls, value: str) -> str:
        return _digest(value, label="development completion digest")

    @field_validator("descriptor_scope", mode="before")
    @classmethod
    def scope_is_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("development descriptor scope must be an exact tuple")
        return value

    @model_validator(mode="after")
    def receipt_references_are_closed(self) -> DevelopmentCompletionReceipt:
        _ref_from_document(self.ensemble_ref, media_type=_ENSEMBLE_MEDIA_TYPE)
        _ref_from_document(
            self.corpus_envelope_ref, media_type=_CORPUS_ENVELOPE_MEDIA_TYPE
        )
        _ref_from_document(self.scorecard_ref, media_type=_SCORECARD_MEDIA_TYPE)
        _ref_from_document(
            self.evaluation_bundle_ref, media_type=_EVALUATION_BUNDLE_MEDIA_TYPE
        )
        _ref_from_document(
            self.development_evidence_ref,
            media_type=_DEVELOPMENT_EVIDENCE_MEDIA_TYPE,
            max_bytes=128 * 1024 * 1024,
        )
        _ref_from_document(
            self.restricted_publication_receipt_ref,
            media_type=RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
        )
        if (
            self.descriptor_scope != _DEVELOPMENT_DESCRIPTOR_SCOPE
            or any(item.startswith("hidden:") for item in self.descriptor_scope)
        ):
            raise ValueError("development receipt hidden descriptor scope is invalid")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


def seal_hidden_context(
    *,
    store: ArtifactStore,
    signer: RunSigningIdentity,
    context: ReplayEvaluationContext,
    profile_sha256: str,
    development_corpus_digest: str,
) -> ArtifactRef:
    """Seal an authority-owned hidden context behind a signed restricted pointer."""
    if (
        type(store) is not ArtifactStore
        or type(signer) is not RunSigningIdentity
        or type(context) is not ReplayEvaluationContext
        or context.evaluation.kind is not EvaluationKind.HIDDEN
        or context.evaluation.value != "hidden"
    ):
        raise ValueError("hidden context sealing inputs are invalid")
    _digest(profile_sha256, label="profile digest")
    _digest(development_corpus_digest, label="development corpus digest")
    context_ref = store.put_bytes(context.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE)
    observations_payload = canonical_json_bytes(
        [row.model_dump(mode="json") for row in context.observations]
    )
    observations_ref = store.put_bytes(
        observations_payload, _HIDDEN_OBSERVATIONS_MEDIA_TYPE
    )
    unsigned = {
        "as_of": context.as_of.isoformat().replace("+00:00", "Z"),
        "assurance_mode": "production_measured",
        "context_sha256": context_ref.sha256,
        "development_corpus_digest": development_corpus_digest,
        "observations_ref": _ref_document(observations_ref),
        "observations_sha256": observations_ref.sha256,
        "phase": "hidden",
        "profile_sha256": profile_sha256,
        "public_key_base64": signer.public_key_base64,
        "restricted_context_ref": _ref_document(context_ref),
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    envelope = RestrictedHiddenContextEnvelope.model_validate(
        {**unsigned, "signature_base64": signer.sign(unsigned)}
    )
    return store.put_bytes(
        canonical_json_bytes(envelope.model_dump(mode="json")),
        _HIDDEN_ENVELOPE_MEDIA_TYPE,
    )


def verify_hidden_context(
    *,
    store: ArtifactStore,
    envelope_ref: ArtifactRef,
    signer: RunSigningIdentity,
    profile_sha256: str,
    development_corpus_digest: str,
    development_event_ids: tuple[str, ...],
) -> tuple[
    RestrictedHiddenContextEnvelope, tuple[ObservedEvent, ...], ArtifactRef
]:
    """Authenticate hidden metadata and truth-blind observations without opening truth."""
    if (
        type(store) is not ArtifactStore
        or type(envelope_ref) is not ArtifactRef
        or envelope_ref.media_type != _HIDDEN_ENVELOPE_MEDIA_TYPE
        or type(signer) is not RunSigningIdentity
        or type(development_event_ids) is not tuple
    ):
        raise ValueError("hidden context reference is invalid")
    payload = store.read(envelope_ref)
    try:
        document = strict_json_loads(payload)
        if type(document) is not dict:
            raise ValueError("hidden envelope must be an object")
        envelope = RestrictedHiddenContextEnvelope.model_validate(document)
    except (WireContractError, ValueError) as error:
        raise ValueError("hidden context envelope is invalid") from error
    if canonical_json_bytes(envelope.model_dump(mode="json")) != payload:
        raise ValueError("hidden context envelope must be canonical")
    if (
        envelope.profile_sha256 != profile_sha256
        or envelope.development_corpus_digest != development_corpus_digest
    ):
        raise ValueError("hidden context profile or corpus lineage differs")
    if (
        envelope.signer_key_id != signer.key_id
        or envelope.public_key_base64 != signer.public_key_base64
        or not signer.verify(envelope.unsigned_document(), envelope.signature_base64)
    ):
        raise ValueError("hidden context signer identity differs")
    observations_ref = _ref_from_document(
        envelope.observations_ref, media_type=_HIDDEN_OBSERVATIONS_MEDIA_TYPE
    )
    observation_payload = store.read(observations_ref)
    try:
        rows = strict_json_loads(observation_payload)
        if type(rows) is not list:
            raise ValueError("hidden observations must be a list")
        observations = tuple(ObservedEvent.model_validate(row) for row in rows)
    except (WireContractError, ValueError) as error:
        raise ValueError("hidden observations are invalid") from error
    if canonical_json_bytes(
        [row.model_dump(mode="json") for row in observations]
    ) != observation_payload:
        raise ValueError("hidden observations must be canonical")
    hidden_ids = {row.event_id for row in observations}
    if (
        not observations
        or len(hidden_ids) != len(observations)
        or hidden_ids.intersection(development_event_ids)
        or envelope.context_sha256 == development_corpus_digest
    ):
        raise ValueError("hidden context is not independent from development")
    restricted_ref = _ref_from_document(
        envelope.restricted_context_ref, media_type=HIDDEN_CONTEXT_MEDIA_TYPE
    )
    return envelope, observations, restricted_ref


def seal_development_completion(
    *,
    store: ArtifactStore,
    signer: RunSigningIdentity,
    ensemble_ref: ArtifactRef,
    profile_sha256: str,
    corpus_envelope_ref: ArtifactRef,
    run_ledger_sha256: str,
    scorecard_ref: ArtifactRef,
    evaluation_bundle_ref: ArtifactRef,
    development_evidence_ref: ArtifactRef,
    restricted_publication_receipt_ref: ArtifactRef,
    promotion_envelope_digest: str,
    descriptor_scope: tuple[str, ...],
) -> ArtifactRef:
    """Publish an immutable signed development-only hidden-release receipt."""
    unsigned = {
        "corpus_envelope_ref": _ref_document(corpus_envelope_ref),
        "descriptor_scope": list(tuple(sorted(descriptor_scope))),
        "ensemble_ref": _ref_document(ensemble_ref),
        "evaluation_bundle_ref": _ref_document(evaluation_bundle_ref),
        "development_evidence_ref": _ref_document(development_evidence_ref),
        "hidden_included": False,
        "phase": "development",
        "profile_sha256": profile_sha256,
        "promotion_envelope_digest": promotion_envelope_digest,
        "public_key_base64": signer.public_key_base64,
        "restricted_publication_receipt_ref": _ref_document(
            restricted_publication_receipt_ref
        ),
        "run_ledger_sha256": run_ledger_sha256,
        "schema_version": "1.0.0",
        "scorecard_ref": _ref_document(scorecard_ref),
        "signer_key_id": signer.key_id,
    }
    receipt = DevelopmentCompletionReceipt.model_validate(
        {
            **unsigned,
            "descriptor_scope": tuple(cast(list[str], unsigned["descriptor_scope"])),
            "signature_base64": signer.sign(unsigned),
        }
    )
    return store.put_bytes(
        canonical_json_bytes(receipt.model_dump(mode="json")),
        _DEVELOPMENT_COMPLETION_MEDIA_TYPE,
    )


def verify_development_completion(
    *,
    store: ArtifactStore,
    receipt_ref: ArtifactRef,
    signer: RunSigningIdentity,
    ensemble_ref: ArtifactRef,
    profile_sha256: str,
    corpus_envelope_ref: ArtifactRef,
    run_ledger_sha256: str,
    evaluator_verifier: EvaluatorReplayVerifier,
    pooled_ref: ArtifactRef,
    held_family_refs: dict[Family, ArtifactRef],
    split: EvaluationSplit,
) -> DevelopmentCompletionReceipt:
    """Verify the exact development-only receipt before hidden refs can resolve."""
    if receipt_ref.media_type != _DEVELOPMENT_COMPLETION_MEDIA_TYPE:
        raise ValueError("development completion reference media type differs")
    payload = store.read(receipt_ref)
    try:
        document = strict_json_loads(payload)
        if type(document) is not dict:
            raise ValueError("development completion must be an object")
        if document.get("hidden_included") is not False or document.get("phase") != "development":
            raise ValueError("development completion hidden evidence is forbidden")
        for field, media_type in (
            ("ensemble_ref", _ENSEMBLE_MEDIA_TYPE),
            ("corpus_envelope_ref", _CORPUS_ENVELOPE_MEDIA_TYPE),
            ("scorecard_ref", _SCORECARD_MEDIA_TYPE),
            ("evaluation_bundle_ref", _EVALUATION_BUNDLE_MEDIA_TYPE),
            ("development_evidence_ref", _DEVELOPMENT_EVIDENCE_MEDIA_TYPE),
            (
                "restricted_publication_receipt_ref",
                RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
            ),
        ):
            try:
                _ref_from_document(document.get(field), media_type=media_type)
            except ValueError as error:
                raise ValueError(
                    "development completion reference is invalid"
                ) from error
        if type(document.get("descriptor_scope")) is list:
            document["descriptor_scope"] = tuple(document["descriptor_scope"])
        receipt = DevelopmentCompletionReceipt.model_validate(document)
    except WireContractError as error:
        raise ValueError("development completion receipt is invalid") from error
    if canonical_json_bytes(receipt.model_dump(mode="json")) != payload:
        raise ValueError("development completion receipt must be canonical")
    if (
        _ref_from_document(receipt.ensemble_ref) != ensemble_ref
        or _ref_from_document(receipt.corpus_envelope_ref) != corpus_envelope_ref
    ):
        raise ValueError("development completion reference lineage differs")
    if receipt.profile_sha256 != profile_sha256:
        raise ValueError("development completion profile lineage differs")
    if receipt.run_ledger_sha256 != run_ledger_sha256:
        raise ValueError("development completion run ledger lineage differs")
    if (
        receipt.signer_key_id != signer.key_id
        or receipt.public_key_base64 != signer.public_key_base64
        or not signer.verify(receipt.unsigned_document(), receipt.signature_base64)
    ):
        raise ValueError("development completion signer identity differs")
    scorecard_ref = _ref_from_document(
        receipt.scorecard_ref, media_type=_SCORECARD_MEDIA_TYPE
    )
    bundle_ref = _ref_from_document(
        receipt.evaluation_bundle_ref, media_type=_EVALUATION_BUNDLE_MEDIA_TYPE
    )
    development_evidence_ref = _ref_from_document(
        receipt.development_evidence_ref,
        media_type=_DEVELOPMENT_EVIDENCE_MEDIA_TYPE,
        max_bytes=128 * 1024 * 1024,
    )
    development_request = ScorecardPublicationRequest.from_worker_json(
        store.read(development_evidence_ref)
    )
    if (
        development_request.promotion_envelope.envelope_digest
        != receipt.promotion_envelope_digest
        or tuple(
            sorted(
                f"{batch.results[0].evaluation.kind.value}:"
                f"{batch.results[0].evaluation.value}"
                for batch in development_request.promotion_envelope.component_batches
            )
        )
        != receipt.descriptor_scope
        or development_request.promotion_envelope.hidden_proofs
    ):
        raise ValueError("development evidence lineage differs")
    defender_verifier = DefenderBundleVerifier(
        store,
        signer_key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
    )
    pooled = _candidate_runtime(
        store=store,
        publication_signer=signer,
        defender_verifier=defender_verifier,
        reference=pooled_ref,
        split=split,
    )
    lofo = {
        family: _candidate_runtime(
            store=store,
            publication_signer=signer,
            defender_verifier=defender_verifier,
            reference=held_family_refs[family],
            split=make_leave_one_family_out(split, family),
        )
        for family in _FAMILIES
    }
    _verify_frozen_development_request(
        development_request,
        evaluator_verifier=evaluator_verifier,
        pooled=pooled,
        lofo=lofo,
    )
    verifier = PublicArtifactVerifier.from_signer(signer)
    scorecard = DefenseScorecard.from_json(
        store.read(scorecard_ref), artifact_store=store, verifier=verifier
    )
    bundle = load_evaluation_bundle(
        bundle_ref, artifact_store=store, verifier=verifier
    )
    if (
        bundle.scorecard_sha256 != scorecard_ref.sha256
        or bundle.evaluation_id != scorecard.evaluation_id
    ):
        raise ValueError("development completion public bundle lineage differs")
    restricted_ref = _ref_from_document(
        receipt.restricted_publication_receipt_ref,
        media_type=RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
    )
    restricted_payload = store.read(restricted_ref)
    try:
        restricted = strict_json_loads(restricted_payload)
        if type(restricted) is not dict:
            raise ValueError("restricted publication receipt must be an object")
        if canonical_json_bytes(restricted) != restricted_payload:
            raise ValueError("restricted publication receipt must be canonical")
        expected_fields = {
            "schema_version",
            "privacy_classification",
            "evaluation_id",
            "corpus_attestation_ref",
            "corpus_evidence_ref",
            "corpus_content_digest",
            "split_digest",
            "defender_attestation",
            "promotion_envelope_digest",
            "champion_decision_digest",
            "threshold_set_digest",
            "threshold_set",
            "metric_lineage",
            "public_bundle_digest",
            "public_bundle_ref",
            "signer_key_id",
            "public_key_base64",
            "signature_base64",
            "receipt_digest",
        }
        if set(restricted) != expected_fields:
            raise ValueError("restricted publication receipt fields differ")
        receipt_digest = cast(dict[str, object], restricted).get("receipt_digest")
        signed = {
            key: value
            for key, value in cast(dict[str, object], restricted).items()
            if key != "receipt_digest"
        }
        signature = signed.pop("signature_base64", None)
        corpus_attestation_ref = _ref_from_document(
            signed.get("corpus_attestation_ref"),
            media_type=CORPUS_ATTESTATION_MEDIA_TYPE,
        )
        verified_corpus = load_corpus_attestation(
            corpus_attestation_ref,
            artifact_store=store,
            verifier=evaluator_verifier,
        )
        primary = tuple(
            row
            for row in development_request.promotion_envelope.combined_batch.results
            if row.evaluation.kind is EvaluationKind.CHRONOLOGICAL
            and row.evaluation.value == "development"
        )
        if tuple(row.arm for row in primary) != tuple(DefenseArm):
            raise ValueError("development primary evidence is incomplete")
        primary_corpus_digests = {row.evaluation_lineage.corpus_digest for row in primary}
        primary_split_digests = {row.evaluation_lineage.split_digest for row in primary}
        expected_metric_lineage = [
            {
                "arm": row.arm.value,
                "result_digest": row.result_digest,
                "metric_report_digest": row.metric_report.report_digest,
                "metric_evidence_digest": cast(
                    MetricDerivationEvidence, row.metric_derivation_evidence
                ).evidence_digest,
                "confidence_intervals_digest": row.confidence_intervals.intervals_digest,
                "bootstrap_evidence_digest": cast(
                    BootstrapDerivationEvidence, row.bootstrap_derivation_evidence
                ).evidence_digest,
            }
            for row in development_request.metric_evidence
        ]
        if (
            receipt_digest
            != hashlib.sha256(
                canonical_json_bytes(
                    {**signed, "signature_base64": signature}
                )
            ).hexdigest()
            or type(signature) is not str
            or not signer.verify(signed, signature)
            or signed.get("schema_version") != "1.0.0"
            or signed.get("privacy_classification")
            != "restricted_evaluation_evidence"
            or signed.get("signer_key_id") != signer.key_id
            or signed.get("public_key_base64") != signer.public_key_base64
            or signed.get("promotion_envelope_digest")
            != receipt.promotion_envelope_digest
            or signed.get("champion_decision_digest")
            != development_request.champion_decision.decision_digest
            or signed.get("threshold_set_digest")
            != development_request.threshold_set.threshold_set_digest
            or signed.get("threshold_set")
            != development_request.threshold_set.model_dump(mode="json")
            or signed.get("metric_lineage") != expected_metric_lineage
            or _ref_from_document(
                signed.get("public_bundle_ref"),
                media_type=_EVALUATION_BUNDLE_MEDIA_TYPE,
            )
            != bundle_ref
            or signed.get("public_bundle_digest") != bundle.bundle_digest
            or signed.get("evaluation_id") != scorecard.evaluation_id
            or verified_corpus.top_ref != corpus_attestation_ref
            or len(primary_corpus_digests) != 1
            or verified_corpus.corpus_digest not in primary_corpus_digests
            or primary_split_digests != {split.split_digest}
            or verified_corpus.split_digest != split.split_digest
            or signed.get("corpus_content_digest") != verified_corpus.corpus_digest
            or signed.get("split_digest") != verified_corpus.split_digest
            or _ref_from_document(
                signed.get("corpus_evidence_ref"),
                media_type=CORPUS_EVIDENCE_MEDIA_TYPE,
            )
            != verified_corpus.evidence_ref
            or type(signed.get("defender_attestation")) is not dict
            or canonical_json_bytes(signed.get("defender_attestation"))
            != pooled.attestation.to_json()
            or pooled.attestation.top_ref != pooled_ref
        ):
            raise ValueError("restricted publication receipt lineage differs")
    except (WireContractError, ValueError) as error:
        raise ValueError("restricted publication receipt is invalid") from error
    return receipt


@dataclass(frozen=True, slots=True)
class PublishedG3Evaluation:
    """Reloadable public references from a real reduced Task12/Task13 run."""

    scorecard_ref: ArtifactRef
    evaluation_bundle_ref: ArtifactRef
    threshold_set_ref: ArtifactRef
    public_artifacts: dict[str, ArtifactRef]
    champion_decision: PublicChampionDecision
    promotion_envelope_digest: str
    descriptor_scope: tuple[str, ...]
    restricted_publication_receipt_ref: ArtifactRef
    development_evidence_ref: ArtifactRef | None = None


def publish_reduced_g3_evaluation(
    *,
    store: ArtifactStore,
    publication_signer: RunSigningIdentity,
    defender_ref: ArtifactRef,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    profile_sha256: str,
    authenticated_run_ids: tuple[str, ...],
    include_hidden: bool = True,
) -> PublishedG3Evaluation:
    """Run pooled chronological + isolated hidden evidence and publish no-promotion truthfully."""
    if (
        type(corpus) is not FrozenCorpus
        or corpus.manifest.profile_id != _FIXTURE_PROFILE_ID
    ):
        raise ValueError("reduced G3 publication is fixture-only")
    _digest(profile_sha256, label="fixture profile digest")
    if authenticated_run_ids != corpus.manifest.run_ids[: len(authenticated_run_ids)]:
        raise ValueError("fixture authenticated run lineage differs")
    evaluator_signer = EvaluatorSigningIdentity.from_private_bytes(_EVALUATOR_SEED)
    evaluator_verifier = EvaluatorReplayVerifier.from_signer(evaluator_signer)
    hidden_signer = EvaluatorSigningIdentity.from_private_bytes(_HIDDEN_SEED)
    hidden_verifier = EvaluatorReplayVerifier.from_signer(hidden_signer)
    publication_verifier = PublicArtifactVerifier.from_signer(publication_signer)
    source_root = Path(__file__).resolve().parents[3]

    with DefenderBundlePublisher(
        store, publication_signer, source_root
    ) as publisher:
        defender = publisher.load(defender_ref)
        defender.verify_reload()
    defender_verifier = DefenderBundleVerifier(
        store,
        signer_key_id=publication_signer.key_id,
        public_key_base64=publication_signer.public_key_base64,
    )
    defender_attestation = defender_verifier.attest(defender_ref)

    selection_ids = tuple(row.event_id for row in defender.threshold_matrix.rows)
    selection_as_of = split.config.development_end + timedelta(days=7)
    selection_binding = bind_replay_case_counter(
        defender.threshold_matrix.events,
        selection_ids,
        as_of=selection_as_of,
    )
    selection_labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in selection_ids], dtype=np.int64
    )
    selection_values = (
        None
        if defender.threshold_binding.values_digest is None
        else np.asarray(
            [float(split.row_net_settled_values[item]) for item in selection_ids],
            dtype=np.float64,
        )
    )
    threshold_set = ReplayThresholdSet.from_selection(
        defender,
        selection_binding,
        labels=selection_labels,
        values=selection_values,
    )
    threshold_set_ref = store.put_bytes(
        canonical_json_bytes(threshold_set.model_dump(mode="json")),
        "application/vnd.apar.replay-threshold-set+json",
    )

    corpus_evidence = ReplayCorpusEvidence.create(
        corpus=corpus, split=split, signer=evaluator_signer
    )
    _corpus_attestation, corpus_ref = publish_corpus_attestation(
        corpus_evidence,
        artifact_store=store,
        signer=evaluator_signer,
    )
    verified_inputs = verify_evaluation_inputs(
        corpus_ref=corpus_ref,
        defender_ref=defender_ref,
        artifact_store=store,
        evaluator_verifier=evaluator_verifier,
        defender_verifier=defender_verifier,
    )

    development = _development_context(corpus, split)
    replay_ids = tuple(row.event_id for row in defender.reload_matrix.rows)
    replay_binding = bind_replay_case_counter(
        development.observations,
        replay_ids,
        as_of=development.as_of,
    )
    development_result = replay_defense_arms(
        matrix=defender.reload_matrix,
        defender=defender,
        defender_verifier=defender_verifier,
        defender_attestation=defender_attestation,
        thresholds=threshold_set,
        threshold_labels=selection_labels,
        threshold_values=selection_values,
        case_counter=replay_binding,
        evaluation_split=split,
        corpus_evidence=corpus_evidence,
        evaluation=development,
        evaluator_signer=evaluator_signer,
        evaluator_verifier=evaluator_verifier,
    )
    if type(development_result) is not VerifiedReplayBatch:
        raise TypeError("development replay returned a hidden outcome")
    development_batch = development_result

    component_batches: tuple[VerifiedReplayBatch, ...] = (development_batch,)
    hidden_proofs: tuple[HiddenPublicProof, ...] = ()
    if include_hidden:
        hidden_context = ReplayEvaluationContext(
            **{
                **development.model_dump(mode="python"),
                "evaluation": EvaluationDescriptor(
                    kind=EvaluationKind.HIDDEN, value="hidden"
                ),
                "truth": tuple(
                    row.model_copy(
                        update={"viewpoint": "hidden", "label_source": "hidden_truth"}
                    )
                    for row in development.truth
                ),
            }
        )
        hidden_ref = store.put_bytes(hidden_context.to_json(), HIDDEN_CONTEXT_MEDIA_TYPE)
        authority = HiddenEvaluationAuthority(defender_verifier, store, hidden_signer)
        capability = authority.freeze_and_issue(  # type: ignore[attr-defined]
            defender_attestation, issued_at=defender.manifest.frozen_at
        )
        hidden_result = replay_defense_arms(
            matrix=defender.reload_matrix,
            defender=defender,
            defender_verifier=defender_verifier,
            defender_attestation=defender_attestation,
            thresholds=threshold_set,
            threshold_labels=selection_labels,
            threshold_values=selection_values,
            case_counter=replay_binding,
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=hidden_ref,
            hidden_released_at=defender.manifest.frozen_at,
            hidden_sealed_at=development.as_of,
        )
        if type(hidden_result) is not HiddenReplayOutcome:
            raise TypeError("hidden replay returned a development batch")
        component_batches = (*component_batches, hidden_result.batch)
        hidden_proofs = (hidden_result.public_proof,)
    envelope = VerifiedPromotionEnvelope.create(
        component_batches=component_batches,
        hidden_proofs=hidden_proofs,
        signer=evaluator_signer,
        hidden_proof_verifier=hidden_verifier,
    )
    decision = evaluate_promotion_gates(
        envelope,
        GateConfig.competition(),
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_verifier,
    )
    metric_evidence = _metric_publication_evidence(
        defender=defender,
        defender_verifier=defender_verifier,
        defender_attestation=defender_attestation,
        threshold_set=threshold_set,
        threshold_labels=selection_labels,
        threshold_values=selection_values,
        case_binding=replay_binding,
        matrix=defender.reload_matrix,
        context=development,
        development_batch=development_batch,
    )
    request = ScorecardPublicationRequest(
        promotion_envelope=envelope,
        champion_decision=decision,
        metric_evidence=metric_evidence,
        threshold_set=threshold_set,
    )
    scorecard, bundle = publish_scorecard(
        request,
        verified_inputs=verified_inputs,
        artifact_store=store,
        signer=publication_signer,
        publication_verifier=publication_verifier,
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_verifier,
    )
    restricted_receipt_ref = store_restricted_publication_receipt(
        request,
        verified_inputs=verified_inputs,
        scorecard=scorecard,
        bundle=bundle,
        artifact_store=store,
        signer=publication_signer,
    )
    descriptor_scope = tuple(
        sorted(
            f"{batch.results[0].evaluation.kind.value}:{batch.results[0].evaluation.value}"
            for batch in component_batches
        )
    )
    return _published_result(
        scorecard,
        bundle,
        threshold_set_ref,
        promotion_envelope_digest=envelope.envelope_digest,
        descriptor_scope=descriptor_scope,
        restricted_publication_receipt_ref=restricted_receipt_ref,
    )


@dataclass(frozen=True, slots=True)
class _CandidateRuntime:
    reference: ArtifactRef
    defender: LoadedDefenderBundle
    attestation: VerifiedDefenderAttestation
    thresholds: ReplayThresholdSet
    threshold_labels: np.ndarray
    threshold_values: np.ndarray | None


def _candidate_runtime(
    *,
    store: ArtifactStore,
    publication_signer: RunSigningIdentity,
    defender_verifier: DefenderBundleVerifier,
    reference: ArtifactRef,
    split: EvaluationSplit,
) -> _CandidateRuntime:
    source_root = Path(__file__).resolve().parents[3]
    with DefenderBundlePublisher(store, publication_signer, source_root) as publisher:
        defender = publisher.load(reference)
        defender.verify_reload()
    attestation = defender_verifier.attest(reference)
    selection_ids = tuple(row.event_id for row in defender.threshold_matrix.rows)
    selection_as_of = split.config.development_end + timedelta(days=7)
    selection_binding = bind_replay_case_counter(
        defender.threshold_matrix.events,
        selection_ids,
        as_of=selection_as_of,
    )
    labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in selection_ids], dtype=np.int64
    )
    values = (
        None
        if defender.threshold_binding.values_digest is None
        else np.asarray(
            [float(split.row_net_settled_values[item]) for item in selection_ids],
            dtype=np.float64,
        )
    )
    thresholds = ReplayThresholdSet.from_selection(
        defender,
        selection_binding,
        labels=labels,
        values=values,
    )
    return _CandidateRuntime(reference, defender, attestation, thresholds, labels, values)


def _descriptor_context(
    *,
    descriptor: EvaluationDescriptor,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    event_ids: tuple[str, ...],
    matrix: FeatureMatrix,
    defender: LoadedDefenderBundle,
) -> ReplayEvaluationContext:
    truth_by_id = {row.event_id: row for row in corpus.truth}
    truth = tuple(truth_by_id[item] for item in event_ids)
    as_of = max(
        split.config.development_end + timedelta(days=7),
        *(row.label_mature_at for row in truth),
    )
    original = tuple(
        row for row in corpus.observations if row.available_at <= as_of
    )
    selected = set(event_ids)
    observations = tuple(
        row
        if (row.event_id in selected and row.is_decision_point) or (
            not row.is_decision_point and row.decision_at is None
        )
        else row.model_copy(
            update={"is_decision_point": False, "decision_at": None}
        )
        for row in original
    )
    audit = audit_feature_matrix(
        original,
        matrix,
        matrix.catalog,
        allow_decision_event_subset=True,
    )
    assert_online_offline_parity(original, matrix.catalog)
    independently_rebuilt = build_feature_matrix(original, matrix.catalog)
    rebuilt_by_id = {row.event_id: row for row in independently_rebuilt.rows}
    if tuple(rebuilt_by_id[row.event_id] for row in matrix.rows) != matrix.rows:
        raise ValueError(
            "full-prefix offline/online parity differs from selected replay features"
        )
    audit_document = {
        "catalog_valid": audit.catalog_valid,
        "feature_order_matches": audit.feature_order_matches,
        "forbidden_sources": list(audit.forbidden_sources),
        "source_ids_resolve": audit.source_ids_resolve,
        "strictly_past_only": audit.strictly_past_only,
    }
    leakage_digest = hashlib.sha256(canonical_json_bytes(audit_document)).hexdigest()
    parity_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "catalog_digest": matrix.catalog_digest,
                "event_ids": [row.event_id for row in original],
                "row_ids": [row.event_id for row in matrix.rows],
                "status": "offline_online_exact",
            }
        )
    ).hexdigest()
    return ReplayEvaluationContext(
        evaluation=descriptor,
        truth=truth,
        observations=observations,
        as_of=as_of,
        slice_assignments=tuple(
            SliceAssignment(
                event_id=item,
                regime=(
                    descriptor.value
                    if descriptor.kind is EvaluationKind.REGIME
                    else "baseline"
                ),
                entity_cohorts=split.entity_cohorts[item],
            )
            for item in event_ids
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=_measured_latency_samples(
            matrix=matrix,
            defender=defender,
            observations=original,
            event_ids=event_ids,
        ),
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=audit.passed,
            parity_passed=True,
            leakage_evidence_digest=leakage_digest,
            parity_evidence_digest=parity_digest,
        ),
    )


def _measured_latency_samples(
    *,
    matrix: FeatureMatrix,
    defender: LoadedDefenderBundle,
    observations: tuple[ObservedEvent, ...],
    event_ids: tuple[str, ...],
) -> tuple[ReplayLatencySamples, ...]:
    """Measure real loaded components; Task13 binds their frozen environment lock."""
    if not event_ids:
        raise ValueError("competition latency evidence cannot be empty")
    row_by_id = {row.event_id: row for row in matrix.rows}
    event_by_id = {row.event_id: row for row in matrix.events}
    rules = RuleEngine(defender.rule_manifest)
    feature_by_id: dict[str, float] = {}
    microbatch_size = 8
    for offset in range(0, len(event_ids), microbatch_size):
        batch_ids = event_ids[offset : offset + microbatch_size]
        latest_decision = max(
            cast(datetime, event_by_id[event_id].decision_at)
            for event_id in batch_ids
        )
        batch_observations = tuple(
            row
            for row in observations
            if row.available_at <= latest_decision
        )
        started = time.perf_counter_ns()
        rebuilt = build_feature_matrix(batch_observations, matrix.catalog)
        elapsed = max((time.perf_counter_ns() - started) / 1_000_000, 0.000001)
        rebuilt_by_id = {row.event_id: row for row in rebuilt.rows}
        if any(rebuilt_by_id.get(event_id) != row_by_id[event_id] for event_id in batch_ids):
            raise ValueError("latency feature replay values differ")
        per_sample = elapsed / len(batch_ids)
        feature_by_id.update({event_id: per_sample for event_id in batch_ids})

    component: dict[str, tuple[float, float, float]] = {}
    for event_id in event_ids:
        row_matrix = FeatureMatrix(
            events=(event_by_id[event_id],),
            catalog=matrix.catalog,
            catalog_digest=matrix.catalog_digest,
            rows=(row_by_id[event_id],),
        )
        started = time.perf_counter_ns()
        rules.evaluate(event_by_id[event_id], row_by_id[event_id])
        rules_ms = max((time.perf_counter_ns() - started) / 1_000_000, 0.000001)
        started = time.perf_counter_ns()
        raw = defender.scorer.predict(row_matrix)
        model_ms = max((time.perf_counter_ns() - started) / 1_000_000, 0.000001)
        started = time.perf_counter_ns()
        defender.calibrator.predict(raw)
        calibration_ms = max(
            (time.perf_counter_ns() - started) / 1_000_000, 0.000001
        )
        component[event_id] = (rules_ms, model_ms, calibration_ms)

    return tuple(
        ReplayLatencySamples(
            arm=arm,
            samples=tuple(
                LatencySample(
                    event_id=event_id,
                    feature_ms=feature_by_id[event_id],
                    rules_ms=component[event_id][0],
                    model_ms=(
                        0.0
                        if arm is DefenseArm.RULES_ONLY
                        else component[event_id][1]
                    ),
                    calibration_policy_ms=(
                        0.0
                        if arm is DefenseArm.RULES_ONLY
                        else component[event_id][2]
                    ),
                    end_to_end_ms=(
                        feature_by_id[event_id] + component[event_id][0]
                        if arm is DefenseArm.RULES_ONLY
                        else feature_by_id[event_id]
                        + component[event_id][0]
                        + component[event_id][1]
                        + component[event_id][2]
                    ),
                )
                for event_id in event_ids
            ),
        )
        for arm in DefenseArm
    )


def _evaluation_matrix(
    *,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    event_ids: tuple[str, ...],
    catalog: object,
) -> FeatureMatrix:
    from apar.features.catalog import FeatureCatalog

    if type(catalog) is not FeatureCatalog:
        raise ValueError("competition feature catalog is invalid")
    truth_by_id = {row.event_id: row for row in corpus.truth}
    as_of = max(
        split.config.development_end + timedelta(days=7),
        *(truth_by_id[event_id].label_mature_at for event_id in event_ids),
    )
    observations = tuple(
        row for row in corpus.observations if row.available_at <= as_of
    )
    complete = build_feature_matrix(observations, catalog)
    row_by_id = {row.event_id: row for row in complete.rows}
    selected_ids = frozenset(event_ids)
    return FeatureMatrix(
        events=tuple(
            row for row in complete.events if row.event_id in selected_ids
        ),
        catalog=complete.catalog,
        catalog_digest=complete.catalog_digest,
        rows=tuple(row_by_id[item] for item in event_ids),
    )


def _replay_component(
    *,
    candidate: _CandidateRuntime,
    defender_verifier: DefenderBundleVerifier,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    descriptor: EvaluationDescriptor,
    event_ids: tuple[str, ...],
    evaluator_signer: EvaluatorSigningIdentity,
    evaluator_verifier: EvaluatorReplayVerifier,
    regime_evidence: ReplayRegimeEvidence | None = None,
) -> tuple[VerifiedReplayBatch, ReplayEvaluationContext]:
    matrix = _evaluation_matrix(
        corpus=corpus,
        split=split,
        event_ids=event_ids,
        catalog=candidate.defender.catalog,
    )
    context = _descriptor_context(
        descriptor=descriptor,
        corpus=corpus,
        split=split,
        event_ids=event_ids,
        matrix=matrix,
        defender=candidate.defender,
    )
    binding = bind_replay_case_counter(
        context.observations, event_ids, as_of=context.as_of
    )
    evidence = ReplayCorpusEvidence.create(
        corpus=corpus, split=split, signer=evaluator_signer
    )
    result = replay_defense_arms(
        matrix=matrix,
        defender=candidate.defender,
        defender_verifier=defender_verifier,
        defender_attestation=candidate.attestation,
        thresholds=candidate.thresholds,
        threshold_labels=candidate.threshold_labels,
        threshold_values=candidate.threshold_values,
        case_counter=binding,
        evaluation_split=split,
        regime_evidence=regime_evidence,
        corpus_evidence=evidence,
        evaluation=context,
        evaluator_signer=evaluator_signer,
        evaluator_verifier=evaluator_verifier,
    )
    if type(result) is not VerifiedReplayBatch:
        raise TypeError("development descriptor returned a hidden outcome")
    return result, context


def _empty_cold_component(
    *,
    candidate: _CandidateRuntime,
    defender_verifier: DefenderBundleVerifier,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    cohort: EntityCohort,
    evaluator_signer: EvaluatorSigningIdentity,
    evaluator_verifier: EvaluatorReplayVerifier,
) -> VerifiedReplayBatch:
    descriptor = EvaluationDescriptor(
        kind=EvaluationKind.COLD_ENTITY, value=cohort.value
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "cohort": cohort.value,
                "kind": "verified_zero_row_nonapplicable",
                "split_digest": split.split_digest,
            }
        )
    ).hexdigest()
    context = ReplayEvaluationContext(
        evaluation=descriptor,
        truth=(),
        observations=(),
        as_of=split.config.development_end + timedelta(days=7),
        slice_assignments=(),
        slice_manifest=SliceManifest.closed(),
        latency_samples=tuple(
            ReplayLatencySamples(arm=arm, samples=()) for arm in DefenseArm
        ),
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=True,
            parity_passed=True,
            leakage_evidence_digest=digest,
            parity_evidence_digest=digest,
        ),
    )
    matrix = FeatureMatrix(
        events=(),
        catalog=candidate.defender.catalog,
        catalog_digest=candidate.defender.reload_matrix.catalog_digest,
        rows=(),
    )
    corpus_evidence = ReplayCorpusEvidence.create(
        corpus=corpus, split=split, signer=evaluator_signer
    )
    return replay_empty_cold_entity(
        matrix=matrix,
        defender=candidate.defender,
        defender_verifier=defender_verifier,
        defender_attestation=candidate.attestation,
        thresholds=candidate.thresholds,
        threshold_labels=candidate.threshold_labels,
        threshold_values=candidate.threshold_values,
        evaluation_split=split,
        corpus_evidence=corpus_evidence,
        evaluation=context,
        evaluator_signer=evaluator_signer,
        evaluator_verifier=evaluator_verifier,
    )


def _regime_specs(corpus: FrozenCorpus, split: EvaluationSplit) -> tuple[RegimeSpec, ...]:
    controls = _benign_control_corpus(corpus, split)
    campaign_ids = tuple(sorted({row.campaign_id for row in controls.truth}))
    return (
        RegimeSpec.prevalence_dilution(campaign_ids),
        RegimeSpec.missing_optional(),
        RegimeSpec.availability_delay(),
        RegimeSpec.compressed_bursts(),
        RegimeSpec.benign_amount_shift(),
        RegimeSpec.cold_id_remap(),
    )


def _benign_control_corpus(corpus: FrozenCorpus, split: EvaluationSplit) -> FrozenCorpus:
    benign = tuple(
        sorted(
            (
                row
                for row in corpus.truth
                if row.event_id in split.row_ids["development"] and not row.is_fraud
            ),
            key=lambda row: row.event_id,
        )
    )
    if not benign:
        raise ValueError("prevalence regime requires a benign development payment")
    selected_truth = (benign[0],)
    campaign = selected_truth[0].campaign_id
    payments = {row.payment_id for row in selected_truth}
    selected_observations = tuple(
        row for row in corpus.observations if row.payment_id in payments
    )
    namespace = uuid5(NAMESPACE_URL, f"apar:competition-control:{campaign}")
    event_map = {
        row.event_id: str(uuid5(namespace, f"event:{row.event_id}"))
        for row in selected_observations
    }
    payment_map = {
        payment: f"control:{uuid5(namespace, f'payment:{payment}')}"
        for payment in payments
    }
    campaign_id = str(uuid5(namespace, "campaign"))
    identity_map = {
        identity: str(uuid5(namespace, f"identity:{identity}"))
        for row in selected_observations
        for identity in (row.actor_id, row.counterparty_id, *row.optional_refs.values())
    }
    observations = tuple(
        row.model_copy(
            update={
                "actor_id": identity_map[row.actor_id],
                "counterparty_id": identity_map[row.counterparty_id],
                "event_id": event_map[row.event_id],
                "optional_refs": {
                    key: identity_map[value] for key, value in row.optional_refs.items()
                },
                "payment_id": payment_map[row.payment_id],
            }
        )
        for row in selected_observations
    )
    truth = tuple(
        row.model_copy(
            update={
                "campaign_id": campaign_id,
                "event_id": event_map[row.event_id],
                "lifecycle_event_ids": tuple(
                    event_map[item] for item in row.lifecycle_event_ids
                ),
                "payment_id": payment_map[row.payment_id],
            }
        )
        for row in selected_truth
    )
    return FrozenCorpus(
        observations=tuple(sorted(observations, key=lambda row: row.event_id)),
        truth=tuple(sorted(truth, key=lambda row: row.event_id)),
        manifest=CorpusManifest(
            profile_id="competition-derived-benign-control-v1",
            run_ids=(f"control:{campaign_id}",),
            run_lineage_digests=(
                hashlib.sha256(
                    canonical_json_bytes([row.event_id for row in truth])
                ).hexdigest(),
            ),
            observation_count=len(observations),
            truth_count=len(truth),
        ),
    )


def _competition_development_components(
    *,
    pooled: _CandidateRuntime,
    lofo: dict[Family, _CandidateRuntime],
    defender_verifier: DefenderBundleVerifier,
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    evaluator_signer: EvaluatorSigningIdentity,
    evaluator_verifier: EvaluatorReplayVerifier,
) -> tuple[
    list[VerifiedReplayBatch],
    VerifiedReplayBatch,
    ReplayEvaluationContext,
    FeatureMatrix,
]:
    """Build the exact closed 16-descriptor development matrix once."""
    components: list[VerifiedReplayBatch] = []
    development_batch, development_context = _replay_component(
        candidate=pooled,
        defender_verifier=defender_verifier,
        corpus=corpus,
        split=split,
        descriptor=EvaluationDescriptor(
            kind=EvaluationKind.CHRONOLOGICAL, value="development"
        ),
        event_ids=split.row_ids["development"],
        evaluator_signer=evaluator_signer,
        evaluator_verifier=evaluator_verifier,
    )
    components.append(development_batch)
    development_matrix = _evaluation_matrix(
        corpus=corpus,
        split=split,
        event_ids=split.row_ids["development"],
        catalog=pooled.defender.catalog,
    )
    for cohort in EntityCohort:
        ids = tuple(
            event_id
            for event_id in split.row_ids["development"]
            if cohort in split.entity_cohorts[event_id]
        )
        if ids:
            batch, _ = _replay_component(
                candidate=pooled,
                defender_verifier=defender_verifier,
                corpus=corpus,
                split=split,
                descriptor=EvaluationDescriptor(
                    kind=EvaluationKind.COLD_ENTITY, value=cohort.value
                ),
                event_ids=ids,
                evaluator_signer=evaluator_signer,
                evaluator_verifier=evaluator_verifier,
            )
        else:
            batch = _empty_cold_component(
                candidate=pooled,
                defender_verifier=defender_verifier,
                corpus=corpus,
                split=split,
                cohort=cohort,
                evaluator_signer=evaluator_signer,
                evaluator_verifier=evaluator_verifier,
            )
        components.append(batch)
    for family in _FAMILIES:
        family_split = make_leave_one_family_out(split, family)
        batch, _ = _replay_component(
            candidate=lofo[family],
            defender_verifier=defender_verifier,
            corpus=corpus,
            split=family_split,
            descriptor=EvaluationDescriptor(
                kind=EvaluationKind.HELD_FAMILY, value=family
            ),
            event_ids=family_split.held_out_evaluation_row_ids,
            evaluator_signer=evaluator_signer,
            evaluator_verifier=evaluator_verifier,
        )
        components.append(batch)
    control = _benign_control_corpus(corpus, split)
    for spec in _regime_specs(corpus, split):
        derived, manifest = derive_regime(
            corpus,
            spec,
            control_corpus=(
                control if spec.kind is RegimeKind.PREVALENCE_DILUTION else None
            ),
        )
        derived_split = make_evaluation_split(derived, split.config)
        evidence = ReplayRegimeEvidence.create(
            parent_corpus=corpus,
            derived_corpus=derived,
            spec=spec,
            manifest=manifest,
            control_corpus=(
                control if spec.kind is RegimeKind.PREVALENCE_DILUTION else None
            ),
        )
        batch, _ = _replay_component(
            candidate=pooled,
            defender_verifier=defender_verifier,
            corpus=derived,
            split=derived_split,
            descriptor=EvaluationDescriptor(
                kind=EvaluationKind.REGIME, value=spec.kind.value
            ),
            event_ids=derived_split.row_ids["development"],
            evaluator_signer=evaluator_signer,
            evaluator_verifier=evaluator_verifier,
            regime_evidence=evidence,
        )
        components.append(batch)
    return components, development_batch, development_context, development_matrix


def _verify_frozen_development_request(
    request: ScorecardPublicationRequest,
    *,
    evaluator_verifier: EvaluatorReplayVerifier,
    pooled: _CandidateRuntime,
    lofo: dict[Family, _CandidateRuntime],
) -> None:
    """Verify frozen development signatures, gates, and exact ensemble candidates."""
    envelope = request.promotion_envelope
    scope = tuple(
        sorted(
            f"{batch.results[0].evaluation.kind.value}:"
            f"{batch.results[0].evaluation.value}"
            for batch in envelope.component_batches
        )
    )
    if (
        scope != _DEVELOPMENT_DESCRIPTOR_SCOPE
        or envelope.hidden_proofs
        or request.threshold_set != pooled.thresholds
        or not evaluator_verifier.verify_promotion_envelope(envelope)
        or evaluate_promotion_gates(
            envelope,
            GateConfig.competition(),
            evaluator_verifier=evaluator_verifier,
            hidden_proof_verifier=evaluator_verifier,
        )
        != request.champion_decision
    ):
        raise ValueError("frozen development evidence failed exact verification")
    for batch in envelope.component_batches:
        descriptor = batch.results[0].evaluation
        expected = (
            lofo[cast(Family, descriptor.value)]
            if descriptor.kind is EvaluationKind.HELD_FAMILY
            else pooled
        )
        if any(
            row.candidate_role.defender_top_ref_digest != expected.reference.sha256
            or row.candidate_role.threshold_set_digest
            != expected.thresholds.threshold_set_digest
            or row.threshold_set_digest != expected.thresholds.threshold_set_digest
            for row in batch.results
        ):
            raise ValueError("frozen development candidate lineage differs")


def publish_competition_evaluation(
    *,
    store: ArtifactStore,
    publication_signer: RunSigningIdentity,
    evaluator_signer: EvaluatorSigningIdentity,
    hidden_signer: EvaluatorSigningIdentity | None,
    pooled_ref: ArtifactRef,
    held_family_refs: dict[Family, ArtifactRef],
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    profile_sha256: str,
    authenticated_run_ids: tuple[str, ...],
    hidden_context_ref: ArtifactRef | None = None,
    hidden_context_signer: RunSigningIdentity | None = None,
    development_evidence_ref: ArtifactRef | None = None,
) -> PublishedG3Evaluation:
    """Run the full pooled/LOFO robustness matrix and publish Task13 evidence."""
    if corpus.manifest.profile_id == _FIXTURE_PROFILE_ID:
        raise ValueError("competition evaluation rejects reduced fixture corpora")
    _digest(profile_sha256, label="competition profile digest")
    if authenticated_run_ids != corpus.manifest.run_ids:
        raise ValueError("competition authenticated run lineage differs")
    if set(held_family_refs) != set(_FAMILIES):
        raise ValueError("competition LOFO roster is incomplete")
    if len({pooled_ref.sha256, *(ref.sha256 for ref in held_family_refs.values())}) != 5:
        raise ValueError("competition candidate roles require five distinct bundles")
    evaluator_verifier = EvaluatorReplayVerifier.from_signer(evaluator_signer)
    hidden_requested = (
        hidden_context_ref is not None or development_evidence_ref is not None
    )
    if hidden_requested != (
        hidden_context_ref is not None and development_evidence_ref is not None
    ):
        raise ValueError("hidden evaluation requires frozen development evidence and context")
    if hidden_requested:
        if not EvaluatorSigningIdentity.is_exact(hidden_signer):
            raise ValueError("hidden evaluation requires its private authority")
        hidden_verifier = EvaluatorReplayVerifier.from_signer(
            cast(EvaluatorSigningIdentity, hidden_signer)
        )
    else:
        if hidden_signer is not None or hidden_context_signer is not None:
            raise ValueError("development evaluation cannot receive hidden authorities")
        hidden_verifier = evaluator_verifier
    defender_verifier = DefenderBundleVerifier(
        store,
        signer_key_id=publication_signer.key_id,
        public_key_base64=publication_signer.public_key_base64,
    )
    pooled = _candidate_runtime(
        store=store,
        publication_signer=publication_signer,
        defender_verifier=defender_verifier,
        reference=pooled_ref,
        split=split,
    )
    lofo: dict[Family, _CandidateRuntime] = {}
    for family in _FAMILIES:
        family_split = make_leave_one_family_out(split, family)
        lofo[family] = _candidate_runtime(
            store=store,
            publication_signer=publication_signer,
            defender_verifier=defender_verifier,
            reference=held_family_refs[family],
            split=family_split,
        )

    development_context: ReplayEvaluationContext | None = None
    development_matrix: FeatureMatrix | None = None
    if development_evidence_ref is None:
        (
            components,
            development_batch,
            development_context,
            development_matrix,
        ) = _competition_development_components(
            pooled=pooled,
            lofo=lofo,
            defender_verifier=defender_verifier,
            corpus=corpus,
            split=split,
            evaluator_signer=evaluator_signer,
            evaluator_verifier=evaluator_verifier,
        )
        metric_evidence: tuple[MetricPublicationEvidence, ...] | None = None
    else:
        if development_evidence_ref.media_type != _DEVELOPMENT_EVIDENCE_MEDIA_TYPE:
            raise ValueError("frozen development evidence media type differs")
        prior_request = ScorecardPublicationRequest.from_worker_json(
            store.read(development_evidence_ref)
        )
        prior_envelope = prior_request.promotion_envelope
        _verify_frozen_development_request(
            prior_request,
            evaluator_verifier=evaluator_verifier,
            pooled=pooled,
            lofo=lofo,
        )
        components = list(prior_envelope.component_batches)
        development_batch = next(
            batch
            for batch in components
            if batch.results[0].evaluation
            == EvaluationDescriptor(
                kind=EvaluationKind.CHRONOLOGICAL, value="development"
            )
        )
        metric_evidence = prior_request.metric_evidence

    hidden_proofs: tuple[HiddenPublicProof, ...] = ()
    if hidden_context_ref is not None:
        if type(hidden_context_signer) is not RunSigningIdentity:
            raise ValueError("hidden context requires its pinned authority signer")
        hidden_envelope, hidden_observations, restricted_hidden_ref = verify_hidden_context(
            store=store,
            envelope_ref=hidden_context_ref,
            signer=hidden_context_signer,
            profile_sha256=profile_sha256,
            development_corpus_digest=frozen_corpus_digest(corpus),
            development_event_ids=tuple(row.event_id for row in corpus.observations),
        )
        hidden_matrix = build_feature_matrix(
            hidden_observations, pooled.defender.catalog
        )
        hidden_ids = tuple(row.event_id for row in hidden_matrix.rows)
        hidden_binding = bind_replay_case_counter(
            hidden_observations, hidden_ids, as_of=hidden_envelope.as_of
        )
        authority = HiddenEvaluationAuthority(
            defender_verifier, store, cast(EvaluatorSigningIdentity, hidden_signer)
        )
        capability = authority.freeze_and_issue(  # type: ignore[attr-defined]
            pooled.attestation, issued_at=pooled.defender.manifest.frozen_at
        )
        hidden_result = replay_defense_arms(
            matrix=hidden_matrix,
            defender=pooled.defender,
            defender_verifier=defender_verifier,
            defender_attestation=pooled.attestation,
            thresholds=pooled.thresholds,
            threshold_labels=pooled.threshold_labels,
            threshold_values=pooled.threshold_values,
            case_counter=hidden_binding,
            hidden_authority=authority,
            hidden_capability=capability,
            hidden_ref=restricted_hidden_ref,
            hidden_released_at=pooled.defender.manifest.frozen_at,
            hidden_sealed_at=hidden_envelope.as_of,
        )
        if type(hidden_result) is not HiddenReplayOutcome:
            raise TypeError("hidden descriptor returned a development batch")
        components.append(hidden_result.batch)
        hidden_proofs = (hidden_result.public_proof,)
    components.sort(
        key=lambda batch: (
            tuple(EvaluationKind).index(batch.results[0].evaluation.kind),
            batch.results[0].evaluation.value,
        )
    )
    envelope = VerifiedPromotionEnvelope.create(
        component_batches=tuple(components),
        hidden_proofs=hidden_proofs,
        signer=evaluator_signer,
        hidden_proof_verifier=hidden_verifier,
    )
    decision = evaluate_promotion_gates(
        envelope,
        GateConfig.competition(),
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_verifier,
    )
    if metric_evidence is None:
        if development_context is None or development_matrix is None:
            raise ValueError("development metric context is unavailable")
        primary_binding = bind_replay_case_counter(
            development_context.observations,
            tuple(row.event_id for row in development_matrix.rows),
            as_of=development_context.as_of,
        )
        metric_evidence = _metric_publication_evidence(
            defender=pooled.defender,
            defender_verifier=defender_verifier,
            defender_attestation=pooled.attestation,
            threshold_set=pooled.thresholds,
            threshold_labels=pooled.threshold_labels,
            threshold_values=pooled.threshold_values,
            case_binding=primary_binding,
            matrix=development_matrix,
            context=development_context,
            development_batch=development_batch,
        )
    corpus_evidence = ReplayCorpusEvidence.create(
        corpus=corpus, split=split, signer=evaluator_signer
    )
    _attestation, corpus_ref = publish_corpus_attestation(
        corpus_evidence, artifact_store=store, signer=evaluator_signer
    )
    verified_inputs = verify_evaluation_inputs(
        corpus_ref=corpus_ref,
        defender_ref=pooled_ref,
        artifact_store=store,
        evaluator_verifier=evaluator_verifier,
        defender_verifier=defender_verifier,
    )
    request = ScorecardPublicationRequest(
        promotion_envelope=envelope,
        champion_decision=decision,
        metric_evidence=metric_evidence,
        threshold_set=pooled.thresholds,
    )
    frozen_development_ref = development_evidence_ref
    if frozen_development_ref is None:
        frozen_development_ref = store.put_bytes(
            request.to_worker_json(), _DEVELOPMENT_EVIDENCE_MEDIA_TYPE
        )
    publication_verifier = PublicArtifactVerifier.from_signer(publication_signer)
    scorecard, bundle = publish_scorecard(
        request,
        verified_inputs=verified_inputs,
        artifact_store=store,
        signer=publication_signer,
        publication_verifier=publication_verifier,
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_verifier,
    )
    restricted_receipt_ref = store_restricted_publication_receipt(
        request,
        verified_inputs=verified_inputs,
        scorecard=scorecard,
        bundle=bundle,
        artifact_store=store,
        signer=publication_signer,
    )
    threshold_ref = store.put_bytes(
        canonical_json_bytes(pooled.thresholds.model_dump(mode="json")),
        "application/vnd.apar.replay-threshold-set+json",
    )
    scope = tuple(
        sorted(
            f"{batch.results[0].evaluation.kind.value}:{batch.results[0].evaluation.value}"
            for batch in components
        )
    )
    return _published_result(
        scorecard,
        bundle,
        threshold_ref,
        promotion_envelope_digest=envelope.envelope_digest,
        descriptor_scope=scope,
        restricted_publication_receipt_ref=restricted_receipt_ref,
        development_evidence_ref=frozen_development_ref,
    )


def _development_context(
    corpus: FrozenCorpus, split: EvaluationSplit
) -> ReplayEvaluationContext:
    event_ids = split.row_ids["development"]
    truth_by_id = {row.event_id: row for row in corpus.truth}
    truth = tuple(truth_by_id[item] for item in event_ids)
    lifecycle_ids = {
        lifecycle_id
        for row in truth
        for lifecycle_id in row.lifecycle_event_ids
    }
    observations = tuple(
        (
            event
            if event.is_decision_point or event.decision_at is None
            else event.model_copy(update={"decision_at": None})
        )
        for event in corpus.observations
        if event.event_id in lifecycle_ids
    )
    latency = tuple(
        ReplayLatencySamples(
            arm=arm,
            samples=tuple(
                LatencySample(
                    event_id=item,
                    feature_ms=1.0,
                    rules_ms=1.0,
                    model_ms=0.0 if arm is DefenseArm.RULES_ONLY else 1.0,
                    calibration_policy_ms=1.0,
                    end_to_end_ms=3.0 if arm is DefenseArm.RULES_ONLY else 4.0,
                )
                for item in event_ids
            ),
        )
        for arm in DefenseArm
    )
    return ReplayEvaluationContext(
        evaluation=EvaluationDescriptor(
            kind=EvaluationKind.CHRONOLOGICAL, value="development"
        ),
        truth=truth,
        observations=observations,
        as_of=split.config.development_end + timedelta(days=7),
        slice_assignments=tuple(
            SliceAssignment(
                event_id=item,
                regime="baseline",
                entity_cohorts=split.entity_cohorts[item],
            )
            for item in event_ids
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=latency,
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=True,
            parity_passed=True,
            leakage_evidence_digest=hashlib.sha256(b"g3-causal-fixture").hexdigest(),
            parity_evidence_digest=hashlib.sha256(b"g3-parity-fixture").hexdigest(),
        ),
    )


def _metric_publication_evidence(
    *,
    defender: LoadedDefenderBundle,
    defender_verifier: DefenderBundleVerifier,
    defender_attestation: VerifiedDefenderAttestation,
    threshold_set: ReplayThresholdSet,
    threshold_labels: np.ndarray,
    threshold_values: np.ndarray | None,
    case_binding: ReplayCaseCounterBinding,
    matrix: FeatureMatrix,
    context: ReplayEvaluationContext,
    development_batch: VerifiedReplayBatch,
) -> tuple[MetricPublicationEvidence, ...]:
    invocation = _HiddenReplayInvocation(
        matrix,
        defender,
        defender_verifier,
        defender_attestation,
        threshold_set,
        threshold_labels,
        threshold_values,
        case_binding,
        None,
    )
    frozen = _freeze_replay_inputs(
        invocation,
        pinned_verifier=defender_verifier,
        pinned_attestation=defender_attestation,
    )
    results = {row.arm: row for row in development_batch.results}
    evidence: list[MetricPublicationEvidence] = []
    for arm in DefenseArm:
        decisions = frozen.decisions_by_arm[arm]
        cases = group_cases(context.observations, decisions, as_of=context.as_of)
        queue = simulate_case_queue(cases, context.queue_config)
        inputs = MetricReportInputs(
            truth=context.truth,
            observations=context.observations,
            decisions=decisions,
            cases=queue.case_inputs,
            queue_report=queue,
            latency_samples=next(
                item.samples for item in context.latency_samples if item.arm is arm
            ),
            as_of=context.as_of,
            slice_assignments=context.slice_assignments,
            slice_manifest=context.slice_manifest,
        )
        report = compute_metric_report(inputs)
        evidence.append(
            MetricPublicationEvidence(
                arm=arm,
                result_digest=results[arm].result_digest,
                metric_report=report,
                metric_derivation_evidence=MetricDerivationEvidence.from_inputs(inputs),
                confidence_intervals=campaign_bootstrap(inputs),
                bootstrap_derivation_evidence=BootstrapDerivationEvidence.from_inputs(inputs),
            )
        )
    return tuple(evidence)


def _published_result(
    scorecard: DefenseScorecard,
    bundle: EvaluationArtifactBundle,
    threshold_set_ref: ArtifactRef,
    *,
    promotion_envelope_digest: str,
    descriptor_scope: tuple[str, ...],
    restricted_publication_receipt_ref: ArtifactRef,
    development_evidence_ref: ArtifactRef | None = None,
) -> PublishedG3Evaluation:
    public = {
        name: reference.as_artifact_ref()
        for name, reference in bundle.public_artifacts.items()
    }
    return PublishedG3Evaluation(
        scorecard_ref=public["defense-scorecard.json"],
        evaluation_bundle_ref=bundle.bundle_ref(),
        threshold_set_ref=threshold_set_ref,
        public_artifacts=public,
        champion_decision=scorecard.champion_decision,
        promotion_envelope_digest=promotion_envelope_digest,
        descriptor_scope=descriptor_scope,
        restricted_publication_receipt_ref=restricted_publication_receipt_ref,
        development_evidence_ref=development_evidence_ref,
    )


__all__ = [
    "DevelopmentCompletionReceipt",
    "PublishedG3Evaluation",
    "RestrictedHiddenContextEnvelope",
    "publish_competition_evaluation",
    "publish_reduced_g3_evaluation",
    "seal_development_completion",
    "seal_hidden_context",
    "verify_development_completion",
    "verify_hidden_context",
]
