"""Canonical signed judge reports with an absolute public/restricted boundary."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import math
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.gates import (
    ArmGateResult,
    ChampionDecision,
    ChampionStatus,
    DefenseArm,
    EvaluationKind,
    EvaluatorReplayVerifier,
    GateConfig,
    ReplayResult,
    VerifiedPromotionEnvelope,
    evaluate_promotion_gates,
)
from apar.evaluation.metrics import (
    BootstrapDerivationEvidence,
    ConfidenceInterval,
    ConfidenceIntervals,
    MetricDerivationEvidence,
    MetricReport,
)
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

SCORECARD_ARTIFACT_NAME = "defense-scorecard.json"
SCORECARD_MEDIA_TYPE = "application/vnd.apar.defense-scorecard+json"
EVALUATION_BUNDLE_MEDIA_TYPE = "application/vnd.apar.evaluation-artifact-bundle+json"
PUBLIC_ARTIFACT_MEDIA_TYPES: dict[str, str] = {
    "calibration.csv": "text/csv; charset=utf-8",
    "data-card.md": "text/markdown; charset=utf-8",
    "defense-scorecard.md": "text/markdown; charset=utf-8",
    "feature-manifest.json": "application/json",
    "latency-evidence.json": "application/vnd.apar.defense-latency+json",
    "leaderboard.csv": "text/csv; charset=utf-8",
    "limitations.md": "text/markdown; charset=utf-8",
    "model-card.md": "text/markdown; charset=utf-8",
    "slice-metrics.csv": "text/csv; charset=utf-8",
    "thresholds.json": "application/json",
    "value-workload.csv": "text/csv; charset=utf-8",
}

_ALL_PUBLIC_MEDIA_TYPES = {
    **PUBLIC_ARTIFACT_MEDIA_TYPES,
    SCORECARD_ARTIFACT_NAME: SCORECARD_MEDIA_TYPE,
}
_MAX_PUBLIC_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_SCORECARD_BYTES = 8 * 1024 * 1024
_MAX_BUNDLE_BYTES = 16 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")
_EXTERNAL_VALIDITY = (
    "Synthetic APAR evaluation only; these results are not evidence of external "
    "validity or live-payment, Mastercard production, or real-world performance."
)
_LIMITATIONS = (
    "All training and evaluation data are synthetic.",
    "The generators cover four implemented families, not the complete threat registry.",
    "Independent code reduces but cannot eliminate shared conceptual assumptions.",
    "Synthetic prevalence and economic distributions are not production estimates.",
    "Seed-specific identities can make cold-entity evaluation dominant.",
    "Unavailable device or merchant enrichment is not claimed.",
    "Challenges receive no prevented-value credit without a frozen outcome.",
    "A strong CatBoost baseline is not evidence of the best model for real payments.",
    "Campaign bootstrap intervals measure simulator variation, not uncertainty over "
    "real payment populations.",
    "This subsystem recommends only a competition champion; later deployment remains "
    "a named human decision.",
)
_FORBIDDEN_PUBLIC_TOKENS = tuple(
    token.encode("ascii")
    for token in (
        "metricderivationevidence",
        "bootstrapderivationevidence",
        "evaluated_result_digests",
        "decision_digest",
        "evaluator_input_digest",
        "derivation_evidence_digest",
        "decision_event_ids",
        "decision_content_digest",
        "batch_content_digest",
        "component_batches",
        "hidden_proofs",
        "evaluator_context_token",
        "cohort_mapping_token",
        "proof_digest",
        "payment_id",
        "campaign_id",
        "event_id",
        "hidden_public_proof_id",
        "hpf_",
        "evaluator_context_digest",
        "cohort_mapping_digest",
        "decision_bindings_digest",
        "worker_manifest_digest",
        "restricted_hidden",
        "evaluation_truth",
        "per_decision_predictions",
        "relative_path",
        "private_key",
        '"hostname":',
        '"process_id":',
        '"pid":',
        '"path":',
        "/users/",
        "file://",
        "private-run-state",
    )
)


class ReportingContractError(ValueError):
    """Judge evidence failed canonical, signature, lineage, or privacy validation."""


def _validate_digest(value: object, *, label: str = "digest") -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_document(document: object) -> str:
    return _digest_bytes(canonical_json_bytes(document))


def _validate_public_name(name: object) -> str:
    if (
        type(name) is not str
        or not name
        or len(name) > 128
        or unicodedata.normalize("NFC", name) != name
        or name != name.casefold()
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or ".." in name
        or name not in _ALL_PUBLIC_MEDIA_TYPES
    ):
        raise ValueError("public artifact name is not an exact allowlisted name")
    return name


@dataclass(frozen=True, slots=True)
class PublicArtifactReference:
    """Path-free public content address and exact media descriptor."""

    name: str
    sha256: str
    media_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_public_name(self.name)
        _validate_digest(self.sha256, label="public artifact digest")
        if self.media_type != _ALL_PUBLIC_MEDIA_TYPES[self.name]:
            raise ValueError("public artifact media type differs from the allowlist")
        if (
            type(self.size_bytes) is not int
            or not 0 <= self.size_bytes <= _MAX_SCORECARD_BYTES
        ):
            raise ValueError("public artifact size is outside its resource cap")

    @classmethod
    def from_artifact_ref(cls, name: str, ref: ArtifactRef) -> PublicArtifactReference:
        if type(ref) is not ArtifactRef:
            raise ReportingContractError("stored public artifact reference is invalid")
        try:
            return cls(name, ref.sha256, ref.media_type, ref.size_bytes)
        except ValueError as error:
            raise ReportingContractError(str(error)) from error

    def as_artifact_ref(self) -> ArtifactRef:
        """Build the internal store path only at the storage boundary."""
        return ArtifactRef(
            sha256=self.sha256,
            media_type=self.media_type,
            size_bytes=self.size_bytes,
            relative_path=f"{self.sha256}/payload",
        )


class PublicArtifactIndex(ExternalContract, Mapping[str, PublicArtifactReference]):
    """Intrinsically immutable, exact-name public artifact lookup."""

    entries: tuple[PublicArtifactReference, ...]

    @field_validator("entries", mode="before")
    @classmethod
    def entries_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("public artifact entries must be an exact tuple")
        return value

    @model_validator(mode="after")
    def index_is_canonical(self) -> PublicArtifactIndex:
        if any(type(item) is not PublicArtifactReference for item in self.entries):
            raise ValueError("public artifact index contains a nonexact reference")
        names = tuple(item.name for item in self.entries)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("public artifact entries must be sorted and unique")
        aliases = tuple(unicodedata.normalize("NFC", name).casefold() for name in names)
        if len(aliases) != len(set(aliases)):
            raise ValueError("public artifact entries contain a Unicode or case alias")
        return self

    @classmethod
    def from_refs(
        cls, references: Mapping[str, ArtifactRef]
    ) -> PublicArtifactIndex:
        try:
            return cls(
                entries=tuple(
                    PublicArtifactReference.from_artifact_ref(name, references[name])
                    for name in sorted(references)
                )
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise ReportingContractError("public artifact index is invalid") from error

    def with_entry(self, entry: PublicArtifactReference) -> PublicArtifactIndex:
        try:
            return PublicArtifactIndex(entries=(*self.entries, entry))
        except (TypeError, ValueError, ValidationError) as error:
            raise ReportingContractError("public artifact alias or duplicate rejected") from error

    def __getitem__(self, key: str) -> PublicArtifactReference:
        if type(key) is not str:
            raise KeyError(key)
        for item in self.entries:
            if item.name == key:
                return item
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return (item.name for item in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def values(self) -> tuple[PublicArtifactReference, ...]:  # type: ignore[override]
        return self.entries

    def items(self) -> tuple[tuple[str, PublicArtifactReference], ...]:  # type: ignore[override]
        return tuple((item.name, item) for item in self.entries)


class MetricPublicationEvidence(ExternalContract):
    """Transient evaluator evidence; none of these restricted objects are published."""

    arm: DefenseArm
    result_digest: str
    metric_report: MetricReport
    metric_derivation_evidence: object
    confidence_intervals: ConfidenceIntervals
    bootstrap_derivation_evidence: object

    @field_validator("result_digest")
    @classmethod
    def result_digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="result digest")

    @field_validator("metric_derivation_evidence", mode="before")
    @classmethod
    def metric_evidence_is_exact(cls, value: object) -> object:
        if type(value) is not MetricDerivationEvidence:
            raise ValueError("metric derivation evidence must have its exact restricted type")
        return value

    @field_validator("bootstrap_derivation_evidence", mode="before")
    @classmethod
    def bootstrap_evidence_is_exact(cls, value: object) -> object:
        if type(value) is not BootstrapDerivationEvidence:
            raise ValueError("bootstrap evidence must have its exact restricted type")
        return value


class ScorecardPublicationRequest(ExternalContract):
    """Closed evaluator result plus deterministic public handoff inputs."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    corpus_artifact_digest: str
    defender_artifact_digest: str
    promotion_envelope: VerifiedPromotionEnvelope
    champion_decision: ChampionDecision
    metric_evidence: tuple[MetricPublicationEvidence, ...]
    feature_manifest: bytes
    thresholds: bytes
    latency_environment: bytes

    @field_validator("corpus_artifact_digest", "defender_artifact_digest")
    @classmethod
    def artifact_digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="evaluation input digest")

    @field_validator("metric_evidence", mode="before")
    @classmethod
    def metric_evidence_is_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("metric publication evidence must be an exact tuple")
        return value

    @field_validator("feature_manifest", "thresholds", "latency_environment", mode="before")
    @classmethod
    def public_documents_are_canonical_bytes(cls, value: object) -> bytes:
        if type(value) is dict:
            payload = canonical_json_bytes(value)
        elif type(value) is bytes:
            payload = value
        else:
            raise ValueError("public input documents must be exact objects or bytes")
        if not 0 < len(payload) <= _MAX_PUBLIC_ARTIFACT_BYTES:
            raise ValueError("public input document exceeds its resource cap")
        _validate_canonical_json(payload)
        _privacy_scan(payload)
        return payload

    @model_validator(mode="after")
    def request_is_closed(self) -> ScorecardPublicationRequest:
        if type(self.promotion_envelope) is not VerifiedPromotionEnvelope:
            raise ValueError("promotion envelope must have its exact type")
        if type(self.champion_decision) is not ChampionDecision:
            raise ValueError("champion decision must have its exact type")
        if tuple(item.arm for item in self.metric_evidence) != tuple(DefenseArm):
            raise ValueError("metric evidence must contain the three arms in exact order")
        if any(type(item) is not MetricPublicationEvidence for item in self.metric_evidence):
            raise ValueError("metric publication evidence must have exact contracts")
        return self

    def rebuild(self, **updates: object) -> ScorecardPublicationRequest:
        fields = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name != "schema_version"
        }
        fields.update(updates)
        return ScorecardPublicationRequest.model_validate(fields)


class LeaderboardRow(ExternalContract):
    """One matched-budget primary arm summary with immutable metric lineage."""

    arm: DefenseArm
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None
    campaign_recall: float | None
    pr_auc: float | None
    roc_auc: float | None
    ece: float | None
    p95_latency_ms: float | None
    time_to_alert_p50_seconds: float | None
    time_to_alert_p95_seconds: float | None
    challenge_rate: float
    false_decline_rate: float | None
    review_case_rate: float
    false_intervention_count: int
    false_interventions_per_10k: float | None
    challenge_count: int
    review_case_count: int
    analyst_minutes: int
    preventable_settled_value: Decimal
    value_escaped: Decimal
    confidence_intervals: tuple[ConfidenceInterval, ...]
    metric_artifact_sha256: str

    @field_validator(
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "campaign_recall",
        "pr_auc",
        "roc_auc",
        "ece",
        "p95_latency_ms",
        "time_to_alert_p50_seconds",
        "time_to_alert_p95_seconds",
        "challenge_rate",
        "false_decline_rate",
        "review_case_rate",
        "false_interventions_per_10k",
        mode="before",
    )
    @classmethod
    def finite_numbers_only(cls, value: object) -> object:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("leaderboard values must be exact finite floats or None")
        return value

    @field_validator(
        "false_intervention_count",
        "challenge_count",
        "review_case_count",
        "analyst_minutes",
        mode="before",
    )
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        if type(value) is not int or value < 0:
            raise ValueError("leaderboard counts must be exact nonnegative integers")
        return value

    @field_validator("confidence_intervals", mode="before")
    @classmethod
    def intervals_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("leaderboard confidence intervals must be an exact tuple")
        return value

    @model_validator(mode="after")
    def interval_items_are_exact(self) -> LeaderboardRow:
        if any(type(item) is not ConfidenceInterval for item in self.confidence_intervals):
            raise ValueError("leaderboard confidence intervals must be exact")
        return self

    @field_validator("metric_artifact_sha256")
    @classmethod
    def artifact_digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="leaderboard artifact digest")


class SliceSummary(ExternalContract):
    """One closed slice recall with exact artifact lineage."""

    arm: DefenseArm
    kind: Literal["family", "rail", "regime", "entity_cohort"]
    value: str
    row_count: int
    fraud_count: int
    recall: float | None
    metric_artifact_sha256: str

    @field_validator("metric_artifact_sha256")
    @classmethod
    def artifact_digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="slice artifact digest")

    @field_validator("row_count", "fraud_count", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        if type(value) is not int or value < 0:
            raise ValueError("slice counts must be exact nonnegative integers")
        return value

    @field_validator("recall", mode="before")
    @classmethod
    def recall_is_finite(cls, value: object) -> object:
        if value is not None and (
            type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError("slice recall must be a finite rate or None")
        return value


class PublicChampionDecision(ExternalContract):
    """Aggregate-only champion outcome with hidden result fingerprints removed."""

    status: ChampionStatus
    champion: DefenseArm | None
    failed_gate_codes: tuple[str, ...]
    arm_gate_results: tuple[ArmGateResult, ...]

    @field_validator("failed_gate_codes", "arm_gate_results", mode="before")
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("public champion decision collections must be exact tuples")
        return value

    @model_validator(mode="after")
    def decision_is_closed(self) -> PublicChampionDecision:
        if self.failed_gate_codes != tuple(sorted(set(self.failed_gate_codes))):
            raise ValueError("public champion gate codes must be sorted and unique")
        if tuple(item.arm for item in self.arm_gate_results) != tuple(DefenseArm):
            raise ValueError("public arm gate results must be complete and ordered")
        if (self.status is ChampionStatus.NO_PROMOTION) != (self.champion is None):
            raise ValueError("public no-promotion status must have no champion")
        if (
            self.status is ChampionStatus.PROMOTED
            and self.champion is not DefenseArm.LAYERED_HYBRID
        ):
            raise ValueError("only the public layered hybrid can be promoted")
        if self.status is ChampionStatus.RETAINED and self.champion not in {
            DefenseArm.RULES_ONLY,
            DefenseArm.GBDT_ONLY,
        }:
            raise ValueError("public retained champion must be a comparator")
        return self


class DefenseScorecard(ExternalContract):
    """Signed aggregate-only public scorecard without self-reference."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation_id: str
    defender_bundle_id: str
    corpus_digest: str
    split_digest: str
    promotion_envelope_digest: str
    champion_decision: PublicChampionDecision
    leaderboard: tuple[LeaderboardRow, ...]
    slice_summaries: tuple[SliceSummary, ...]
    public_artifacts: PublicArtifactIndex
    failed_checks: tuple[str, ...]
    limitations: tuple[str, ...]
    external_validity_statement: str
    core_digest: str
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator(
        "evaluation_id",
        "corpus_digest",
        "split_digest",
        "promotion_envelope_digest",
        "core_digest",
        "signer_key_id",
    )
    @classmethod
    def digest_fields_are_sha256(cls, value: str) -> str:
        return _validate_digest(value)

    @field_validator(
        "leaderboard", "slice_summaries", "failed_checks", "limitations", mode="before"
    )
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("scorecard collections must be exact tuples")
        return value

    @model_validator(mode="after")
    def scorecard_is_closed(self) -> DefenseScorecard:
        if type(self.champion_decision) is not PublicChampionDecision:
            raise ValueError("scorecard champion decision must be exact")
        if tuple(item.arm for item in self.leaderboard) != tuple(DefenseArm):
            raise ValueError("scorecard leaderboard must contain all arms in exact order")
        slice_keys = tuple((item.arm, item.kind, item.value) for item in self.slice_summaries)
        if slice_keys != tuple(sorted(slice_keys, key=_slice_sort_key)) or len(
            slice_keys
        ) != len(set(slice_keys)):
            raise ValueError("scorecard slices must be canonical and unique")
        if set(self.public_artifacts) != set(PUBLIC_ARTIFACT_MEDIA_TYPES):
            raise ValueError("scorecard public artifact allowlist is incomplete")
        if self.failed_checks != self.champion_decision.failed_gate_codes:
            raise ValueError("scorecard failed checks differ from champion evidence")
        if (
            self.limitations != _LIMITATIONS
            or self.external_validity_statement != _EXTERNAL_VALIDITY
        ):
            raise ValueError("scorecard limitations or external-validity warning changed")
        if self.core_digest != self.compute_core_digest():
            raise ValueError("scorecard core digest is inconsistent")
        _validate_signature_identity(self.signer_key_id, self.public_key_base64)
        if not _verify_signature(
            self.public_key_base64, self.unsigned_document(), self.signature_base64
        ):
            raise ValueError("scorecard signature is invalid")
        artifact_digests = {item.sha256 for item in self.public_artifacts.values()}
        if any(item.metric_artifact_sha256 not in artifact_digests for item in self.leaderboard):
            raise ValueError("leaderboard metric lineage is unresolved")
        if any(
            item.metric_artifact_sha256 not in artifact_digests
            for item in self.slice_summaries
        ):
            raise ValueError("slice metric lineage is unresolved")
        return self

    def core_document(self) -> dict[str, object]:
        return self._core_document(self.model_dump(mode="json"))

    @staticmethod
    def _core_document(document: Mapping[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in document.items()
            if key not in {
                "core_digest",
                "public_artifacts",
                "signer_key_id",
                "public_key_base64",
                "signature_base64",
            }
        }

    @staticmethod
    def digest_core_document(document: Mapping[str, object]) -> str:
        return _digest_document(DefenseScorecard._core_document(document))

    def compute_core_digest(self) -> str:
        return _digest_document(self.core_document())

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def to_json(self) -> bytes:
        try:
            checked = DefenseScorecard.model_validate(
                {
                    **self.model_dump(
                        mode="python", warnings=False, exclude={"public_artifacts"}
                    ),
                    "public_artifacts": self.public_artifacts,
                },
                strict=True,
            )
            payload = canonical_json_bytes(checked.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise ReportingContractError("scorecard failed semantic revalidation") from error
        if len(payload) > _MAX_SCORECARD_BYTES:
            raise ReportingContractError("scorecard exceeds its resource cap")
        _privacy_scan(payload)
        return payload

    @classmethod
    def from_json(
        cls,
        payload: bytes,
        *,
        artifact_store: ArtifactStore,
        signer: RunSigningIdentity,
    ) -> DefenseScorecard:
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_SCORECARD_BYTES:
            raise ReportingContractError("scorecard payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise ReportingContractError("scorecard must be a JSON object")
            _tupleize_scorecard_document(document)
            scorecard = cls.model_validate(document)
            _verify_pinned_signer(scorecard, signer)
            if scorecard.to_json() != payload:
                raise ReportingContractError("scorecard JSON is not canonical")
            _validate_public_artifacts(
                scorecard.public_artifacts,
                artifact_store=artifact_store,
                include_scorecard=False,
            )
            _validate_scorecard_cross_references(scorecard)
            return scorecard
        except ReportingContractError:
            raise
        except (TypeError, ValueError, ValidationError, WireContractError) as error:
            raise ReportingContractError("scorecard failed closed validation") from error


class EvaluationArtifactBundle(ExternalContract):
    """Signed outer manifest adding the final non-self-referential scorecard ref."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation_id: str
    scorecard_sha256: str
    public_artifacts: PublicArtifactIndex
    signer_key_id: str
    public_key_base64: str
    signature_base64: str
    bundle_digest: str

    @field_validator("evaluation_id", "scorecard_sha256", "signer_key_id", "bundle_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value)

    @model_validator(mode="after")
    def bundle_is_closed(self) -> EvaluationArtifactBundle:
        if set(self.public_artifacts) != set(_ALL_PUBLIC_MEDIA_TYPES):
            raise ValueError("evaluation bundle public allowlist is incomplete")
        if self.public_artifacts[SCORECARD_ARTIFACT_NAME].sha256 != self.scorecard_sha256:
            raise ValueError("evaluation bundle scorecard reference differs")
        _validate_signature_identity(self.signer_key_id, self.public_key_base64)
        expected = _digest_document(self.unsigned_document())
        if self.bundle_digest != expected:
            raise ValueError("evaluation bundle digest is inconsistent")
        if not _verify_signature(
            self.public_key_base64, self.signing_document(), self.signature_base64
        ):
            raise ValueError("evaluation bundle signature is invalid")
        return self

    def signing_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "bundle_digest"}
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"bundle_digest"})

    def to_json(self) -> bytes:
        try:
            checked = EvaluationArtifactBundle.model_validate(
                {
                    **self.model_dump(
                        mode="python", warnings=False, exclude={"public_artifacts"}
                    ),
                    "public_artifacts": self.public_artifacts,
                },
                strict=True,
            )
            payload = canonical_json_bytes(checked.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise ReportingContractError(
                "evaluation bundle failed semantic revalidation"
            ) from error
        if len(payload) > _MAX_BUNDLE_BYTES:
            raise ReportingContractError("evaluation bundle exceeds its resource cap")
        _privacy_scan(payload)
        return payload

    @classmethod
    def from_json(
        cls,
        payload: bytes,
        *,
        artifact_store: ArtifactStore,
        signer: RunSigningIdentity,
    ) -> EvaluationArtifactBundle:
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_BUNDLE_BYTES:
            raise ReportingContractError("evaluation bundle payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise ReportingContractError("evaluation bundle must be a JSON object")
            artifacts = document.get("public_artifacts")
            if type(artifacts) is dict and type(artifacts.get("entries")) is list:
                artifacts["entries"] = tuple(
                    _reference_from_document(item) for item in artifacts["entries"]
                )
            bundle = cls.model_validate(document)
            _verify_pinned_signer(bundle, signer)
            if bundle.to_json() != payload:
                raise ReportingContractError("evaluation bundle JSON is not canonical")
            _validate_public_artifacts(
                bundle.public_artifacts,
                artifact_store=artifact_store,
                include_scorecard=True,
            )
            scorecard = bundle.scorecard(artifact_store=artifact_store, signer=signer)
            if scorecard.evaluation_id != bundle.evaluation_id:
                raise ReportingContractError("bundle and scorecard evaluation IDs differ")
            return bundle
        except ReportingContractError:
            raise
        except (TypeError, ValueError, ValidationError, WireContractError) as error:
            raise ReportingContractError("evaluation bundle failed closed validation") from error

    def bundle_ref(self) -> ArtifactRef:
        payload = self.to_json()
        digest = _digest_bytes(payload)
        return ArtifactRef(
            digest,
            EVALUATION_BUNDLE_MEDIA_TYPE,
            len(payload),
            f"{digest}/payload",
        )

    def scorecard(
        self, *, artifact_store: ArtifactStore, signer: RunSigningIdentity
    ) -> DefenseScorecard:
        reference = self.public_artifacts[SCORECARD_ARTIFACT_NAME]
        payload = artifact_store.read(reference.as_artifact_ref())
        return DefenseScorecard.from_json(
            payload, artifact_store=artifact_store, signer=signer
        )


def publish_scorecard(
    request: ScorecardPublicationRequest,
    *,
    artifact_store: ArtifactStore,
    signer: RunSigningIdentity,
    evaluator_verifier: EvaluatorReplayVerifier,
    hidden_proof_verifier: EvaluatorReplayVerifier,
) -> tuple[DefenseScorecard, EvaluationArtifactBundle]:
    """Verify restricted inputs transiently, then atomically publish aggregate artifacts."""
    try:
        checked = _validate_publication_request(
            request,
            evaluator_verifier=evaluator_verifier,
            hidden_proof_verifier=hidden_proof_verifier,
        )
        if type(artifact_store) is not ArtifactStore or type(signer) is not RunSigningIdentity:
            raise ReportingContractError("publication dependencies must have exact types")
        signer_key_id = signer.key_id
        signer_public_key = signer.public_key_base64
        _validate_signature_identity(signer_key_id, signer_public_key)
        primary_results = _primary_results(checked.promotion_envelope)
        report_by_arm = _validated_metric_evidence(checked.metric_evidence, primary_results)
        _validate_defender_binding(checked, primary_results)
        artifact_payloads = _render_public_artifacts(
            checked,
            primary_results=primary_results,
            report_by_arm=report_by_arm,
        )
        stored: dict[str, ArtifactRef] = {}
        for name in sorted(artifact_payloads):
            payload = artifact_payloads[name]
            _privacy_scan(payload)
            if len(payload) > _MAX_PUBLIC_ARTIFACT_BYTES:
                raise ReportingContractError("public artifact exceeds resource cap")
            stored[name] = artifact_store.put_bytes(
                payload, PUBLIC_ARTIFACT_MEDIA_TYPES[name]
            )
        artifact_index = PublicArtifactIndex.from_refs(stored)
        scorecard = _build_scorecard(
            checked,
            primary_results=primary_results,
            report_by_arm=report_by_arm,
            public_artifacts=artifact_index,
            signer=signer,
            signer_key_id=signer_key_id,
            signer_public_key=signer_public_key,
        )
        scorecard_payload = scorecard.to_json()
        scorecard_ref = artifact_store.put_bytes(scorecard_payload, SCORECARD_MEDIA_TYPE)
        full_refs = {**stored, SCORECARD_ARTIFACT_NAME: scorecard_ref}
        full_index = PublicArtifactIndex.from_refs(full_refs)
        bundle = _build_evaluation_bundle(
            scorecard=scorecard,
            public_artifacts=full_index,
            signer=signer,
            signer_key_id=signer_key_id,
            signer_public_key=signer_public_key,
        )
        bundle_ref = artifact_store.put_bytes(bundle.to_json(), EVALUATION_BUNDLE_MEDIA_TYPE)
        if bundle_ref != bundle.bundle_ref():
            raise ReportingContractError("stored evaluation bundle reference differs")
        loaded = load_evaluation_bundle(bundle_ref, artifact_store=artifact_store, signer=signer)
        if loaded != bundle:
            raise ReportingContractError("published evaluation bundle did not reload exactly")
        return scorecard, bundle
    except ReportingContractError:
        raise
    except (
        ArithmeticError,
        AttributeError,
        MemoryError,
        OverflowError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise ReportingContractError("scorecard publication failed closed") from error


def load_evaluation_bundle(
    ref: ArtifactRef,
    *,
    artifact_store: ArtifactStore,
    signer: RunSigningIdentity,
) -> EvaluationArtifactBundle:
    """Re-read and fully revalidate one signed public evaluation bundle."""
    if type(ref) is not ArtifactRef or ref.media_type != EVALUATION_BUNDLE_MEDIA_TYPE:
        raise ReportingContractError("evaluation bundle reference is invalid")
    if ref.size_bytes > _MAX_BUNDLE_BYTES:
        raise ReportingContractError("evaluation bundle reference exceeds its resource cap")
    try:
        payload = artifact_store.read(ref)
    except (TypeError, ValueError) as error:
        raise ReportingContractError("evaluation bundle artifact is invalid") from error
    return EvaluationArtifactBundle.from_json(
        payload, artifact_store=artifact_store, signer=signer
    )


def _validate_publication_request(
    request: object,
    *,
    evaluator_verifier: EvaluatorReplayVerifier,
    hidden_proof_verifier: EvaluatorReplayVerifier,
) -> ScorecardPublicationRequest:
    if type(request) is not ScorecardPublicationRequest:
        raise ReportingContractError("publication request must have its exact type")
    if type(evaluator_verifier) is not EvaluatorReplayVerifier or type(
        hidden_proof_verifier
    ) is not EvaluatorReplayVerifier:
        raise ReportingContractError("publication requires exact pinned evaluator verifiers")
    try:
        checked = ScorecardPublicationRequest.model_validate(
            {
                **request.model_dump(
                    mode="python",
                    warnings=False,
                    exclude={
                        "promotion_envelope",
                        "champion_decision",
                        "metric_evidence",
                    },
                ),
                "promotion_envelope": request.promotion_envelope,
                "champion_decision": request.champion_decision,
                "metric_evidence": request.metric_evidence,
            },
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ReportingContractError("publication request failed semantic revalidation") from error
    if not evaluator_verifier.verify_promotion_envelope(checked.promotion_envelope):
        raise ReportingContractError("promotion envelope signature is invalid")
    expected = evaluate_promotion_gates(
        checked.promotion_envelope,
        GateConfig.competition(),
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_proof_verifier,
    )
    if checked.champion_decision != expected:
        raise ReportingContractError("champion decision differs from verified promotion evidence")
    return checked


def _primary_results(envelope: VerifiedPromotionEnvelope) -> tuple[ReplayResult, ...]:
    primary = tuple(
        row
        for row in envelope.combined_batch.results
        if row.evaluation.kind is EvaluationKind.CHRONOLOGICAL
        and row.evaluation.value == "development"
    )
    if tuple(row.arm for row in primary) != tuple(DefenseArm):
        raise ReportingContractError("primary leaderboard evidence is incomplete")
    return primary


def _validated_metric_evidence(
    evidence_rows: tuple[MetricPublicationEvidence, ...],
    results: tuple[ReplayResult, ...],
) -> dict[DefenseArm, tuple[MetricReport, ConfidenceIntervals]]:
    result_by_arm = {row.arm: row for row in results}
    output: dict[DefenseArm, tuple[MetricReport, ConfidenceIntervals]] = {}
    for evidence in evidence_rows:
        if type(evidence) is not MetricPublicationEvidence:
            raise ReportingContractError("metric publication evidence is not exact")
        try:
            report = MetricReport.from_json(
                evidence.metric_report.to_json(),
                evidence=cast(MetricDerivationEvidence, evidence.metric_derivation_evidence),
            )
            confidence = ConfidenceIntervals.from_json(
                evidence.confidence_intervals.to_json(),
                evidence=cast(
                    BootstrapDerivationEvidence,
                    evidence.bootstrap_derivation_evidence,
                ),
            )
        except (TypeError, ValueError) as error:
            raise ReportingContractError("Task11 metric derivation evidence is invalid") from error
        result = result_by_arm[evidence.arm]
        if evidence.result_digest != result.result_digest:
            raise ReportingContractError("metric evidence result binding differs")
        if report.report_digest != result.metric_report_digest:
            raise ReportingContractError("metric report digest differs from signed replay")
        if report.evaluator_input_digest != confidence.evaluator_input_digest:
            raise ReportingContractError("metric report and bootstrap inputs differ")
        _validate_report_projection(report, result)
        output[evidence.arm] = (report, confidence)
    if tuple(output) != tuple(DefenseArm):
        raise ReportingContractError("metric publication evidence is incomplete")
    return output


def _validate_report_projection(report: MetricReport, result: ReplayResult) -> None:
    classification = report.classification
    operations = report.operations
    metrics = result.metrics
    expected_slices = tuple(
        sorted(
            ((item.kind, item.value, item.recall.value) for item in classification.slices),
            key=lambda item: (item[0], item[1]),
        )
    )
    actual_slices = tuple(
        (item.kind, item.value, item.recall) for item in metrics.slice_performance
    )
    false_decline = (
        operations.false_decline_count / classification.legitimate_count
        if classification.legitimate_count
        else None
    )
    expected = (
        classification.row_count,
        classification.recall.value,
        report.calibration.ece.value,
        report.engineering.end_to_end_ms.p95.value,
        report.value.preventable_settled_value,
        report.value.value_escaped,
        operations.review_case_count,
        operations.challenge_count / classification.row_count,
        false_decline,
        operations.review_case_count / classification.row_count,
        expected_slices,
    )
    actual = (
        metrics.row_count,
        metrics.recall,
        metrics.ece,
        metrics.p95_latency_ms,
        metrics.preventable_settled_value,
        metrics.value_escaped,
        metrics.review_case_count,
        metrics.challenge_rate,
        metrics.false_decline.value,
        metrics.review_case_rate,
        actual_slices,
    )
    if expected != actual:
        raise ReportingContractError("Task11 aggregates differ from signed Task12 projection")


def _validate_defender_binding(
    request: ScorecardPublicationRequest, primary: tuple[ReplayResult, ...]
) -> None:
    identities = {
        (
            row.candidate_role.bundle_id,
            row.bundle_manifest_digest,
            row.evaluation_lineage.defender_top_ref_digest,
        )
        for row in primary
    }
    if len(identities) != 1:
        raise ReportingContractError("primary defender identity is inconsistent")
    _, manifest_digest, top_ref_digest = next(iter(identities))
    if (
        manifest_digest != request.defender_artifact_digest
        or top_ref_digest != request.defender_artifact_digest
    ):
        raise ReportingContractError("requested defender artifact differs from signed replay")


def _render_public_artifacts(
    request: ScorecardPublicationRequest,
    *,
    primary_results: tuple[ReplayResult, ...],
    report_by_arm: dict[DefenseArm, tuple[MetricReport, ConfidenceIntervals]],
) -> dict[str, bytes]:
    leaderboard_rows = []
    calibration_rows = []
    slice_rows = []
    value_rows = []
    latency_rows = []
    for result in primary_results:
        report, confidence = report_by_arm[result.arm]
        classification = report.classification
        operations = report.operations
        leaderboard_rows.append(
            (
                result.arm.value,
                _float_text(classification.precision.value),
                _float_text(classification.recall.value),
                _float_text(classification.f1.value),
                _float_text(classification.false_positive_rate.value),
                _float_text(classification.campaign_recall.value),
                _float_text(classification.pr_auc.value),
                _float_text(classification.roc_auc.value),
                _float_text(report.calibration.ece.value),
                _float_text(report.engineering.end_to_end_ms.p95.value),
                _float_text(report.alerts.p50_seconds.value),
                _float_text(report.alerts.p95_seconds.value),
                _float_text(result.metrics.challenge_rate),
                _float_text(result.metrics.false_decline.value),
                _float_text(result.metrics.review_case_rate),
                str(operations.false_intervention_count),
                _float_text(operations.false_interventions_per_10k.value),
                str(operations.challenge_count),
                str(operations.review_case_count),
                str(operations.analyst_minutes),
                str(report.value.preventable_settled_value),
                str(report.value.value_escaped),
                result.metric_report_digest,
                canonical_json_bytes(
                    [item.model_dump(mode="json") for item in confidence.intervals]
                ).decode("utf-8"),
            )
        )
        for item in classification.slices:
            slice_rows.append(
                (
                    result.arm.value,
                    item.kind,
                    item.value,
                    str(item.row_count),
                    str(item.fraud_count),
                    _float_text(item.precision.value),
                    _float_text(item.recall.value),
                    _float_text(item.f1.value),
                    _float_text(item.pr_auc.value),
                    _float_text(item.roc_auc.value),
                    _float_text(item.campaign_recall.value),
                )
            )
        for reliability_bin in report.calibration.reliability_bins:
            calibration_rows.append(
                (
                    result.arm.value,
                    str(reliability_bin.bin_index),
                    str(reliability_bin.count),
                    _float_text(reliability_bin.lower_score),
                    _float_text(reliability_bin.upper_score),
                    _float_text(reliability_bin.mean_prediction),
                    _float_text(reliability_bin.observed_frequency),
                )
            )
        value_rows.append(
            (
                result.arm.value,
                str(report.value.fraudulent_net_settled_value),
                str(report.value.preventable_settled_value),
                str(report.value.value_escaped),
                str(report.value.value_before_first_alert),
                str(report.value.remaining_preventable_at_alert),
                str(operations.false_intervention_count),
                str(operations.challenge_count),
                str(operations.review_case_count),
                str(operations.analyst_minutes),
                str(operations.peak_backlog_count),
                str(operations.sla_breaches),
            )
        )
        latency_rows.append(
            {
                "arm": result.arm.value,
                "sample_count": report.engineering.end_to_end_ms.sample_count,
                "p50_ms": report.engineering.end_to_end_ms.p50.value,
                "p90_ms": report.engineering.end_to_end_ms.p90.value,
                "p95_ms": report.engineering.end_to_end_ms.p95.value,
                "p99_ms": report.engineering.end_to_end_ms.p99.value,
            }
        )
    payloads = {
        "leaderboard.csv": _csv_bytes(
            (
                "arm",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "campaign_recall",
                "pr_auc",
                "roc_auc",
                "ece",
                "p95_latency_ms",
                "time_to_alert_p50_seconds",
                "time_to_alert_p95_seconds",
                "challenge_rate",
                "false_decline_rate",
                "review_case_rate",
                "false_intervention_count",
                "false_interventions_per_10k",
                "challenge_count",
                "review_case_count",
                "analyst_minutes",
                "preventable_settled_value",
                "value_escaped",
                "metric_report_digest",
                "confidence_intervals_json",
            ),
            tuple(leaderboard_rows),
        ),
        "slice-metrics.csv": _csv_bytes(
            (
                "arm",
                "kind",
                "value",
                "row_count",
                "fraud_count",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "campaign_recall",
            ),
            tuple(sorted(slice_rows)),
        ),
        "calibration.csv": _csv_bytes(
            (
                "arm",
                "bin_index",
                "count",
                "lower_score",
                "upper_score",
                "mean_prediction",
                "observed_frequency",
            ),
            tuple(calibration_rows),
        ),
        "value-workload.csv": _csv_bytes(
            (
                "arm",
                "fraudulent_net_settled_value",
                "preventable_settled_value",
                "value_escaped",
                "value_before_first_alert",
                "remaining_preventable_at_alert",
                "false_interventions",
                "challenges",
                "review_cases",
                "analyst_minutes",
                "peak_backlog",
                "sla_breaches",
            ),
            tuple(value_rows),
        ),
        "feature-manifest.json": request.feature_manifest,
        "thresholds.json": request.thresholds,
        "latency-evidence.json": canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                "environment": strict_json_loads(request.latency_environment),
                "observational_evidence": latency_rows,
                "raw_samples_published": False,
            }
        ),
        "data-card.md": _markdown(
            "APAR synthetic defense data card",
            (
                _EXTERNAL_VALIDITY,
                "The corpus contains authorized APAR-generated synthetic events only.",
                "Prevalence, identity mix, and value distributions are simulator evidence.",
            ),
        ),
        "model-card.md": _markdown(
            "APAR defense model card",
            (
                _EXTERNAL_VALIDITY,
                "The compared arms are deterministic rules, calibrated GBDT, and a layered hybrid.",
                "Promotion is controlled by the frozen signed evaluation envelope and hard gates.",
            ),
        ),
        "limitations.md": _markdown("Limitations", (_EXTERNAL_VALIDITY, *_LIMITATIONS)),
    }
    evidence_lines = tuple(
        f"- {name}: `{_digest_bytes(payloads[name])}`"
        for name in sorted(payloads)
    )
    payloads["defense-scorecard.md"] = _markdown(
        "APAR Defend judge scorecard",
        (
            _EXTERNAL_VALIDITY,
            f"Champion status: `{request.champion_decision.status.value}`.",
            "All displayed metrics are in the signed immutable artifacts below.",
            *evidence_lines,
        ),
    )
    if set(payloads) != set(PUBLIC_ARTIFACT_MEDIA_TYPES):
        raise ReportingContractError("rendered public artifact allowlist differs")
    return payloads


def _build_scorecard(
    request: ScorecardPublicationRequest,
    *,
    primary_results: tuple[ReplayResult, ...],
    report_by_arm: dict[DefenseArm, tuple[MetricReport, ConfidenceIntervals]],
    public_artifacts: PublicArtifactIndex,
    signer: RunSigningIdentity,
    signer_key_id: str,
    signer_public_key: str,
) -> DefenseScorecard:
    leaderboard_ref = public_artifacts["leaderboard.csv"].sha256
    slice_ref = public_artifacts["slice-metrics.csv"].sha256
    leaderboard = tuple(
        LeaderboardRow(
            arm=result.arm,
            precision=report_by_arm[result.arm][0].classification.precision.value,
            recall=report_by_arm[result.arm][0].classification.recall.value,
            f1=report_by_arm[result.arm][0].classification.f1.value,
            false_positive_rate=(
                report_by_arm[result.arm][0].classification.false_positive_rate.value
            ),
            campaign_recall=(
                report_by_arm[result.arm][0].classification.campaign_recall.value
            ),
            pr_auc=report_by_arm[result.arm][0].classification.pr_auc.value,
            roc_auc=report_by_arm[result.arm][0].classification.roc_auc.value,
            ece=report_by_arm[result.arm][0].calibration.ece.value,
            p95_latency_ms=report_by_arm[result.arm][0].engineering.end_to_end_ms.p95.value,
            time_to_alert_p50_seconds=(
                report_by_arm[result.arm][0].alerts.p50_seconds.value
            ),
            time_to_alert_p95_seconds=(
                report_by_arm[result.arm][0].alerts.p95_seconds.value
            ),
            challenge_rate=result.metrics.challenge_rate,
            false_decline_rate=result.metrics.false_decline.value,
            review_case_rate=result.metrics.review_case_rate,
            false_intervention_count=(
                report_by_arm[result.arm][0].operations.false_intervention_count
            ),
            false_interventions_per_10k=(
                report_by_arm[result.arm][0]
                .operations.false_interventions_per_10k.value
            ),
            challenge_count=report_by_arm[result.arm][0].operations.challenge_count,
            review_case_count=report_by_arm[result.arm][0].operations.review_case_count,
            analyst_minutes=report_by_arm[result.arm][0].operations.analyst_minutes,
            preventable_settled_value=result.metrics.preventable_settled_value,
            value_escaped=result.metrics.value_escaped,
            confidence_intervals=report_by_arm[result.arm][1].intervals,
            metric_artifact_sha256=leaderboard_ref,
        )
        for result in primary_results
    )
    slice_summaries = tuple(
        sorted(
            (
                SliceSummary(
                    arm=result.arm,
                    kind=item.kind,
                    value=item.value,
                    row_count=item.row_count,
                    fraud_count=item.fraud_count,
                    recall=item.recall.value,
                    metric_artifact_sha256=slice_ref,
                )
                for result in primary_results
                for item in report_by_arm[result.arm][0].classification.slices
            ),
            key=lambda item: _slice_sort_key((item.arm, item.kind, item.value)),
        )
    )
    primary_identity = primary_results[0].candidate_role
    split_digest = primary_results[0].evaluation_lineage.split_digest
    public_decision = PublicChampionDecision(
        status=request.champion_decision.status,
        champion=request.champion_decision.champion,
        failed_gate_codes=request.champion_decision.failed_gate_codes,
        arm_gate_results=request.champion_decision.arm_gate_results,
    )
    core_fields: dict[str, object] = {
        "schema_version": "1.0.0",
        "defender_bundle_id": primary_identity.bundle_id,
        "corpus_digest": request.corpus_artifact_digest,
        "split_digest": split_digest,
        "promotion_envelope_digest": request.promotion_envelope.envelope_digest,
        "champion_decision": public_decision,
        "leaderboard": leaderboard,
        "slice_summaries": slice_summaries,
        "failed_checks": request.champion_decision.failed_gate_codes,
        "limitations": _LIMITATIONS,
        "external_validity_statement": _EXTERNAL_VALIDITY,
    }
    evaluation_id = _digest_document(
        {
            "schema_version": "1.0.0",
            "corpus_digest": request.corpus_artifact_digest,
            "defender_digest": request.defender_artifact_digest,
            "promotion_envelope_digest": request.promotion_envelope.envelope_digest,
            "public_champion_decision": _json_tree(public_decision),
            "metric_report_digests": [row.metric_report_digest for row in primary_results],
            "feature_manifest_digest": _digest_bytes(request.feature_manifest),
            "thresholds_digest": _digest_bytes(request.thresholds),
        }
    )
    core_fields["evaluation_id"] = evaluation_id
    json_core = _json_tree(core_fields)
    core_digest = _digest_document(json_core)
    unsigned = {
        **core_fields,
        "public_artifacts": public_artifacts,
        "core_digest": core_digest,
        "signer_key_id": signer_key_id,
        "public_key_base64": signer_public_key,
    }
    signature = signer.sign(_json_tree(unsigned))
    _verify_signer_snapshot(signer, signer_key_id, signer_public_key, unsigned, signature)
    return DefenseScorecard.model_validate(
        {**unsigned, "signature_base64": signature}
    )


def _build_evaluation_bundle(
    *,
    scorecard: DefenseScorecard,
    public_artifacts: PublicArtifactIndex,
    signer: RunSigningIdentity,
    signer_key_id: str,
    signer_public_key: str,
) -> EvaluationArtifactBundle:
    fields: dict[str, object] = {
        "schema_version": "1.0.0",
        "evaluation_id": scorecard.evaluation_id,
        "scorecard_sha256": public_artifacts[SCORECARD_ARTIFACT_NAME].sha256,
        "public_artifacts": public_artifacts,
        "signer_key_id": signer_key_id,
        "public_key_base64": signer_public_key,
    }
    signature = signer.sign(_json_tree(fields))
    signed = {**fields, "signature_base64": signature}
    _verify_signer_snapshot(signer, signer_key_id, signer_public_key, fields, signature)
    return EvaluationArtifactBundle.model_validate(
        {**signed, "bundle_digest": _digest_document(_json_tree(signed))}
    )


def _verify_pinned_signer(
    value: DefenseScorecard | EvaluationArtifactBundle,
    signer: RunSigningIdentity,
) -> None:
    if type(signer) is not RunSigningIdentity:
        raise ReportingContractError("public artifact signer is not exact")
    if (
        value.signer_key_id != signer.key_id
        or value.public_key_base64 != signer.public_key_base64
    ):
        raise ReportingContractError("public artifact signer differs from pinned authority")
    if type(value) is DefenseScorecard:
        document = value.unsigned_document()
    else:
        document = cast(EvaluationArtifactBundle, value).signing_document()
    if not signer.verify(document, value.signature_base64):
        raise ReportingContractError("pinned public artifact signature is invalid")


def _verify_signer_snapshot(
    signer: RunSigningIdentity,
    key_id: str,
    public_key: str,
    document: object,
    signature: str,
) -> None:
    if signer.key_id != key_id or signer.public_key_base64 != public_key:
        raise ReportingContractError("signing authority identity changed during publication")
    if not signer.verify(_json_tree(document), signature) or not _verify_signature(
        public_key, _json_tree(document), signature
    ):
        raise ReportingContractError("public artifact signature did not verify")


def _validate_signature_identity(key_id: str, public_key_base64: str) -> None:
    _validate_digest(key_id, label="signer key ID")
    if type(public_key_base64) is not str:
        raise ValueError("public signing key is invalid")
    try:
        public = base64.b64decode(public_key_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("public signing key is invalid") from error
    if len(public) != 32 or _digest_bytes(public) != key_id:
        raise ValueError("public signing identity is inconsistent")


def _verify_signature(public_key_base64: str, document: object, signature_base64: str) -> bool:
    try:
        public = base64.b64decode(public_key_base64, validate=True)
        signature = base64.b64decode(signature_base64, validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(
            signature, canonical_json_bytes(document)
        )
    except (InvalidSignature, TypeError, ValueError, binascii.Error):
        return False
    return True


def _validate_public_artifacts(
    index: PublicArtifactIndex,
    *,
    artifact_store: ArtifactStore,
    include_scorecard: bool,
) -> None:
    expected = set(_ALL_PUBLIC_MEDIA_TYPES if include_scorecard else PUBLIC_ARTIFACT_MEDIA_TYPES)
    if set(index) != expected:
        raise ReportingContractError("public artifact allowlist differs")
    for name, reference in index.items():
        try:
            payload = artifact_store.read(reference.as_artifact_ref())
        except (TypeError, ValueError) as error:
            raise ReportingContractError("public artifact is missing or invalid") from error
        if len(payload) != reference.size_bytes or _digest_bytes(payload) != reference.sha256:
            raise ReportingContractError("public artifact content address differs")
        _privacy_scan(payload)
        if name.endswith(".json") and name != SCORECARD_ARTIFACT_NAME:
            _validate_canonical_json(payload)
        elif name.endswith(".csv"):
            _validate_csv(payload)
        elif name.endswith(".md"):
            _validate_markdown(payload)


def _validate_scorecard_cross_references(scorecard: DefenseScorecard) -> None:
    leaderboard_digest = scorecard.public_artifacts["leaderboard.csv"].sha256
    slice_digest = scorecard.public_artifacts["slice-metrics.csv"].sha256
    if any(item.metric_artifact_sha256 != leaderboard_digest for item in scorecard.leaderboard):
        raise ReportingContractError("leaderboard rows reference the wrong artifact")
    if any(item.metric_artifact_sha256 != slice_digest for item in scorecard.slice_summaries):
        raise ReportingContractError("slice rows reference the wrong artifact")


def _validate_canonical_json(payload: bytes) -> None:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_PUBLIC_ARTIFACT_BYTES:
        raise ValueError("public JSON payload is invalid")
    document = strict_json_loads(payload)
    if canonical_json_bytes(document) != payload:
        raise ValueError("public JSON payload is not canonical")


def _validate_csv(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportingContractError("public CSV is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text:
        raise ReportingContractError("public CSV newline contract differs")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ReportingContractError("public CSV shape is invalid")
    if _csv_bytes(tuple(rows[0]), tuple(tuple(row) for row in rows[1:])) != payload:
        raise ReportingContractError("public CSV is not canonical")


def _validate_markdown(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportingContractError("public Markdown is not UTF-8") from error
    if not text.startswith("# ") or not text.endswith("\n") or "\r" in text:
        raise ReportingContractError("public Markdown is not canonical")


def _privacy_scan(payload: bytes) -> None:
    if type(payload) is not bytes:
        raise ReportingContractError("public payload must be exact bytes")
    lowered = payload.lower()
    representation = repr(payload).lower().encode("ascii", errors="ignore")
    try:
        decoded = payload.decode("utf-8").lower().encode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportingContractError("public payload is not UTF-8") from error
    semantic = b""
    if payload.startswith((b"{", b"[")):
        try:
            semantic = repr(strict_json_loads(payload)).lower().encode(
                "utf-8", errors="ignore"
            )
        except WireContractError:
            semantic = decoded
    if any(
        token in lowered
        or token in representation
        or token in decoded
        or token in semantic
        for token in _FORBIDDEN_PUBLIC_TOKENS
    ):
        raise ReportingContractError("public payload contains restricted evaluator semantics")


def _csv_bytes(header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _markdown(title: str, paragraphs: tuple[str, ...]) -> bytes:
    text = f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"
    return text.encode("utf-8")


def _float_text(value: float | None) -> str:
    if value is None:
        return "undefined"
    if type(value) is not float or not math.isfinite(value):
        raise ReportingContractError("public metric must be finite or undefined")
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _slice_sort_key(value: tuple[DefenseArm, str, str]) -> tuple[int, str, str]:
    arm, kind, label = value
    return tuple(DefenseArm).index(arm), kind, label


def _tupleize_scorecard_document(document: dict[str, object]) -> None:
    for name in ("leaderboard", "slice_summaries", "failed_checks", "limitations"):
        value = document.get(name)
        if type(value) is list:
            document[name] = tuple(cast(list[object], value))
    leaderboard = document.get("leaderboard")
    if type(leaderboard) is tuple:
        for item in leaderboard:
            if type(item) is dict and type(item.get("confidence_intervals")) is list:
                item["confidence_intervals"] = tuple(item["confidence_intervals"])
    artifacts = document.get("public_artifacts")
    if type(artifacts) is dict and type(artifacts.get("entries")) is list:
        artifacts["entries"] = tuple(
            _reference_from_document(item) for item in artifacts["entries"]
        )
    champion = document.get("champion_decision")
    if type(champion) is dict:
        for name in ("failed_gate_codes", "arm_gate_results", "evaluated_result_digests"):
            if type(champion.get(name)) is list:
                champion[name] = tuple(champion[name])
        arm_results = champion.get("arm_gate_results")
        if type(arm_results) is tuple:
            for item in arm_results:
                if type(item) is dict and type(item.get("failed_gate_codes")) is list:
                    item["failed_gate_codes"] = tuple(item["failed_gate_codes"])


def _reference_from_document(value: object) -> PublicArtifactReference:
    if type(value) is not dict:
        raise ValueError("public artifact reference must be an object")
    document = cast(dict[str, object], value)
    if set(document) != {"name", "sha256", "media_type", "size_bytes"}:
        raise ValueError("public artifact reference fields differ")
    return PublicArtifactReference(
        name=cast(str, document["name"]),
        sha256=cast(str, document["sha256"]),
        media_type=cast(str, document["media_type"]),
        size_bytes=cast(int, document["size_bytes"]),
    )


def _json_tree(value: object) -> object:
    if isinstance(value, ExternalContract):
        return value.model_dump(mode="json")
    if isinstance(value, PublicArtifactReference):
        return {
            "name": value.name,
            "sha256": value.sha256,
            "media_type": value.media_type,
            "size_bytes": value.size_bytes,
        }
    if type(value) is dict:
        return {str(key): _json_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_json_tree(item) for item in value]
    return value


__all__ = [
    "EVALUATION_BUNDLE_MEDIA_TYPE",
    "PUBLIC_ARTIFACT_MEDIA_TYPES",
    "SCORECARD_ARTIFACT_NAME",
    "SCORECARD_MEDIA_TYPE",
    "DefenseScorecard",
    "EvaluationArtifactBundle",
    "LeaderboardRow",
    "MetricPublicationEvidence",
    "PublicArtifactIndex",
    "PublicArtifactReference",
    "PublicChampionDecision",
    "ReportingContractError",
    "ScorecardPublicationRequest",
    "SliceSummary",
    "load_evaluation_bundle",
    "publish_scorecard",
]
