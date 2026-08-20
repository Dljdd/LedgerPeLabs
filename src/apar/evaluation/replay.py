"""Verified identical-row replay for rules, GBDT, and layered hybrid arms."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator

from apar.cases import (
    QueueConfig,
    ReviewCaseCounter,
    bind_review_case_counter,
    group_cases,
    simulate_case_queue,
)
from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.defense.bundle import DefenderBundleManifest, LoadedDefenderBundle
from apar.defense.contracts import ObservedEvent, PolicyThresholds
from apar.defense.policy import ActionPolicy, DefenseDecision
from apar.defense.rules import DefenseReason, RuleEngine, RuleManifest, RuleResult
from apar.defense.thresholds import (
    ThresholdReport,
    select_policy_thresholds,
)
from apar.evaluation.contracts import EvaluationTruthRow, Family, FrozenCorpus
from apar.evaluation.defender_attestation import (
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.evaluation.gates import (
    AssuranceEvidence,
    CandidateBundleRole,
    CandidateRoleEvidence,
    DefenseArm,
    EvaluationDescriptor,
    EvaluationKind,
    EvaluationLineage,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    PromotionMetrics,
    RateEvidence,
    ReplayFailure,
    ReplayResult,
    SlicePerformance,
    VerifiedReplayBatch,
)
from apar.evaluation.metrics import (
    LatencySample,
    MetricDerivationEvidence,
    MetricReport,
    MetricReportInputs,
    SliceAssignment,
    SliceManifest,
    compute_metric_report,
)
from apar.evaluation.regimes import (
    DerivedRegimeManifest,
    RegimeSpec,
    derive_regime,
    frozen_corpus_digest,
)
from apar.evaluation.splits import EntityCohort, EvaluationSplit, make_evaluation_split
from apar.evaluation_hidden.defense_authority import (
    HiddenArmEvidenceBinding,
    HiddenBoundaryError,
    HiddenDecisionBinding,
    HiddenEvaluationAuthority,
    HiddenEvaluationCapability,
    HiddenReplayOutcome,
)
from apar.features.builders import FeatureMatrix
from apar.features.parity import audit_feature_matrix
from apar.features.state import FeatureVector
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef

_MAX_REPLAY_ROWS = 100_000
_MAX_REPLAY_BYTES = 32_000_000
_MODEL_FAILURES = {DefenseReason.MODEL_UNAVAILABLE, DefenseReason.MODEL_TIMEOUT}


class ReplayContractError(ValueError):
    """Replay rows, artifacts, or evaluator lineage are inconsistent."""


@dataclass(frozen=True, slots=True)
class _EvaluatedArm:
    result: ReplayResult
    hidden_evidence: HiddenArmEvidenceBinding


class _HiddenReplayInvocation(NamedTuple):
    matrix: FeatureMatrix
    defender: LoadedDefenderBundle
    defender_verifier: DefenderBundleVerifier
    defender_attestation: VerifiedDefenderAttestation
    thresholds: ReplayThresholdSet
    threshold_labels: np.ndarray
    threshold_values: np.ndarray | None
    case_counter: ReplayCaseCounterBinding
    model_failure: ModelFailure | None


class _FrozenDefenseReplay(NamedTuple):
    matrix: FeatureMatrix
    defender: LoadedDefenderBundle
    defender_verifier: DefenderBundleVerifier
    defender_attestation: VerifiedDefenderAttestation
    threshold_set: ReplayThresholdSet
    case_counter: ReplayCaseCounterBinding
    rows: tuple[FeatureVector, ...]
    events: tuple[ObservedEvent, ...]
    event_ids: tuple[str, ...]
    mandatory: tuple[bool, ...]
    decisions_by_arm: dict[DefenseArm, tuple[DefenseDecision, ...]]
    arm_scores: dict[DefenseArm, np.ndarray]
    feature_audit_passed: bool
    actual_failure: ModelFailure | None
    manifest_digest: str


class _HiddenEvaluationProduct(NamedTuple):
    evaluated: tuple[_EvaluatedArm, ...]
    evaluator_context_digest: str
    restricted_cohort_mapping_digest: str
    evaluation_lineage_digest: str
    evaluator_as_of: datetime


class ModelFailure(ExternalContract):
    """Declared model failure used only for audited GBDT failure/hybrid fallback."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    reason: DefenseReason
    failed_component_version: str

    @field_validator("failed_component_version")
    @classmethod
    def component_is_bounded_nonblank(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value or len(value) > 256:
            raise ValueError("failed component identity must be bounded nonblank text")
        return value

    @model_validator(mode="after")
    def reason_is_model_failure(self) -> ModelFailure:
        if self.reason not in _MODEL_FAILURES:
            raise ValueError("declared failure must be model unavailable or timeout")
        return self


class ReplayFeatureAssurance(ExternalContract):
    """Explicit Task4 leakage/parity evidence supplied to the replay gate."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    leakage_passed: bool
    parity_passed: bool
    leakage_evidence_digest: str
    parity_evidence_digest: str

    @field_validator("leakage_passed", "parity_passed", mode="before")
    @classmethod
    def flags_are_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("feature-assurance flags must be exact bools")
        return value

    @field_validator("leakage_evidence_digest", "parity_evidence_digest")
    @classmethod
    def evidence_is_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value


class ReplayRegimeEvidence(ExternalContract):
    """Evaluator-owned, re-derived proof of one exact robustness corpus."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    parent_corpus: FrozenCorpus
    derived_corpus: FrozenCorpus
    spec: RegimeSpec
    manifest: DerivedRegimeManifest
    control_corpus: FrozenCorpus | None = None
    evidence_digest: str

    @field_validator("evidence_digest")
    @classmethod
    def evidence_is_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def derivation_is_exact(self) -> ReplayRegimeEvidence:
        if (
            type(self.parent_corpus) is not FrozenCorpus
            or type(self.derived_corpus) is not FrozenCorpus
            or type(self.spec) is not RegimeSpec
            or type(self.manifest) is not DerivedRegimeManifest
            or (
                self.control_corpus is not None
                and type(self.control_corpus) is not FrozenCorpus
            )
        ):
            raise ValueError("regime evidence components must have exact types")
        derived, manifest = derive_regime(
            self.parent_corpus,
            self.spec,
            control_corpus=self.control_corpus,
        )
        if derived != self.derived_corpus or manifest != self.manifest:
            raise ValueError("regime evidence differs from exact upstream derivation")
        if (
            self.manifest.parent_corpus_digest
            != frozen_corpus_digest(self.parent_corpus)
            or self.manifest.output_corpus_digest
            != frozen_corpus_digest(self.derived_corpus)
        ):
            raise ValueError("regime manifest corpus digests are inconsistent")
        expected = _digest_document(
            self.model_dump(mode="json", exclude={"evidence_digest"})
        )
        if self.evidence_digest != expected:
            raise ValueError("regime evidence digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        parent_corpus: FrozenCorpus,
        derived_corpus: FrozenCorpus,
        spec: RegimeSpec,
        manifest: DerivedRegimeManifest,
        control_corpus: FrozenCorpus | None = None,
    ) -> ReplayRegimeEvidence:
        fields: dict[str, object] = {
            "parent_corpus": parent_corpus,
            "derived_corpus": derived_corpus,
            "spec": spec,
            "manifest": manifest,
            "control_corpus": control_corpus,
        }
        provisional = cast(Any, cls).model_construct(
            **fields, evidence_digest="0" * 64
        )
        digest = _digest_document(
            provisional.model_dump(mode="json", exclude={"evidence_digest"})
        )
        return cls.model_validate({**fields, "evidence_digest": digest})


class ReplayCorpusEvidence(ExternalContract):
    """Evaluator-signed exact corpus from which a development split is rederived."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    corpus: FrozenCorpus
    corpus_digest: str
    split_digest: str
    signer_key_id: str
    signature_base64: str
    evidence_digest: str

    @field_validator("corpus_digest", "split_digest", "signer_key_id", "evidence_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def evidence_is_closed(self) -> ReplayCorpusEvidence:
        if type(self.corpus) is not FrozenCorpus:
            raise ValueError("replay corpus evidence requires an exact frozen corpus")
        if self.corpus_digest != frozen_corpus_digest(self.corpus):
            raise ValueError("replay corpus digest is inconsistent")
        expected = _digest_document(
            {
                **self.unsigned_document(),
                "signature_base64": self.signature_base64,
            }
        )
        if self.evidence_digest != expected:
            raise ValueError("replay corpus evidence digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        corpus: FrozenCorpus,
        split: EvaluationSplit,
        signer: EvaluatorSigningIdentity,
    ) -> ReplayCorpusEvidence:
        if type(corpus) is not FrozenCorpus or type(split) is not EvaluationSplit:
            raise ReplayContractError("corpus evidence inputs must have exact types")
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise ReplayContractError("corpus evidence requires exact evaluator signer")
        if make_evaluation_split(corpus, split.config) != split:
            raise ReplayContractError("signed corpus does not rederive the exact split")
        fields = {
            "schema_version": "1.0.0",
            "corpus": corpus,
            "corpus_digest": frozen_corpus_digest(corpus),
            "split_digest": split.split_digest,
            "signer_key_id": signer.key_id,
        }
        provisional = cast(Any, cls).model_construct(
            **fields,
            signature_base64="",
            evidence_digest="0" * 64,
        )
        unsigned = provisional.unsigned_document()
        signature = signer._sign(unsigned)
        evidence_digest = _digest_document(
            {**unsigned, "signature_base64": signature}
        )
        return cls.model_validate(
            {
                **fields,
                "signature_base64": signature,
                "evidence_digest": evidence_digest,
            }
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "evidence_digest"}
        )

    def verify(self, verifier: EvaluatorReplayVerifier) -> bool:
        return (
            type(verifier) is EvaluatorReplayVerifier
            and self.signer_key_id == verifier.key_id
            and verifier.verify_document(self.unsigned_document(), self.signature_base64)
        )

class ReplayLatencySamples(ExternalContract):
    """Observational per-arm latency evidence kept outside core score lineage."""

    arm: DefenseArm
    samples: tuple[LatencySample, ...]

    @field_validator("samples", mode="before")
    @classmethod
    def samples_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("replay latency samples must be an exact tuple")
        return value


class ReplayEvaluationContext(ExternalContract):
    """Restricted evaluator inputs consumed only after all arm decisions freeze."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation: EvaluationDescriptor
    truth: tuple[EvaluationTruthRow, ...]
    observations: tuple[ObservedEvent, ...]
    as_of: datetime
    slice_assignments: tuple[SliceAssignment, ...]
    slice_manifest: SliceManifest
    latency_samples: tuple[ReplayLatencySamples, ...]
    feature_assurance: ReplayFeatureAssurance
    queue_config: QueueConfig = Field(default_factory=QueueConfig)

    @field_validator(
        "truth",
        "observations",
        "slice_assignments",
        "latency_samples",
        mode="before",
    )
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("replay evaluator collections must be exact tuples")
        return value

    @field_validator("as_of")
    @classmethod
    def as_of_is_exact_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("replay as_of must be an exact datetime")
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def context_is_closed(self) -> ReplayEvaluationContext:
        if tuple(item.arm for item in self.latency_samples) != tuple(DefenseArm):
            raise ValueError("latency evidence must contain all arms in canonical order")
        return self

    def to_json(self) -> bytes:
        if type(self) is not ReplayEvaluationContext:
            raise ReplayContractError("evaluation context must have its exact type")
        checked = ReplayEvaluationContext.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > _MAX_REPLAY_BYTES:
            raise ReplayContractError("evaluation context exceeds its resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> ReplayEvaluationContext:
        if type(payload) is not bytes or len(payload) > _MAX_REPLAY_BYTES:
            raise ReplayContractError("evaluation context payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise ReplayContractError("evaluation context must be an object")
            _tupleize_context_document(document)
            context = cls.model_validate(document)
            if context.to_json() != payload:
                raise ReplayContractError("evaluation context JSON is not canonical")
            return context
        except (ValidationError, WireContractError) as error:
            raise ReplayContractError(
                "evaluation context failed canonical validation"
            ) from error


class ArmThresholdEvidence(ExternalContract):
    """One arm's immutable matched-budget operating point."""

    arm: DefenseArm
    report: ThresholdReport


class ReplayThresholdSet(ExternalContract):
    """Three arm reports bound to one verified bundle and Task10 callback."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_manifest_digest: str
    case_callback_digest: str
    selection_row_ids_digest: str
    selection_as_of: datetime
    reports: tuple[ArmThresholdEvidence, ...]
    threshold_set_digest: str

    @field_validator(
        "bundle_manifest_digest",
        "case_callback_digest",
        "selection_row_ids_digest",
        "threshold_set_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("reports", mode="before")
    @classmethod
    def reports_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("arm threshold reports must be an exact tuple")
        return value

    @field_validator("selection_as_of")
    @classmethod
    def selection_time_is_exact_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("threshold selection time must be exact")
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def threshold_evidence_is_complete(self) -> ReplayThresholdSet:
        if tuple(item.arm for item in self.reports) != tuple(DefenseArm):
            raise ValueError("threshold reports must contain all arms in canonical order")
        if any(not item.report.feasible or item.report.thresholds is None for item in self.reports):
            raise ValueError("every replay arm requires a feasible frozen operating point")
        budgets = {item.report.budget for item in self.reports}
        if len(budgets) != 1:
            raise ValueError("all replay arms must use an identical matched budget")
        selection_lineage = {
            (
                item.report.row_count,
                item.report.legitimate_count,
                item.report.fraud_count,
                item.report.input_labels_digest,
                item.report.input_mandatory_actions_digest,
                item.report.input_values_digest,
            )
            for item in self.reports
        }
        if len(selection_lineage) != 1:
            raise ValueError("all replay arms must share exact threshold selection lineage")
        expected = _digest_document(
            self.model_dump(mode="json", exclude={"threshold_set_digest"})
        )
        if self.threshold_set_digest != expected:
            raise ValueError("replay threshold-set digest is inconsistent")
        return self

    @classmethod
    def from_reports(
        cls,
        defender: LoadedDefenderBundle,
        case_counter: ReplayCaseCounterBinding,
        reports: object,
    ) -> ReplayThresholdSet:
        """Reject caller-authored reports; all arms must be independently rederived."""
        del cls, defender, case_counter, reports
        raise ReplayContractError(
            "threshold reports require exact rederived selection evidence"
        )

    @classmethod
    def from_selection(
        cls,
        defender: LoadedDefenderBundle,
        case_counter: ReplayCaseCounterBinding,
        *,
        labels: np.ndarray,
        values: np.ndarray | None,
    ) -> ReplayThresholdSet:
        """Rederive all arm reports from signed rows and the Task10 callback."""
        if type(defender) is not LoadedDefenderBundle:
            raise ReplayContractError("threshold set requires an exact loaded defender")
        if type(case_counter) is not ReplayCaseCounterBinding:
            raise ReplayContractError("threshold set requires exact case callback lineage")
        if type(labels) is not np.ndarray or (
            values is not None and type(values) is not np.ndarray
        ):
            raise ReplayContractError("threshold labels and values must be exact arrays")
        matrix = defender.threshold_matrix
        rows, events = _validated_replay_rows(matrix, defender)
        row_ids = tuple(row.event_id for row in rows)
        selection_counter = case_counter.reconstruct(
            matrix.events, row_ids, case_counter.as_of
        )
        binding = defender.threshold_binding
        if _digest_document(row_ids) != binding.row_ids_digest:
            raise ReplayContractError("threshold selection row IDs differ from signed lineage")
        if _numeric_array_digest(labels) != binding.labels_digest:
            raise ReplayContractError("threshold labels differ from signed selection evidence")
        if (values is None) != (binding.values_digest is None) or (
            values is not None
            and _numeric_array_digest(values) != binding.values_digest
        ):
            raise ReplayContractError("threshold values differ from signed selection evidence")
        rule_engine = RuleEngine(defender.rule_manifest)
        rule_results = tuple(
            rule_engine.evaluate(event, row)
            for event, row in zip(events, rows, strict=True)
        )
        mandatory = tuple(
            any(hit.mandatory for hit in result.hits) for result in rule_results
        )
        common = _common_mandatory_decisions(
            events, mandatory, defender.rule_manifest
        )
        mandatory_actions = np.asarray(
            [
                Action.DECLINE if selected else Action.APPROVE
                for selected in mandatory
            ],
            dtype=object,
        )
        if tuple(mandatory_actions) != binding.mandatory_actions or (
            _action_array_digest(mandatory_actions) != binding.mandatory_actions_digest
        ):
            raise ReplayContractError(
                "derived mandatory actions differ from signed selection evidence"
            )
        if any(
            selected and (decision is None or decision.action is not Action.DECLINE)
            for selected, decision in zip(mandatory, common, strict=True)
        ):
            raise ReplayContractError("mandatory selection decisions are inconsistent")
        raw_rule = np.asarray([item.score for item in rule_results], dtype=np.float64)
        raw_model = defender.scorer.predict(matrix)
        calibrated = defender.calibrator.predict(raw_model)
        if _numeric_array_digest(calibrated) != binding.calibrated_scores_digest:
            raise ReplayContractError("calibrated scores differ from signed selection evidence")
        raw_by_arm = {
            DefenseArm.RULES_ONLY: raw_rule,
            DefenseArm.GBDT_ONLY: calibrated,
            DefenseArm.LAYERED_HYBRID: np.maximum(raw_rule, calibrated),
        }
        signed_report = defender.threshold_report
        checked = tuple(
            ArmThresholdEvidence(
                arm=arm,
                report=select_policy_thresholds(
                    raw_by_arm[arm],
                    labels,
                    mandatory_actions,
                    cast(Callable[[NDArray[np.object_]], int], selection_counter),
                    signed_report.budget,
                    values,
                ),
            )
            for arm in DefenseArm
        )
        if checked[-1].report.report_digest != signed_report.report_digest or (
            checked[-1].report.report_digest != binding.threshold_report_digest
        ):
            raise ReplayContractError(
                "rederived layered threshold differs from signed defender evidence"
            )
        fields: dict[str, object] = {
            "bundle_manifest_digest": _manifest_digest(defender.manifest),
            "case_callback_digest": case_counter.callback_digest,
            "selection_row_ids_digest": binding.row_ids_digest,
            "selection_as_of": case_counter.as_of,
            "reports": checked,
        }
        digest_fields = {
            "schema_version": "1.0.0",
            "bundle_manifest_digest": fields["bundle_manifest_digest"],
            "case_callback_digest": fields["case_callback_digest"],
            "selection_row_ids_digest": fields["selection_row_ids_digest"],
            "selection_as_of": case_counter.as_of.isoformat().replace("+00:00", "Z"),
            "reports": [item.model_dump(mode="json") for item in checked],
        }
        return cls.model_validate(
            {**fields, "threshold_set_digest": _digest_document(digest_fields)}
        )

    def report_for(self, arm: DefenseArm) -> ThresholdReport:
        """Return one freshly validated arm report."""
        for item in self.reports:
            if item.arm is arm:
                return ThresholdReport.from_json(item.report.to_json())
        raise ReplayContractError("threshold set is incomplete")

    def to_json(self) -> bytes:
        checked = ReplayThresholdSet.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        return canonical_json_bytes(checked.model_dump(mode="json"))

    @classmethod
    def from_json(
        cls,
        payload: bytes,
        *,
        defender: LoadedDefenderBundle,
        case_counter: ReplayCaseCounterBinding,
        labels: np.ndarray,
        values: np.ndarray | None,
    ) -> ReplayThresholdSet:
        if type(payload) is not bytes or len(payload) > _MAX_REPLAY_BYTES:
            raise ReplayContractError("threshold set payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise ReplayContractError("threshold set must be a JSON object")
            if type(document.get("reports")) is list:
                document["reports"] = tuple(document["reports"])
            value = cls.model_validate(document)
            if value.to_json() != payload:
                raise ReplayContractError("threshold set JSON is not canonical")
            expected = cls.from_selection(
                defender, case_counter, labels=labels, values=values
            )
            if value != expected:
                raise ReplayContractError(
                    "serialized thresholds differ from rederived selection evidence"
                )
            return expected
        except (ValidationError, WireContractError) as error:
            raise ReplayContractError(str(error)) from error


class ReplayCaseCounterBinding(ExternalContract):
    """Serializable recipe that always rebuilds the production Task10 callback."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_ids: tuple[str, ...]
    rows_digest: str
    as_of: datetime
    callback_digest: str

    @field_validator("event_ids", mode="before")
    @classmethod
    def event_ids_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("case callback event IDs must be an exact tuple")
        return value

    @field_validator("event_ids")
    @classmethod
    def event_ids_are_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(type(item) is not str or not item for item in value):
            raise ValueError("case callback event IDs must be nonempty exact text")
        if len(value) != len(set(value)):
            raise ValueError("case callback event IDs must be unique")
        return value

    @field_validator("rows_digest", "callback_digest")
    @classmethod
    def callback_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("as_of")
    @classmethod
    def callback_time_is_exact_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("case callback time must be exact")
        return validate_utc_timestamp(value)

    def reconstruct(
        self,
        observations: tuple[ObservedEvent, ...],
        event_ids: tuple[str, ...],
        as_of: datetime,
    ) -> ReviewCaseCounter:
        """Validate the recipe and rebuild Task10 from concrete rows every time."""
        if type(self) is not ReplayCaseCounterBinding:
            raise ReplayContractError("case callback binding must have its exact type")
        document = _case_binding_document(observations, event_ids, as_of)
        if (
            event_ids != self.event_ids
            or as_of != self.as_of
            or _digest_document(document) != self.rows_digest
        ):
            raise ReplayContractError("case callback lineage does not match replay rows")
        expected_callback_digest = _digest_document(
            {"schema_version": "1.0.0", "binding": document, "adapter": "Task10"}
        )
        if expected_callback_digest != self.callback_digest:
            raise ReplayContractError("case callback lineage digest is inconsistent")
        observation_by_id = {row.event_id: row for row in observations}
        canonical_ids = tuple(
            sorted(
                event_ids,
                key=lambda event_id: (
                    cast(datetime, observation_by_id[event_id].decision_at), event_id
                ),
            )
        )
        placeholder = tuple(_placeholder_decision(event_id) for event_id in canonical_ids)
        ordered_observations = tuple(
            sorted(observations, key=lambda row: (row.available_at, row.event_id))
        )
        try:
            return bind_review_case_counter(
                ordered_observations, placeholder, as_of=as_of
            )
        except Exception as error:
            raise ReplayContractError("production case callback reconstruction failed") from error

    def validate_context(
        self,
        observations: tuple[ObservedEvent, ...],
        event_ids: tuple[str, ...],
        as_of: datetime,
    ) -> None:
        self.reconstruct(observations, event_ids, as_of)


def bind_replay_case_counter(
    observations: tuple[ObservedEvent, ...],
    decision_event_ids: tuple[str, ...],
    *,
    as_of: datetime,
) -> ReplayCaseCounterBinding:
    """Bind the real Task10 callback and its exact public replay-row lineage."""
    if type(observations) is not tuple or type(decision_event_ids) is not tuple:
        raise ReplayContractError("case callback rows must be exact tuples")
    if type(as_of) is not datetime:
        raise ReplayContractError("case callback as_of must be exact datetime")
    validate_utc_timestamp(as_of)
    if not decision_event_ids or len(decision_event_ids) != len(set(decision_event_ids)):
        raise ReplayContractError("case callback decision IDs must be nonempty and unique")
    observation_by_id = {row.event_id: row for row in observations}
    if len(observation_by_id) != len(observations) or any(
        event_id not in observation_by_id for event_id in decision_event_ids
    ):
        raise ReplayContractError("case callback observations do not cover decisions")
    canonical_ids = tuple(
        sorted(
            decision_event_ids,
            key=lambda event_id: (
                cast(datetime, observation_by_id[event_id].decision_at), event_id
            ),
        )
    )
    placeholder = tuple(_placeholder_decision(event_id) for event_id in canonical_ids)
    ordered_observations = tuple(
        sorted(observations, key=lambda row: (row.available_at, row.event_id))
    )
    try:
        bind_review_case_counter(ordered_observations, placeholder, as_of=as_of)
    except Exception as error:
        raise ReplayContractError("production case callback binding failed") from error
    document = _case_binding_document(observations, decision_event_ids, as_of)
    rows_digest = _digest_document(document)
    callback_digest = _digest_document(
        {"schema_version": "1.0.0", "binding": document, "adapter": "Task10"}
    )
    return ReplayCaseCounterBinding(
        event_ids=decision_event_ids,
        rows_digest=rows_digest,
        as_of=as_of,
        callback_digest=callback_digest,
    )


def replay_defense_arms(
    *,
    matrix: FeatureMatrix,
    defender: LoadedDefenderBundle,
    defender_verifier: DefenderBundleVerifier,
    defender_attestation: VerifiedDefenderAttestation,
    thresholds: ReplayThresholdSet,
    threshold_labels: np.ndarray,
    threshold_values: np.ndarray | None,
    case_counter: ReplayCaseCounterBinding,
    evaluation_split: EvaluationSplit | None = None,
    regime_evidence: ReplayRegimeEvidence | None = None,
    corpus_evidence: ReplayCorpusEvidence | None = None,
    evaluation: ReplayEvaluationContext | None = None,
    evaluator_signer: EvaluatorSigningIdentity | None = None,
    evaluator_verifier: EvaluatorReplayVerifier | None = None,
    hidden_authority: HiddenEvaluationAuthority | None = None,
    hidden_capability: HiddenEvaluationCapability | None = None,
    hidden_ref: ArtifactRef | None = None,
    hidden_released_at: datetime | None = None,
    hidden_sealed_at: datetime | None = None,
    model_failure: ModelFailure | None = None,
) -> VerifiedReplayBatch | HiddenReplayOutcome:
    """Replay development evidence or delegate the sealed hidden lifecycle."""
    try:
        hidden_values = (
            hidden_authority,
            hidden_capability,
            hidden_ref,
            hidden_released_at,
            hidden_sealed_at,
        )
        hidden_requested = any(item is not None for item in hidden_values)
        if hidden_requested:
            if (
                evaluation is not None
                or evaluator_signer is not None
                or evaluator_verifier is not None
                or corpus_evidence is not None
                or evaluation_split is not None
                or regime_evidence is not None
                or any(item is None for item in hidden_values)
            ):
                raise ReplayContractError(
                    "hidden replay requires one complete authority-owned lifecycle"
                )
            if not isinstance(hidden_authority, HiddenEvaluationAuthority):
                raise ReplayContractError("hidden replay authority is invalid")
            invocation = _HiddenReplayInvocation(
                matrix,
                defender,
                defender_verifier,
                defender_attestation,
                thresholds,
                threshold_labels,
                threshold_values,
                case_counter,
                model_failure,
            )
            return cast(
                HiddenReplayOutcome,
                hidden_authority.evaluate_hidden_replay(  # type: ignore[attr-defined]
                    invocation,
                    capability=hidden_capability,
                    restricted_ref=hidden_ref,
                    released_at=hidden_released_at,
                    sealed_at=hidden_sealed_at,
                ),
            )
        if evaluation is None:
            raise ReplayContractError("development replay requires an evaluation context")
        if not EvaluatorSigningIdentity.is_exact(evaluator_signer):
            raise ReplayContractError(
                "development replay requires an explicit evaluator signing identity"
            )
        checked_evaluator_signer = cast(EvaluatorSigningIdentity, evaluator_signer)
        if (
            type(evaluator_verifier) is not EvaluatorReplayVerifier
            or evaluator_verifier.key_id != checked_evaluator_signer.key_id
        ):
            raise ReplayContractError(
                "development replay requires the pinned evaluator verifier"
            )
        if (
            type(corpus_evidence) is not ReplayCorpusEvidence
            or not corpus_evidence.verify(evaluator_verifier)
        ):
            raise ReplayContractError(
                "development replay requires signed exact corpus evidence"
            )
        frozen = _freeze_replay_inputs(
            _HiddenReplayInvocation(
                matrix,
                defender,
                defender_verifier,
                defender_attestation,
                thresholds,
                threshold_labels,
                threshold_values,
                case_counter,
                model_failure,
            ),
            pinned_verifier=defender_verifier,
            pinned_attestation=defender_attestation,
        )
        context = _exact_model(evaluation, ReplayEvaluationContext, "evaluation context")
        if context.evaluation.kind is EvaluationKind.HIDDEN:
            raise ReplayContractError("hidden evaluation requires its sealed authority")
        lineage = _validate_descriptor_lineage(
            context.evaluation,
            frozen.event_ids,
            frozen.defender,
            evaluation_split,
            frozen.matrix,
            frozen.defender_attestation,
            context,
            regime_evidence,
            corpus_evidence,
        )
        evaluated, _ = _evaluate_frozen_context(
            frozen, context, lineage, hidden_access_clean=True
        )
        results = tuple(item.result for item in evaluated)
        _validate_identical_result_rows(results)
        return checked_evaluator_signer.sign_batch(results)
    except ReplayContractError:
        raise
    except HiddenBoundaryError as error:
        raise ReplayContractError(str(error)) from error
    except (
        ArithmeticError,
        AttributeError,
        MemoryError,
        OverflowError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        raise ReplayContractError("defense replay failed deterministically") from error


def _freeze_hidden_invocation(
    invocation: object,
    *,
    pinned_verifier: DefenderBundleVerifier,
    pinned_attestation: VerifiedDefenderAttestation,
) -> _FrozenDefenseReplay:
    """Evaluator-only entry: materialize all actions without touching truth."""
    if type(invocation) is not _HiddenReplayInvocation:
        raise ReplayContractError("hidden replay invocation must have its exact type")
    return _freeze_replay_inputs(
        invocation,
        pinned_verifier=pinned_verifier,
        pinned_attestation=pinned_attestation,
    )


def _freeze_replay_inputs(
    invocation: _HiddenReplayInvocation,
    *,
    pinned_verifier: DefenderBundleVerifier,
    pinned_attestation: VerifiedDefenderAttestation,
) -> _FrozenDefenseReplay:
    matrix_value = _exact_model(invocation.matrix, FeatureMatrix, "feature matrix")
    threshold_set = _exact_model(
        invocation.thresholds, ReplayThresholdSet, "threshold set"
    )
    defender = invocation.defender
    if type(defender) is not LoadedDefenderBundle:
        raise ReplayContractError("defender must be an exact verified loaded bundle")
    if (
        type(invocation.defender_verifier) is not DefenderBundleVerifier
        or invocation.defender_verifier is not pinned_verifier
        or type(pinned_verifier) is not DefenderBundleVerifier
    ):
        raise ReplayContractError("replay requires the exact pinned defender verifier")
    if (
        type(invocation.defender_attestation) is not VerifiedDefenderAttestation
        or invocation.defender_attestation is not pinned_attestation
        or not pinned_verifier.verify(pinned_attestation)
    ):
        raise ReplayContractError("defender attestation failed exact re-attestation")
    case_counter = invocation.case_counter
    if type(case_counter) is not ReplayCaseCounterBinding:
        raise ReplayContractError("case callback must have exact replay lineage")
    declared_failure = (
        None
        if invocation.model_failure is None
        else _exact_model(invocation.model_failure, ModelFailure, "model failure")
    )
    rows, events = _validated_replay_rows(matrix_value, defender)
    event_ids = tuple(row.event_id for row in rows)
    case_counter.reconstruct(matrix_value.events, event_ids, case_counter.as_of)
    selection_ids = tuple(row.event_id for row in defender.threshold_matrix.rows)
    selection_binding = bind_replay_case_counter(
        defender.threshold_matrix.events,
        selection_ids,
        as_of=threshold_set.selection_as_of,
    )
    rederived = ReplayThresholdSet.from_selection(
        defender,
        selection_binding,
        labels=invocation.threshold_labels,
        values=invocation.threshold_values,
    )
    if threshold_set != rederived:
        raise ReplayContractError(
            "threshold set differs from exact rederived selection evidence"
        )
    manifest_digest = _manifest_digest(defender.manifest)
    if (
        pinned_attestation.bundle_manifest_digest != manifest_digest
        or pinned_attestation.top_ref.sha256 != manifest_digest
        or pinned_attestation.threshold_digest != defender.manifest.threshold_digest
    ):
        raise ReplayContractError("defender attestation does not bind loaded artifacts")
    if threshold_set.bundle_manifest_digest != manifest_digest:
        raise ReplayContractError("threshold bundle lineage does not match defender")
    if (
        threshold_set.selection_row_ids_digest
        != defender.threshold_binding.row_ids_digest
        or threshold_set.report_for(DefenseArm.LAYERED_HYBRID).report_digest
        != defender.threshold_report.report_digest
    ):
        raise ReplayContractError(
            "threshold selection lineage does not match the signed defender"
        )
    audit = audit_feature_matrix(matrix_value.events, matrix_value, defender.catalog)
    defender.verify_reload()
    rule_engine = RuleEngine(defender.rule_manifest)
    rule_results = tuple(
        rule_engine.evaluate(event, row)
        for event, row in zip(events, rows, strict=True)
    )
    mandatory = tuple(any(hit.mandatory for hit in row.hits) for row in rule_results)
    common_mandatory = _common_mandatory_decisions(
        events, mandatory, defender.rule_manifest
    )
    actual_failure = declared_failure
    calibrated: np.ndarray | None = None
    if actual_failure is None:
        try:
            calibrated = defender.calibrator.predict(defender.scorer.predict(matrix_value))
        except Exception:
            actual_failure = ModelFailure(
                reason=DefenseReason.MODEL_UNAVAILABLE,
                failed_component_version=f"model:{defender.manifest.model_digest}",
            )
    decisions_by_arm = {
        arm: _arm_decisions(
            arm=arm,
            events=events,
            vectors=rows,
            rules=rule_results,
            calibrated=calibrated,
            mandatory=mandatory,
            common_mandatory=common_mandatory,
            thresholds=threshold_set,
            failure=actual_failure,
            rule_manifest=defender.rule_manifest,
            rule_fallback_thresholds=threshold_set.report_for(
                DefenseArm.RULES_ONLY
            ).thresholds,
        )
        for arm in DefenseArm
    }
    arm_scores = {
        arm: np.asarray(
            [decision.score for decision in decisions_by_arm[arm]], dtype=np.float64
        )
        for arm in DefenseArm
    }
    return _FrozenDefenseReplay(
        matrix_value,
        defender,
        pinned_verifier,
        pinned_attestation,
        threshold_set,
        case_counter,
        rows,
        events,
        event_ids,
        mandatory,
        decisions_by_arm,
        arm_scores,
        audit.passed,
        actual_failure,
        manifest_digest,
    )


def _hidden_decision_bindings(
    frozen: object,
) -> tuple[HiddenDecisionBinding, ...]:
    if type(frozen) is not _FrozenDefenseReplay:
        raise ReplayContractError("hidden frozen replay must have its exact type")
    bindings: list[HiddenDecisionBinding] = []
    for arm in DefenseArm:
        decisions = frozen.decisions_by_arm[arm]
        mandatory_document = tuple(
            decisions[index].model_dump(mode="json")
            for index, selected in enumerate(frozen.mandatory)
            if selected
        )
        bindings.append(
            HiddenDecisionBinding(
                arm=arm.value,
                decision_event_ids_digest=_digest_document(frozen.event_ids),
                decision_artifact_digest=_digest_document(
                    tuple(item.model_dump(mode="json") for item in decisions)
                ),
                action_digest=_digest_document(
                    tuple(item.action.value for item in decisions)
                ),
                score_digest=_array_digest(frozen.arm_scores[arm]),
                common_integrity_digest=_digest_document(mandatory_document),
                threshold_report_digest=frozen.threshold_set.report_for(
                    arm
                ).report_digest,
                threshold_set_digest=frozen.threshold_set.threshold_set_digest,
                case_callback_digest=frozen.case_counter.callback_digest,
            )
        )
    result = tuple(bindings)
    if len({item.decision_event_ids_digest for item in result}) != 1 or len(
        {item.common_integrity_digest for item in result}
    ) != 1:
        raise ReplayContractError("hidden arms lack an identical common freeze")
    return result


def _hidden_invocation_digest(frozen: object) -> str:
    if type(frozen) is not _FrozenDefenseReplay:
        raise ReplayContractError("hidden frozen replay must have its exact type")
    return _digest_document(
        {
            "schema_version": "1.0.0",
            "matrix": frozen.matrix.model_dump(mode="json"),
            "bundle_manifest_digest": frozen.manifest_digest,
            "attestation_digest": frozen.defender_attestation.attestation_digest,
            "threshold_set_digest": frozen.threshold_set.threshold_set_digest,
            "case_callback_digest": frozen.case_counter.callback_digest,
            "decisions": [
                item.model_dump(mode="json") for item in _hidden_decision_bindings(frozen)
            ],
            "model_failure": (
                None
                if frozen.actual_failure is None
                else frozen.actual_failure.model_dump(mode="json")
            ),
        }
    )


def _hidden_worker_frozen_document(
    frozen: object,
    *,
    proof_id: str,
) -> dict[str, object]:
    """Serialize only truth-blind frozen decisions for the isolated evaluator."""
    if type(frozen) is not _FrozenDefenseReplay:
        raise ReplayContractError("hidden worker freeze must have its exact type")
    binding = frozen.defender.training_binding
    training_document = {
        "requested": list(binding.requested_row_ids),
        "excluded": list(binding.excluded_row_ids),
        "final_fit": list(binding.final_fit_row_ids),
        "receipt": frozen.defender.manifest.training_receipt_digest,
        "binding": frozen.defender.manifest.training_binding_digest,
    }
    return {
        "schema_version": "1.0.0",
        "proof_id": proof_id,
        "observations": [
            item.model_dump(mode="json") for item in frozen.matrix.events
        ],
        "decision_event_ids": list(frozen.event_ids),
        "mandatory": list(frozen.mandatory),
        "decisions": {
            arm.value: [
                item.model_dump(mode="json")
                for item in frozen.decisions_by_arm[arm]
            ]
            for arm in DefenseArm
        },
        "scores": {
            arm.value: [float(item) for item in frozen.arm_scores[arm]]
            for arm in DefenseArm
        },
        "threshold_report_digests": {
            arm.value: frozen.threshold_set.report_for(arm).report_digest
            for arm in DefenseArm
        },
        "threshold_set_digest": frozen.threshold_set.threshold_set_digest,
        "case_callback_digest": frozen.case_counter.callback_digest,
        "case_as_of": frozen.case_counter.as_of.isoformat().replace("+00:00", "Z"),
        "feature_audit_passed": frozen.feature_audit_passed,
        "model_failure": (
            None
            if frozen.actual_failure is None
            else frozen.actual_failure.model_dump(mode="json")
        ),
        "rollback_available": frozen.defender_attestation.rollback_available,
        "bundle_id": frozen.defender.manifest.bundle_id,
        "bundle_manifest_digest": frozen.manifest_digest,
        "defender_top_ref_digest": frozen.defender_attestation.top_ref.sha256,
        "decision_content_digest": _digest_document(
            frozen.matrix.model_dump(mode="json")
        ),
        "corpus_digest": frozen.defender.manifest.corpus_digest,
        "split_digest": frozen.defender.manifest.split_manifest_digest,
        "training_population_digest": _digest_document(training_document),
    }


def _hidden_worker_decision_bindings_from_document(
    frozen_document: object,
) -> tuple[HiddenDecisionBinding, ...]:
    """Revalidate serialized decisions before the worker may read restricted truth."""
    if type(frozen_document) is not dict:
        raise ReplayContractError("isolated hidden freeze must be an exact object")
    document = cast(dict[str, object], frozen_document)
    raw_ids = document.get("decision_event_ids")
    raw_mandatory = document.get("mandatory")
    raw_decisions = document.get("decisions")
    raw_scores = document.get("scores")
    raw_reports = document.get("threshold_report_digests")
    if (
        type(raw_ids) is not list
        or type(raw_mandatory) is not list
        or type(raw_decisions) is not dict
        or type(raw_scores) is not dict
        or type(raw_reports) is not dict
    ):
        raise ReplayContractError("isolated hidden decision freeze is invalid")
    event_ids = tuple(raw_ids)
    mandatory = tuple(raw_mandatory)
    if (
        not event_ids
        or len(event_ids) > _MAX_REPLAY_ROWS
        or any(type(item) is not str for item in event_ids)
        or len(set(event_ids)) != len(event_ids)
        or len(mandatory) != len(event_ids)
        or any(type(item) is not bool for item in mandatory)
    ):
        raise ReplayContractError("isolated hidden decision rows are invalid")
    bindings: list[HiddenDecisionBinding] = []
    for arm in DefenseArm:
        raw_arm_decisions = raw_decisions.get(arm.value)
        raw_arm_scores = raw_scores.get(arm.value)
        report_digest = raw_reports.get(arm.value)
        if type(raw_arm_decisions) is not list or type(raw_arm_scores) is not list:
            raise ReplayContractError("isolated hidden arm decision rows are invalid")
        decisions = tuple(
            DefenseDecision.model_validate(item) for item in raw_arm_decisions
        )
        scores = np.asarray(raw_arm_scores, dtype=np.float64)
        if (
            tuple(item.event_id for item in decisions) != event_ids
            or scores.shape != (len(event_ids),)
            or not np.isfinite(scores).all()
            or any(
                decision.score != float(score)
                for decision, score in zip(decisions, scores, strict=True)
            )
        ):
            raise ReplayContractError("isolated hidden arm decisions are inconsistent")
        _validate_digest(cast(str, report_digest))
        mandatory_document = tuple(
            decisions[index].model_dump(mode="json")
            for index, selected in enumerate(mandatory)
            if selected
        )
        bindings.append(
            HiddenDecisionBinding(
                arm=arm.value,
                decision_event_ids_digest=_digest_document(event_ids),
                decision_artifact_digest=_digest_document(
                    tuple(item.model_dump(mode="json") for item in decisions)
                ),
                action_digest=_digest_document(
                    tuple(item.action.value for item in decisions)
                ),
                score_digest=_array_digest(scores),
                common_integrity_digest=_digest_document(mandatory_document),
                threshold_report_digest=cast(str, report_digest),
                threshold_set_digest=cast(str, document.get("threshold_set_digest")),
                case_callback_digest=cast(str, document.get("case_callback_digest")),
            )
        )
    result = tuple(bindings)
    if len({item.common_integrity_digest for item in result}) != 1:
        raise ReplayContractError("isolated hidden mandatory decisions differ by arm")
    return result


def _evaluate_hidden_worker_document(
    frozen_document: object,
    payload: bytes,
) -> _HiddenEvaluationProduct:
    """Evaluator-worker-only parser and metric derivation over restricted bytes."""
    if type(frozen_document) is not dict or type(payload) is not bytes:
        raise ReplayContractError("isolated hidden evaluator inputs are invalid")
    document = cast(dict[str, object], frozen_document)
    required = {
        "schema_version",
        "proof_id",
        "observations",
        "decision_event_ids",
        "mandatory",
        "decisions",
        "scores",
        "threshold_report_digests",
        "threshold_set_digest",
        "case_callback_digest",
        "case_as_of",
        "feature_audit_passed",
        "model_failure",
        "rollback_available",
        "bundle_id",
        "bundle_manifest_digest",
        "defender_top_ref_digest",
        "decision_content_digest",
        "corpus_digest",
        "split_digest",
        "training_population_digest",
    }
    if set(document) != required or document["schema_version"] != "1.0.0":
        raise ReplayContractError("isolated hidden freeze field set is invalid")
    bundle_id = document["bundle_id"]
    if type(bundle_id) is not str or not bundle_id or len(bundle_id) > 128:
        raise ReplayContractError("isolated hidden bundle ID is invalid")
    proof_id = document["proof_id"]
    if (
        type(proof_id) is not str
        or not proof_id.startswith("hpf_")
        or len(proof_id) != 36
    ):
        raise ReplayContractError("isolated hidden proof identity is invalid")
    raw_observations = document["observations"]
    raw_ids = document["decision_event_ids"]
    raw_mandatory = document["mandatory"]
    if (
        type(raw_observations) is not list
        or type(raw_ids) is not list
        or type(raw_mandatory) is not list
    ):
        raise ReplayContractError("isolated hidden row collections are invalid")
    observations = tuple(ObservedEvent.model_validate(item) for item in raw_observations)
    event_ids = tuple(raw_ids)
    mandatory = tuple(raw_mandatory)
    if (
        not event_ids
        or len(event_ids) > _MAX_REPLAY_ROWS
        or any(type(item) is not str for item in event_ids)
        or any(type(item) is not bool for item in mandatory)
        or len(mandatory) != len(event_ids)
    ):
        raise ReplayContractError("isolated hidden ordered rows are invalid")
    observation_by_id = {item.event_id: item for item in observations}
    events = tuple(observation_by_id[item] for item in event_ids)
    raw_decisions = document["decisions"]
    raw_scores = document["scores"]
    raw_reports = document["threshold_report_digests"]
    if not all(type(item) is dict for item in (raw_decisions, raw_scores, raw_reports)):
        raise ReplayContractError("isolated hidden arm maps are invalid")
    decisions_by_arm: dict[DefenseArm, tuple[DefenseDecision, ...]] = {}
    scores_by_arm: dict[DefenseArm, np.ndarray] = {}
    report_digests: dict[DefenseArm, str] = {}
    for arm in DefenseArm:
        decision_rows = cast(dict[str, object], raw_decisions).get(arm.value)
        score_rows = cast(dict[str, object], raw_scores).get(arm.value)
        report_digest = cast(dict[str, object], raw_reports).get(arm.value)
        if type(decision_rows) is not list or type(score_rows) is not list:
            raise ReplayContractError("isolated hidden arm rows are invalid")
        decisions = tuple(DefenseDecision.model_validate(item) for item in decision_rows)
        scores = np.asarray(score_rows, dtype=np.float64)
        _validate_digest(cast(str, report_digest))
        if (
            len(decisions) != len(event_ids)
            or tuple(item.event_id for item in decisions) != event_ids
            or scores.shape != (len(event_ids),)
            or not np.isfinite(scores).all()
        ):
            raise ReplayContractError("isolated hidden arm freeze is inconsistent")
        decisions_by_arm[arm] = decisions
        scores_by_arm[arm] = scores
        report_digests[arm] = cast(str, report_digest)
    case_as_of = _parse_utc_wire(document["case_as_of"])
    case_counter = bind_replay_case_counter(observations, event_ids, as_of=case_as_of)
    if case_counter.callback_digest != document["case_callback_digest"]:
        raise ReplayContractError("isolated hidden case callback lineage changed")
    context = ReplayEvaluationContext.from_json(payload)
    if context.evaluation.kind is not EvaluationKind.HIDDEN:
        raise ReplayContractError("restricted context is not a hidden descriptor")
    _validate_evaluator_context(context, event_ids)
    case_counter.validate_context(context.observations, event_ids, context.as_of)
    if document["feature_audit_passed"] is not context.feature_assurance.leakage_passed:
        raise ReplayContractError("hidden feature assurance differs from frozen audit")
    cohort_document = {
        row.event_id: [item.value for item in row.entity_cohorts]
        for row in context.slice_assignments
    }
    for name in (
        "decision_content_digest",
        "corpus_digest",
        "split_digest",
        "training_population_digest",
        "bundle_manifest_digest",
        "defender_top_ref_digest",
        "threshold_set_digest",
        "case_callback_digest",
    ):
        _validate_digest(cast(str, document[name]))
    restricted_cohort_mapping_digest = _digest_document(cohort_document)
    lineage_fields: dict[str, object] = {
        "descriptor": context.evaluation,
        "decision_rows_digest": _digest_document(event_ids),
        "decision_content_digest": cast(str, document["decision_content_digest"]),
        "corpus_digest": cast(str, document["corpus_digest"]),
        "split_digest": cast(str, document["split_digest"]),
        "training_population_digest": cast(
            str, document["training_population_digest"]
        ),
        "bundle_manifest_digest": cast(str, document["bundle_manifest_digest"]),
        "defender_top_ref_digest": cast(str, document["defender_top_ref_digest"]),
    }
    restricted_lineage = EvaluationLineage.create(
        **lineage_fields,
        cohort_mapping_digest=restricted_cohort_mapping_digest,
    )
    public_context_digest = _digest_document(
        {"scope": "hidden-public-context", "proof_id": proof_id}
    )
    public_cohort_mapping_digest = _digest_document(
        {"scope": "hidden-public-cohorts", "proof_id": proof_id}
    )
    lineage = EvaluationLineage.create(
        **lineage_fields,
        cohort_mapping_digest=public_cohort_mapping_digest,
    )
    context_digest = _evaluator_context_digest(payload)
    failure_document = document["model_failure"]
    failure = (
        None
        if failure_document is None
        else ModelFailure.model_validate(failure_document)
    )
    evaluated = tuple(
        _evaluate_frozen_arm(
            arm=arm,
            event_ids=event_ids,
            events=events,
            decisions=decisions_by_arm[arm],
            scores=scores_by_arm[arm],
            mandatory=mandatory,
            threshold_report_digest=report_digests[arm],
            threshold_set_digest=cast(str, document["threshold_set_digest"]),
            bundle_manifest_digest=cast(str, document["bundle_manifest_digest"]),
            candidate_bundle_id=bundle_id,
            case_counter=case_counter,
            context=context,
            evaluation_lineage=lineage,
            evaluation_context_digest=public_context_digest,
            rollback_available=cast(bool, document["rollback_available"]),
            hidden_access_clean=True,
            hidden_public_proof_id=proof_id,
            failure=failure if arm is DefenseArm.GBDT_ONLY else None,
        )
        for arm in DefenseArm
    )
    return _HiddenEvaluationProduct(
        evaluated=evaluated,
        evaluator_context_digest=context_digest,
        restricted_cohort_mapping_digest=restricted_cohort_mapping_digest,
        evaluation_lineage_digest=restricted_lineage.lineage_digest,
        evaluator_as_of=context.as_of,
    )


def _evaluate_hidden_frozen(
    frozen: object, payload: bytes
) -> _HiddenEvaluationProduct:
    if type(frozen) is not _FrozenDefenseReplay or type(payload) is not bytes:
        raise ReplayContractError("hidden evaluator inputs must have exact types")
    context = ReplayEvaluationContext.from_json(payload)
    if context.evaluation.kind is not EvaluationKind.HIDDEN:
        raise ReplayContractError("restricted context is not a hidden descriptor")
    lineage = _hidden_descriptor_lineage(
        context.evaluation,
        frozen.event_ids,
        frozen.defender,
        frozen.matrix,
        frozen.defender_attestation,
        context,
    )
    evaluated, context_digest = _evaluate_frozen_context(
        frozen, context, lineage, hidden_access_clean=False
    )
    return _HiddenEvaluationProduct(
        evaluated,
        context_digest,
        lineage.cohort_mapping_digest,
        lineage.lineage_digest,
        context.as_of,
    )


def _hidden_product_evidence(
    product: object,
) -> tuple[HiddenArmEvidenceBinding, ...]:
    if type(product) is not _HiddenEvaluationProduct:
        raise ReplayContractError("hidden evaluation product must have its exact type")
    return tuple(item.hidden_evidence for item in product.evaluated)


def _evaluate_frozen_context(
    frozen: _FrozenDefenseReplay,
    context: ReplayEvaluationContext,
    lineage: EvaluationLineage,
    *,
    hidden_access_clean: bool,
) -> tuple[tuple[_EvaluatedArm, ...], str]:
    _validate_evaluator_context(context, frozen.event_ids)
    frozen.case_counter.validate_context(
        context.observations, frozen.event_ids, context.as_of
    )
    if frozen.feature_audit_passed != context.feature_assurance.leakage_passed:
        raise ReplayContractError("feature leakage evidence disagrees with replay audit")
    context_digest = _evaluator_context_digest(context.to_json())
    evaluated = tuple(
        _evaluate_frozen_arm(
            arm=arm,
            event_ids=frozen.event_ids,
            events=frozen.events,
            decisions=frozen.decisions_by_arm[arm],
            scores=frozen.arm_scores[arm],
            mandatory=frozen.mandatory,
            threshold_report_digest=frozen.threshold_set.report_for(arm).report_digest,
            threshold_set_digest=frozen.threshold_set.threshold_set_digest,
            bundle_manifest_digest=frozen.manifest_digest,
            candidate_bundle_id=frozen.defender.manifest.bundle_id,
            case_counter=frozen.case_counter,
            context=context,
            evaluation_lineage=lineage,
            evaluation_context_digest=context_digest,
            rollback_available=frozen.defender_attestation.rollback_available,
            hidden_access_clean=hidden_access_clean,
            hidden_public_proof_id=None,
            failure=(
                frozen.actual_failure if arm is DefenseArm.GBDT_ONLY else None
            ),
        )
        for arm in DefenseArm
    )
    return evaluated, context_digest


def _validate_identical_result_rows(results: tuple[ReplayResult, ...]) -> None:
    if len(results) != len(DefenseArm) or len(
        {item.decision_event_ids for item in results}
    ) != 1:
        raise ReplayContractError("defense arms did not replay identical rows")


def _validated_replay_rows(
    matrix: FeatureMatrix, defender: LoadedDefenderBundle
) -> tuple[tuple[FeatureVector, ...], tuple[ObservedEvent, ...]]:
    if not matrix.rows or len(matrix.rows) > _MAX_REPLAY_ROWS:
        raise ReplayContractError("replay rows must be bounded and nonempty")
    if matrix.catalog != defender.catalog or matrix.catalog_digest != matrix.rows[0].catalog_digest:
        raise ReplayContractError("feature catalog lineage does not match defender")
    event_by_id = {event.event_id: event for event in matrix.events}
    if len(event_by_id) != len(matrix.events):
        raise ReplayContractError("feature matrix contains duplicate observation IDs")
    events: list[ObservedEvent] = []
    for row in matrix.rows:
        event = event_by_id.get(row.event_id)
        if event is None or not event.is_decision_point or event.decision_at != row.decision_at:
            raise ReplayContractError("feature row is not bound to its decision event")
        events.append(event)
    expected = tuple(
        sorted(
            matrix.rows,
            key=lambda row: (row.decision_at, row.event_id),
        )
    )
    if matrix.rows != expected:
        raise ReplayContractError("ordered decision rows must be chronological and stable")
    return matrix.rows, tuple(events)


def _validate_evaluator_context(
    context: ReplayEvaluationContext, event_ids: tuple[str, ...]
) -> None:
    truth_ids = tuple(row.event_id for row in context.truth)
    if truth_ids != event_ids or len(truth_ids) != len(set(truth_ids)):
        raise ReplayContractError("evaluator truth must bijectively match ordered replay rows")
    assignment_ids = tuple(item.event_id for item in context.slice_assignments)
    if assignment_ids != event_ids:
        raise ReplayContractError("slice assignments must match ordered replay rows")
    observation_ids = {row.event_id for row in context.observations}
    if not set(event_ids) <= observation_ids:
        raise ReplayContractError("evaluator observations do not cover replay rows")
    for latency in context.latency_samples:
        if tuple(item.event_id for item in latency.samples) != event_ids:
            raise ReplayContractError("latency samples must match ordered replay rows")
    family_by_campaign: dict[str, str] = {}
    for row in context.truth:
        owner = family_by_campaign.setdefault(row.campaign_id, row.family)
        if owner != row.family:
            raise ReplayContractError(
                "competition campaigns must have exactly one family owner"
            )


def _validate_descriptor_lineage(
    descriptor: EvaluationDescriptor,
    event_ids: tuple[str, ...],
    defender: LoadedDefenderBundle,
    split: EvaluationSplit | None,
    matrix: FeatureMatrix,
    attestation: VerifiedDefenderAttestation,
    context: ReplayEvaluationContext,
    regime_evidence: ReplayRegimeEvidence | None,
    corpus_evidence: ReplayCorpusEvidence,
) -> EvaluationLineage:
    if type(split) is not EvaluationSplit:
        raise ReplayContractError("evaluation descriptor lineage requires an exact split")
    checked = _exact_model(split, EvaluationSplit, "evaluation split")
    if type(corpus_evidence) is not ReplayCorpusEvidence:
        raise ReplayContractError("replay corpus evidence must have its exact type")
    try:
        checked_corpus = ReplayCorpusEvidence.model_validate(
            {
                **corpus_evidence.model_dump(
                    mode="python", warnings=False, exclude={"corpus"}
                ),
                "corpus": corpus_evidence.corpus,
            },
            strict=True,
        )
    except ValidationError as error:
        raise ReplayContractError(
            "replay corpus evidence failed semantic revalidation"
        ) from error
    if (
        checked_corpus.split_digest != checked.split_digest
        or make_evaluation_split(checked_corpus.corpus, checked.config) != checked
    ):
        raise ReplayContractError(
            "signed corpus evidence does not rederive the exact evaluation split"
        )
    checked_regime: ReplayRegimeEvidence | None = None
    if descriptor.kind is EvaluationKind.REGIME:
        checked_regime = _exact_regime_evidence(regime_evidence)
    elif regime_evidence is not None:
        raise ReplayContractError("regime evidence is irrelevant to this descriptor")
    binding = defender.training_binding
    manifest = defender.manifest
    training_split = (
        make_evaluation_split(checked_regime.parent_corpus, checked.config)
        if checked_regime is not None
        else checked
    )
    if (
        training_split.split_digest != manifest.split_manifest_digest
        or _digest_document(
            training_split.model_dump(mode="json", exclude={"split_digest"})
        )
        != binding.split_semantic_digest
        or binding.split_artifact_digest != manifest.split_artifact_digest
    ):
        raise ReplayContractError("evaluation descriptor lineage differs from signed split")
    if descriptor.kind is EvaluationKind.CHRONOLOGICAL:
        expected = checked.row_ids["development"]
    elif descriptor.kind is EvaluationKind.COLD_ENTITY:
        cohort = EntityCohort(descriptor.value)
        expected = tuple(
            event_id
            for event_id in checked.row_ids["development"]
            if cohort in checked.entity_cohorts[event_id]
        )
    elif descriptor.kind is EvaluationKind.HELD_FAMILY:
        if checked.held_out_family != descriptor.value:
            raise ReplayContractError(
                "held-family descriptor lineage lacks matching LOFO split"
            )
        expected = checked.held_out_evaluation_row_ids
        requested = binding.requested_row_ids
        final_fit = binding.final_fit_row_ids
        if any(checked.row_families[row_id] == descriptor.value for row_id in requested) or any(
            checked.row_families[row_id] == descriptor.value for row_id in final_fit
        ):
            raise ReplayContractError(
                "held-family descriptor lineage lacks training-exclusion proof"
            )
    elif descriptor.kind is EvaluationKind.REGIME:
        assert checked_regime is not None
        regime_manifest = checked_regime.manifest
        if (
            regime_manifest.transformer.value != descriptor.value
            or regime_manifest.parent_corpus_digest != manifest.corpus_digest
        ):
            raise ReplayContractError(
                "regime descriptor lineage differs from its signed parent corpus"
            )
        expected = checked.row_ids["development"]
        derived_truth = {
            row.event_id: row for row in checked_regime.derived_corpus.truth
        }
        expected_observation_ids = {
            lifecycle_id
            for event_id in expected
            for lifecycle_id in derived_truth[event_id].lifecycle_event_ids
        }
        expected_events = tuple(
            sorted(
                (
                    row
                    for row in checked_regime.derived_corpus.observations
                    if row.event_id in expected_observation_ids
                ),
                key=lambda row: row.event_id,
            )
        )
        if matrix.events != expected_events:
            raise ReplayContractError(
                "regime feature matrix differs from the exact derived corpus"
            )
    else:
        raise ReplayContractError("hidden descriptor lineage is authority-only")
    if event_ids != expected:
        raise ReplayContractError(
            "evaluation descriptor lineage row order does not match its source"
        )
    truth_by_id = {row.event_id: row for row in context.truth}
    if set(truth_by_id) != set(event_ids):
        raise ReplayContractError("signed split truth does not cover replay rows")
    for event_id in event_ids:
        truth = truth_by_id[event_id]
        if (
            truth.family != checked.row_families[event_id]
            or truth.campaign_id != checked.row_campaigns[event_id]
            or truth.is_fraud is not checked.row_is_fraud[event_id]
            or truth.net_settled_value
            != checked.row_net_settled_values[event_id]
        ):
            raise ReplayContractError(
                "evaluator truth differs from signed split truth"
            )
    assignment_by_id = {row.event_id: row for row in context.slice_assignments}
    if set(assignment_by_id) != set(event_ids):
        raise ReplayContractError("evaluator cohorts do not cover signed split rows")
    if descriptor.kind is EvaluationKind.REGIME:
        assert checked_regime is not None
        if checked_corpus.corpus != checked_regime.derived_corpus:
            raise ReplayContractError(
                "signed regime corpus differs from exact derived-corpus evidence"
            )
        derived_truth = {
            row.event_id: row for row in checked_regime.derived_corpus.truth
        }
        if tuple(truth_by_id[event_id] for event_id in event_ids) != tuple(
            derived_truth[event_id] for event_id in event_ids
        ):
            raise ReplayContractError(
                "regime evaluator truth differs from the exact derived corpus"
            )
    else:
        corpus_truth = {row.event_id: row for row in checked_corpus.corpus.truth}
        corpus_observations = {
            row.event_id: row for row in checked_corpus.corpus.observations
        }
        if tuple(truth_by_id[event_id] for event_id in event_ids) != tuple(
            corpus_truth[event_id] for event_id in event_ids
        ) or any(
            corpus_observations.get(event.event_id) != event for event in matrix.events
        ):
            raise ReplayContractError(
                "evaluator context differs from signed corpus truth or observations"
            )
    if any(
        assignment_by_id[event_id].entity_cohorts
        != checked.entity_cohorts[event_id]
        for event_id in event_ids
    ):
        raise ReplayContractError(
            "evaluator cohorts differ from signed split cohort mapping"
        )
    cohort_document = {
        event_id: [item.value for item in checked.entity_cohorts[event_id]]
        for event_id in event_ids
    }
    training_document = {
        "requested": list(binding.requested_row_ids),
        "excluded": list(binding.excluded_row_ids),
        "final_fit": list(binding.final_fit_row_ids),
        "receipt": defender.manifest.training_receipt_digest,
        "binding": defender.manifest.training_binding_digest,
    }
    return EvaluationLineage.create(
        descriptor=descriptor,
        decision_rows_digest=_digest_document(event_ids),
        decision_content_digest=_digest_document(matrix.model_dump(mode="json")),
        corpus_digest=(
            checked_regime.manifest.output_corpus_digest
            if checked_regime is not None
            else checked_corpus.corpus_digest
        ),
        split_digest=checked.split_digest,
        cohort_mapping_digest=_digest_document(cohort_document),
        training_population_digest=_digest_document(training_document),
        bundle_manifest_digest=_manifest_digest(defender.manifest),
        defender_top_ref_digest=attestation.top_ref.sha256,
        regime_parent_digest=(
            checked_regime.manifest.parent_corpus_digest
            if checked_regime is not None
            else None
        ),
        regime_output_digest=(
            checked_regime.manifest.output_corpus_digest
            if checked_regime is not None
            else None
        ),
        regime_parameters_digest=(
            _digest_document(checked_regime.manifest.parameters)
            if checked_regime is not None
            else None
        ),
        regime_truth_unchanged=(
            checked_regime.manifest.truth_bytes_unchanged
            if checked_regime is not None
            else None
        ),
        held_family=(
            cast(Family, descriptor.value)
            if descriptor.kind is EvaluationKind.HELD_FAMILY
            else None
        ),
        training_exclusion_verified=descriptor.kind is EvaluationKind.HELD_FAMILY,
    )


def _arm_decisions(
    *,
    arm: DefenseArm,
    events: tuple[ObservedEvent, ...],
    vectors: tuple[FeatureVector, ...],
    rules: tuple[RuleResult, ...],
    calibrated: np.ndarray | None,
    mandatory: tuple[bool, ...],
    common_mandatory: tuple[DefenseDecision | None, ...],
    thresholds: ReplayThresholdSet,
    failure: ModelFailure | None,
    rule_manifest: RuleManifest,
    rule_fallback_thresholds: PolicyThresholds | None,
) -> tuple[DefenseDecision, ...]:
    report = thresholds.report_for(arm)
    arm_thresholds = report.thresholds
    if arm_thresholds is None or rule_fallback_thresholds is None:
        raise ReplayContractError("replay requires feasible arm and fallback thresholds")
    policy = ActionPolicy(rule_manifest=rule_manifest)
    mode: Literal["rules_only", "model_only", "layered"]
    if arm is DefenseArm.RULES_ONLY:
        mode = "rules_only"
    elif arm is DefenseArm.GBDT_ONLY:
        mode = "model_only"
    else:
        mode = "layered"
    output: list[DefenseDecision] = []
    for index, (event, vector, rule_result) in enumerate(
        zip(events, vectors, rules, strict=True)
    ):
        if mandatory[index]:
            shared = common_mandatory[index]
            if shared is None:
                raise ReplayContractError("mandatory decision was not materialized")
            output.append(shared)
            continue
        calibrated_score = (
            None
            if arm is DefenseArm.RULES_ONLY or failure is not None
            else float(cast(np.ndarray, calibrated)[index])
        )
        output.append(
            policy.choose(
                event,
                rule_result,
                calibrated_score=calibrated_score,
                thresholds=arm_thresholds,
                model_failure=None if failure is None else failure.reason,
                failed_component_version=(
                    None if failure is None else failure.failed_component_version
                ),
                latency_ms=0.0,
                vector=vector,
                score_mode=mode,
                fallback_thresholds=rule_fallback_thresholds,
            )
        )
    if any(
        selected and output[index].action is not Action.DECLINE
        for index, selected in enumerate(mandatory)
    ):
        raise ReplayContractError("mandatory decision differs from production policy")
    return tuple(output)


def _hidden_descriptor_lineage(
    descriptor: EvaluationDescriptor,
    event_ids: tuple[str, ...],
    defender: LoadedDefenderBundle,
    matrix: FeatureMatrix,
    attestation: VerifiedDefenderAttestation,
    context: ReplayEvaluationContext,
) -> EvaluationLineage:
    binding = defender.training_binding
    cohort_document = {
        row.event_id: [item.value for item in row.entity_cohorts]
        for row in context.slice_assignments
    }
    training_document = {
        "requested": list(binding.requested_row_ids),
        "excluded": list(binding.excluded_row_ids),
        "final_fit": list(binding.final_fit_row_ids),
        "receipt": defender.manifest.training_receipt_digest,
        "binding": defender.manifest.training_binding_digest,
    }
    return EvaluationLineage.create(
        descriptor=descriptor,
        decision_rows_digest=_digest_document(event_ids),
        decision_content_digest=_digest_document(matrix.model_dump(mode="json")),
        corpus_digest=defender.manifest.corpus_digest,
        split_digest=defender.manifest.split_manifest_digest,
        cohort_mapping_digest=_digest_document(cohort_document),
        training_population_digest=_digest_document(training_document),
        bundle_manifest_digest=_manifest_digest(defender.manifest),
        defender_top_ref_digest=attestation.top_ref.sha256,
    )


def _common_mandatory_decisions(
    events: tuple[ObservedEvent, ...],
    mandatory: tuple[bool, ...],
    rule_manifest: RuleManifest,
) -> tuple[DefenseDecision | None, ...]:
    """Compute each mandatory decision once for literal reuse by every arm."""
    policy = ActionPolicy(rule_manifest=rule_manifest)
    return tuple(
        policy.choose(
            event,
            RuleResult.clear(),
            calibrated_score=None,
            thresholds=None,
            latency_ms=0.0,
        )
        if selected
        else None
        for event, selected in zip(events, mandatory, strict=True)
    )


def _decision(
    event: ObservedEvent,
    rule_result: RuleResult,
    *,
    action: Action,
    score: float,
    calibrated_score: float | None,
    reasons: tuple[DefenseReason, ...],
    fallback: ModelFailure | None,
) -> DefenseDecision:
    evidence = tuple(
        sorted(
            {
                event.event_id,
                *(
                    source
                    for hit in rule_result.hits
                    for source in hit.evidence_source_ids
                ),
            }
        )
    )
    return DefenseDecision(
        event_id=event.event_id,
        action=action,
        score=score,
        rule_score=rule_result.score,
        calibrated_score=calibrated_score,
        reason_codes=tuple(dict.fromkeys(reasons)),
        evidence_source_ids=evidence,
        fallback_used=fallback is not None,
        fallback_reason=None if fallback is None else fallback.reason,
        failed_component_version=(
            None if fallback is None else fallback.failed_component_version
        ),
        latency_ms=0.0,
        policy_version="1.0.0",
    )


def _action(score: float, thresholds: PolicyThresholds) -> Action:
    if score >= thresholds.decline:
        return Action.DECLINE
    if score >= thresholds.challenge:
        return Action.CHALLENGE
    return Action.APPROVE


def _evaluate_frozen_arm(
    *,
    arm: DefenseArm,
    event_ids: tuple[str, ...],
    events: tuple[ObservedEvent, ...],
    decisions: tuple[DefenseDecision, ...],
    scores: np.ndarray,
    mandatory: tuple[bool, ...],
    threshold_report_digest: str,
    threshold_set_digest: str,
    bundle_manifest_digest: str,
    candidate_bundle_id: str,
    case_counter: ReplayCaseCounterBinding,
    context: ReplayEvaluationContext,
    evaluation_lineage: EvaluationLineage,
    evaluation_context_digest: str,
    rollback_available: bool,
    hidden_access_clean: bool,
    hidden_public_proof_id: str | None,
    failure: ModelFailure | None,
) -> _EvaluatedArm:
    actions = np.asarray([item.action for item in decisions], dtype=object)
    cases = group_cases(context.observations, decisions, as_of=context.as_of)
    production_counter = case_counter.reconstruct(
        context.observations, event_ids, context.as_of
    )
    if production_counter(actions.copy()) != len(cases):
        raise ReplayContractError("production review-case callback differs from full grouping")
    queue_report = simulate_case_queue(cases, context.queue_config)
    latency = next(item.samples for item in context.latency_samples if item.arm is arm)
    metric_inputs = MetricReportInputs(
        truth=context.truth,
        observations=context.observations,
        decisions=decisions,
        cases=queue_report.case_inputs,
        queue_report=queue_report,
        latency_samples=latency,
        as_of=context.as_of,
        slice_assignments=context.slice_assignments,
        slice_manifest=context.slice_manifest,
    )
    report = compute_metric_report(metric_inputs)
    evidence = MetricDerivationEvidence.from_inputs(metric_inputs)
    if evidence.evidence_digest != report.derivation_evidence_digest:
        raise ReplayContractError("metric report lost restricted derivation lineage")
    metrics = _promotion_metrics(report)
    assurance = AssuranceEvidence(
        leakage_passed=context.feature_assurance.leakage_passed,
        parity_passed=context.feature_assurance.parity_passed,
        artifact_signature_valid=True,
        rollback_available=rollback_available,
        hidden_access_clean=hidden_access_clean,
        campaign_family_ownership_valid=True,
    )
    mandatory_document = tuple(
        decisions[index].model_dump(mode="json")
        for index, selected in enumerate(mandatory)
        if selected
    )
    replay_failure = (
        None
        if failure is None
        else ReplayFailure(
            code=cast(Literal["MODEL_UNAVAILABLE", "MODEL_TIMEOUT"], failure.reason.value),
            failed_component_version=failure.failed_component_version,
        )
    )
    result = ReplayResult.create(
        arm=arm,
        evaluation=context.evaluation,
        evaluation_lineage=evaluation_lineage,
        candidate_role=CandidateRoleEvidence.create(
            role=(
                CandidateBundleRole.HELD_FAMILY_LOFO
                if context.evaluation.kind is EvaluationKind.HELD_FAMILY
                else CandidateBundleRole.POOLED
            ),
            held_family=(
                cast(Family, context.evaluation.value)
                if context.evaluation.kind is EvaluationKind.HELD_FAMILY
                else None
            ),
            bundle_id=candidate_bundle_id,
            bundle_manifest_digest=bundle_manifest_digest,
            defender_top_ref_digest=evaluation_lineage.defender_top_ref_digest,
            threshold_set_digest=threshold_set_digest,
        ),
        decision_event_ids=event_ids,
        decision_rows_digest=_digest_document(event_ids),
        common_integrity_digest=_digest_document(mandatory_document),
        action_digest=_digest_document(tuple(item.action.value for item in decisions)),
        score_digest=_array_digest(scores),
        threshold_report_digest=threshold_report_digest,
        threshold_set_digest=threshold_set_digest,
        bundle_manifest_digest=bundle_manifest_digest,
        case_callback_digest=case_counter.callback_digest,
        evaluation_context_digest=evaluation_context_digest,
        hidden_public_proof_id=hidden_public_proof_id,
        metric_report_digest=report.report_digest,
        metrics=metrics,
        assurance=assurance,
        failure=replay_failure,
        fallback_count=sum(item.fallback_used for item in decisions),
        mandatory_decline_count=sum(mandatory),
    )
    evidence_binding = HiddenArmEvidenceBinding(
        arm=arm.value,
        evaluator_input_digest=report.evaluator_input_digest,
        derivation_evidence_digest=evidence.evidence_digest,
        metric_report_digest=report.report_digest,
    )
    return _EvaluatedArm(result=result, hidden_evidence=evidence_binding)


def _promotion_metrics(report: MetricReport) -> PromotionMetrics:
    classification = report.classification
    operations = report.operations
    row_count = classification.row_count
    legitimate_count = classification.legitimate_count
    slices = tuple(
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
    )
    return PromotionMetrics(
        row_count=row_count,
        recall=classification.recall.value,
        ece=report.calibration.ece.value,
        p95_latency_ms=report.engineering.end_to_end_ms.p95.value,
        preventable_settled_value=report.value.preventable_settled_value,
        value_escaped=report.value.value_escaped,
        review_case_count=operations.review_case_count,
        challenge_rate=operations.challenge_count / row_count,
        false_decline=RateEvidence(
            numerator=operations.false_decline_count,
            denominator=legitimate_count,
            value=(
                operations.false_decline_count / legitimate_count
                if legitimate_count
                else None
            ),
            defined=legitimate_count > 0,
        ),
        review_case_rate=operations.review_case_count / row_count,
        slice_performance=slices,
    )


def _case_binding_document(
    observations: tuple[ObservedEvent, ...],
    event_ids: tuple[str, ...],
    as_of: datetime,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "observations": [
            row.model_dump(mode="json")
            for row in sorted(observations, key=lambda item: (item.available_at, item.event_id))
        ],
        "decision_event_ids": list(event_ids),
        "as_of": as_of.isoformat().replace("+00:00", "Z"),
    }


def _placeholder_decision(event_id: str) -> DefenseDecision:
    return DefenseDecision(
        event_id=event_id,
        action=Action.APPROVE,
        score=0.0,
        rule_score=0.0,
        calibrated_score=0.0,
        reason_codes=(),
        evidence_source_ids=(event_id,),
        fallback_used=False,
        fallback_reason=None,
        failed_component_version=None,
        latency_ms=0.0,
        policy_version="1.0.0",
    )


def _manifest_digest(manifest: DefenderBundleManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest.model_dump(mode="json"))).hexdigest()


def _array_digest(values: np.ndarray) -> str:
    checked = np.asarray(values, dtype=np.float64)
    if checked.ndim != 1 or not np.isfinite(checked).all():
        raise ReplayContractError("replay score array is invalid")
    return _digest_document(
        {
            "dtype": checked.dtype.str,
            "shape": list(checked.shape),
            "values_hex": checked.tobytes(order="C").hex(),
        }
    )


def _numeric_array_digest(values: np.ndarray) -> str:
    if type(values) is not np.ndarray or values.ndim != 1 or not values.size:
        raise ReplayContractError(
            "selection arrays must be exact nonempty one-dimensional arrays"
        )
    if np.issubdtype(values.dtype, np.complexfloating) or np.issubdtype(
        values.dtype, np.object_
    ):
        raise ReplayContractError("selection arrays have an unsupported dtype")
    contiguous = np.ascontiguousarray(values)
    numeric = np.asarray(contiguous, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ReplayContractError("selection arrays must be finite")
    return _digest_document(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "bytes_hex": contiguous.tobytes(order="C").hex(),
        }
    )


def _action_array_digest(actions: np.ndarray) -> str:
    if type(actions) is not np.ndarray or actions.ndim != 1 or not actions.size:
        raise ReplayContractError("mandatory actions must be an exact nonempty array")
    values: list[str] = []
    for action in actions:
        if type(action) is not Action:
            raise ReplayContractError("mandatory action array contains a non-action")
        values.append(action.value)
    return _digest_document(values)


def _exact_model[T: ExternalContract](value: object, expected: type[T], label: str) -> T:
    if type(value) is not expected:
        raise ReplayContractError(f"{label} must have its exact contract type")
    try:
        return expected.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True
        )
    except ValidationError as error:
        raise ReplayContractError(f"{label} failed semantic revalidation") from error


def _exact_regime_evidence(value: object) -> ReplayRegimeEvidence:
    if type(value) is not ReplayRegimeEvidence:
        raise ReplayContractError(
            "regime descriptor lineage requires exact derived-corpus evidence"
        )
    try:
        return ReplayRegimeEvidence.model_validate(
            {
                **value.model_dump(
                    mode="python",
                    warnings=False,
                    exclude={
                        "parent_corpus",
                        "derived_corpus",
                        "control_corpus",
                        "spec",
                        "manifest",
                    },
                ),
                "parent_corpus": value.parent_corpus,
                "derived_corpus": value.derived_corpus,
                "control_corpus": value.control_corpus,
                "spec": value.spec,
                "manifest": value.manifest,
            },
            strict=True,
        )
    except ValidationError as error:
        raise ReplayContractError(
            "regime evidence failed semantic revalidation"
        ) from error


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("digest must be lowercase SHA-256")


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_json_tree(document))).hexdigest()


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evaluator_context_digest(payload: bytes) -> str:
    return _digest_bytes(b"apar-hidden-evaluator-context-v1\x00" + payload)


def _parse_utc_wire(value: object) -> datetime:
    if type(value) is not str or len(value) > 40 or not value.endswith("Z"):
        raise ReplayContractError("UTC timestamp wire value is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        return validate_utc_timestamp(parsed)
    except ValueError as error:
        raise ReplayContractError("UTC timestamp wire value is invalid") from error


def _tupleize_context_document(document: dict[str, object]) -> None:
    for name in ("truth", "observations", "slice_assignments", "latency_samples"):
        value = document.get(name)
        if type(value) is list:
            document[name] = tuple(value)
    truth = document.get("truth")
    if type(truth) is tuple:
        for row in cast(tuple[object, ...], truth):
            if type(row) is dict and type(row.get("lifecycle_event_ids")) is list:
                row["lifecycle_event_ids"] = tuple(row["lifecycle_event_ids"])
    assignments = document.get("slice_assignments")
    if type(assignments) is tuple:
        for row in cast(tuple[object, ...], assignments):
            if type(row) is dict and type(row.get("entity_cohorts")) is list:
                cohort_values = row["entity_cohorts"]
                if any(type(item) is not str for item in cohort_values):
                    raise ReplayContractError("entity cohort JSON is invalid")
                row["entity_cohorts"] = tuple(
                    EntityCohort(cast(str, item)) for item in cohort_values
                )
    manifest = document.get("slice_manifest")
    if type(manifest) is dict:
        for name in ("regimes", "entity_cohorts"):
            if type(manifest.get(name)) is list:
                manifest[name] = tuple(manifest[name])
    latency = document.get("latency_samples")
    if type(latency) is tuple:
        for row in cast(tuple[object, ...], latency):
            if type(row) is dict and type(row.get("samples")) is list:
                row["samples"] = tuple(row["samples"])


def _json_tree(value: object) -> object:
    if isinstance(value, ExternalContract):
        return value.model_dump(mode="json")
    if isinstance(value, StrEnum):
        return value.value
    if type(value) is tuple:
        return [_json_tree(item) for item in cast(tuple[object, ...], value)]
    if type(value) is list:
        return [_json_tree(item) for item in cast(list[object], value)]
    if type(value) is dict:
        return {
            cast(str, key): _json_tree(item)
            for key, item in cast(dict[object, object], value).items()
        }
    return value


__all__ = [
    "DefenseArm",
    "HiddenReplayOutcome",
    "ModelFailure",
    "ReplayCaseCounterBinding",
    "ReplayContractError",
    "ReplayEvaluationContext",
    "ReplayFeatureAssurance",
    "ReplayLatencySamples",
    "ReplayRegimeEvidence",
    "ReplayThresholdSet",
    "bind_replay_case_counter",
    "replay_defense_arms",
]
