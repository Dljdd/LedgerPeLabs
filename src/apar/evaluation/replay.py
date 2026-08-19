"""Verified identical-row replay for rules, GBDT, and layered hybrid arms."""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, cast

import numpy as np
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
    normalize_operating_scores,
    select_policy_thresholds,
)
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.defender_attestation import VerifiedDefenderAttestation
from apar.evaluation.gates import (
    AssuranceEvidence,
    DefenseArm,
    EvaluationDescriptor,
    EvaluationKind,
    PromotionMetrics,
    ReplayFailure,
    ReplayResult,
    SlicePerformance,
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
from apar.evaluation.splits import EntityCohort
from apar.evaluation_hidden.defense_authority import (
    HiddenArmEvidenceBinding,
    HiddenReleaseRequest,
    ResolvedHiddenEvaluation,
    resolve_hidden_release,
    seal_hidden_evaluation,
    verify_hidden_receipt,
)
from apar.features.builders import FeatureMatrix
from apar.features.parity import audit_feature_matrix
from apar.features.state import FeatureVector
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_MAX_REPLAY_ROWS = 100_000
_MAX_REPLAY_BYTES = 32_000_000
_MODEL_FAILURES = {DefenseReason.MODEL_UNAVAILABLE, DefenseReason.MODEL_TIMEOUT}
_CASE_BINDING_TOKEN = object()
_THRESHOLD_SETS: dict[int, weakref.ReferenceType[ReplayThresholdSet]] = {}


class ReplayContractError(ValueError):
    """Replay rows, artifacts, or evaluator lineage are inconsistent."""


@dataclass(frozen=True, slots=True)
class _EvaluatedArm:
    result: ReplayResult
    hidden_evidence: HiddenArmEvidenceBinding


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
        case_counter.validate_context(matrix.events, row_ids, case_counter.as_of)
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
                    case_counter,
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
            "reports": checked,
        }
        digest_fields = {
            "schema_version": "1.0.0",
            "bundle_manifest_digest": fields["bundle_manifest_digest"],
            "case_callback_digest": fields["case_callback_digest"],
            "selection_row_ids_digest": fields["selection_row_ids_digest"],
            "reports": [item.model_dump(mode="json") for item in checked],
        }
        value = cls.model_validate(
            {**fields, "threshold_set_digest": _digest_document(digest_fields)}
        )
        _register_threshold_set(value)
        return value

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


@dataclass(frozen=True, slots=True)
class _CaseBindingState:
    counter: ReviewCaseCounter
    event_ids: tuple[str, ...]
    rows_digest: str
    as_of: datetime
    callback_digest: str


_CASE_BINDINGS: dict[
    int,
    tuple[
        weakref.ReferenceType[ReplayCaseCounterBinding],
        _CaseBindingState,
    ],
] = {}


class ReplayCaseCounterBinding:
    """Immutable lineage wrapper around the production Task10 callback."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        *,
        counter: ReviewCaseCounter,
        event_ids: tuple[str, ...],
        rows_digest: str,
        as_of: datetime,
        callback_digest: str,
        _token: object = None,
    ) -> None:
        if _token is not _CASE_BINDING_TOKEN:
            raise ReplayContractError(
                "case callback bindings must come from the production factory"
            )
        if id(self) in _CASE_BINDINGS:
            raise ReplayContractError("case callback binding is already initialized")
        if type(counter) is not ReviewCaseCounter:
            raise ReplayContractError("case callback must be the exact production adapter")
        _validate_digest(rows_digest)
        _validate_digest(callback_digest)
        state = _CaseBindingState(
            counter=counter,
            event_ids=event_ids,
            rows_digest=rows_digest,
            as_of=as_of,
            callback_digest=callback_digest,
        )
        identity = id(self)

        def cleanup(reference: weakref.ReferenceType[ReplayCaseCounterBinding]) -> None:
            current = _CASE_BINDINGS.get(identity)
            if current is not None and current[0] is reference:
                _CASE_BINDINGS.pop(identity, None)

        reference = weakref.ref(self, cleanup)
        _CASE_BINDINGS[identity] = (reference, state)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("replay case callback binding is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("replay case callback binding is immutable")

    @property
    def callback_digest(self) -> str:
        return _case_binding_state(self).callback_digest

    @property
    def as_of(self) -> datetime:
        return _case_binding_state(self).as_of

    def __call__(self, actions: np.ndarray) -> int:
        return _case_binding_state(self).counter(actions)

    def validate_context(
        self,
        observations: tuple[ObservedEvent, ...],
        event_ids: tuple[str, ...],
        as_of: datetime,
    ) -> None:
        state = _case_binding_state(self)
        document = _case_binding_document(observations, event_ids, as_of)
        if (
            event_ids != state.event_ids
            or as_of != state.as_of
            or _digest_document(document) != state.rows_digest
        ):
            raise ReplayContractError("case callback lineage does not match replay rows")
        expected_callback_digest = _digest_document(
            {"schema_version": "1.0.0", "binding": document, "adapter": "Task10"}
        )
        if expected_callback_digest != self.callback_digest:
            raise ReplayContractError("case callback lineage digest is inconsistent")


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
    placeholder = tuple(
        DefenseDecision(
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
        for event_id in canonical_ids
    )
    ordered_observations = tuple(
        sorted(observations, key=lambda row: (row.available_at, row.event_id))
    )
    try:
        counter = bind_review_case_counter(ordered_observations, placeholder, as_of=as_of)
    except Exception as error:
        raise ReplayContractError("production case callback binding failed") from error
    document = _case_binding_document(observations, decision_event_ids, as_of)
    rows_digest = _digest_document(document)
    callback_digest = _digest_document(
        {"schema_version": "1.0.0", "binding": document, "adapter": "Task10"}
    )
    return ReplayCaseCounterBinding(
        counter=counter,
        event_ids=decision_event_ids,
        rows_digest=rows_digest,
        as_of=as_of,
        callback_digest=callback_digest,
        _token=_CASE_BINDING_TOKEN,
    )


def replay_defense_arms(
    *,
    matrix: FeatureMatrix,
    defender: LoadedDefenderBundle,
    defender_attestation: VerifiedDefenderAttestation,
    thresholds: ReplayThresholdSet,
    case_counter: ReplayCaseCounterBinding,
    evaluation: ReplayEvaluationContext | None = None,
    hidden_release: HiddenReleaseRequest | None = None,
    model_failure: ModelFailure | None = None,
) -> tuple[ReplayResult, ...]:
    """Score identical rows, freeze decisions, then resolve evaluator-side metrics."""
    try:
        matrix_value = _exact_model(matrix, FeatureMatrix, "feature matrix")
        _exact_model(thresholds, ReplayThresholdSet, "threshold set")
        if not _is_registered_threshold_set(thresholds):
            raise ReplayContractError(
                "threshold set must come from exact rederived selection evidence"
            )
        threshold_set = thresholds
        if type(defender) is not LoadedDefenderBundle:
            raise ReplayContractError("defender must be an exact verified loaded bundle")
        if type(defender_attestation) is not VerifiedDefenderAttestation:
            raise ReplayContractError("replay requires an exact defender attestation")
        if type(case_counter) is not ReplayCaseCounterBinding:
            raise ReplayContractError("case callback must have exact replay lineage")
        if (evaluation is None) == (hidden_release is None):
            raise ReplayContractError(
                "replay requires exactly one development context or hidden release"
            )
        if evaluation is not None and type(evaluation) is not ReplayEvaluationContext:
            raise ReplayContractError("evaluation context must have its exact contract type")
        if hidden_release is not None and type(hidden_release) is not HiddenReleaseRequest:
            raise ReplayContractError("hidden release request must have its exact type")
        declared_failure = (
            None
            if model_failure is None
            else _exact_model(model_failure, ModelFailure, "model failure")
        )
        rows, events = _validated_replay_rows(matrix_value, defender)
        event_ids = tuple(row.event_id for row in rows)
        case_counter.validate_context(matrix_value.events, event_ids, case_counter.as_of)
        manifest_digest = _manifest_digest(defender.manifest)
        if (
            defender_attestation.bundle_manifest_digest != manifest_digest
            or defender_attestation.top_ref.sha256 != manifest_digest
            or defender_attestation.threshold_digest
            != defender.manifest.threshold_digest
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
        mandatory = tuple(
            any(hit.mandatory for hit in result.hits) for result in rule_results
        )
        common_mandatory = _common_mandatory_decisions(
            events, mandatory, defender.rule_manifest
        )
        actual_failure = declared_failure
        calibrated: np.ndarray | None = None
        if actual_failure is None:
            try:
                raw_model_scores = defender.scorer.predict(matrix_value)
                calibrated = defender.calibrator.predict(raw_model_scores)
            except Exception:
                actual_failure = ModelFailure(
                    reason=DefenseReason.MODEL_UNAVAILABLE,
                    failed_component_version=f"model:{defender.manifest.model_digest}",
                )

        raw_rule_scores = np.asarray(
            [item.score for item in rule_results], dtype=np.float64
        )
        rule_scores = normalize_operating_scores(raw_rule_scores)
        if calibrated is None:
            gbdt_scores = normalize_operating_scores(
                np.zeros(len(rows), dtype=np.float64)
            )
            hybrid_scores = rule_scores.copy()
        else:
            gbdt_scores = normalize_operating_scores(calibrated)
            hybrid_scores = normalize_operating_scores(
                np.maximum(raw_rule_scores, calibrated)
            )
        arm_scores = {
            DefenseArm.RULES_ONLY: rule_scores,
            DefenseArm.GBDT_ONLY: gbdt_scores,
            DefenseArm.LAYERED_HYBRID: hybrid_scores,
        }
        # All scoring and action construction completes before the first truth row is read.
        decisions_by_arm = {
            arm: _arm_decisions(
                arm=arm,
                events=events,
                rules=rule_results,
                scores=arm_scores[arm],
                calibrated=calibrated,
                mandatory=mandatory,
                common_mandatory=common_mandatory,
                thresholds=threshold_set,
                failure=actual_failure,
                rule_fallback_thresholds=threshold_set.report_for(
                    DefenseArm.RULES_ONLY
                ).thresholds,
            )
            for arm in DefenseArm
        }
        frozen_decision_digest = _digest_document(
            tuple(
                (
                    arm.value,
                    tuple(item.model_dump(mode="json") for item in decisions_by_arm[arm]),
                )
                for arm in DefenseArm
            )
        )
        if not frozen_decision_digest:
            raise ReplayContractError("decision freeze failed")
        resolved: ResolvedHiddenEvaluation | None = None
        if hidden_release is not None:
            resolved = resolve_hidden_release(hidden_release)
            context = ReplayEvaluationContext.from_json(resolved.payload)
            if context.evaluation.kind is not EvaluationKind.HIDDEN:
                raise ReplayContractError(
                    "sealed hidden release must contain a hidden evaluation descriptor"
                )
        else:
            assert evaluation is not None
            context = _exact_model(
                evaluation, ReplayEvaluationContext, "evaluation context"
            )
            if context.evaluation.kind is EvaluationKind.HIDDEN:
                raise ReplayContractError(
                    "hidden evaluation requires an exact sealed hidden release receipt"
                )
        _validate_evaluator_context(context, event_ids)
        case_counter.validate_context(context.observations, event_ids, context.as_of)
        if audit.passed != context.feature_assurance.leakage_passed:
            raise ReplayContractError("feature leakage evidence disagrees with replay audit")
        context_digest = _evaluator_context_digest(context.to_json())
        evaluated = tuple(
            _evaluate_frozen_arm(
                arm=arm,
                event_ids=event_ids,
                events=events,
                decisions=decisions_by_arm[arm],
                scores=arm_scores[arm],
                mandatory=mandatory,
                threshold_set=threshold_set,
                manifest=defender.manifest,
                case_counter=case_counter,
                context=context,
                evaluation_context_digest=context_digest,
                rollback_available=defender_attestation.rollback_available,
                hidden_access_clean=resolved is None,
                failure=actual_failure if arm is DefenseArm.GBDT_ONLY else None,
            )
            for arm in DefenseArm
        )
        results = tuple(item.result for item in evaluated)
        if resolved is not None:
            evidence = tuple(item.hidden_evidence for item in evaluated)
            receipt = seal_hidden_evaluation(
                resolved, evidence, sealed_at=context.as_of
            )
            if (
                receipt.bundle_manifest_digest != manifest_digest
                or receipt.defender_attestation_digest
                != defender_attestation.attestation_digest
                or receipt.defender_top_ref_digest
                != defender_attestation.top_ref.sha256
                or receipt.evaluator_context_digest != context_digest
                or not verify_hidden_receipt(receipt, resolved, evidence)
            ):
                raise ReplayContractError("sealed hidden release receipt failed verification")
            results = tuple(
                item.result.rebuild(
                    hidden_release_receipt_digest=receipt.receipt_digest,
                    assurance=item.result.assurance.model_copy(
                        update={"hidden_access_clean": True}
                    ),
                )
                for item in evaluated
            )
        if len({item.decision_event_ids for item in results}) != 1:
            raise ReplayContractError("defense arms did not replay identical rows")
        return results
    except ReplayContractError:
        raise
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


def _arm_decisions(
    *,
    arm: DefenseArm,
    events: tuple[ObservedEvent, ...],
    rules: tuple[RuleResult, ...],
    scores: np.ndarray,
    calibrated: np.ndarray | None,
    mandatory: tuple[bool, ...],
    common_mandatory: tuple[DefenseDecision | None, ...],
    thresholds: ReplayThresholdSet,
    failure: ModelFailure | None,
    rule_fallback_thresholds: PolicyThresholds | None,
) -> tuple[DefenseDecision, ...]:
    report = thresholds.report_for(arm)
    arm_thresholds = report.thresholds
    if arm_thresholds is None or rule_fallback_thresholds is None:
        raise ReplayContractError("replay requires feasible arm and fallback thresholds")
    output: list[DefenseDecision] = []
    for index, (event, rule_result) in enumerate(zip(events, rules, strict=True)):
        if mandatory[index]:
            shared = common_mandatory[index]
            if shared is None:
                raise ReplayContractError("mandatory decision cache is incomplete")
            output.append(shared)
            continue
        if arm is DefenseArm.GBDT_ONLY and failure is not None:
            output.append(
                _decision(
                    event,
                    rule_result,
                    action=Action.APPROVE,
                    score=float(scores[index]),
                    calibrated_score=None,
                    reasons=(failure.reason,),
                    fallback=None,
                )
            )
            continue
        if arm is DefenseArm.LAYERED_HYBRID and failure is not None:
            output.append(
                _decision(
                    event,
                    rule_result,
                    action=_action(float(scores[index]), rule_fallback_thresholds),
                    score=float(scores[index]),
                    calibrated_score=None,
                    reasons=tuple(
                        dict.fromkeys(
                            (item.reason for item in rule_result.hits),
                        )
                    )
                    + (failure.reason,),
                    fallback=failure,
                )
            )
            continue
        output.append(
            _decision(
                event,
                rule_result,
                action=_action(float(scores[index]), arm_thresholds),
                score=float(scores[index]),
                calibrated_score=(
                    None
                    if arm is DefenseArm.RULES_ONLY
                    else float(cast(np.ndarray, calibrated)[index])
                ),
                reasons=(
                    tuple(item.reason for item in rule_result.hits)
                    if arm is not DefenseArm.GBDT_ONLY
                    else ()
                ),
                fallback=None,
            )
        )
    return tuple(output)


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
    threshold_set: ReplayThresholdSet,
    manifest: DefenderBundleManifest,
    case_counter: ReplayCaseCounterBinding,
    context: ReplayEvaluationContext,
    evaluation_context_digest: str,
    rollback_available: bool,
    hidden_access_clean: bool,
    failure: ModelFailure | None,
) -> _EvaluatedArm:
    actions = np.asarray([item.action for item in decisions], dtype=object)
    cases = group_cases(context.observations, decisions, as_of=context.as_of)
    if case_counter(actions.copy()) != len(cases):
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
    threshold_report = threshold_set.report_for(arm)
    result = ReplayResult.create(
        arm=arm,
        evaluation=context.evaluation,
        decision_event_ids=event_ids,
        decision_rows_digest=_digest_document(event_ids),
        common_integrity_digest=_digest_document(mandatory_document),
        action_digest=_digest_document(tuple(item.action.value for item in decisions)),
        score_digest=_array_digest(scores),
        threshold_report_digest=threshold_report.report_digest,
        threshold_set_digest=threshold_set.threshold_set_digest,
        bundle_manifest_digest=_manifest_digest(manifest),
        case_callback_digest=case_counter.callback_digest,
        evaluation_context_digest=evaluation_context_digest,
        hidden_release_receipt_digest=None,
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
        false_decline_rate=(
            operations.false_decline_count / legitimate_count
            if legitimate_count
            else 0.0
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


def _register_threshold_set(value: ReplayThresholdSet) -> None:
    identity = id(value)

    def cleanup(reference: weakref.ReferenceType[ReplayThresholdSet]) -> None:
        if _THRESHOLD_SETS.get(identity) is reference:
            _THRESHOLD_SETS.pop(identity, None)

    _THRESHOLD_SETS[identity] = weakref.ref(value, cleanup)


def _case_binding_state(value: ReplayCaseCounterBinding) -> _CaseBindingState:
    if type(value) is not ReplayCaseCounterBinding:
        raise ReplayContractError("case callback binding must have its exact type")
    entry = _CASE_BINDINGS.get(id(value))
    if entry is None or entry[0]() is not value:
        raise ReplayContractError("case callback binding is not factory issued")
    return entry[1]


def _is_registered_threshold_set(value: object) -> bool:
    if type(value) is not ReplayThresholdSet:
        return False
    reference = _THRESHOLD_SETS.get(id(value))
    return reference is not None and reference() is value


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
    "ModelFailure",
    "ReplayCaseCounterBinding",
    "ReplayContractError",
    "ReplayEvaluationContext",
    "ReplayFeatureAssurance",
    "ReplayLatencySamples",
    "ReplayThresholdSet",
    "bind_replay_case_counter",
    "replay_defense_arms",
]
