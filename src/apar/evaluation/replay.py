"""Verified identical-row replay for rules, GBDT, and layered hybrid arms."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
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
)
from apar.evaluation.contracts import EvaluationTruthRow
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
from apar.features.builders import FeatureMatrix
from apar.features.parity import audit_feature_matrix
from apar.features.state import FeatureVector
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_MAX_REPLAY_ROWS = 100_000
_MAX_REPLAY_BYTES = 32_000_000
_MODEL_FAILURES = {DefenseReason.MODEL_UNAVAILABLE, DefenseReason.MODEL_TIMEOUT}
_CASE_BINDING_TOKEN = object()


class ReplayContractError(ValueError):
    """Replay rows, artifacts, or evaluator lineage are inconsistent."""


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
    hidden_release_digest: str | None = None

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

    @field_validator("hidden_release_digest")
    @classmethod
    def hidden_release_is_digest_or_none(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_digest(value)
        return value

    @model_validator(mode="after")
    def context_is_closed(self) -> ReplayEvaluationContext:
        if tuple(item.arm for item in self.latency_samples) != tuple(DefenseArm):
            raise ValueError("latency evidence must contain all arms in canonical order")
        if self.evaluation.kind is EvaluationKind.HIDDEN:
            if self.hidden_release_digest is None:
                raise ValueError("hidden evaluation requires frozen release evidence")
        elif self.hidden_release_digest is not None:
            raise ValueError("non-hidden evaluation cannot claim hidden release evidence")
        return self


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
        reports: Mapping[DefenseArm, ThresholdReport],
    ) -> ReplayThresholdSet:
        """Freeze exact reports after signed bundle and callback lineage checks."""
        if type(defender) is not LoadedDefenderBundle:
            raise ReplayContractError("threshold set requires an exact loaded defender")
        if type(case_counter) is not ReplayCaseCounterBinding:
            raise ReplayContractError("threshold set requires exact case callback lineage")
        if type(reports) is not dict or set(reports) != set(DefenseArm):
            raise ReplayContractError("threshold reports must be an exact complete dict")
        checked: list[ArmThresholdEvidence] = []
        for arm in DefenseArm:
            report = reports[arm]
            if type(report) is not ThresholdReport:
                raise ReplayContractError("threshold report must have its exact type")
            try:
                validated = ThresholdReport.from_json(report.to_json())
            except Exception as error:
                raise ReplayContractError(
                    "threshold report failed semantic revalidation"
                ) from error
            checked.append(ArmThresholdEvidence(arm=arm, report=validated))
        signed_report = defender.threshold_report
        signed_binding = defender.threshold_binding
        if checked[-1].report.report_digest != signed_report.report_digest:
            raise ReplayContractError(
                "layered threshold report is not the signed defender operating point"
            )
        hybrid_report = checked[-1].report
        if (
            hybrid_report.input_labels_digest != signed_binding.labels_digest
            or hybrid_report.input_mandatory_actions_digest
            != signed_binding.mandatory_actions_digest
            or hybrid_report.input_values_digest != signed_binding.values_digest
            or hybrid_report.report_digest != signed_binding.threshold_report_digest
        ):
            raise ReplayContractError(
                "layered threshold selection lineage does not match its signed binding"
            )
        selection_lineage = {
            (
                item.report.row_count,
                item.report.legitimate_count,
                item.report.fraud_count,
                item.report.input_labels_digest,
                item.report.input_mandatory_actions_digest,
                item.report.input_values_digest,
            )
            for item in checked
        }
        if len(selection_lineage) != 1:
            raise ReplayContractError(
                "all replay arms must share exact threshold selection lineage"
            )
        fields: dict[str, object] = {
            "bundle_manifest_digest": _manifest_digest(defender.manifest),
            "case_callback_digest": case_counter.callback_digest,
            "selection_row_ids_digest": signed_binding.row_ids_digest,
            "reports": tuple(checked),
        }
        digest_fields = {
            "schema_version": "1.0.0",
            "bundle_manifest_digest": fields["bundle_manifest_digest"],
            "case_callback_digest": fields["case_callback_digest"],
            "selection_row_ids_digest": fields["selection_row_ids_digest"],
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
    def from_json(cls, payload: bytes) -> ReplayThresholdSet:
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
            return value
        except (ValidationError, WireContractError) as error:
            raise ReplayContractError(str(error)) from error


class ReplayCaseCounterBinding:
    """Immutable lineage wrapper around the production Task10 callback."""

    __slots__ = ("_as_of", "_callback_digest", "_counter", "_event_ids", "_rows_digest")

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
        if hasattr(self, "_counter"):
            raise ReplayContractError("case callback binding is already initialized")
        if type(counter) is not ReviewCaseCounter:
            raise ReplayContractError("case callback must be the exact production adapter")
        _validate_digest(rows_digest)
        _validate_digest(callback_digest)
        object.__setattr__(self, "_counter", counter)
        object.__setattr__(self, "_event_ids", event_ids)
        object.__setattr__(self, "_rows_digest", rows_digest)
        object.__setattr__(self, "_as_of", as_of)
        object.__setattr__(self, "_callback_digest", callback_digest)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("replay case callback binding is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("replay case callback binding is immutable")

    @property
    def callback_digest(self) -> str:
        return cast(str, object.__getattribute__(self, "_callback_digest"))

    def __call__(self, actions: np.ndarray) -> int:
        return cast(ReviewCaseCounter, object.__getattribute__(self, "_counter"))(actions)

    def validate_context(
        self,
        observations: tuple[ObservedEvent, ...],
        event_ids: tuple[str, ...],
        as_of: datetime,
    ) -> None:
        document = _case_binding_document(observations, event_ids, as_of)
        if (
            event_ids != object.__getattribute__(self, "_event_ids")
            or as_of != object.__getattribute__(self, "_as_of")
            or _digest_document(document) != object.__getattribute__(self, "_rows_digest")
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
    thresholds: ReplayThresholdSet,
    case_counter: ReplayCaseCounterBinding,
    evaluation: ReplayEvaluationContext,
    model_failure: ModelFailure | None = None,
) -> tuple[ReplayResult, ...]:
    """Score identical rows, freeze decisions, then resolve evaluator-side metrics."""
    try:
        matrix_value = _exact_model(matrix, FeatureMatrix, "feature matrix")
        context = _exact_model(evaluation, ReplayEvaluationContext, "evaluation context")
        threshold_set = _exact_model(thresholds, ReplayThresholdSet, "threshold set")
        if type(defender) is not LoadedDefenderBundle:
            raise ReplayContractError("defender must be an exact verified loaded bundle")
        if type(case_counter) is not ReplayCaseCounterBinding:
            raise ReplayContractError("case callback must have exact replay lineage")
        declared_failure = (
            None
            if model_failure is None
            else _exact_model(model_failure, ModelFailure, "model failure")
        )
        rows, events = _validated_replay_rows(matrix_value, defender)
        event_ids = tuple(row.event_id for row in rows)
        case_counter.validate_context(context.observations, event_ids, context.as_of)
        manifest_digest = _manifest_digest(defender.manifest)
        if threshold_set.bundle_manifest_digest != manifest_digest:
            raise ReplayContractError("threshold bundle lineage does not match defender")
        if threshold_set.case_callback_digest != case_counter.callback_digest:
            raise ReplayContractError("threshold case callback lineage does not match replay")
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
        if audit.passed != context.feature_assurance.leakage_passed:
            raise ReplayContractError("feature leakage evidence disagrees with replay audit")
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
        _validate_evaluator_context(context, event_ids)
        results = tuple(
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
                failure=actual_failure if arm is DefenseArm.GBDT_ONLY else None,
            )
            for arm in DefenseArm
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
    fraud_family_by_campaign: dict[str, str] = {}
    for row in context.truth:
        if not row.is_fraud:
            continue
        owner = fraud_family_by_campaign.setdefault(row.campaign_id, row.family)
        if owner != row.family:
            raise ReplayContractError(
                "competition campaigns must have exactly one fraud family owner"
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
    failure: ModelFailure | None,
) -> ReplayResult:
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
        rollback_available=True,
        hidden_access_clean=(
            context.evaluation.kind is not EvaluationKind.HIDDEN
            or context.hidden_release_digest is not None
        ),
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
    return ReplayResult.create(
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
        metric_report_digest=report.report_digest,
        metrics=metrics,
        assurance=assurance,
        failure=replay_failure,
        fallback_count=sum(item.fallback_used for item in decisions),
        mandatory_decline_count=sum(mandatory),
    )


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


def _exact_model[T: ExternalContract](value: object, expected: type[T], label: str) -> T:
    if type(value) is not expected:
        raise ReplayContractError(f"{label} must have its exact contract type")
    try:
        return expected.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True
        )
    except ValidationError as error:
        raise ReplayContractError(f"{label} failed semantic revalidation") from error


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("digest must be lowercase SHA-256")


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_json_tree(document))).hexdigest()


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
