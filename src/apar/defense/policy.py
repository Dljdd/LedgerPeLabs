"""Immutable action policy with mandatory gates and rules-only fallback."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_semantic_version
from apar.contracts.decisions import Action
from apar.defense.contracts import ObservedEvent, PolicyThresholds
from apar.defense.rules import (
    DefenseReason,
    RuleEngine,
    RuleManifest,
    RuleResult,
    feature_vector_digest,
    mandatory_reasons,
    rule_manifest_digest,
)
from apar.features.state import FeatureVector


class OperatingBudget(ExternalContract):
    """Synthetic competition operating caps, not production recommendations."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    challenge_rate_max: float = Field(default=0.02, ge=0.0, le=1.0)
    false_decline_rate_max: float = Field(default=0.001, ge=0.0, le=1.0)
    review_case_rate_max: float = Field(default=0.01, ge=0.0, le=1.0)

    @field_validator("challenge_rate_max", "false_decline_rate_max", "review_case_rate_max")
    @classmethod
    def rates_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("operating-budget rates must be finite")
        return value


class DefenseDecision(ExternalContract):
    """Deterministic, JSON-safe policy output with explicit fallback audit."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str
    action: Action
    score: float = Field(ge=0.0, le=1.0)
    rule_score: float = Field(ge=0.0, le=1.0)
    calibrated_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_codes: tuple[DefenseReason, ...]
    evidence_source_ids: tuple[str, ...]
    fallback_used: bool
    fallback_reason: DefenseReason | None
    failed_component_version: str | None
    latency_ms: float = Field(ge=0.0)
    policy_version: str

    @field_validator("score", "rule_score", "calibrated_score", "latency_ms")
    @classmethod
    def numeric_fields_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("decision numeric fields must be finite")
        return value

    @field_validator("evidence_source_ids")
    @classmethod
    def evidence_is_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("evidence source IDs must be unique and sorted")
        return value

    @field_validator("reason_codes")
    @classmethod
    def reasons_are_unique(cls, value: tuple[DefenseReason, ...]) -> tuple[DefenseReason, ...]:
        if len(value) != len(set(value)):
            raise ValueError("decision reason codes must be unique")
        return value

    @field_validator("policy_version")
    @classmethod
    def policy_version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="policy_version")

    @model_validator(mode="after")
    def fallback_audit_is_consistent(self) -> DefenseDecision:
        fallback_reasons = {DefenseReason.MODEL_UNAVAILABLE, DefenseReason.MODEL_TIMEOUT}
        if self.fallback_used:
            if self.fallback_reason not in fallback_reasons:
                raise ValueError("fallback decision must identify the model failure")
            if self.fallback_reason not in self.reason_codes:
                raise ValueError("fallback reason must appear in decision reason codes")
            if self.calibrated_score is not None:
                raise ValueError("fallback decision cannot claim a calibrated score")
            if (
                self.failed_component_version is None
                or not self.failed_component_version.strip()
            ):
                raise ValueError("fallback decision must identify a nonblank failed component")
        elif self.fallback_reason is not None or self.failed_component_version is not None:
            raise ValueError("non-fallback decision cannot identify a failed component")
        return self


class ActionPolicy(ExternalContract):
    """Choose actions from mandatory gates and continuous hybrid risk."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    version: str = "1.0.0"
    operating_budget: OperatingBudget = Field(default_factory=OperatingBudget)
    rules_challenge_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    rules_decline_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    rule_manifest: RuleManifest = Field(default_factory=RuleManifest.default)

    @field_validator("version")
    @classmethod
    def version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="version")

    @field_validator("rules_challenge_threshold", "rules_decline_threshold")
    @classmethod
    def thresholds_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rules-only thresholds must be finite")
        return value

    @model_validator(mode="after")
    def rules_thresholds_are_ordered(self) -> ActionPolicy:
        if self.rules_challenge_threshold > self.rules_decline_threshold:
            raise ValueError("rules challenge threshold must not exceed decline threshold")
        return self

    @classmethod
    def default(cls) -> ActionPolicy:
        """Return the fixed competition action policy."""
        return cls()

    def choose(
        self,
        event: ObservedEvent,
        rule_result: RuleResult,
        calibrated_score: float | None,
        thresholds: PolicyThresholds | None,
        *,
        model_failure: DefenseReason | None = None,
        failed_component_version: str | None = None,
        latency_ms: float = 0.0,
        vector: FeatureVector | None = None,
        score_mode: Literal["rules_only", "model_only", "layered"] = "layered",
        fallback_thresholds: PolicyThresholds | None = None,
    ) -> DefenseDecision:
        """Apply mandatory decline then one closed production score mode."""
        if type(event) is not ObservedEvent:
            raise TypeError("event must be an exact ObservedEvent")
        if type(rule_result) is not RuleResult:
            raise TypeError("rule_result must be an exact RuleResult")
        rule_result = RuleResult.model_validate(rule_result)
        current_mandatory = mandatory_reasons(event)
        _validate_rule_result(
            event,
            rule_result,
            vector,
            manifest=self.rule_manifest,
            current_mandatory=current_mandatory,
        )
        _validate_latency(latency_ms)
        if calibrated_score is not None:
            _validate_score(calibrated_score)
        if thresholds is not None and type(thresholds) is not PolicyThresholds:
            raise TypeError("thresholds must be exact PolicyThresholds or None")
        if score_mode not in {"rules_only", "model_only", "layered"}:
            raise ValueError("score_mode must be one closed production mode")
        if fallback_thresholds is not None and type(fallback_thresholds) is not PolicyThresholds:
            raise TypeError("fallback_thresholds must be exact PolicyThresholds or None")
        if model_failure is not None and type(model_failure) is not DefenseReason:
            raise TypeError("model_failure must be a DefenseReason or None")
        if model_failure not in {
            None,
            DefenseReason.MODEL_UNAVAILABLE,
            DefenseReason.MODEL_TIMEOUT,
        }:
            raise ValueError("model failure must be unavailable or timeout")
        if failed_component_version is not None and not failed_component_version.strip():
            raise ValueError("failed component identity must be nonblank")
        if model_failure is not None and failed_component_version is None:
            raise ValueError("explicit model failure requires a failed component identity")
        model_available = calibrated_score is not None and thresholds is not None
        if model_available and model_failure is not None:
            raise ValueError("model failure cannot accompany an available calibrated score")
        if model_available and failed_component_version is not None:
            raise ValueError("available model cannot identify a failed component")

        if current_mandatory:
            return self._decision(
                event,
                rule_result,
                action=Action.DECLINE,
                score=1.0,
                calibrated_score=None,
                reasons=current_mandatory,
                fallback_reason=None,
                failed_component_version=None,
                latency_ms=latency_ms,
            )

        if score_mode == "rules_only":
            if thresholds is None:
                raise ValueError("rules-only score mode requires frozen thresholds")
            return self._scored_decision(
                event,
                rule_result,
                raw_score=rule_result.score,
                calibrated_score=None,
                thresholds=thresholds,
                reasons=tuple(hit.reason for hit in rule_result.hits),
                fallback_reason=None,
                failed_component_version=None,
                latency_ms=latency_ms,
            )

        if score_mode == "model_only" and model_available:
            assert calibrated_score is not None
            assert thresholds is not None
            return self._scored_decision(
                event,
                rule_result,
                raw_score=calibrated_score,
                calibrated_score=calibrated_score,
                thresholds=thresholds,
                reasons=(),
                fallback_reason=None,
                failed_component_version=None,
                latency_ms=latency_ms,
            )

        if score_mode == "model_only":
            failure = model_failure or DefenseReason.MODEL_UNAVAILABLE
            return self._decision(
                event,
                rule_result,
                action=Action.APPROVE,
                score=0.0,
                calibrated_score=None,
                reasons=(failure,),
                fallback_reason=None,
                failed_component_version=None,
                latency_ms=latency_ms,
            )

        if model_available:
            assert calibrated_score is not None
            assert thresholds is not None
            return self._scored_decision(
                event,
                rule_result,
                raw_score=max(rule_result.score, calibrated_score),
                calibrated_score=calibrated_score,
                thresholds=thresholds,
                reasons=tuple(hit.reason for hit in rule_result.hits),
                fallback_reason=None,
                failed_component_version=None,
                latency_ms=latency_ms,
            )

        failure = model_failure or DefenseReason.MODEL_UNAVAILABLE
        component_identity = failed_component_version or "unknown"
        selected_fallback = fallback_thresholds or PolicyThresholds(
            challenge=self.rules_challenge_threshold,
            decline=self.rules_decline_threshold,
        )
        fallback_reasons = _unique_reasons(
            *(hit.reason for hit in rule_result.hits), failure
        )
        return self._scored_decision(
            event,
            rule_result,
            raw_score=rule_result.score,
            calibrated_score=None,
            thresholds=selected_fallback,
            reasons=fallback_reasons,
            fallback_reason=failure,
            failed_component_version=component_identity,
            latency_ms=latency_ms,
        )

    def _scored_decision(
        self,
        event: ObservedEvent,
        rule_result: RuleResult,
        *,
        raw_score: float,
        calibrated_score: float | None,
        thresholds: PolicyThresholds,
        reasons: tuple[DefenseReason, ...],
        fallback_reason: DefenseReason | None,
        failed_component_version: str | None,
        latency_ms: float,
    ) -> DefenseDecision:
        from apar.defense.thresholds import normalize_operating_scores

        score = float(
            normalize_operating_scores(np.asarray([raw_score], dtype=np.float64))[0]
        )
        if score >= thresholds.decline:
            action = Action.DECLINE
        elif score >= thresholds.challenge:
            action = Action.CHALLENGE
        else:
            action = Action.APPROVE
        return self._decision(
            event,
            rule_result,
            action=action,
            score=score,
            calibrated_score=calibrated_score,
            reasons=reasons,
            fallback_reason=fallback_reason,
            failed_component_version=failed_component_version,
            latency_ms=latency_ms,
        )

    def _decision(
        self,
        event: ObservedEvent,
        rule_result: RuleResult,
        *,
        action: Action,
        score: float,
        calibrated_score: float | None,
        reasons: tuple[DefenseReason, ...],
        fallback_reason: DefenseReason | None,
        failed_component_version: str | None,
        latency_ms: float,
    ) -> DefenseDecision:
        evidence_sources = {event.event_id}
        evidence_sources.update(
            source_id
            for hit in rule_result.hits
            for source_id in hit.evidence_source_ids
        )
        evidence = tuple(sorted(evidence_sources))
        return DefenseDecision(
            event_id=event.event_id,
            action=action,
            score=score,
            rule_score=rule_result.score,
            calibrated_score=calibrated_score,
            reason_codes=reasons,
            evidence_source_ids=evidence,
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
            failed_component_version=failed_component_version,
            latency_ms=latency_ms,
            policy_version=self.version,
        )


def _unique_reasons(*reasons: DefenseReason) -> tuple[DefenseReason, ...]:
    return tuple(dict.fromkeys(reasons))


def _validate_score(value: float) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("calibrated_score must be a finite float in [0, 1]")


def _validate_latency(value: float) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")


def _validate_rule_result(
    event: ObservedEvent,
    result: RuleResult,
    vector: FeatureVector | None,
    *,
    manifest: RuleManifest,
    current_mandatory: tuple[DefenseReason, ...],
) -> None:
    if result.event_id is None:
        if vector is not None:
            raise ValueError("neutral unbound rule sentinel cannot claim vector provenance")
        if not current_mandatory:
            raise ValueError(
                "neutral unbound rule sentinel is only valid for mandatory reconstruction"
            )
        return
    if result.manifest_digest != rule_manifest_digest(manifest):
        raise ValueError("rule result manifest provenance does not match the policy")
    if vector is None:
        raise ValueError("bound rule result requires its feature vector provenance")
    if type(vector) is not FeatureVector:
        raise TypeError("vector must be an exact FeatureVector")
    if result.event_id != event.event_id or vector.event_id != event.event_id:
        raise ValueError("rule result event binding does not match the current event")
    if result.decision_at != event.decision_at or vector.decision_at != event.decision_at:
        raise ValueError("rule result decision-time binding does not match the current event")
    if result.catalog_digest != vector.catalog_digest:
        raise ValueError("rule result catalog provenance does not match the feature vector")
    if result.vector_digest != feature_vector_digest(vector):
        raise ValueError("rule result vector provenance does not match the feature vector")
    expected_evidence = tuple(sorted({event.event_id, *vector.source_event_ids}))
    if any(hit.evidence_source_ids != expected_evidence for hit in result.hits):
        raise ValueError("rule hit evidence does not match the feature vector provenance")
    supplied_mandatory = tuple(hit.reason for hit in result.hits if hit.mandatory)
    unjustified = set(supplied_mandatory) - set(current_mandatory)
    if unjustified:
        raise ValueError("rule result contains an unjustified mandatory hit")
    recomputed = RuleEngine(manifest).evaluate(event, vector)
    if result != recomputed:
        raise ValueError("rule result failed deterministic semantic re-evaluation")
