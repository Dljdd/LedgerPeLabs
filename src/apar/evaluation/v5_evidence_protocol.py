"""Frozen metric, control, economics, and verifier semantics for Sentinel v5."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class V5MetricApplicability(StrEnum):
    DEFINED = "defined"
    UNDEFINED = "undefined"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class V5LabelShuffleSpec(_FrozenModel):
    permutation_seed: int = Field(gt=0)
    max_roc_auc: float
    max_pr_auc_excess_over_prevalence: float
    min_roc_auc_delta: float

    @model_validator(mode="after")
    def criteria_are_frozen(self) -> Self:
        expected = (1707, 0.70, 0.20, 0.05)
        observed = (
            self.permutation_seed,
            self.max_roc_auc,
            self.max_pr_auc_excess_over_prevalence,
            self.min_roc_auc_delta,
        )
        if observed != expected:
            raise ValueError("label-shuffle criterion differs from frozen values")
        return self


class V5IdentityRenameSpec(_FrozenModel):
    namespace: Literal["apar.sentinel-v5.control.identity-rename.v1"]
    require_exact_numeric_invariance: Literal[True]


class V5FutureCausalitySpec(_FrozenModel):
    offset_seconds: Literal[86400]
    require_exact_invariance: Literal[True]


class V5EqualTimeSpec(_FrozenModel):
    peer_count: Literal[2]
    require_exact_invariance: Literal[True]


class V5BenignOnlySpec(_FrozenModel):
    false_decline_rate_max: float
    challenge_rate_max: float
    review_rate_max: float
    recall_semantics: Literal[V5MetricApplicability.UNDEFINED]

    @model_validator(mode="after")
    def gates_are_exact(self) -> Self:
        if (
            self.false_decline_rate_max,
            self.challenge_rate_max,
            self.review_rate_max,
        ) != (0.001, 0.02, 0.01):
            raise ValueError("benign-only gates differ from frozen values")
        return self


class V5FraudOnlySpec(_FrozenModel):
    qualifies_for_readiness: Literal[False]
    workload_semantics: Literal[V5MetricApplicability.UNDEFINED]


class V5FeatureLeakageSpec(_FrozenModel):
    forbidden_fields: tuple[str, ...]
    require_exact_numeric_invariance: Literal[True]

    @field_validator("forbidden_fields")
    @classmethod
    def forbidden_fields_are_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "is_fraud",
            "family",
            "campaign_id",
            "split",
            "seed",
            "generator",
            "lifecycle_state",
            "final_outcome",
        )
        if value != expected:
            raise ValueError("forbidden feature-field contract differs from frozen values")
        return value


class V5ControlProtocol(_FrozenModel):
    label_shuffle: V5LabelShuffleSpec
    identity_rename: V5IdentityRenameSpec
    future_causality: V5FutureCausalitySpec
    equal_time: V5EqualTimeSpec
    benign_only: V5BenignOnlySpec
    fraud_only: V5FraudOnlySpec
    feature_leakage: V5FeatureLeakageSpec


class V5CalibrationProtocol(_FrozenModel):
    bin_boundaries: tuple[float, ...]
    interval_closure: Literal["left_closed_right_open_final_closed"]
    rules_only: Literal[V5MetricApplicability.NOT_APPLICABLE]

    @field_validator("bin_boundaries")
    @classmethod
    def bins_are_frozen(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        expected = tuple(index / 10 for index in range(11))
        if value != expected:
            raise ValueError("calibration bins differ from frozen boundaries")
        return value


class V5BootstrapProtocol(_FrozenModel):
    replicates: Literal[2000]
    seed: Literal[707]
    confidence_level: float
    interval_method: Literal["percentile"]
    fraud_unit: Literal["campaign"]
    legitimate_unit: Literal["campaign"]
    stratification: Literal["legitimate_and_each_fraud_family"]
    metrics: tuple[str, ...]

    @field_validator("confidence_level")
    @classmethod
    def confidence_is_exact(cls, value: float) -> float:
        if value != 0.95:
            raise ValueError("bootstrap confidence differs from frozen value")
        return value

    @field_validator("metrics")
    @classmethod
    def metrics_are_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "recall",
            "false_decline_rate",
            "challenge_rate",
            "review_rate",
            "captured_value_fraction",
            "campaign_detection_rate",
            "expected_calibration_error",
        )
        if value != expected:
            raise ValueError("bootstrap metric set differs from frozen values")
        return value


class V5EconomicProtocol(_FrozenModel):
    intervention_actions: tuple[str, ...]
    rail_movement_events: Mapping[str, tuple[str, ...]]
    authorization_events: tuple[str, ...]
    value_reversal_events: tuple[str, ...]
    family_rails: Mapping[str, str]
    aggregate_value_fraction: Literal["unweighted_mean_of_currency_fractions"]
    capture_semantics: Literal[
        "first_intervention_at_or_before_value_movement_or_lifecycle_reversal"
    ]
    escape_semantics: Literal[
        "malicious_value_moved_without_prior_intervention_and_without_lifecycle_reversal"
    ]

    @field_validator("rail_movement_events", "family_rails", mode="after")
    @classmethod
    def mappings_are_immutable(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_serializer("rail_movement_events", "family_rails")
    def serialize_mapping(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @model_validator(mode="after")
    def semantics_are_exact(self) -> Self:
        if self.intervention_actions != ("challenge", "review_hold", "decline_hold"):
            raise ValueError("economic intervention actions differ from frozen values")
        if dict(self.rail_movement_events) != {
            "a2a": ("transfer_posted",),
            "agentic": ("authorization",),
            "card": ("settlement",),
        }:
            raise ValueError("economic rail movement events differ from frozen values")
        if self.value_reversal_events != ("transfer_returned", "refund", "recovery"):
            raise ValueError("economic reversal events differ from frozen values")
        return self


class V5MetricDefinitions(_FrozenModel):
    detection_actions: tuple[str, ...]
    recall: str
    precision: str
    f1: str
    false_decline_rate: str
    challenge_rate: str
    review_rate: str
    decline_rate: str
    brier: str
    expected_calibration_error: str
    maximum_calibration_error: str
    family_precision: str
    campaign_detection_rate: str
    time_to_first_alert: str

    @model_validator(mode="after")
    def formulas_are_frozen(self) -> Self:
        expected = {
            "detection_actions": ("challenge", "review_hold", "decline_hold"),
            "recall": "detected_fraud_rows/fraud_rows",
            "precision": "detected_fraud_rows/all_detected_rows",
            "f1": "2*true_positive/(2*true_positive+false_positive+false_negative)",
            "false_decline_rate": "declined_legitimate_rows/legitimate_rows",
            "challenge_rate": "challenged_legitimate_rows/legitimate_rows",
            "review_rate": "reviewed_legitimate_rows/legitimate_rows",
            "decline_rate": "declined_rows/all_rows",
            "brier": "sum((probability-label)^2)/all_rows",
            "expected_calibration_error": (
                "sum(bin_count/all_rows*abs("
                "bin_mean_probability-bin_empirical_rate))"
            ),
            "maximum_calibration_error": "max(abs(bin_mean_probability-bin_empirical_rate))",
            "family_precision": (
                "detected_family_rows/"
                "(detected_family_rows+detected_legitimate_rows)"
            ),
            "campaign_detection_rate": "detected_fraud_campaigns/fraud_campaigns",
            "time_to_first_alert": "first_intervention_decision_at-first_campaign_decision_at",
        }
        if self.model_dump(mode="python") != expected:
            raise ValueError("metric definitions differ from frozen formulas")
        return self


class V5GateProtocol(_FrozenModel):
    metric: str
    comparison: Literal["lower_bound_gte", "upper_bound_lte", "point_lte", "defined_interval"]
    target: float | None


class V5EvidenceBounds(_FrozenModel):
    max_rows: Literal[100000]
    max_execution_artifacts: Literal[4096]
    max_single_execution_bytes: Literal[16777216]
    max_aggregate_execution_bytes: Literal[536870912]
    max_control_rows: Literal[100000]
    max_bootstrap_replicates: Literal[10000]
    max_serialized_evidence_bytes: Literal[536870912]


class V5RunModeSpec(_FrozenModel):
    profile: Literal["smoke", "production"]
    development_test_seed: Literal[404, 2404]
    repeatable: bool
    authorization_required: bool


class V5RunModeProtocol(_FrozenModel):
    safe_validation: V5RunModeSpec
    locked_development: V5RunModeSpec

    @model_validator(mode="after")
    def modes_are_exact(self) -> Self:
        observed = {
            "safe_validation": self.safe_validation.model_dump(mode="json"),
            "locked_development": self.locked_development.model_dump(mode="json"),
        }
        expected = {
            "safe_validation": {
                "profile": "smoke",
                "development_test_seed": 404,
                "repeatable": True,
                "authorization_required": False,
            },
            "locked_development": {
                "profile": "production",
                "development_test_seed": 2404,
                "repeatable": False,
                "authorization_required": True,
            },
        }
        if observed != expected:
            raise ValueError("closed run-mode contract differs")
        return self


class V5LockedArtifactStorage(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-chunked-evidence/1"]
    candidate_manifest_path: Literal[
        "docs/experiments/defense-v5-locked-development-candidate.manifest.json"
    ]
    judge_summary_path: Literal[
        "docs/experiments/defense-v5-locked-development-summary.json"
    ]
    chunk_size_bytes: Literal[67108864]
    expected_envelope_upper_bound_bytes: Literal[805306368]
    maximum_envelope_bytes: Literal[1073741824]
    maximum_chunk_count: Literal[16]
    normal_git_blob_limit_bytes: Literal[104857600]
    publication: Literal["content_chunks_then_atomic_exclusive_manifest"]

    @model_validator(mode="after")
    def storage_is_bounded(self) -> Self:
        if self.chunk_size_bytes >= self.normal_git_blob_limit_bytes:
            raise ValueError("evidence chunk reaches the normal Git blob limit")
        if self.expected_envelope_upper_bound_bytes > self.maximum_envelope_bytes:
            raise ValueError("expected envelope estimate exceeds the hard bound")
        if self.chunk_size_bytes * self.maximum_chunk_count < self.maximum_envelope_bytes:
            raise ValueError("chunk count cannot contain the maximum envelope")
        return self


class V5EvidenceProtocol(_FrozenModel):
    schema_version: Literal["1.1.0"]
    protocol_id: Literal["apar-sentinel-v5-development-evidence"]
    base_protocol_path: Literal["config/defense/defense-v5-development.json"]
    arm_protocol_path: Literal["config/defense/defense-v5-arms.json"]
    safe_development_test_seed: Literal[404]
    locked_development_test_seed: Literal[2404]
    existing_development_result_path: Literal["docs/experiments/defense-v5-development-result.json"]
    existing_development_result_sha256: Literal[
        "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185"
    ]
    controls: V5ControlProtocol
    calibration: V5CalibrationProtocol
    bootstrap: V5BootstrapProtocol
    economics: V5EconomicProtocol
    metric_definitions: V5MetricDefinitions
    gates: tuple[V5GateProtocol, ...]
    bounds: V5EvidenceBounds
    run_modes: V5RunModeProtocol
    locked_artifact_storage: V5LockedArtifactStorage
    implementation_paths: tuple[str, ...]
    evidence_protocol_sha256: str = ""
    base_protocol_sha256: str = ""
    arm_protocol_sha256: str = ""
    implementation_sha256: str = ""

    @model_validator(mode="after")
    def seed_and_gate_contract_is_exact(self) -> Self:
        if int(self.safe_development_test_seed) == int(self.locked_development_test_seed):
            raise ValueError("safe development-test seed aliases locked seed")
        expected_gates = (
            ("family_recall", "lower_bound_gte", 0.75),
            ("false_decline_rate", "upper_bound_lte", 0.001),
            ("manual_review_rate", "upper_bound_lte", 0.01),
            ("challenge_rate", "upper_bound_lte", 0.02),
            ("captured_value_fraction", "lower_bound_gte", 0.70),
            ("expected_calibration_error", "upper_bound_lte", 0.10),
            ("p95_decision_latency_ms", "point_lte", 50.0),
            ("campaign_detection_rate", "defined_interval", None),
        )
        observed = tuple((gate.metric, gate.comparison, gate.target) for gate in self.gates)
        if observed != expected_gates:
            raise ValueError("readiness gate semantics differ from frozen values")
        if self.implementation_paths != tuple(sorted(set(self.implementation_paths))):
            raise ValueError("evidence implementation paths must be unique and canonical")
        return self


def _canonical_digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v5_evidence_protocol(path: Path, *, root: Path) -> V5EvidenceProtocol:
    """Load frozen Round 3B semantics and bind unchanged configs and source bytes."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("evidence protocol must be a JSON object")
    parsed = V5EvidenceProtocol.model_validate(raw)
    if int(parsed.safe_development_test_seed) == int(parsed.locked_development_test_seed):
        raise ValueError("safe seed cannot equal locked development-test seed")
    implementation = []
    for relative in parsed.implementation_paths:
        source = root / relative
        if not source.is_file():
            raise ValueError(f"evidence implementation path is missing: {relative}")
        implementation.append((relative, _file_digest(source)))
    values = parsed.model_dump(
        mode="json",
        exclude={
            "evidence_protocol_sha256",
            "base_protocol_sha256",
            "arm_protocol_sha256",
            "implementation_sha256",
        },
    )
    return parsed.model_copy(
        update={
            "evidence_protocol_sha256": _canonical_digest(values),
            "base_protocol_sha256": _file_digest(root / parsed.base_protocol_path),
            "arm_protocol_sha256": _file_digest(root / parsed.arm_protocol_path),
            "implementation_sha256": _canonical_digest(implementation),
        }
    )


__all__ = [
    "V5EvidenceProtocol",
    "V5MetricApplicability",
    "load_v5_evidence_protocol",
]
