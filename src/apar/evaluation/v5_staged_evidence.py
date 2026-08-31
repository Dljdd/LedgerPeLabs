"""Typed execution boundaries for the Sentinel v5 staged evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from apar.evaluation.v5_arms import score_v5_arm_set, train_v5_arm_set
from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    V5CheckpointManifest,
    iter_v5_checkpoint_observational_records,
    iter_v5_checkpoint_records,
    read_v5_checkpoint_manifest,
)
from apar.evaluation.v5_controls import (
    V5ControlGroup,
    V5ExecutedControlGroup,
    V5ExecutedControlSuite,
    assemble_v5_control_suite,
    execute_v5_control_group,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    V5TrainingPartitionEvidence,
    bind_v5_evaluation_result,
    build_v5_arm_support_rows,
    build_v5_execution_artifacts,
    build_v5_training_partition_evidence,
    derive_v5_trust_failures,
    evaluate_v5_arm,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_bundle import (
    V5ReadinessEvidence,
    _build_v5_readiness_evidence_from_source,
    build_v5_readiness_evidence,
)
from apar.evaluation.v5_evidence_layers import (
    _stable_complete_metrics,
    _stable_controls,
    _stable_readiness,
)
from apar.evaluation.v5_evidence_protocol import (
    V5EvidenceProtocol,
    load_v5_evidence_protocol,
)
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleMode,
    V5KaggleProtocol,
    V5KaggleResourceGates,
    V5KaggleStage,
    V5KaggleSupportPlan,
    build_v5_kaggle_support_plan,
    load_v5_kaggle_protocol,
    resolve_next_v5_kaggle_stage,
)
from apar.evaluation.v5_locked_evidence import (
    V5CheckpointChainBinding,
    V5StagedEvidencePayload,
    build_v5_staged_evidence_payload,
)
from apar.evaluation.v5_metrics import (
    V5BootstrapInterval,
    V5CalibrationEvidence,
    V5CompleteArmMetrics,
    V5EconomicEvidence,
    V5FamilyMetrics,
    V5MetricEstimate,
    evaluate_v5_complete_result,
)
from apar.evaluation.v5_population import (
    V5Corpus,
    V5DecisionRow,
    V5ExecutionManifest,
    V5PartitionCorpus,
    build_v5_corpus,
)
from apar.evaluation.v5_protocol import (
    V5DevelopmentProtocol,
    V5Profile,
    load_v5_development_protocol,
    v5_protocol_digest,
)
from apar.evaluation.v5_run_mode import V5RunMode
from apar.features.sentinel import (
    SentinelFeatureBatch,
    SentinelFeatureCatalog,
    build_sentinel_features,
)

_PROTOCOL_PATH = Path("config/defense/defense-v5-kaggle-recovery.json")
_STAGE_CAPABILITY_SEAL = object()
_STAGED_ARMS = (
    V5Arm.RULES_ONLY,
    V5Arm.ENSEMBLE_NO_GRAPH,
    V5Arm.ENSEMBLE_WITH_GRAPH,
    V5Arm.FULL_SENTINEL,
)
_CONTROL_GROUP_BY_STAGE = {
    V5KaggleStage.LABEL_SHUFFLE: V5ControlGroup.LABEL_SHUFFLE,
    V5KaggleStage.IDENTITY_RENAME: V5ControlGroup.IDENTITY_RENAME,
    V5KaggleStage.FUTURE_CAUSALITY: V5ControlGroup.FUTURE_CAUSALITY,
    V5KaggleStage.EQUAL_TIME_ISOLATION: V5ControlGroup.EQUAL_TIME_ISOLATION,
    V5KaggleStage.FEATURE_LEAKAGE: V5ControlGroup.FEATURE_LEAKAGE,
    V5KaggleStage.SINGLE_CLASS_CONTROLS: V5ControlGroup.SINGLE_CLASS,
}
_INVARIANCE_CONTROL_NAMES = {
    "identity_rename",
    "future_causality",
    "equal_time_isolation",
    "feature_leakage",
}
_WORKLOAD_CONTROL_NAMES = {"benign_only", "fraud_only_diagnostic"}
_MAX_ARM_SECTION_RECORD_BYTES = 16_777_216
_ARM_SECTION_ENVELOPE_BYTES = 512
_ARM_HEADER_SCHEMA = "apar-sentinel-v5-kaggle-arms/2"
_ARM_RESULT_META_SCHEMA = "apar-sentinel-v5-kaggle-arm-result-meta/1"
_ARM_LATENCY_META_SCHEMA = "apar-sentinel-v5-kaggle-arm-latency-meta/1"
_ARM_SECTION_SCHEMA = "apar-sentinel-v5-kaggle-arm-section/1"


@dataclass(frozen=True, slots=True)
class V5StageCapability:
    stage: V5KaggleStage
    mode: V5KaggleMode
    run_binding_sha256: str
    attempt_receipt_sha256: str
    predecessor_manifest_sha256: str | None
    seal: object
    execution_manifest_sha256: str = "0" * 64


@dataclass(frozen=True, slots=True)
class V5MetricWorkerArmResult:
    """One fully authenticated Stage 30 arm restored for a fresh metric worker."""

    arm: str
    arm_index: int
    deterministic_result_sha256: str
    support_event_ids_sha256: str
    result: V5EvaluationResult


@dataclass(frozen=True, slots=True)
class V5PreparedPartition:
    partition: str
    matrix: NDArray[np.float64]
    labels: NDArray[np.int64]
    event_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    amounts: NDArray[np.float64]
    trust_failures: tuple[bool, ...]
    feature_batch: SentinelFeatureBatch
    training_evidence: V5TrainingPartitionEvidence | None


@dataclass(frozen=True, slots=True)
class V5MetricStageEvidence:
    complete_metrics: tuple[V5CompleteArmMetrics, ...]
    controls: V5ExecutedControlSuite
    readiness: V5ReadinessEvidence


@dataclass(frozen=True, slots=True)
class V5CompactMetricStageEvidence:
    """Authenticated Stage 70 evidence without retained bootstrap draw arrays."""

    worker_receipts: tuple[V5MetricArmWorkerReceipt, ...]
    controls: V5ExecutedControlSuite
    readiness: V5ReadinessEvidence


class V5MetricBootstrapSummary(BaseModel):
    """Complete bootstrap design and intervals with the bulky draw stream addressed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-bootstrap-summary/1"]
    seed: int
    replicates: int = Field(gt=0, le=10_000)
    confidence_level: float
    interval_method: str
    resampling_unit: str
    stratification: str
    strata: tuple[tuple[str, tuple[str, ...]], ...]
    sample_count: int = Field(gt=0, le=10_000)
    sample_stream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intervals: tuple[V5BootstrapInterval, ...]
    bootstrap_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def summary_is_bound(self) -> Self:
        if self.sample_count != self.replicates:
            raise ValueError("bootstrap summary sample count differs")
        expected = _sha256(
            _canonical_bytes(self.model_dump(mode="json", exclude={"summary_sha256"}))
        )
        if self.summary_sha256 != expected:
            raise ValueError("bootstrap summary digest differs")
        return self


class V5CompleteArmMetricSummary(BaseModel):
    """Exact arm metrics with deterministic bootstrap draws retained by address."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-complete-metric-summary/1"]
    deterministic_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm: V5Arm
    arm_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregate: Mapping[str, V5MetricEstimate]
    calibration: V5CalibrationEvidence
    economics: V5EconomicEvidence
    by_family: tuple[V5FamilyMetrics, ...]
    bootstrap: V5MetricBootstrapSummary
    complete_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("aggregate", mode="after")
    @classmethod
    def aggregate_is_immutable(
        cls, value: Mapping[str, V5MetricEstimate]
    ) -> Mapping[str, V5MetricEstimate]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("aggregate")
    def serialize_aggregate(self, value: Mapping[str, V5MetricEstimate]) -> dict[str, object]:
        return {name: metric.model_dump(mode="json") for name, metric in value.items()}

    @model_validator(mode="after")
    def summary_is_bound(self) -> Self:
        expected = _sha256(
            _canonical_bytes(self.model_dump(mode="json", exclude={"summary_sha256"}))
        )
        if self.summary_sha256 != expected:
            raise ValueError("complete metric summary digest differs")
        return self

    def stable_document(self) -> dict[str, object]:
        """Return the existing frozen deterministic metric projection."""
        document = self.model_dump(
            mode="json",
            exclude={
                "schema_version",
                "deterministic_result_sha256",
                "summary_sha256",
            },
        )
        return _stable_complete_metrics(
            document,
            deterministic_result_sha256=self.deterministic_result_sha256,
        )


class V5MetricWorkerResourceTelemetry(BaseModel):
    """Resource measurements reported by one fresh Stage 70 interpreter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fresh_interpreter: Literal[True]
    wall_seconds: float = Field(gt=0.0, le=21_600.0)
    peak_rss_bytes: int = Field(gt=0)
    artifact_bytes: int = Field(gt=0)


class V5MetricArmWorkerReceipt(BaseModel):
    """Authenticated compact output from one source-bound Stage 70 candidate worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-source-bound-metric-arm-worker/1"]
    stage: Literal["70_metrics"]
    mode: V5KaggleMode
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_manifest_deterministic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm: V5Arm
    arm_index: int = Field(ge=0, lt=len(_STAGED_ARMS))
    arm_order: tuple[V5Arm, ...]
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_count: int = Field(gt=0)
    support_event_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_summary: V5CompleteArmMetricSummary
    resource_telemetry: V5MetricWorkerResourceTelemetry
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def receipt_is_bound(self) -> Self:
        expected = _sha256(
            _canonical_bytes(self.model_dump(mode="json", exclude={"receipt_sha256"}))
        )
        if self.receipt_sha256 != expected:
            raise ValueError("metric worker receipt digest differs")
        if self.arm_order != _STAGED_ARMS:
            raise ValueError("metric worker arm order differs")
        if self.arm is not self.arm_order[self.arm_index]:
            raise ValueError("metric worker arm index differs")
        if self.metric_summary.arm is not self.arm:
            raise ValueError("metric worker summary arm differs")
        if self.metric_summary.support_sha256 != self.support_sha256:
            raise ValueError("metric worker summary support differs")
        if (
            self.metric_summary.deterministic_result_sha256
            != self.deterministic_result_sha256
        ):
            raise ValueError("metric worker deterministic result differs")
        return self


class V5CompactFinalPayload(BaseModel):
    """Official Stage 80 index over the authenticated compact Stage 70 chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-kaggle-compact-final-payload/2"]
    mode: V5KaggleMode
    profile: Literal["production"]
    development_test_seed: int
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_plan: V5KaggleSupportPlan
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_chain: V5CheckpointChainBinding
    evidence_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_manifest_deterministic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_manifest_deterministic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_metric_stage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_worker_receipt_sha256: tuple[str, ...]
    worker_resources: tuple[V5MetricWorkerResourceTelemetry, ...]
    controls_suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness: V5ReadinessEvidence
    final_core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def compact_final_is_bound(self) -> Self:
        expected_seed = {
            V5KaggleMode.CAPACITY_VALIDATION: 404,
            V5KaggleMode.LOCKED_SUCCESSOR: 2404,
        }[self.mode]
        if (
            self.development_test_seed != expected_seed
            or self.support_plan.mode is not self.mode
            or self.checkpoint_chain.attempt_receipt_sha256
            != self.attempt_receipt_sha256
            or len(self.metric_worker_receipt_sha256) != len(_STAGED_ARMS)
            or len(set(self.metric_worker_receipt_sha256)) != len(_STAGED_ARMS)
            or len(self.worker_resources) != len(_STAGED_ARMS)
        ):
            raise ValueError("compact final chain shape differs")
        expected = _sha256(
            _canonical_bytes(self.model_dump(mode="json", exclude={"payload_sha256"}))
        )
        if self.payload_sha256 != expected:
            raise ValueError("compact final payload digest differs")
        return self


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not a lowercase SHA-256") from error
    if value != value.lower():
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _issue_stage_capability(
    *,
    protocol: V5KaggleProtocol,
    mode: V5KaggleMode,
    attempt_receipt_sha256: str,
    predecessor: V5CheckpointManifest | None,
    execution_manifest_sha256: str = "0" * 64,
) -> V5StageCapability:
    """Issue the only stage capability implied by a verified predecessor."""
    if type(protocol) is not V5KaggleProtocol:
        raise TypeError("stage authority requires an exact V5KaggleProtocol")
    if type(mode) is not V5KaggleMode:
        raise TypeError("stage authority requires an exact V5KaggleMode")
    _require_sha256(attempt_receipt_sha256, label="attempt receipt digest")
    _require_sha256(execution_manifest_sha256, label="execution manifest digest")
    expected_run_binding = protocol.run_binding_sha256(mode)
    if predecessor is not None and (
        type(predecessor) is not V5CheckpointManifest
        or predecessor.run_binding_sha256 != expected_run_binding
        or predecessor.attempt_receipt_sha256 != attempt_receipt_sha256
    ):
        raise ValueError("stage predecessor authority differs")
    expected_stage = resolve_next_v5_kaggle_stage(predecessor)
    return V5StageCapability(
        stage=expected_stage,
        mode=mode,
        run_binding_sha256=expected_run_binding,
        attempt_receipt_sha256=attempt_receipt_sha256,
        predecessor_manifest_sha256=(None if predecessor is None else predecessor.manifest_sha256),
        seal=_STAGE_CAPABILITY_SEAL,
        execution_manifest_sha256=execution_manifest_sha256,
    )


def _validate_capability(
    capability: V5StageCapability,
    *,
    required_stage: V5KaggleStage,
) -> None:
    if type(capability) is not V5StageCapability or capability.seal is not _STAGE_CAPABILITY_SEAL:
        raise PermissionError("invalid staged execution capability")
    if capability.stage is not required_stage:
        raise PermissionError("staged execution capability has the wrong stage")
    _require_sha256(capability.run_binding_sha256, label="run binding digest")
    _require_sha256(
        capability.attempt_receipt_sha256,
        label="attempt receipt digest",
    )
    _require_sha256(
        capability.execution_manifest_sha256,
        label="execution manifest digest",
    )


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _iter_bounded_arm_section_records(
    *,
    kind: str,
    arm: str,
    section: str,
    items: Sequence[dict[str, object]],
    max_record_bytes: int,
) -> Iterator[V5CheckpointInput]:
    """Encode one ordered arm section without exceeding a record byte bound."""
    ranges = _bounded_arm_section_ranges(
        items=items,
        max_record_bytes=max_record_bytes,
    )
    for index, (start, end) in enumerate(ranges):
        document = {
            "schema_version": _ARM_SECTION_SCHEMA,
            "arm": arm,
            "section": section,
            "index": index,
            "start": start,
            "items": items[start:end],
        }
        encoded = _canonical_bytes(document)
        if len(encoded) > max_record_bytes:
            raise ValueError("arm section record exceeds its byte bound")
        yield V5CheckpointInput(
            kind=kind,
            key=f"{arm}:{index:04d}",
            canonical_bytes=encoded,
        )


def _bounded_arm_section_ranges(
    *,
    items: Sequence[dict[str, object]],
    max_record_bytes: int,
) -> tuple[tuple[int, int], ...]:
    if max_record_bytes <= _ARM_SECTION_ENVELOPE_BYTES:
        raise ValueError("arm section record byte bound is too small")
    ranges: list[tuple[int, int]] = []
    start = 0
    encoded_items_bytes = 2
    for index, item in enumerate(items):
        item_bytes = len(_canonical_bytes(item))
        if item_bytes + _ARM_SECTION_ENVELOPE_BYTES > max_record_bytes:
            raise ValueError("one arm section item exceeds the record byte bound")
        separator_bytes = int(index > start)
        if (
            index > start
            and encoded_items_bytes
            + separator_bytes
            + item_bytes
            + _ARM_SECTION_ENVELOPE_BYTES
            > max_record_bytes
        ):
            ranges.append((start, index))
            start = index
            encoded_items_bytes = 2
            separator_bytes = 0
        encoded_items_bytes += separator_bytes + item_bytes
    if start < len(items):
        ranges.append((start, len(items)))
    return tuple(ranges)


def _canonical_sequence_sha256(items: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, item in enumerate(items):
        if index:
            digest.update(b",")
        digest.update(_canonical_bytes(item))
    digest.update(b"]")
    return digest.hexdigest()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _bootstrap_sample_stream_sha256(metrics: V5CompleteArmMetrics) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, sample in enumerate(metrics.bootstrap.samples):
        if index:
            digest.update(b",")
        digest.update(_canonical_bytes(sample.model_dump(mode="json")))
    digest.update(b"]")
    return digest.hexdigest()


def summarize_v5_complete_arm_metrics(
    *,
    metric: V5CompleteArmMetrics,
    deterministic_result_sha256: str,
) -> V5CompleteArmMetricSummary:
    """Address complete deterministic draws while retaining all reported semantics."""
    _require_sha256(
        deterministic_result_sha256,
        label="metric summary deterministic result digest",
    )
    bootstrap_values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-bootstrap-summary/1",
        "seed": metric.bootstrap.seed,
        "replicates": metric.bootstrap.replicates,
        "confidence_level": metric.bootstrap.confidence_level,
        "interval_method": metric.bootstrap.interval_method,
        "resampling_unit": metric.bootstrap.resampling_unit,
        "stratification": metric.bootstrap.stratification,
        "strata": metric.bootstrap.strata,
        "sample_count": len(metric.bootstrap.samples),
        "sample_stream_sha256": _bootstrap_sample_stream_sha256(metric),
        "intervals": tuple(
            item.model_dump(mode="json") for item in metric.bootstrap.intervals
        ),
        "bootstrap_sha256": metric.bootstrap.bootstrap_sha256,
    }
    bootstrap_values["summary_sha256"] = _sha256(_canonical_bytes(bootstrap_values))
    bootstrap = V5MetricBootstrapSummary.model_validate(bootstrap_values)
    summary_values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-complete-metric-summary/1",
        "deterministic_result_sha256": deterministic_result_sha256,
        "arm": metric.arm,
        "arm_result_sha256": metric.arm_result_sha256,
        "support_sha256": metric.support_sha256,
        "aggregate": {
            name: item.model_dump(mode="json")
            for name, item in metric.aggregate.items()
        },
        "calibration": metric.calibration.model_dump(mode="json"),
        "economics": metric.economics.model_dump(mode="json"),
        "by_family": tuple(
            item.model_dump(mode="json") for item in metric.by_family
        ),
        "bootstrap": bootstrap.model_dump(mode="json"),
        "complete_metrics_sha256": metric.complete_metrics_sha256,
    }
    summary_values["summary_sha256"] = _sha256(_canonical_bytes(summary_values))
    return V5CompleteArmMetricSummary.model_validate(summary_values)


def build_v5_metric_arm_worker_receipt(
    *,
    mode: V5KaggleMode,
    run_binding_sha256: str,
    attempt_receipt_sha256: str,
    execution_manifest_sha256: str,
    arm_manifest_sha256: str,
    arm_manifest_deterministic_sha256: str,
    arm_index: int,
    arm_order: tuple[str, ...],
    support_count: int,
    support_event_ids_sha256: str,
    summary: V5CompleteArmMetricSummary,
    evidence_protocol_sha256: str,
    implementation_sha256: str,
    wall_seconds: float,
    peak_rss_bytes: int,
    artifact_bytes: int,
) -> V5MetricArmWorkerReceipt:
    """Bind one compact metric summary to its exact run, arm, and resources."""
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-source-bound-metric-arm-worker/1",
        "stage": V5KaggleStage.METRICS.value,
        "mode": mode.value,
        "run_binding_sha256": run_binding_sha256,
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "arm_manifest_sha256": arm_manifest_sha256,
        "arm_manifest_deterministic_sha256": arm_manifest_deterministic_sha256,
        "arm": summary.arm.value,
        "arm_index": arm_index,
        "arm_order": arm_order,
        "support_sha256": summary.support_sha256,
        "support_count": support_count,
        "support_event_ids_sha256": support_event_ids_sha256,
        "deterministic_result_sha256": summary.deterministic_result_sha256,
        "evidence_protocol_sha256": evidence_protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "metric_summary": summary.model_dump(mode="json"),
        "resource_telemetry": {
            "fresh_interpreter": True,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "artifact_bytes": artifact_bytes,
        },
    }
    values["receipt_sha256"] = _sha256(_canonical_bytes(values))
    return V5MetricArmWorkerReceipt.model_validate(values)


def validate_v5_metric_arm_worker_receipt(
    *,
    receipt: V5MetricArmWorkerReceipt,
    expected: Mapping[str, object],
    max_peak_rss_bytes: int,
    max_artifact_bytes: int,
) -> V5MetricArmWorkerReceipt:
    """Fail closed unless a worker receipt matches every coordinator binding."""
    actual = receipt.model_dump(mode="json")
    for name, expected_value in expected.items():
        normalized_expected = (
            expected_value.value
            if isinstance(expected_value, (V5Arm, V5KaggleMode, V5KaggleStage))
            else expected_value
        )
        if isinstance(normalized_expected, tuple):
            normalized_expected = list(normalized_expected)
        if actual.get(name) != normalized_expected:
            raise ValueError(f"metric worker {name} differs")
    telemetry = receipt.resource_telemetry
    if (
        telemetry.peak_rss_bytes >= max_peak_rss_bytes
        or telemetry.artifact_bytes >= max_artifact_bytes
    ):
        raise ValueError("metric worker resource gate failed")
    return receipt


def execute_v5_metric_arm_worker(
    *,
    root: Path,
    mode: V5KaggleMode,
    run_binding_sha256: str,
    attempt_receipt_sha256: str,
    execution_manifest_sha256: str,
    arm_checkpoint_root: Path,
    arm_manifest_sha256: str,
    arm_manifest_deterministic_sha256: str,
    target_arm: str,
) -> V5MetricArmWorkerReceipt:
    """Compute one exact arm metric inside the caller's fresh interpreter."""
    started = time.perf_counter()
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    if protocol.run_binding_sha256(mode) != run_binding_sha256:
        raise PermissionError("metric worker run binding differs")
    arm_manifest = read_v5_checkpoint_manifest(
        output_root=arm_checkpoint_root,
        limits=protocol.resources,
    )
    if (
        arm_manifest.stage is not V5KaggleStage.ARMS
        or arm_manifest.run_binding_sha256 != run_binding_sha256
        or arm_manifest.attempt_receipt_sha256 != attempt_receipt_sha256
        or arm_manifest.manifest_sha256 != arm_manifest_sha256
        or arm_manifest.deterministic_sha256
        != arm_manifest_deterministic_sha256
    ):
        raise PermissionError("metric worker Stage 30 manifest differs")
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path,
        root=root,
    )
    isolated = load_v5_metric_worker_arm_result(
        checkpoint_root=arm_checkpoint_root,
        limits=protocol.resources,
        target_arm=target_arm,
    )
    metric = evaluate_v5_complete_result(
        result=isolated.result,
        protocol=evidence_protocol,
    )
    summary = summarize_v5_complete_arm_metrics(
        metric=metric,
        deterministic_result_sha256=isolated.deterministic_result_sha256,
    )
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum_rss *= 1024
    receipt = build_v5_metric_arm_worker_receipt(
        mode=mode,
        run_binding_sha256=run_binding_sha256,
        attempt_receipt_sha256=attempt_receipt_sha256,
        execution_manifest_sha256=execution_manifest_sha256,
        arm_manifest_sha256=arm_manifest_sha256,
        arm_manifest_deterministic_sha256=arm_manifest_deterministic_sha256,
        arm_index=isolated.arm_index,
        arm_order=tuple(arm.value for arm in _STAGED_ARMS),
        support_count=isolated.result.support_total,
        support_event_ids_sha256=isolated.support_event_ids_sha256,
        summary=summary,
        evidence_protocol_sha256=evidence_protocol.evidence_protocol_sha256,
        implementation_sha256=evidence_protocol.implementation_sha256,
        wall_seconds=max(time.perf_counter() - started, 1e-9),
        peak_rss_bytes=max(maximum_rss, 1),
        artifact_bytes=len(_canonical_bytes(summary.model_dump(mode="json"))),
    )
    validate_v5_metric_arm_worker_receipt(
        receipt=receipt,
        expected={
            "mode": mode,
            "run_binding_sha256": run_binding_sha256,
            "attempt_receipt_sha256": attempt_receipt_sha256,
            "execution_manifest_sha256": execution_manifest_sha256,
            "arm_manifest_sha256": arm_manifest_sha256,
            "arm_manifest_deterministic_sha256": (
                arm_manifest_deterministic_sha256
            ),
            "arm": target_arm,
            "arm_index": isolated.arm_index,
            "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
            "support_sha256": isolated.result.support_sha256,
            "support_count": isolated.result.support_total,
            "support_event_ids_sha256": isolated.support_event_ids_sha256,
            "deterministic_result_sha256": isolated.deterministic_result_sha256,
            "evidence_protocol_sha256": (
                evidence_protocol.evidence_protocol_sha256
            ),
            "implementation_sha256": evidence_protocol.implementation_sha256,
        },
        max_peak_rss_bytes=protocol.resources.max_peak_rss_bytes,
        max_artifact_bytes=min(
            protocol.resources.max_stage_output_bytes,
            256 * 1024**2,
        ),
    )
    return receipt


def build_v5_readiness_from_metric_summary(
    *,
    metrics: V5CompleteArmMetricSummary,
    controls: V5ExecutedControlSuite,
) -> V5ReadinessEvidence:
    """Evaluate unchanged readiness gates from an authenticated metric summary."""
    return _build_v5_readiness_evidence_from_source(
        metrics=metrics,
        controls=controls,
    )


def _development_protocol_for_stage(
    *,
    root: Path,
    protocol: V5KaggleProtocol,
    mode: V5KaggleMode,
) -> tuple[V5DevelopmentProtocol, V5Profile]:
    """Bind the frozen production profile to only the selected closed seed."""
    locked = load_v5_development_protocol(root / protocol.source_bindings.base_protocol_path)
    selected = protocol.capacity if mode is V5KaggleMode.CAPACITY_VALIDATION else protocol.locked
    if selected.profile != V5Profile.PRODUCTION.value:
        raise ValueError("staged corpus profile differs from production")
    if locked.seeds.development_test == selected.development_test_seed:
        selected_protocol = locked
    else:
        selected_protocol = locked.model_copy(
            update={
                "seeds": locked.seeds.model_copy(
                    update={
                        "development_test": selected.development_test_seed,
                    }
                ),
                "protocol_sha256": "",
            }
        )
        selected_protocol = selected_protocol.model_copy(
            update={"protocol_sha256": v5_protocol_digest(selected_protocol)}
        )
    return selected_protocol, V5Profile.PRODUCTION


def _corpus_digest(partitions: dict[str, V5PartitionCorpus]) -> str:
    document = {
        name: {
            "decisions": [row.model_dump(mode="json") for row in partition.decisions],
            "executions": [execution.model_dump(mode="json") for execution in partition.executions],
        }
        for name, partition in partitions.items()
    }
    return _sha256(_canonical_bytes(document))


def _iter_v5_corpus_records(
    *,
    corpus: V5Corpus,
    mode: V5KaggleMode,
    development_test_seed: int,
    support_plan_sha256: str,
) -> Iterator[V5CheckpointInput]:
    """Serialize the full immutable corpus without live simulator objects."""
    partition_order = tuple(corpus.partitions)
    header = {
        "schema_version": "apar-sentinel-v5-kaggle-corpus/1",
        "mode": mode,
        "profile": corpus.profile,
        "development_test_seed": development_test_seed,
        "support_plan_sha256": support_plan_sha256,
        "partition_order": partition_order,
        "partition_support": [
            {
                "partition": name,
                "decisions": len(corpus.partitions[name].decisions),
                "executions": len(corpus.partitions[name].executions),
            }
            for name in partition_order
        ],
        "corpus_sha256": corpus.corpus_sha256,
        "is_production": corpus.is_production,
    }
    yield V5CheckpointInput(
        kind="corpus_header",
        key="corpus",
        canonical_bytes=_canonical_bytes(header),
    )
    for partition_name in partition_order:
        partition = corpus.partitions[partition_name]
        yield V5CheckpointInput(
            kind="partition_header",
            key=partition_name,
            canonical_bytes=_canonical_bytes(
                {
                    "schema_version": "apar-sentinel-v5-kaggle-corpus-partition/1",
                    "partition": partition_name,
                    "decisions": len(partition.decisions),
                    "executions": len(partition.executions),
                }
            ),
        )
        for row in partition.decisions:
            yield V5CheckpointInput(
                kind="decision_row",
                key=f"{partition_name}:{row.event_id}",
                canonical_bytes=_canonical_bytes(
                    {
                        "partition": partition_name,
                        "decision": row.model_dump(mode="json"),
                    }
                ),
            )
        for execution in partition.executions:
            yield V5CheckpointInput(
                kind="execution_manifest",
                key=f"{partition_name}:{execution.evidence_sha256}",
                canonical_bytes=_canonical_bytes(
                    {
                        "partition": partition_name,
                        "execution": execution.model_dump(mode="json"),
                    }
                ),
            )


def _read_json_record(record: V5CheckpointInput, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(record.canonical_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(document, dict) or record.canonical_bytes != _canonical_bytes(document):
        raise ValueError(f"{label} is not canonical JSON")
    return document


def load_v5_corpus_checkpoint(*, checkpoint_root: Path, limits: V5KaggleResourceGates) -> V5Corpus:
    """Reconstruct and revalidate a complete executed corpus checkpoint."""
    records = iter(
        iter_v5_checkpoint_records(
            output_root=checkpoint_root,
            limits=limits,
        )
    )
    try:
        header_record = next(records)
    except StopIteration as error:
        raise ValueError("corpus checkpoint is empty") from error
    if header_record.kind != "corpus_header" or header_record.key != "corpus":
        raise ValueError("corpus checkpoint header is missing")
    header = _read_json_record(header_record, label="corpus header")
    if header.get("schema_version") != "apar-sentinel-v5-kaggle-corpus/1":
        raise ValueError("corpus checkpoint schema differs")
    partition_order_value = header.get("partition_order")
    if not isinstance(partition_order_value, list) or not partition_order_value:
        raise ValueError("corpus checkpoint partition order is missing")
    partition_order = tuple(str(item) for item in partition_order_value)
    if len(set(partition_order)) != len(partition_order):
        raise ValueError("corpus checkpoint partitions are duplicated")
    support_value = header.get("partition_support")
    if not isinstance(support_value, list) or len(support_value) != len(partition_order):
        raise ValueError("corpus checkpoint support is incomplete")
    expected_support: dict[str, tuple[int, int]] = {}
    for item in support_value:
        if not isinstance(item, dict):
            raise ValueError("corpus checkpoint support row is malformed")
        partition = item.get("partition")
        expected_decisions = item.get("decisions")
        expected_executions = item.get("executions")
        if (
            type(partition) is not str
            or type(expected_decisions) is not int
            or type(expected_executions) is not int
            or expected_decisions <= 0
            or expected_executions <= 0
            or partition in expected_support
        ):
            raise ValueError("corpus checkpoint support row differs")
        expected_support[partition] = (expected_decisions, expected_executions)
    if tuple(expected_support) != partition_order:
        raise ValueError("corpus checkpoint support order differs")

    partitions: dict[str, V5PartitionCorpus] = {}
    for partition_name in partition_order:
        try:
            partition_record = next(records)
        except StopIteration as error:
            raise ValueError("corpus checkpoint partition header is missing") from error
        partition_header = _read_json_record(partition_record, label="corpus partition header")
        decision_count, execution_count = expected_support[partition_name]
        expected_header = {
            "schema_version": "apar-sentinel-v5-kaggle-corpus-partition/1",
            "partition": partition_name,
            "decisions": decision_count,
            "executions": execution_count,
        }
        if (
            partition_record.kind != "partition_header"
            or partition_record.key != partition_name
            or partition_header != expected_header
        ):
            raise ValueError("corpus checkpoint partition header differs")
        decisions: list[V5DecisionRow] = []
        for _ in range(decision_count):
            try:
                record = next(records)
            except StopIteration as error:
                raise ValueError("corpus checkpoint decision is missing") from error
            document = _read_json_record(record, label="corpus decision")
            if record.kind != "decision_row" or document.get("partition") != partition_name:
                raise ValueError("corpus checkpoint decision partition differs")
            decision = V5DecisionRow.model_validate(document.get("decision"))
            if record.key != f"{partition_name}:{decision.event_id}":
                raise ValueError("corpus checkpoint decision key differs")
            decisions.append(decision)
        executions: list[V5ExecutionManifest] = []
        for _ in range(execution_count):
            try:
                record = next(records)
            except StopIteration as error:
                raise ValueError("corpus checkpoint execution is missing") from error
            document = _read_json_record(record, label="corpus execution")
            if record.kind != "execution_manifest" or document.get("partition") != partition_name:
                raise ValueError("corpus checkpoint execution partition differs")
            execution = V5ExecutionManifest.model_validate(document.get("execution"))
            if record.key != f"{partition_name}:{execution.evidence_sha256}":
                raise ValueError("corpus checkpoint execution key differs")
            executions.append(execution)
        partitions[partition_name] = V5PartitionCorpus(
            partition_name=partition_name,
            decisions=tuple(decisions),
            executions=tuple(executions),
        )
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("corpus checkpoint has extra records")
    corpus_digest = _corpus_digest(partitions)
    if corpus_digest != header.get("corpus_sha256"):
        raise ValueError("corpus checkpoint digest differs")
    try:
        profile = V5Profile(str(header.get("profile")))
    except ValueError as error:
        raise ValueError("corpus checkpoint profile differs") from error
    is_production = header.get("is_production")
    if type(is_production) is not bool or is_production is not (profile is V5Profile.PRODUCTION):
        raise ValueError("corpus checkpoint production flag differs")
    return V5Corpus(
        profile=profile,
        partitions=partitions,
        corpus_sha256=corpus_digest,
        is_production=is_production,
    )


def _validate_corpus_support(*, corpus: V5Corpus, support_plan: V5KaggleSupportPlan) -> None:
    for item in support_plan.partitions:
        partition = corpus.partitions.get(item.partition)
        if partition is None:
            raise ValueError("production corpus support partition is missing")
        fraud_by_family = tuple(
            (
                family,
                sum(1 for row in partition.decisions if row.is_fraud and row.family == family),
            )
            for family, _count in item.fraud_rows_by_family
        )
        observed = {
            "legitimate_rows": partition.benign_count,
            "fraud_rows_by_family": fraud_by_family,
            "total_rows": len(partition.decisions),
            "execution_artifacts": len(partition.executions),
        }
        expected = {
            "legitimate_rows": item.legitimate_rows,
            "fraud_rows_by_family": item.fraud_rows_by_family,
            "total_rows": item.total_rows,
            "execution_artifacts": item.execution_artifacts,
        }
        if observed != expected:
            diagnostic = {
                "partition": item.partition,
                "observed": observed,
                "expected": expected,
            }
            raise ValueError(
                "production corpus support differs from frozen plan: "
                + json.dumps(
                    diagnostic,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )


def execute_v5_corpus_stage(
    *, root: Path, capability: V5StageCapability
) -> Iterator[V5CheckpointInput]:
    """Execute and serialize the real production-size corpus for one closed mode."""
    _validate_capability(capability, required_stage=V5KaggleStage.CORPUS)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    if capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode):
        raise PermissionError("corpus capability run binding differs")
    development, profile = _development_protocol_for_stage(
        root=root, protocol=protocol, mode=capability.mode
    )
    support_plan = build_v5_kaggle_support_plan(root=root, protocol=protocol, mode=capability.mode)
    corpus = build_v5_corpus(development, profile=profile)
    _validate_corpus_support(corpus=corpus, support_plan=support_plan)
    yield from _iter_v5_corpus_records(
        corpus=corpus,
        mode=capability.mode,
        development_test_seed=development.seeds.development_test,
        support_plan_sha256=support_plan.support_plan_sha256,
    )


def _immutable_array(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _prepare_partition(
    *,
    partition_name: str,
    partition: V5PartitionCorpus,
    catalog: SentinelFeatureCatalog,
) -> V5PreparedPartition:
    canonical = tuple(sorted(partition.decisions, key=lambda row: (row.decision_at, row.event_id)))
    if partition.decisions != canonical:
        raise ValueError("partition decisions must use canonical feature-row order")
    batch = build_sentinel_features(partition.decisions, catalog=catalog)
    matrix = np.ascontiguousarray(batch.matrix, dtype="<f8")
    labels = np.ascontiguousarray([int(row.is_fraud) for row in partition.decisions], dtype="<i8")
    amounts = np.ascontiguousarray([float(row.amount) for row in partition.decisions], dtype="<f8")
    event_ids = tuple(row.event_id for row in partition.decisions)
    campaign_ids = tuple(row.campaign_id for row in partition.decisions)
    support = build_v5_arm_support_rows(partition.decisions)
    artifacts = build_v5_execution_artifacts(partition.executions)
    trust_failures = tuple(derive_v5_trust_failures(support, artifacts))
    training_evidence: V5TrainingPartitionEvidence | None = None
    if partition_name in {"train", "calibration", "threshold"}:
        training_evidence = build_v5_training_partition_evidence(
            partition=cast(Literal["train", "calibration", "threshold"], partition_name),
            event_ids=event_ids,
            labels=labels,
            support=support,
            feature_batch_sha256=batch.batch_sha256,
            feature_matrix=matrix,
            feature_names=catalog.feature_names,
            catalog_sha256=catalog.catalog_sha256,
            execution_manifests=partition.executions,
            feature_batch_source_matrix=batch.matrix,
        )
    _immutable_array(matrix)
    _immutable_array(labels)
    _immutable_array(amounts)
    return V5PreparedPartition(
        partition=partition_name,
        matrix=matrix,
        labels=labels,
        event_ids=event_ids,
        campaign_ids=campaign_ids,
        amounts=amounts,
        trust_failures=trust_failures,
        feature_batch=batch,
        training_evidence=training_evidence,
    )


def _array_metadata(array: NDArray[np.generic]) -> dict[str, object]:
    content = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": len(content),
        "sha256": _sha256(content),
    }


def _iter_prepared_partition_records(
    prepared: V5PreparedPartition,
) -> Iterator[V5CheckpointInput]:
    prefix = prepared.partition
    trust_bytes = bytes(int(value) for value in prepared.trust_failures)
    metadata = {
        "schema_version": "apar-sentinel-v5-kaggle-prepared-partition/1",
        "partition": prefix,
        "event_ids": prepared.event_ids,
        "campaign_ids": prepared.campaign_ids,
        "matrix": _array_metadata(prepared.matrix),
        "labels": _array_metadata(prepared.labels),
        "amounts": _array_metadata(prepared.amounts),
        "trust_failures": {
            "dtype": "|u1",
            "shape": [len(prepared.trust_failures)],
            "bytes": len(trust_bytes),
            "sha256": _sha256(trust_bytes),
        },
        "feature_batch_sha256": prepared.feature_batch.batch_sha256,
        "catalog_sha256": prepared.feature_batch.catalog_sha256,
        "has_training_evidence": prepared.training_evidence is not None,
    }
    yield V5CheckpointInput(
        kind="prepared_partition",
        key=prefix,
        canonical_bytes=_canonical_bytes(metadata),
    )
    for kind, content in (
        ("feature_matrix", prepared.matrix.tobytes(order="C")),
        ("labels", prepared.labels.tobytes(order="C")),
        ("amounts", prepared.amounts.tobytes(order="C")),
        ("trust_failures", trust_bytes),
    ):
        yield V5CheckpointInput(
            kind=kind,
            key=f"{prefix}:{kind}",
            canonical_bytes=content,
        )
    yield V5CheckpointInput(
        kind="feature_batch",
        key=f"{prefix}:feature_batch",
        canonical_bytes=_canonical_bytes(prepared.feature_batch.model_dump(mode="json")),
    )
    if prepared.training_evidence is not None:
        yield V5CheckpointInput(
            kind="training_evidence",
            key=f"{prefix}:training_evidence",
            canonical_bytes=_canonical_bytes(prepared.training_evidence.model_dump(mode="json")),
        )


def execute_v5_feature_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    corpus_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]:
    """Build causal feature evidence solely from the validated corpus checkpoint."""
    _validate_capability(capability, required_stage=V5KaggleStage.FEATURES)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    corpus_manifest = read_v5_checkpoint_manifest(
        output_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    if (
        capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode)
        or capability.predecessor_manifest_sha256 != corpus_manifest.manifest_sha256
        or corpus_manifest.stage is not V5KaggleStage.CORPUS
        or corpus_manifest.run_binding_sha256 != capability.run_binding_sha256
        or corpus_manifest.attempt_receipt_sha256 != capability.attempt_receipt_sha256
    ):
        raise PermissionError("feature capability or corpus predecessor differs")
    corpus = load_v5_corpus_checkpoint(
        checkpoint_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.source_bindings.feature_catalog_path
    )
    partition_order = ("train", "calibration", "threshold", "development_test")
    if any(name not in corpus.partitions for name in partition_order):
        raise ValueError("feature corpus lacks a retained partition")
    yield V5CheckpointInput(
        kind="feature_header",
        key="features",
        canonical_bytes=_canonical_bytes(
            {
                "schema_version": "apar-sentinel-v5-kaggle-features/1",
                "partition_order": partition_order,
                "catalog_sha256": catalog.catalog_sha256,
                "feature_names": catalog.feature_names,
                "corpus_sha256": corpus.corpus_sha256,
            }
        ),
    )
    for partition_name in partition_order:
        yield from _iter_prepared_partition_records(
            _prepare_partition(
                partition_name=partition_name,
                partition=corpus.partitions[partition_name],
                catalog=catalog,
            )
        )


def _decode_array(*, record: V5CheckpointInput, metadata: object) -> NDArray[np.generic]:
    if not isinstance(metadata, dict) or set(metadata) != {
        "dtype",
        "shape",
        "bytes",
        "sha256",
    }:
        raise ValueError("feature array metadata differs")
    dtype = metadata["dtype"]
    shape = metadata["shape"]
    byte_count = metadata["bytes"]
    digest = metadata["sha256"]
    if (
        type(dtype) is not str
        or not isinstance(shape, list)
        or not shape
        or any(type(value) is not int or value < 0 for value in shape)
        or type(byte_count) is not int
        or type(digest) is not str
        or len(record.canonical_bytes) != byte_count
        or _sha256(record.canonical_bytes) != digest
    ):
        raise ValueError("feature array bytes differ")
    if dtype not in {"<f8", "<i8", "|u1"}:
        raise ValueError("feature array dtype differs")
    array = np.frombuffer(record.canonical_bytes, dtype=np.dtype(dtype)).copy()
    expected_values = int(np.prod(shape, dtype=np.int64))
    if len(array) != expected_values:
        raise ValueError("feature array shape differs")
    result = array.reshape(tuple(shape))
    _immutable_array(result)
    return result


def load_v5_feature_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> dict[str, V5PreparedPartition]:
    """Reconstruct exact feature arrays, provenance, labels, and training evidence."""
    records = iter(
        iter_v5_checkpoint_records(
            output_root=checkpoint_root,
            limits=limits,
        )
    )
    try:
        header_record = next(records)
    except StopIteration as error:
        raise ValueError("feature checkpoint is empty") from error
    header = _read_json_record(header_record, label="feature header")
    if (
        header_record.kind != "feature_header"
        or header_record.key != "features"
        or header.get("schema_version") != "apar-sentinel-v5-kaggle-features/1"
    ):
        raise ValueError("feature checkpoint header differs")
    partition_order_value = header.get("partition_order")
    if partition_order_value != [
        "train",
        "calibration",
        "threshold",
        "development_test",
    ]:
        raise ValueError("feature checkpoint partition order differs")
    feature_names_value = header.get("feature_names")
    catalog_sha256 = header.get("catalog_sha256")
    if (
        not isinstance(feature_names_value, list)
        or not all(type(name) is str for name in feature_names_value)
        or type(catalog_sha256) is not str
    ):
        raise ValueError("feature checkpoint catalog binding differs")
    feature_names = tuple(cast(str, name) for name in feature_names_value)
    prepared_by_partition: dict[str, V5PreparedPartition] = {}
    for partition_name in partition_order_value:
        try:
            metadata_record = next(records)
        except StopIteration as error:
            raise ValueError("prepared partition metadata is missing") from error
        metadata = _read_json_record(metadata_record, label="prepared partition metadata")
        if (
            metadata_record.kind != "prepared_partition"
            or metadata_record.key != partition_name
            or metadata.get("partition") != partition_name
            or metadata.get("schema_version") != "apar-sentinel-v5-kaggle-prepared-partition/1"
        ):
            raise ValueError("prepared partition metadata differs")
        binary_records: dict[str, V5CheckpointInput] = {}
        for expected_kind in (
            "feature_matrix",
            "labels",
            "amounts",
            "trust_failures",
        ):
            try:
                binary = next(records)
            except StopIteration as error:
                raise ValueError("prepared partition array is missing") from error
            if binary.kind != expected_kind or binary.key != f"{partition_name}:{expected_kind}":
                raise ValueError("prepared partition array order differs")
            binary_records[expected_kind] = binary
        matrix = _decode_array(
            record=binary_records["feature_matrix"], metadata=metadata.get("matrix")
        )
        labels = _decode_array(record=binary_records["labels"], metadata=metadata.get("labels"))
        amounts = _decode_array(record=binary_records["amounts"], metadata=metadata.get("amounts"))
        trust = _decode_array(
            record=binary_records["trust_failures"],
            metadata=metadata.get("trust_failures"),
        )
        if (
            matrix.dtype != np.dtype("<f8")
            or labels.dtype != np.dtype("<i8")
            or amounts.dtype != np.dtype("<f8")
            or trust.dtype != np.dtype("u1")
            or matrix.ndim != 2
            or labels.ndim != 1
            or amounts.ndim != 1
            or trust.ndim != 1
        ):
            raise ValueError("prepared partition array type differs")
        try:
            feature_batch_record = next(records)
        except StopIteration as error:
            raise ValueError("feature batch is missing") from error
        if (
            feature_batch_record.kind != "feature_batch"
            or feature_batch_record.key != f"{partition_name}:feature_batch"
        ):
            raise ValueError("feature batch order differs")
        feature_batch = SentinelFeatureBatch.model_validate(
            _read_json_record(feature_batch_record, label="feature batch")
        )
        training_evidence: V5TrainingPartitionEvidence | None = None
        has_training = metadata.get("has_training_evidence")
        if type(has_training) is not bool:
            raise ValueError("training evidence declaration differs")
        if has_training:
            try:
                evidence_record = next(records)
            except StopIteration as error:
                raise ValueError("training evidence is missing") from error
            if (
                evidence_record.kind != "training_evidence"
                or evidence_record.key != f"{partition_name}:training_evidence"
            ):
                raise ValueError("training evidence order differs")
            training_evidence = V5TrainingPartitionEvidence.model_validate(
                _read_json_record(evidence_record, label="training evidence")
            )
        event_ids_value = metadata.get("event_ids")
        campaign_ids_value = metadata.get("campaign_ids")
        if (
            not isinstance(event_ids_value, list)
            or not isinstance(campaign_ids_value, list)
            or not all(type(value) is str for value in event_ids_value)
            or not all(type(value) is str for value in campaign_ids_value)
        ):
            raise ValueError("prepared partition identity metadata differs")
        event_ids = tuple(cast(str, value) for value in event_ids_value)
        campaign_ids = tuple(cast(str, value) for value in campaign_ids_value)
        row_count = len(event_ids)
        if (
            len(set(event_ids)) != row_count
            or len(campaign_ids) != row_count
            or matrix.shape != (row_count, len(feature_names))
            or labels.shape != (row_count,)
            or amounts.shape != (row_count,)
            or trust.shape != (row_count,)
            or not np.isin(labels, (0, 1)).all()
            or not np.isin(trust, (0, 1)).all()
            or feature_batch.catalog_sha256 != catalog_sha256
            or feature_batch.batch_sha256 != metadata.get("feature_batch_sha256")
            or tuple(item.event_id for item in feature_batch.provenance) != event_ids
            or not np.array_equal(np.asarray(feature_batch.matrix, dtype="<f8"), matrix)
            or (training_evidence is None) is (partition_name != "development_test")
        ):
            raise ValueError("prepared partition support or feature binding differs")
        prepared_by_partition[partition_name] = V5PreparedPartition(
            partition=partition_name,
            matrix=cast(NDArray[np.float64], matrix),
            labels=cast(NDArray[np.int64], labels),
            event_ids=event_ids,
            campaign_ids=campaign_ids,
            amounts=cast(NDArray[np.float64], amounts),
            trust_failures=tuple(bool(value) for value in trust.tolist()),
            feature_batch=feature_batch,
            training_evidence=training_evidence,
        )
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("feature checkpoint has extra records")
    return prepared_by_partition


def _arm_core_and_observation(
    result: V5EvaluationResult,
) -> tuple[dict[str, object], dict[str, object]]:
    full = result.model_dump(mode="json")
    raw_rows = full.get("row_evidence")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("arm result rows are missing")
    stable_rows: list[dict[str, object]] = []
    latency_samples: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("arm result row is malformed")
        row = cast(dict[str, object], dict(raw_row))
        support = row.get("support")
        if not isinstance(support, dict) or type(support.get("event_id")) is not str:
            raise ValueError("arm result support is malformed")
        latency = row.pop("latency_ms", None)
        row_output_sha256 = row.pop("row_output_sha256", None)
        if not isinstance(latency, (int, float)) or type(row_output_sha256) is not str:
            raise ValueError("arm result observational latency is malformed")
        row["deterministic_row_sha256"] = _sha256(_canonical_bytes(row))
        stable_rows.append(row)
        latency_samples.append(
            {
                "event_id": support["event_id"],
                "latency_ms": float(latency),
                "row_output_sha256": row_output_sha256,
            }
        )
    core = cast(dict[str, object], dict(full))
    for field in (
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "score_sha256",
        "result_sha256",
    ):
        core.pop(field, None)
    core["row_evidence"] = stable_rows
    core["deterministic_result_sha256"] = _sha256(_canonical_bytes(core))
    observation: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-arm-latency/1",
        "arm": result.arm,
        "deterministic_result_sha256": core["deterministic_result_sha256"],
        "samples": latency_samples,
        "p50_latency_ms": result.p50_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "p99_latency_ms": result.p99_latency_ms,
        "score_sha256": result.score_sha256,
        "result_sha256": result.result_sha256,
    }
    observation["observational_sha256"] = _sha256(_canonical_bytes(observation))
    return core, observation


def _pop_document_sequence(
    document: dict[str, object], *, field: str
) -> tuple[dict[str, object], ...]:
    raw_items = document.pop(field, None)
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"arm checkpoint {field} is missing")
    if any(not isinstance(item, dict) for item in raw_items):
        raise ValueError(f"arm checkpoint {field} contains a malformed item")
    return tuple(cast(dict[str, object], item) for item in raw_items)


def _arm_section_index(items: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(items),
        "record_count": len(
            _bounded_arm_section_ranges(
                items=items,
                max_record_bytes=_MAX_ARM_SECTION_RECORD_BYTES,
            )
        ),
        "sha256": _canonical_sequence_sha256(items),
    }


def _iter_arm_result_checkpoint_records(
    *,
    result: V5EvaluationResult,
    core: dict[str, object],
    observation: dict[str, object],
) -> Iterator[V5CheckpointInput]:
    artifacts = _pop_document_sequence(core, field="execution_artifacts")
    rows = _pop_document_sequence(core, field="row_evidence")
    samples = _pop_document_sequence(observation, field="samples")
    core_meta = {
        "schema_version": _ARM_RESULT_META_SCHEMA,
        "arm": result.arm,
        "fields": core,
        "sections": {
            "execution_artifacts": _arm_section_index(artifacts),
            "row_evidence": _arm_section_index(rows),
        },
    }
    yield V5CheckpointInput(
        kind="arm_result_meta",
        key=result.arm,
        canonical_bytes=_canonical_bytes(core_meta),
    )
    yield from _iter_bounded_arm_section_records(
        kind="arm_execution_artifacts",
        arm=result.arm,
        section="execution_artifacts",
        items=artifacts,
        max_record_bytes=_MAX_ARM_SECTION_RECORD_BYTES,
    )
    yield from _iter_bounded_arm_section_records(
        kind="arm_result_rows",
        arm=result.arm,
        section="row_evidence",
        items=rows,
        max_record_bytes=_MAX_ARM_SECTION_RECORD_BYTES,
    )
    latency_meta = {
        "schema_version": _ARM_LATENCY_META_SCHEMA,
        "arm": result.arm,
        "fields": observation,
        "sections": {"samples": _arm_section_index(samples)},
    }
    yield V5CheckpointInput(
        kind="arm_latency_meta",
        key=result.arm,
        canonical_bytes=_canonical_bytes(latency_meta),
        layer="observational",
    )
    for record in _iter_bounded_arm_section_records(
        kind="arm_latency_samples",
        arm=result.arm,
        section="samples",
        items=samples,
        max_record_bytes=_MAX_ARM_SECTION_RECORD_BYTES,
    ):
        yield V5CheckpointInput(
            kind=record.kind,
            key=record.key,
            canonical_bytes=record.canonical_bytes,
            layer="observational",
        )


def _read_arm_section_index(
    value: object, *, label: str
) -> tuple[int, int, str]:
    if not isinstance(value, dict) or set(value) != {"count", "record_count", "sha256"}:
        raise ValueError(f"{label} index is malformed")
    count = value.get("count")
    record_count = value.get("record_count")
    digest = value.get("sha256")
    if (
        type(count) is not int
        or type(record_count) is not int
        or count <= 0
        or record_count <= 0
        or record_count > count
        or type(digest) is not str
    ):
        raise ValueError(f"{label} index is malformed")
    _require_sha256(digest, label=f"{label} digest")
    return count, record_count, digest


def _read_arm_meta_record(
    record: V5CheckpointInput,
    *,
    kind: str,
    arm: str,
    schema: str,
    section_names: tuple[str, ...],
    layer: str,
) -> tuple[dict[str, object], dict[str, tuple[int, int, str]]]:
    if record.kind != kind or record.key != arm or record.layer != layer:
        raise ValueError("arm result metadata order differs")
    document = _read_json_record(record, label=f"{arm} {kind}")
    if (
        set(document) != {"schema_version", "arm", "fields", "sections"}
        or document.get("schema_version") != schema
        or document.get("arm") != arm
        or not isinstance(document.get("fields"), dict)
        or not isinstance(document.get("sections"), dict)
    ):
        raise ValueError("arm result metadata differs")
    raw_sections = cast(dict[str, object], document["sections"])
    if tuple(raw_sections) != section_names:
        raise ValueError("arm result metadata sections differ")
    sections = {
        name: _read_arm_section_index(raw_sections[name], label=f"{arm} {name}")
        for name in section_names
    }
    return dict(cast(dict[str, object], document["fields"])), sections


def _read_arm_section_records(
    records: Iterator[V5CheckpointInput],
    *,
    kind: str,
    arm: str,
    section: str,
    index: tuple[int, int, str],
    layer: str,
) -> list[dict[str, object]]:
    expected_count, record_count, expected_digest = index
    items: list[dict[str, object]] = []
    for record_index in range(record_count):
        try:
            record = next(records)
        except StopIteration as error:
            raise ValueError(f"{arm} {section} record is missing") from error
        if (
            record.kind != kind
            or record.key != f"{arm}:{record_index:04d}"
            or record.layer != layer
            or len(record.canonical_bytes) > _MAX_ARM_SECTION_RECORD_BYTES
        ):
            raise ValueError(f"{arm} {section} record order differs")
        document = _read_json_record(record, label=f"{arm} {section} record")
        raw_items = document.get("items")
        if (
            set(document)
            != {"schema_version", "arm", "section", "index", "start", "items"}
            or document.get("schema_version") != _ARM_SECTION_SCHEMA
            or document.get("arm") != arm
            or document.get("section") != section
            or document.get("index") != record_index
            or document.get("start") != len(items)
            or not isinstance(raw_items, list)
            or not raw_items
            or any(not isinstance(item, dict) for item in raw_items)
        ):
            raise ValueError(f"{arm} {section} record differs")
        items.extend(cast(list[dict[str, object]], raw_items))
    if (
        len(items) != expected_count
        or _canonical_sequence_sha256(items) != expected_digest
    ):
        raise ValueError(f"{arm} {section} index binding differs")
    return items


def _restore_arm_result(
    *, core: dict[str, object], observation: dict[str, object]
) -> V5EvaluationResult:
    claimed_core_digest = core.pop("deterministic_result_sha256", None)
    if (
        type(claimed_core_digest) is not str
        or _sha256(_canonical_bytes(core)) != claimed_core_digest
    ):
        raise ValueError("arm deterministic result digest differs")
    claimed_observation_digest = observation.pop("observational_sha256", None)
    if (
        type(claimed_observation_digest) is not str
        or _sha256(_canonical_bytes(observation)) != claimed_observation_digest
    ):
        raise ValueError("arm observational result digest differs")
    if (
        observation.get("schema_version") != "apar-sentinel-v5-kaggle-arm-latency/1"
        or observation.get("arm") != core.get("arm")
        or observation.get("deterministic_result_sha256") != claimed_core_digest
    ):
        raise ValueError("arm deterministic/observational binding differs")
    raw_rows = core.get("row_evidence")
    raw_samples = observation.get("samples")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(raw_samples, list)
        or len(raw_rows) != len(raw_samples)
    ):
        raise ValueError("arm latency sample support differs")
    restored_rows: list[dict[str, object]] = []
    for raw_row, raw_sample in zip(raw_rows, raw_samples, strict=True):
        if not isinstance(raw_row, dict) or not isinstance(raw_sample, dict):
            raise ValueError("arm row or latency sample is malformed")
        row = cast(dict[str, object], dict(raw_row))
        claimed_row_digest = row.pop("deterministic_row_sha256", None)
        if (
            type(claimed_row_digest) is not str
            or _sha256(_canonical_bytes(row)) != claimed_row_digest
        ):
            raise ValueError("arm deterministic row digest differs")
        support = row.get("support")
        if (
            not isinstance(support, dict)
            or raw_sample.get("event_id") != support.get("event_id")
            or not isinstance(raw_sample.get("latency_ms"), (int, float))
            or type(raw_sample.get("row_output_sha256")) is not str
        ):
            raise ValueError("arm latency sample alignment differs")
        row["latency_ms"] = raw_sample["latency_ms"]
        row["row_output_sha256"] = raw_sample["row_output_sha256"]
        restored_rows.append(row)
    restored = cast(dict[str, object], dict(core))
    restored["row_evidence"] = restored_rows
    for field in (
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "score_sha256",
        "result_sha256",
    ):
        restored[field] = observation.get(field)
    return V5EvaluationResult.model_validate(restored)


def execute_v5_arm_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    corpus_checkpoint_root: Path,
    feature_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]:
    """Train and score the four frozen arms from checkpointed causal features."""
    _validate_capability(capability, required_stage=V5KaggleStage.ARMS)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    feature_manifest = read_v5_checkpoint_manifest(
        output_root=feature_checkpoint_root,
        limits=protocol.resources,
    )
    corpus_manifest = read_v5_checkpoint_manifest(
        output_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    if (
        capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode)
        or capability.predecessor_manifest_sha256 != feature_manifest.manifest_sha256
        or feature_manifest.stage is not V5KaggleStage.FEATURES
        or feature_manifest.predecessor_manifest_sha256 != corpus_manifest.manifest_sha256
        or corpus_manifest.stage is not V5KaggleStage.CORPUS
        or feature_manifest.run_binding_sha256 != capability.run_binding_sha256
        or corpus_manifest.run_binding_sha256 != capability.run_binding_sha256
        or feature_manifest.attempt_receipt_sha256 != capability.attempt_receipt_sha256
        or corpus_manifest.attempt_receipt_sha256 != capability.attempt_receipt_sha256
    ):
        raise PermissionError("arm capability or checkpoint lineage differs")
    corpus = load_v5_corpus_checkpoint(
        checkpoint_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    prepared = load_v5_feature_checkpoint(
        checkpoint_root=feature_checkpoint_root,
        limits=protocol.resources,
    )
    development, _profile = _development_protocol_for_stage(
        root=root,
        protocol=protocol,
        mode=capability.mode,
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.source_bindings.feature_catalog_path
    )
    configuration = load_v5_arm_configuration(
        root / protocol.source_bindings.arm_protocol_path,
        catalog=catalog,
        protocol=development,
    )
    train = prepared["train"]
    calibration = prepared["calibration"]
    threshold = prepared["threshold"]
    evaluation = prepared["development_test"]
    if (
        train.training_evidence is None
        or calibration.training_evidence is None
        or threshold.training_evidence is None
        or evaluation.training_evidence is not None
    ):
        raise ValueError("arm training evidence partition semantics differ")
    evaluation_rows = corpus.partitions["development_test"].decisions
    if (
        evaluation.event_ids != tuple(row.event_id for row in evaluation_rows)
        or evaluation.labels.tolist() != [int(row.is_fraud) for row in evaluation_rows]
        or evaluation.campaign_ids != tuple(row.campaign_id for row in evaluation_rows)
        or not np.array_equal(
            evaluation.amounts,
            np.asarray([float(row.amount) for row in evaluation_rows], dtype="<f8"),
        )
    ):
        raise ValueError("arm evaluation support differs from checkpointed corpus")
    trained = train_v5_arm_set(
        configuration=configuration,
        catalog=catalog,
        x_train=train.matrix,
        y_train=train.labels,
        x_calibration=calibration.matrix,
        y_calibration=calibration.labels,
        x_threshold=threshold.matrix,
        y_threshold=threshold.labels,
        bootstrap_seed=development.seeds.bootstrap,
        train_evidence=train.training_evidence,
        calibration_evidence=calibration.training_evidence,
        threshold_evidence=threshold.training_evidence,
    )
    support = build_v5_arm_support_rows(evaluation_rows)
    scores = score_v5_arm_set(
        trained=trained,
        catalog=catalog,
        features_matrix=evaluation.matrix,
        support=support,
        execution_artifacts=build_v5_execution_artifacts(
            corpus.partitions["development_test"].executions
        ),
        trust_failures=list(evaluation.trust_failures),
        feature_provenance=evaluation.feature_batch.provenance,
    )
    results: list[V5EvaluationResult] = []
    campaigns = np.asarray(evaluation.campaign_ids)
    for arm in _STAGED_ARMS:
        score = scores.by_arm[arm]
        base = evaluate_v5_arm(
            arm=arm,
            y_true=evaluation.labels,
            actions=[row.action for row in score.rows],
            probabilities=np.asarray([row.probability for row in score.rows]),
            campaign_ids=campaigns,
            amounts=evaluation.amounts,
        )
        results.append(bind_v5_evaluation_result(base=base, score=score))
    cores_and_observations = tuple(_arm_core_and_observation(result) for result in results)
    yield V5CheckpointInput(
        kind="arm_header",
        key="arms",
        canonical_bytes=_canonical_bytes(
            {
                "schema_version": _ARM_HEADER_SCHEMA,
                "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
                "support_event_ids": evaluation.event_ids,
                "support_sha256": results[0].support_sha256,
                "deterministic_result_sha256": tuple(
                    cast(str, core["deterministic_result_sha256"])
                    for core, _observation in cores_and_observations
                ),
            }
        ),
    )
    for result, (core, observation) in zip(results, cores_and_observations, strict=True):
        yield from _iter_arm_result_checkpoint_records(
            result=result,
            core=core,
            observation=observation,
        )


def _skip_v5_metric_worker_arm_section(
    records: Iterator[V5CheckpointInput],
    *,
    kind: str,
    arm: str,
    record_count: int,
    layer: str,
) -> None:
    """Consume one non-target section without retaining its record payloads."""
    for record_index in range(record_count):
        try:
            record = next(records)
        except StopIteration as error:
            raise ValueError(f"{arm} skipped section record is missing") from error
        if (
            record.kind != kind
            or record.key != f"{arm}:{record_index:04d}"
            or record.layer != layer
            or len(record.canonical_bytes) > _MAX_ARM_SECTION_RECORD_BYTES
        ):
            raise ValueError(f"{arm} skipped section record order differs")


def load_v5_metric_worker_arm_result(
    *,
    checkpoint_root: Path,
    limits: V5KaggleResourceGates,
    target_arm: str,
) -> V5MetricWorkerArmResult:
    """Restore exactly one Stage 30 arm while authenticating both complete streams."""
    deterministic = iter(
        iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits)
    )
    observational = iter(
        iter_v5_checkpoint_observational_records(
            output_root=checkpoint_root,
            limits=limits,
        )
    )
    try:
        header_record = next(deterministic)
    except StopIteration as error:
        raise ValueError("arm checkpoint is empty") from error
    header = _read_json_record(header_record, label="arm header")
    expected_order = tuple(arm.value for arm in _STAGED_ARMS)
    if (
        set(header)
        != {
            "schema_version",
            "arm_order",
            "support_event_ids",
            "support_sha256",
            "deterministic_result_sha256",
        }
        or header_record.kind != "arm_header"
        or header_record.key != "arms"
        or header.get("schema_version") != _ARM_HEADER_SCHEMA
        or header.get("arm_order") != list(expected_order)
    ):
        raise ValueError("arm checkpoint header differs")
    if target_arm not in expected_order:
        raise ValueError("metric worker target arm is not in the frozen arm order")
    expected_core_digests = header.get("deterministic_result_sha256")
    support_event_ids = header.get("support_event_ids")
    support_sha256 = header.get("support_sha256")
    if (
        not isinstance(expected_core_digests, list)
        or len(expected_core_digests) != len(expected_order)
        or not isinstance(support_event_ids, list)
        or not support_event_ids
        or any(type(event_id) is not str for event_id in support_event_ids)
        or type(support_sha256) is not str
    ):
        raise ValueError("arm checkpoint header support differs")
    _require_sha256(support_sha256, label="arm checkpoint support digest")

    restored: V5MetricWorkerArmResult | None = None
    for index, arm in enumerate(expected_order):
        try:
            core_meta_record = next(deterministic)
            latency_meta_record = next(observational)
        except StopIteration as error:
            raise ValueError("arm result or latency evidence is missing") from error
        core, core_sections = _read_arm_meta_record(
            core_meta_record,
            kind="arm_result_meta",
            arm=arm,
            schema=_ARM_RESULT_META_SCHEMA,
            section_names=("execution_artifacts", "row_evidence"),
            layer="deterministic",
        )
        observation, observation_sections = _read_arm_meta_record(
            latency_meta_record,
            kind="arm_latency_meta",
            arm=arm,
            schema=_ARM_LATENCY_META_SCHEMA,
            section_names=("samples",),
            layer="observational",
        )
        expected_digest = expected_core_digests[index]
        if (
            type(expected_digest) is not str
            or core.get("deterministic_result_sha256") != expected_digest
        ):
            raise ValueError("arm deterministic result index binding differs")
        _require_sha256(expected_digest, label=f"{arm} deterministic result digest")

        if arm == target_arm:
            core["execution_artifacts"] = _read_arm_section_records(
                deterministic,
                kind="arm_execution_artifacts",
                arm=arm,
                section="execution_artifacts",
                index=core_sections["execution_artifacts"],
                layer="deterministic",
            )
            core["row_evidence"] = _read_arm_section_records(
                deterministic,
                kind="arm_result_rows",
                arm=arm,
                section="row_evidence",
                index=core_sections["row_evidence"],
                layer="deterministic",
            )
            observation["samples"] = _read_arm_section_records(
                observational,
                kind="arm_latency_samples",
                arm=arm,
                section="samples",
                index=observation_sections["samples"],
                layer="observational",
            )
            result = _restore_arm_result(core=core, observation=observation)
            result_event_ids = [row.support.event_id for row in result.row_evidence]
            if (
                result.arm != arm
                or result.support_sha256 != support_sha256
                or result_event_ids != support_event_ids
            ):
                raise ValueError("metric worker arm support binding differs")
            restored = V5MetricWorkerArmResult(
                arm=arm,
                arm_index=index,
                deterministic_result_sha256=expected_digest,
                support_event_ids_sha256=_sha256(
                    _canonical_bytes(support_event_ids)
                ),
                result=result,
            )
            continue

        _skip_v5_metric_worker_arm_section(
            deterministic,
            kind="arm_execution_artifacts",
            arm=arm,
            record_count=core_sections["execution_artifacts"][1],
            layer="deterministic",
        )
        _skip_v5_metric_worker_arm_section(
            deterministic,
            kind="arm_result_rows",
            arm=arm,
            record_count=core_sections["row_evidence"][1],
            layer="deterministic",
        )
        _skip_v5_metric_worker_arm_section(
            observational,
            kind="arm_latency_samples",
            arm=arm,
            record_count=observation_sections["samples"][1],
            layer="observational",
        )

    for stream, label in (
        (deterministic, "deterministic"),
        (observational, "observational"),
    ):
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise ValueError(f"arm checkpoint has extra {label} records")
    if restored is None:
        raise ValueError("metric worker target arm was not restored")
    return restored


def load_v5_arm_checkpoint(
    *,
    checkpoint_root: Path,
    limits: V5KaggleResourceGates,
) -> tuple[V5EvaluationResult, ...]:
    """Recombine and validate deterministic arm results with aligned real latency."""
    deterministic = iter(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
    observational = iter(
        iter_v5_checkpoint_observational_records(output_root=checkpoint_root, limits=limits)
    )
    try:
        header_record = next(deterministic)
    except StopIteration as error:
        raise ValueError("arm checkpoint is empty") from error
    header = _read_json_record(header_record, label="arm header")
    expected_order = tuple(arm.value for arm in _STAGED_ARMS)
    if (
        set(header)
        != {
            "schema_version",
            "arm_order",
            "support_event_ids",
            "support_sha256",
            "deterministic_result_sha256",
        }
        or header_record.kind != "arm_header"
        or header_record.key != "arms"
        or header.get("schema_version") != _ARM_HEADER_SCHEMA
        or header.get("arm_order") != list(expected_order)
    ):
        raise ValueError("arm checkpoint header differs")
    expected_core_digests = header.get("deterministic_result_sha256")
    if not isinstance(expected_core_digests, list) or len(expected_core_digests) != len(
        expected_order
    ):
        raise ValueError("arm checkpoint deterministic result index differs")
    results: list[V5EvaluationResult] = []
    for index, arm in enumerate(expected_order):
        try:
            core_meta_record = next(deterministic)
            latency_meta_record = next(observational)
        except StopIteration as error:
            raise ValueError("arm result or latency evidence is missing") from error
        core, core_sections = _read_arm_meta_record(
            core_meta_record,
            kind="arm_result_meta",
            arm=arm,
            schema=_ARM_RESULT_META_SCHEMA,
            section_names=("execution_artifacts", "row_evidence"),
            layer="deterministic",
        )
        core["execution_artifacts"] = _read_arm_section_records(
            deterministic,
            kind="arm_execution_artifacts",
            arm=arm,
            section="execution_artifacts",
            index=core_sections["execution_artifacts"],
            layer="deterministic",
        )
        core["row_evidence"] = _read_arm_section_records(
            deterministic,
            kind="arm_result_rows",
            arm=arm,
            section="row_evidence",
            index=core_sections["row_evidence"],
            layer="deterministic",
        )
        observation, observation_sections = _read_arm_meta_record(
            latency_meta_record,
            kind="arm_latency_meta",
            arm=arm,
            schema=_ARM_LATENCY_META_SCHEMA,
            section_names=("samples",),
            layer="observational",
        )
        observation["samples"] = _read_arm_section_records(
            observational,
            kind="arm_latency_samples",
            arm=arm,
            section="samples",
            index=observation_sections["samples"],
            layer="observational",
        )
        if core.get("deterministic_result_sha256") != expected_core_digests[index]:
            raise ValueError("arm deterministic result index binding differs")
        results.append(_restore_arm_result(core=core, observation=observation))
    for stream, label in (
        (deterministic, "deterministic"),
        (observational, "observational"),
    ):
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise ValueError(f"arm checkpoint has extra {label} records")
    if (
        tuple(result.arm for result in results) != expected_order
        or len({result.support_sha256 for result in results}) != 1
        or len({tuple(row.support.event_id for row in result.row_evidence) for result in results})
        != 1
        or header.get("support_sha256") != results[0].support_sha256
        or header.get("support_event_ids")
        != [row.support.event_id for row in results[0].row_evidence]
    ):
        raise ValueError("arm checkpoint support differs across arms")
    return tuple(results)


def _stable_control_row_evidence(*, name: str, raw_json: str) -> dict[str, object]:
    try:
        raw_document = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise ValueError("control row evidence is not JSON") from error
    if not isinstance(raw_document, dict) or raw_json != json.dumps(raw_document, sort_keys=True):
        raise ValueError("control row evidence is not canonical")
    document = cast(dict[str, object], dict(raw_document))
    if name in _INVARIANCE_CONTROL_NAMES:
        if not {
            "before_score_sha256",
            "after_score_sha256",
        } <= set(document):
            raise ValueError("invariance control score evidence is incomplete")
        document.pop("before_score_sha256")
        document.pop("after_score_sha256")
        return document
    if name in _WORKLOAD_CONTROL_NAMES:
        if set(document) != {"arms", "full_score_sha256"}:
            raise ValueError("workload control evidence schema differs")
        raw_arms = document["arms"]
        if not isinstance(raw_arms, dict):
            raise ValueError("workload control arms are malformed")
        stable_arms: dict[str, object] = {}
        for arm, raw_arm in raw_arms.items():
            if type(arm) is not str or not isinstance(raw_arm, dict):
                raise ValueError("workload control arm evidence is malformed")
            arm_document = cast(dict[str, object], dict(raw_arm))
            raw_rows = arm_document.pop("rows", None)
            if arm_document.pop("score_sha256", None) is None or not isinstance(raw_rows, list):
                raise ValueError("workload control arm evidence is incomplete")
            stable_rows: list[dict[str, object]] = []
            for raw_row in raw_rows:
                if not isinstance(raw_row, dict) or set(raw_row) != {
                    "event_id",
                    "action",
                    "probability",
                    "latency_ms",
                }:
                    raise ValueError("workload control row evidence schema differs")
                row = cast(dict[str, object], dict(raw_row))
                row.pop("latency_ms")
                stable_rows.append(row)
            arm_document["rows"] = stable_rows
            stable_arms[arm] = arm_document
        return {"arms": stable_arms}
    if name == "label_shuffle":
        return document
    raise ValueError(f"unknown executed control: {name}")


def _stable_control_document(control: object) -> dict[str, object]:
    if not hasattr(control, "model_dump"):
        raise TypeError("stable control evidence requires a validated control")
    document = cast(dict[str, object], control.model_dump(mode="json"))
    name = document.get("name")
    raw_json = document.pop("row_evidence_json", None)
    document.pop("row_evidence_sha256", None)
    document.pop("control_sha256", None)
    raw_measurements = document.pop("measurements", None)
    if type(name) is not str or type(raw_json) is not str or not isinstance(raw_measurements, list):
        raise ValueError("executed control evidence is malformed")
    document["measurements"] = [
        item
        for item in raw_measurements
        if isinstance(item, dict) and not str(item.get("name", "")).endswith("p95_latency_ms")
    ]
    stable_rows = _stable_control_row_evidence(name=name, raw_json=raw_json)
    document["deterministic_row_evidence_sha256"] = _sha256(_canonical_bytes(stable_rows))
    document["deterministic_control_sha256"] = _sha256(_canonical_bytes(document))
    return document


def _control_group_core_and_observation(
    group: V5ExecutedControlGroup,
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a control group into deterministic semantics and real observations."""
    core = {
        "schema_version": "apar-sentinel-v5-control-group-core/1",
        "group": group.group,
        "run_mode": group.run_mode,
        "controls": [_stable_control_document(item) for item in group.controls],
        "evidence_protocol_sha256": group.evidence_protocol_sha256,
        "support_sha256": group.support_sha256,
        "implementation_sha256": group.implementation_sha256,
    }
    core["deterministic_group_sha256"] = _sha256(_canonical_bytes(core))
    observation = {
        "schema_version": "apar-sentinel-v5-control-group-observation/1",
        "group": group.group,
        "run_mode": group.run_mode,
        "executed_group": group.model_dump(mode="json"),
    }
    observation["observational_group_sha256"] = _sha256(_canonical_bytes(observation))
    return cast(dict[str, object], core), cast(dict[str, object], observation)


def _restore_control_group(
    *, core: dict[str, object], observation: dict[str, object]
) -> V5ExecutedControlGroup:
    claimed_observation = observation.get("observational_group_sha256")
    observation_document = dict(observation)
    observation_document.pop("observational_group_sha256", None)
    if (
        type(claimed_observation) is not str
        or _sha256(_canonical_bytes(observation_document)) != claimed_observation
    ):
        raise ValueError("observational control group digest differs")
    raw_group = observation.get("executed_group")
    if not isinstance(raw_group, dict):
        raise ValueError("observational control group is malformed")
    group = V5ExecutedControlGroup.model_validate(raw_group)
    rebuilt_core, rebuilt_observation = _control_group_core_and_observation(group)
    if rebuilt_core != core:
        raise ValueError("deterministic control group differs from observation")
    if rebuilt_observation != observation:
        raise ValueError("observational control group differs")
    return group


def _control_run_mode(mode: V5KaggleMode) -> V5RunMode:
    return (
        V5RunMode.SAFE_VALIDATION
        if mode is V5KaggleMode.CAPACITY_VALIDATION
        else V5RunMode.LOCKED_DEVELOPMENT
    )


def execute_v5_control_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    corpus_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]:
    """Execute the one control group assigned to this immutable stage."""
    group_name = _CONTROL_GROUP_BY_STAGE.get(capability.stage)
    if group_name is None:
        raise PermissionError("control capability stage differs")
    _validate_capability(capability, required_stage=capability.stage)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    corpus_manifest = read_v5_checkpoint_manifest(
        output_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    if (
        capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode)
        or corpus_manifest.stage is not V5KaggleStage.CORPUS
        or corpus_manifest.run_binding_sha256 != capability.run_binding_sha256
        or corpus_manifest.attempt_receipt_sha256 != capability.attempt_receipt_sha256
    ):
        raise PermissionError("control capability or corpus binding differs")
    corpus = load_v5_corpus_checkpoint(
        checkpoint_root=corpus_checkpoint_root,
        limits=protocol.resources,
    )
    development, _profile = _development_protocol_for_stage(
        root=root,
        protocol=protocol,
        mode=capability.mode,
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path,
        root=root,
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.source_bindings.feature_catalog_path
    )
    configuration = load_v5_arm_configuration(
        root / protocol.source_bindings.arm_protocol_path,
        catalog=catalog,
        protocol=development,
    )
    group = execute_v5_control_group(
        group=group_name,
        protocol=development,
        evidence_protocol=evidence_protocol,
        corpus=corpus,
        catalog=catalog,
        configuration=configuration,
        mode=_control_run_mode(capability.mode),
    )
    core, observation = _control_group_core_and_observation(group)
    yield V5CheckpointInput(
        kind="control_group",
        key=group.group,
        canonical_bytes=_canonical_bytes(core),
    )
    yield V5CheckpointInput(
        kind="control_observation",
        key=group.group,
        canonical_bytes=_canonical_bytes(observation),
        layer="observational",
    )


def load_v5_control_group_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> V5ExecutedControlGroup:
    """Load one exact group and independently rebind its observational evidence."""
    deterministic = tuple(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
    observational = tuple(
        iter_v5_checkpoint_observational_records(output_root=checkpoint_root, limits=limits)
    )
    if len(deterministic) != 1 or len(observational) != 1:
        raise ValueError("control checkpoint requires one record in each layer")
    core_record = deterministic[0]
    observation_record = observational[0]
    if (
        core_record.kind != "control_group"
        or observation_record.kind != "control_observation"
        or core_record.key != observation_record.key
    ):
        raise ValueError("control checkpoint record order differs")
    core = _read_json_record(core_record, label="deterministic control group")
    observation = _read_json_record(observation_record, label="observational control group")
    group = _restore_control_group(core=core, observation=observation)
    if group.group.value != core_record.key:
        raise ValueError("control checkpoint group key differs")
    return group


def build_v5_metric_stage_evidence(
    *,
    arm_results: tuple[V5EvaluationResult, ...],
    control_groups: tuple[V5ExecutedControlGroup, ...],
    evidence_protocol: V5EvidenceProtocol,
) -> V5MetricStageEvidence:
    """Compute frozen complete metrics and readiness from staged evidence."""
    if tuple(result.arm for result in arm_results) != tuple(arm.value for arm in _STAGED_ARMS):
        raise ValueError("metric stage requires exact ordered four arm results")
    support_orders = {
        tuple(row.support.event_id for row in result.row_evidence) for result in arm_results
    }
    if len(support_orders) != 1:
        raise ValueError("metric stage arm support order differs")
    controls = assemble_v5_control_suite(control_groups)
    complete = tuple(
        evaluate_v5_complete_result(result=result, protocol=evidence_protocol)
        for result in arm_results
    )
    if len({item.support_sha256 for item in complete}) != 1:
        raise ValueError("metric stage complete-metric support differs")
    readiness = build_v5_readiness_evidence(
        metrics=complete[_STAGED_ARMS.index(V5Arm.FULL_SENTINEL)],
        controls=controls,
    )
    return V5MetricStageEvidence(
        complete_metrics=complete,
        controls=controls,
        readiness=readiness,
    )


def _metric_stage_core_and_observation(
    *,
    evidence: V5MetricStageEvidence,
    arm_results: tuple[V5EvaluationResult, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    deterministic_result_sha256 = tuple(
        cast(str, _arm_core_and_observation(result)[0]["deterministic_result_sha256"])
        for result in arm_results
    )
    complete_documents = tuple(item.model_dump(mode="json") for item in evidence.complete_metrics)
    controls_document = evidence.controls.model_dump(mode="json")
    readiness_document = evidence.readiness.model_dump(mode="json")
    stable_controls, control_digests = _stable_controls(controls_document)
    core: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-core/1",
        "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
        "deterministic_result_sha256": deterministic_result_sha256,
        "complete_metrics": tuple(
            _stable_complete_metrics(
                document,
                deterministic_result_sha256=result_digest,
            )
            for document, result_digest in zip(
                complete_documents, deterministic_result_sha256, strict=True
            )
        ),
        "controls": stable_controls,
        "readiness": _stable_readiness(
            readiness_document,
            deterministic_control_digests=control_digests,
        ),
    }
    core["deterministic_metric_stage_sha256"] = _sha256(_canonical_bytes(core))
    observation: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-observation/1",
        "deterministic_metric_stage_sha256": core["deterministic_metric_stage_sha256"],
        "complete_metrics": complete_documents,
        "controls": controls_document,
        "readiness": readiness_document,
    }
    observation["observational_metric_stage_sha256"] = _sha256(_canonical_bytes(observation))
    return cast(dict[str, object], json.loads(_canonical_bytes(core))), cast(
        dict[str, object], json.loads(_canonical_bytes(observation))
    )


def _validate_v5_metric_worker_receipt_set(
    receipts: tuple[V5MetricArmWorkerReceipt, ...],
) -> None:
    expected_order = tuple(arm.value for arm in _STAGED_ARMS)
    if (
        len(receipts) != len(expected_order)
        or tuple(receipt.arm.value for receipt in receipts) != expected_order
        or tuple(receipt.arm_index for receipt in receipts)
        != tuple(range(len(expected_order)))
        or any(
            tuple(arm.value for arm in receipt.arm_order) != expected_order
            for receipt in receipts
        )
    ):
        raise ValueError("metric worker receipt order differs")
    common_fields = (
        "mode",
        "run_binding_sha256",
        "attempt_receipt_sha256",
        "execution_manifest_sha256",
        "arm_manifest_sha256",
        "arm_manifest_deterministic_sha256",
        "support_sha256",
        "support_count",
        "support_event_ids_sha256",
        "evidence_protocol_sha256",
        "implementation_sha256",
    )
    for field in common_fields:
        if len({getattr(receipt, field) for receipt in receipts}) != 1:
            raise ValueError(f"metric worker receipt {field} differs across arms")
    if tuple(
        receipt.metric_summary.deterministic_result_sha256 for receipt in receipts
    ) != tuple(receipt.deterministic_result_sha256 for receipt in receipts):
        raise ValueError("metric worker deterministic result order differs")


def build_v5_compact_metric_stage_documents(
    *,
    receipts: tuple[V5MetricArmWorkerReceipt, ...],
    controls: V5ExecutedControlSuite,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the unchanged Stage 70 core from compact authenticated arm summaries."""
    _validate_v5_metric_worker_receipt_set(receipts)
    readiness = build_v5_readiness_from_metric_summary(
        metrics=receipts[_STAGED_ARMS.index(V5Arm.FULL_SENTINEL)].metric_summary,
        controls=controls,
    )
    controls_document = controls.model_dump(mode="json")
    readiness_document = readiness.model_dump(mode="json")
    stable_controls, control_digests = _stable_controls(controls_document)
    core: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-core/1",
        "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
        "deterministic_result_sha256": tuple(
            receipt.deterministic_result_sha256 for receipt in receipts
        ),
        "complete_metrics": tuple(
            receipt.metric_summary.stable_document() for receipt in receipts
        ),
        "controls": stable_controls,
        "readiness": _stable_readiness(
            readiness_document,
            deterministic_control_digests=control_digests,
        ),
    }
    core["deterministic_metric_stage_sha256"] = _sha256(_canonical_bytes(core))
    observation: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-observation/2",
        "deterministic_metric_stage_sha256": core[
            "deterministic_metric_stage_sha256"
        ],
        "worker_receipts": tuple(
            receipt.model_dump(mode="json") for receipt in receipts
        ),
        "controls": controls_document,
        "readiness": readiness_document,
    }
    observation["observational_metric_stage_sha256"] = _sha256(
        _canonical_bytes(observation)
    )
    return cast(dict[str, object], json.loads(_canonical_bytes(core))), cast(
        dict[str, object],
        json.loads(_canonical_bytes(observation)),
    )


def _restore_v5_compact_metric_stage_evidence(
    *,
    core: dict[str, object],
    observation: dict[str, object],
) -> V5CompactMetricStageEvidence:
    """Rebind a compact Stage 70 observation to its unchanged deterministic core."""
    claimed_core = core.get("deterministic_metric_stage_sha256")
    core_document = dict(core)
    core_document.pop("deterministic_metric_stage_sha256", None)
    if (
        type(claimed_core) is not str
        or _sha256(_canonical_bytes(core_document)) != claimed_core
    ):
        raise ValueError("deterministic compact metric stage digest differs")
    claimed_observation = observation.get("observational_metric_stage_sha256")
    observation_document = dict(observation)
    observation_document.pop("observational_metric_stage_sha256", None)
    if (
        type(claimed_observation) is not str
        or _sha256(_canonical_bytes(observation_document)) != claimed_observation
    ):
        raise ValueError("observational compact metric stage digest differs")
    raw_receipts = observation.get("worker_receipts")
    raw_controls = observation.get("controls")
    raw_readiness = observation.get("readiness")
    if (
        observation.get("schema_version")
        != "apar-sentinel-v5-kaggle-metric-observation/2"
        or observation.get("deterministic_metric_stage_sha256") != claimed_core
        or not isinstance(raw_receipts, list)
        or not isinstance(raw_controls, dict)
        or not isinstance(raw_readiness, dict)
    ):
        raise ValueError("compact metric stage observation is malformed")
    receipts = tuple(
        V5MetricArmWorkerReceipt.model_validate(item) for item in raw_receipts
    )
    controls = V5ExecutedControlSuite.model_validate(raw_controls)
    readiness = V5ReadinessEvidence.model_validate(raw_readiness)
    rebuilt_core, rebuilt_observation = build_v5_compact_metric_stage_documents(
        receipts=receipts,
        controls=controls,
    )
    if rebuilt_core != core or rebuilt_observation != observation:
        raise ValueError("compact metric stage differs from authenticated receipts")
    return V5CompactMetricStageEvidence(
        worker_receipts=receipts,
        controls=controls,
        readiness=readiness,
    )


def _restore_metric_stage_evidence(
    *, core: dict[str, object], observation: dict[str, object]
) -> V5MetricStageEvidence:
    claimed_core = core.get("deterministic_metric_stage_sha256")
    core_document = dict(core)
    core_document.pop("deterministic_metric_stage_sha256", None)
    if type(claimed_core) is not str or _sha256(_canonical_bytes(core_document)) != claimed_core:
        raise ValueError("deterministic metric stage digest differs")
    claimed_observation = observation.get("observational_metric_stage_sha256")
    observation_document = dict(observation)
    observation_document.pop("observational_metric_stage_sha256", None)
    if (
        type(claimed_observation) is not str
        or _sha256(_canonical_bytes(observation_document)) != claimed_observation
    ):
        raise ValueError("observational metric stage digest differs")
    if (
        observation.get("schema_version") != "apar-sentinel-v5-kaggle-metric-observation/1"
        or observation.get("deterministic_metric_stage_sha256") != claimed_core
    ):
        raise ValueError("metric stage layer binding differs")
    raw_metrics = observation.get("complete_metrics")
    raw_controls = observation.get("controls")
    raw_readiness = observation.get("readiness")
    if (
        not isinstance(raw_metrics, list)
        or not isinstance(raw_controls, dict)
        or not isinstance(raw_readiness, dict)
    ):
        raise ValueError("metric stage observation is malformed")
    evidence = V5MetricStageEvidence(
        complete_metrics=tuple(V5CompleteArmMetrics.model_validate(item) for item in raw_metrics),
        controls=V5ExecutedControlSuite.model_validate(raw_controls),
        readiness=V5ReadinessEvidence.model_validate(raw_readiness),
    )
    raw_result_digests = core.get("deterministic_result_sha256")
    if not isinstance(raw_result_digests, list) or len(raw_result_digests) != len(
        evidence.complete_metrics
    ):
        raise ValueError("metric stage deterministic arm binding differs")
    stable_controls, control_digests = _stable_controls(evidence.controls.model_dump(mode="json"))
    rebuilt: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-core/1",
        "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
        "deterministic_result_sha256": tuple(raw_result_digests),
        "complete_metrics": tuple(
            _stable_complete_metrics(
                item.model_dump(mode="json"),
                deterministic_result_sha256=result_digest,
            )
            for item, result_digest in zip(
                evidence.complete_metrics, raw_result_digests, strict=True
            )
        ),
        "controls": stable_controls,
        "readiness": _stable_readiness(
            evidence.readiness.model_dump(mode="json"),
            deterministic_control_digests=control_digests,
        ),
    }
    rebuilt["deterministic_metric_stage_sha256"] = _sha256(_canonical_bytes(rebuilt))
    if json.loads(_canonical_bytes(rebuilt)) != core:
        raise ValueError("deterministic metric stage differs from observation")
    return evidence


def execute_v5_metric_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    corpus_checkpoint_root: Path,
    arm_checkpoint_root: Path,
    control_checkpoint_roots: Sequence[Path],
) -> Iterator[V5CheckpointInput]:
    """Build complete metrics from exact arm and six control checkpoints."""
    _validate_capability(capability, required_stage=V5KaggleStage.METRICS)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    corpus_manifest = read_v5_checkpoint_manifest(
        output_root=corpus_checkpoint_root, limits=protocol.resources
    )
    arm_manifest = read_v5_checkpoint_manifest(
        output_root=arm_checkpoint_root, limits=protocol.resources
    )
    control_roots = tuple(control_checkpoint_roots)
    if len(control_roots) != 6:
        raise ValueError("metric stage requires exact six control checkpoints")
    control_manifests = tuple(
        read_v5_checkpoint_manifest(output_root=path, limits=protocol.resources)
        for path in control_roots
    )
    expected_control_stages = (
        V5KaggleStage.LABEL_SHUFFLE,
        V5KaggleStage.IDENTITY_RENAME,
        V5KaggleStage.FUTURE_CAUSALITY,
        V5KaggleStage.EQUAL_TIME_ISOLATION,
        V5KaggleStage.FEATURE_LEAKAGE,
        V5KaggleStage.SINGLE_CLASS_CONTROLS,
    )
    manifests = (corpus_manifest, arm_manifest, *control_manifests)
    if (
        capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode)
        or capability.predecessor_manifest_sha256 != control_manifests[-1].manifest_sha256
        or corpus_manifest.stage is not V5KaggleStage.CORPUS
        or arm_manifest.stage is not V5KaggleStage.ARMS
        or tuple(item.stage for item in control_manifests) != expected_control_stages
        or control_manifests[0].predecessor_manifest_sha256 != arm_manifest.manifest_sha256
        or any(
            current.predecessor_manifest_sha256 != previous.manifest_sha256
            for previous, current in pairwise(control_manifests)
        )
        or any(
            item.run_binding_sha256 != capability.run_binding_sha256
            or item.attempt_receipt_sha256 != capability.attempt_receipt_sha256
            for item in manifests
        )
    ):
        raise PermissionError("metric checkpoint lineage differs")
    control_groups = tuple(
        load_v5_control_group_checkpoint(checkpoint_root=path, limits=protocol.resources)
        for path in control_roots
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path, root=root
    )
    core, observation = _execute_v5_memory_safe_metric_stage(
        root=root,
        capability=capability,
        arm_checkpoint_root=arm_checkpoint_root,
        arm_manifest=arm_manifest,
        control_groups=control_groups,
        evidence_protocol=evidence_protocol,
        limits=protocol.resources,
    )
    yield V5CheckpointInput(
        kind="metric_evidence",
        key="complete",
        canonical_bytes=_canonical_bytes(core),
    )
    yield V5CheckpointInput(
        kind="metric_observation",
        key="complete",
        canonical_bytes=_canonical_bytes(observation),
        layer="observational",
    )


def _run_v5_metric_arm_worker_subprocess(
    *,
    root: Path,
    capability: V5StageCapability,
    arm_checkpoint_root: Path,
    arm_manifest: V5CheckpointManifest,
    target_arm: V5Arm,
    evidence_protocol: V5EvidenceProtocol,
    limits: V5KaggleResourceGates,
    timeout_seconds: float,
) -> V5MetricArmWorkerReceipt:
    """Run one candidate metric arm through an exclusive file-only child channel."""
    if timeout_seconds <= 0.0:
        raise TimeoutError("metric worker stage deadline expired")
    with tempfile.TemporaryDirectory(prefix="apar-v5-stage70-") as temporary:
        receipt_path = Path(temporary) / "metric-arm-receipt.json"
        command = [
            sys.executable,
            str(root / "scripts/run_defense_v5_kaggle_stage.py"),
            "--internal-metric-worker",
            "--root",
            str(root),
            "--mode",
            capability.mode.value,
            "--run-binding-sha256",
            capability.run_binding_sha256,
            "--attempt-receipt-sha256",
            capability.attempt_receipt_sha256,
            "--execution-manifest-sha256",
            capability.execution_manifest_sha256,
            "--arm-checkpoint-root",
            str(arm_checkpoint_root),
            "--arm-manifest-sha256",
            arm_manifest.manifest_sha256,
            "--arm-manifest-deterministic-sha256",
            arm_manifest.deterministic_sha256,
            "--target-arm",
            target_arm.value,
            "--receipt-path",
            str(receipt_path),
            "--max-address-space-bytes",
            str(limits.max_peak_rss_bytes),
        ]
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("metric worker stage deadline expired") from error
        if completed.returncode != 0 or completed.stdout or completed.stderr:
            raise RuntimeError(
                "metric worker failed closed: "
                f"exit={completed.returncode} "
                f"stdout_bytes={len(completed.stdout)} "
                f"stdout_sha256={_sha256(completed.stdout)} "
                f"stderr_bytes={len(completed.stderr)} "
                f"stderr_sha256={_sha256(completed.stderr)}"
            )
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("metric worker receipt file is missing or linked")
        metadata = receipt_path.stat()
        max_artifact_bytes = min(limits.max_stage_output_bytes, 256 * 1024**2)
        if (
            metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size >= max_artifact_bytes
        ):
            raise ValueError("metric worker receipt file exceeds resource gate")
        raw_receipt = receipt_path.read_bytes()
        try:
            raw_document = json.loads(raw_receipt)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metric worker receipt is not JSON") from error
        if raw_receipt != _canonical_bytes(raw_document):
            raise ValueError("metric worker receipt is not canonical")
        receipt = V5MetricArmWorkerReceipt.model_validate(raw_document)
        validate_v5_metric_arm_worker_receipt(
            receipt=receipt,
            expected={
                "mode": capability.mode,
                "run_binding_sha256": capability.run_binding_sha256,
                "attempt_receipt_sha256": capability.attempt_receipt_sha256,
                "execution_manifest_sha256": capability.execution_manifest_sha256,
                "arm_manifest_sha256": arm_manifest.manifest_sha256,
                "arm_manifest_deterministic_sha256": (
                    arm_manifest.deterministic_sha256
                ),
                "arm": target_arm,
                "arm_index": _STAGED_ARMS.index(target_arm),
                "arm_order": tuple(arm.value for arm in _STAGED_ARMS),
                "support_sha256": receipt.support_sha256,
                "support_count": receipt.support_count,
                "support_event_ids_sha256": receipt.support_event_ids_sha256,
                "deterministic_result_sha256": (
                    receipt.deterministic_result_sha256
                ),
                "evidence_protocol_sha256": (
                    evidence_protocol.evidence_protocol_sha256
                ),
                "implementation_sha256": (
                    evidence_protocol.implementation_sha256
                ),
            },
            max_peak_rss_bytes=limits.max_peak_rss_bytes,
            max_artifact_bytes=max_artifact_bytes,
        )
        return receipt


def _execute_v5_memory_safe_metric_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    arm_checkpoint_root: Path,
    arm_manifest: V5CheckpointManifest,
    control_groups: tuple[V5ExecutedControlGroup, ...],
    evidence_protocol: V5EvidenceProtocol,
    limits: V5KaggleResourceGates,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute the source-bound Stage 70 worker protocol without retaining all arms."""
    deadline = time.monotonic() + limits.max_stage_seconds
    receipts: list[V5MetricArmWorkerReceipt] = []
    for arm in _STAGED_ARMS:
        receipts.append(
            _run_v5_metric_arm_worker_subprocess(
                root=root,
                capability=capability,
                arm_checkpoint_root=arm_checkpoint_root,
                arm_manifest=arm_manifest,
                target_arm=arm,
                evidence_protocol=evidence_protocol,
                limits=limits,
                timeout_seconds=max(deadline - time.monotonic(), 1e-9),
            )
        )
    frozen_receipts = tuple(receipts)
    _validate_v5_metric_worker_receipt_set(frozen_receipts)
    controls = assemble_v5_control_suite(control_groups)
    return build_v5_compact_metric_stage_documents(
        receipts=frozen_receipts,
        controls=controls,
    )


def load_v5_metric_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> V5MetricStageEvidence | V5CompactMetricStageEvidence:
    """Load and rebind complete metrics with exact observational evidence."""
    deterministic = tuple(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
    observational = tuple(
        iter_v5_checkpoint_observational_records(output_root=checkpoint_root, limits=limits)
    )
    if len(deterministic) != 1 or len(observational) != 1:
        raise ValueError("metric checkpoint requires one record in each layer")
    if (
        deterministic[0].kind != "metric_evidence"
        or deterministic[0].key != "complete"
        or observational[0].kind != "metric_observation"
        or observational[0].key != "complete"
    ):
        raise ValueError("metric checkpoint record identity differs")
    core = _read_json_record(deterministic[0], label="deterministic metric stage")
    observation = _read_json_record(
        observational[0],
        label="observational metric stage",
    )
    if (
        observation.get("schema_version")
        == "apar-sentinel-v5-kaggle-metric-observation/2"
    ):
        compact = _restore_v5_compact_metric_stage_evidence(
            core=core,
            observation=observation,
        )
        for receipt in compact.worker_receipts:
            validate_v5_metric_arm_worker_receipt(
                receipt=receipt,
                expected={
                    "mode": receipt.mode,
                    "run_binding_sha256": receipt.run_binding_sha256,
                    "attempt_receipt_sha256": receipt.attempt_receipt_sha256,
                    "execution_manifest_sha256": (
                        receipt.execution_manifest_sha256
                    ),
                    "arm_manifest_sha256": receipt.arm_manifest_sha256,
                    "arm_manifest_deterministic_sha256": (
                        receipt.arm_manifest_deterministic_sha256
                    ),
                    "arm": receipt.arm,
                    "arm_index": receipt.arm_index,
                    "arm_order": tuple(arm.value for arm in receipt.arm_order),
                    "support_sha256": receipt.support_sha256,
                    "support_count": receipt.support_count,
                    "support_event_ids_sha256": (
                        receipt.support_event_ids_sha256
                    ),
                    "deterministic_result_sha256": (
                        receipt.deterministic_result_sha256
                    ),
                    "evidence_protocol_sha256": (
                        receipt.evidence_protocol_sha256
                    ),
                    "implementation_sha256": receipt.implementation_sha256,
                },
                max_peak_rss_bytes=limits.max_peak_rss_bytes,
                max_artifact_bytes=min(
                    limits.max_stage_output_bytes,
                    256 * 1024**2,
                ),
            )
        return compact
    return _restore_metric_stage_evidence(core=core, observation=observation)


def _build_checkpoint_chain_binding(
    manifests: Sequence[V5CheckpointManifest],
) -> V5CheckpointChainBinding:
    frozen = tuple(manifests)
    expected = tuple(V5KaggleStage)[:-1]
    if tuple(item.stage for item in frozen) != expected:
        raise ValueError("checkpoint chain requires exact Stage 00-70 manifests")
    if any(
        current.predecessor_manifest_sha256 != previous.manifest_sha256
        for previous, current in pairwise(frozen)
    ):
        raise ValueError("checkpoint chain predecessor linkage differs")
    run_bindings = {item.run_binding_sha256 for item in frozen}
    attempts = {item.attempt_receipt_sha256 for item in frozen}
    if len(run_bindings) != 1 or len(attempts) != 1:
        raise ValueError("checkpoint chain run or attempt binding differs")
    values = {
        "schema_version": "apar-sentinel-v5-checkpoint-chain/1",
        "attempt_receipt_sha256": frozen[0].attempt_receipt_sha256,
        "predecessor_stage_manifest_sha256": tuple(
            (item.stage.value, item.manifest_sha256) for item in frozen
        ),
    }
    values["predecessor_chain_root_sha256"] = _sha256(_canonical_bytes(values))
    return V5CheckpointChainBinding.model_validate(values)


def _build_v5_compact_final_documents(
    *,
    capability: V5StageCapability,
    chain: V5CheckpointChainBinding,
    arm_manifest: V5CheckpointManifest,
    metric_manifest: V5CheckpointManifest,
    metric_evidence: V5CompactMetricStageEvidence,
    support_plan: V5KaggleSupportPlan,
    evidence_protocol: V5EvidenceProtocol,
    catalog_sha256: str,
) -> tuple[dict[str, object], bytes]:
    """Build a bounded Stage 80 index without restoring any Stage 30 arm."""
    receipts = metric_evidence.worker_receipts
    _validate_v5_metric_worker_receipt_set(receipts)
    for receipt in receipts:
        if (
            receipt.mode is not capability.mode
            or receipt.run_binding_sha256 != capability.run_binding_sha256
            or receipt.attempt_receipt_sha256 != capability.attempt_receipt_sha256
            or receipt.execution_manifest_sha256
            != capability.execution_manifest_sha256
            or receipt.arm_manifest_sha256 != arm_manifest.manifest_sha256
            or receipt.arm_manifest_deterministic_sha256
            != arm_manifest.deterministic_sha256
            or receipt.evidence_protocol_sha256
            != evidence_protocol.evidence_protocol_sha256
            or receipt.implementation_sha256
            != evidence_protocol.implementation_sha256
        ):
            raise ValueError("compact final worker lineage differs")
    metric_core, _metric_observation = build_v5_compact_metric_stage_documents(
        receipts=receipts,
        controls=metric_evidence.controls,
    )
    deterministic_metric_stage_sha256 = metric_core.get(
        "deterministic_metric_stage_sha256"
    )
    if type(deterministic_metric_stage_sha256) is not str:
        raise ValueError("compact final metric digest is missing")
    core: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-final-core/2",
        "mode": capability.mode.value,
        "run_binding_sha256": capability.run_binding_sha256,
        "support_plan_sha256": support_plan.support_plan_sha256,
        "checkpoint_chain_root_sha256": chain.predecessor_chain_root_sha256,
        "arm_manifest_deterministic_sha256": arm_manifest.deterministic_sha256,
        "metric_manifest_deterministic_sha256": metric_manifest.deterministic_sha256,
        "deterministic_metric_stage_sha256": deterministic_metric_stage_sha256,
        "evidence_protocol_sha256": evidence_protocol.evidence_protocol_sha256,
        "implementation_sha256": evidence_protocol.implementation_sha256,
        "catalog_sha256": catalog_sha256,
    }
    core["final_core_sha256"] = _sha256(_canonical_bytes(core))
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-compact-final-payload/2",
        "mode": capability.mode.value,
        "profile": "production",
        "development_test_seed": 404
        if capability.mode is V5KaggleMode.CAPACITY_VALIDATION
        else 2404,
        "run_binding_sha256": capability.run_binding_sha256,
        "support_plan": support_plan.model_dump(mode="json"),
        "attempt_receipt_sha256": capability.attempt_receipt_sha256,
        "checkpoint_chain": chain.model_dump(mode="json"),
        "evidence_protocol_sha256": evidence_protocol.evidence_protocol_sha256,
        "implementation_sha256": evidence_protocol.implementation_sha256,
        "catalog_sha256": catalog_sha256,
        "arm_manifest_sha256": arm_manifest.manifest_sha256,
        "arm_manifest_deterministic_sha256": arm_manifest.deterministic_sha256,
        "metric_manifest_sha256": metric_manifest.manifest_sha256,
        "metric_manifest_deterministic_sha256": metric_manifest.deterministic_sha256,
        "deterministic_metric_stage_sha256": deterministic_metric_stage_sha256,
        "metric_worker_receipt_sha256": tuple(
            receipt.receipt_sha256 for receipt in receipts
        ),
        "worker_resources": tuple(
            receipt.resource_telemetry.model_dump(mode="json")
            for receipt in receipts
        ),
        "controls_suite_sha256": metric_evidence.controls.suite_sha256,
        "readiness": metric_evidence.readiness.model_dump(mode="json"),
        "final_core_sha256": core["final_core_sha256"],
    }
    values["payload_sha256"] = _sha256(_canonical_bytes(values))
    payload = V5CompactFinalPayload.model_validate(values)
    return cast(dict[str, object], json.loads(_canonical_bytes(core))), _canonical_bytes(
        payload.model_dump(mode="json")
    )


def execute_v5_finalize_stage(
    *,
    root: Path,
    capability: V5StageCapability,
    predecessor_checkpoint_roots: Sequence[Path],
) -> Iterator[V5CheckpointInput]:
    """Recompute Stage-70 claims and emit the complete chained evidence payload."""
    _validate_capability(capability, required_stage=V5KaggleStage.FINALIZE)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    roots = tuple(predecessor_checkpoint_roots)
    predecessor_stages = tuple(V5KaggleStage)[:-1]
    if len(roots) != len(predecessor_stages):
        raise ValueError("finalization requires exact Stage 00-70 checkpoint roots")
    manifests = tuple(
        read_v5_checkpoint_manifest(output_root=path, limits=protocol.resources) for path in roots
    )
    chain = _build_checkpoint_chain_binding(manifests)
    if (
        capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode)
        or capability.predecessor_manifest_sha256 != manifests[-1].manifest_sha256
        or chain.attempt_receipt_sha256 != capability.attempt_receipt_sha256
    ):
        raise PermissionError("finalization capability or chain differs")
    roots_by_stage = dict(zip(predecessor_stages, roots, strict=True))
    manifests_by_stage = dict(zip(predecessor_stages, manifests, strict=True))
    retained_metrics = load_v5_metric_checkpoint(
        checkpoint_root=roots_by_stage[V5KaggleStage.METRICS],
        limits=protocol.resources,
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path,
        root=root,
    )
    support_plan = build_v5_kaggle_support_plan(
        root=root,
        protocol=protocol,
        mode=capability.mode,
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.source_bindings.feature_catalog_path
    )
    if isinstance(retained_metrics, V5CompactMetricStageEvidence):
        compact_core, compact_payload = _build_v5_compact_final_documents(
            capability=capability,
            chain=chain,
            arm_manifest=manifests_by_stage[V5KaggleStage.ARMS],
            metric_manifest=manifests_by_stage[V5KaggleStage.METRICS],
            metric_evidence=retained_metrics,
            support_plan=support_plan,
            evidence_protocol=evidence_protocol,
            catalog_sha256=catalog.catalog_sha256,
        )
        yield V5CheckpointInput(
            kind="final_core",
            key="complete",
            canonical_bytes=_canonical_bytes(compact_core),
        )
        yield V5CheckpointInput(
            kind="final_payload",
            key="complete",
            canonical_bytes=compact_payload,
            layer="observational",
        )
        return
    arm_results = load_v5_arm_checkpoint(
        checkpoint_root=roots_by_stage[V5KaggleStage.ARMS], limits=protocol.resources
    )
    control_groups = tuple(
        load_v5_control_group_checkpoint(
            checkpoint_root=roots_by_stage[stage], limits=protocol.resources
        )
        for stage in (
            V5KaggleStage.LABEL_SHUFFLE,
            V5KaggleStage.IDENTITY_RENAME,
            V5KaggleStage.FUTURE_CAUSALITY,
            V5KaggleStage.EQUAL_TIME_ISOLATION,
            V5KaggleStage.FEATURE_LEAKAGE,
            V5KaggleStage.SINGLE_CLASS_CONTROLS,
        )
    )
    recomputed = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=control_groups,
        evidence_protocol=evidence_protocol,
    )
    if recomputed != retained_metrics:
        raise ValueError("finalization recomputation differs from Stage 70 evidence")
    payload_bytes = build_v5_staged_evidence_payload(
        mode=capability.mode,
        run_binding_sha256=capability.run_binding_sha256,
        support_plan=support_plan,
        chain=chain,
        evidence_protocol=evidence_protocol,
        catalog_sha256=catalog.catalog_sha256,
        arm_results=arm_results,
        controls=recomputed.controls,
    )
    payload = V5StagedEvidencePayload.model_validate_json(payload_bytes)
    core = {
        "schema_version": "apar-sentinel-v5-kaggle-final-core/1",
        "mode": capability.mode,
        "run_binding_sha256": capability.run_binding_sha256,
        "support_plan_sha256": support_plan.support_plan_sha256,
        "deterministic_core_sha256": payload.deterministic_core.core_sha256,
    }
    core["final_core_sha256"] = _sha256(_canonical_bytes(core))
    yield V5CheckpointInput(
        kind="final_core",
        key="complete",
        canonical_bytes=_canonical_bytes(core),
    )
    yield V5CheckpointInput(
        kind="final_payload",
        key="complete",
        canonical_bytes=payload_bytes,
        layer="observational",
    )


def load_v5_final_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> V5StagedEvidencePayload | V5CompactFinalPayload:
    """Load the final payload and bind it back to its deterministic core record."""
    deterministic = tuple(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
    observational = tuple(
        iter_v5_checkpoint_observational_records(output_root=checkpoint_root, limits=limits)
    )
    if len(deterministic) != 1 or len(observational) != 1:
        raise ValueError("final checkpoint requires one record in each layer")
    if (
        deterministic[0].kind != "final_core"
        or observational[0].kind != "final_payload"
        or deterministic[0].key != "complete"
        or observational[0].key != "complete"
    ):
        raise ValueError("final checkpoint record identity differs")
    core = _read_json_record(deterministic[0], label="final deterministic core")
    claimed = core.get("final_core_sha256")
    core_document = dict(core)
    core_document.pop("final_core_sha256", None)
    if type(claimed) is not str or _sha256(_canonical_bytes(core_document)) != claimed:
        raise ValueError("final deterministic core digest differs")
    if core.get("schema_version") == "apar-sentinel-v5-kaggle-final-core/2":
        compact_payload = V5CompactFinalPayload.model_validate_json(
            observational[0].canonical_bytes
        )
        if (
            core.get("mode") != compact_payload.mode.value
            or core.get("run_binding_sha256") != compact_payload.run_binding_sha256
            or core.get("support_plan_sha256")
            != compact_payload.support_plan.support_plan_sha256
            or core.get("checkpoint_chain_root_sha256")
            != compact_payload.checkpoint_chain.predecessor_chain_root_sha256
            or core.get("arm_manifest_deterministic_sha256")
            != compact_payload.arm_manifest_deterministic_sha256
            or core.get("metric_manifest_deterministic_sha256")
            != compact_payload.metric_manifest_deterministic_sha256
            or core.get("deterministic_metric_stage_sha256")
            != compact_payload.deterministic_metric_stage_sha256
            or core.get("evidence_protocol_sha256")
            != compact_payload.evidence_protocol_sha256
            or core.get("implementation_sha256")
            != compact_payload.implementation_sha256
            or core.get("catalog_sha256") != compact_payload.catalog_sha256
            or claimed != compact_payload.final_core_sha256
        ):
            raise ValueError("compact final deterministic/payload binding differs")
        return compact_payload
    legacy_payload = V5StagedEvidencePayload.model_validate_json(
        observational[0].canonical_bytes
    )
    if (
        core.get("mode") != legacy_payload.mode.value
        or core.get("run_binding_sha256") != legacy_payload.run_binding_sha256
        or core.get("support_plan_sha256")
        != legacy_payload.support_plan.support_plan_sha256
        or core.get("deterministic_core_sha256")
        != legacy_payload.deterministic_core.core_sha256
    ):
        raise ValueError("final deterministic/payload binding differs")
    return legacy_payload


def execute_v5_authorization_stage(
    *, root: Path, capability: V5StageCapability
) -> Iterator[V5CheckpointInput]:
    """Emit the non-executing Stage-00 authority and support binding."""
    _validate_capability(capability, required_stage=V5KaggleStage.AUTHORIZE)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    if capability.run_binding_sha256 != protocol.run_binding_sha256(capability.mode):
        raise PermissionError("staged execution capability run binding differs")
    selected = (
        protocol.capacity
        if capability.mode is V5KaggleMode.CAPACITY_VALIDATION
        else protocol.locked
    )
    support_plan = build_v5_kaggle_support_plan(
        root=root,
        protocol=protocol,
        mode=capability.mode,
    )
    document = {
        "schema_version": "apar-sentinel-v5-kaggle-authorization/1",
        "stage": V5KaggleStage.AUTHORIZE,
        "mode": capability.mode,
        "profile": selected.profile,
        "development_test_seed": selected.development_test_seed,
        "repeatable": selected.repeatable,
        "authorization_required": selected.authorization_required,
        "run_binding_sha256": capability.run_binding_sha256,
        "attempt_receipt_sha256": capability.attempt_receipt_sha256,
        "execution_manifest_sha256": capability.execution_manifest_sha256,
        "protocol_sha256": protocol.protocol_sha256,
        "source_bindings": protocol.source_bindings.model_dump(mode="json"),
        "recovery": protocol.recovery.model_dump(mode="json"),
        "support_plan": support_plan.model_dump(mode="json"),
        "resources": protocol.resources.model_dump(mode="json"),
        "checkpoint": protocol.checkpoint.model_dump(mode="json"),
    }
    yield V5CheckpointInput(
        kind="authorization",
        key=V5KaggleStage.AUTHORIZE,
        canonical_bytes=_canonical_bytes(document),
    )


__all__ = [
    "V5MetricStageEvidence",
    "V5PreparedPartition",
    "V5StageCapability",
    "execute_v5_authorization_stage",
    "execute_v5_arm_stage",
    "execute_v5_control_stage",
    "execute_v5_corpus_stage",
    "execute_v5_feature_stage",
    "execute_v5_finalize_stage",
    "execute_v5_metric_stage",
    "load_v5_arm_checkpoint",
    "load_v5_control_group_checkpoint",
    "load_v5_corpus_checkpoint",
    "load_v5_feature_checkpoint",
    "load_v5_final_checkpoint",
    "load_v5_metric_checkpoint",
]
