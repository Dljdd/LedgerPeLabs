"""Non-authoritative, memory-bounded rescue helpers for the failed v5 Kaggle run.

These helpers do not issue stage capabilities, publish official checkpoint manifests,
or change the frozen protocol.  They exist only to salvage diagnostic metric evidence
from already-accepted checkpoints after the official Stage 70 loader exhausted RAM.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    iter_v5_checkpoint_observational_records,
    iter_v5_checkpoint_records,
)
from apar.evaluation.v5_controls import V5ExecutedControlSuite
from apar.evaluation.v5_evaluation import V5EvaluationResult
from apar.evaluation.v5_evidence_bundle import V5ReadinessEvidence
from apar.evaluation.v5_evidence_layers import (
    _stable_complete_metrics,
    _stable_controls,
    _stable_readiness,
)
from apar.evaluation.v5_kaggle_protocol import V5KaggleResourceGates
from apar.evaluation.v5_metrics import V5CompleteArmMetrics
from apar.evaluation.v5_staged_evidence import (
    _ARM_HEADER_SCHEMA,
    _ARM_LATENCY_META_SCHEMA,
    _ARM_RESULT_META_SCHEMA,
    _MAX_ARM_SECTION_RECORD_BYTES,
    _STAGED_ARMS,
    _canonical_bytes,
    _read_arm_meta_record,
    _read_arm_section_records,
    _read_json_record,
    _restore_arm_result,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class NonAuthoritativeRescueArmResult:
    """One restored arm plus its deterministic checkpoint binding."""

    arm: str
    deterministic_result_sha256: str
    result: V5EvaluationResult


def _open_arm_checkpoint(
    *,
    checkpoint_root: Path,
    limits: V5KaggleResourceGates,
) -> tuple[
    Iterator[V5CheckpointInput],
    Iterator[V5CheckpointInput],
    dict[str, object],
    tuple[str, ...],
    list[object],
]:
    deterministic = iter(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
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
    expected_core_digests = header.get("deterministic_result_sha256")
    if not isinstance(expected_core_digests, list) or len(expected_core_digests) != len(
        expected_order
    ):
        raise ValueError("arm checkpoint deterministic result index differs")
    return deterministic, observational, header, expected_order, expected_core_digests


def _skip_arm_section_records(
    records: Iterator[V5CheckpointInput],
    *,
    kind: str,
    arm: str,
    record_count: int,
    layer: str,
) -> None:
    for record_index in range(record_count):
        try:
            record = next(records)
        except StopIteration as error:
            raise ValueError(f"{arm} skipped section record is missing") from error
        if (
            getattr(record, "kind", None) != kind
            or getattr(record, "key", None) != f"{arm}:{record_index:04d}"
            or getattr(record, "layer", None) != layer
            or len(getattr(record, "canonical_bytes", b"")) > _MAX_ARM_SECTION_RECORD_BYTES
        ):
            raise ValueError(f"{arm} skipped section record order differs")


def load_non_authoritative_rescue_arm_result(
    *,
    checkpoint_root: Path,
    limits: V5KaggleResourceGates,
    target_arm: str,
) -> NonAuthoritativeRescueArmResult:
    """Restore one selected arm while streaming past all earlier arm records."""
    deterministic, observational, header, expected_order, expected_core_digests = (
        _open_arm_checkpoint(checkpoint_root=checkpoint_root, limits=limits)
    )
    if target_arm not in expected_order:
        raise ValueError("rescue target arm is not in the frozen arm order")
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
        if arm != target_arm:
            _skip_arm_section_records(
                deterministic,
                kind="arm_execution_artifacts",
                arm=arm,
                record_count=core_sections["execution_artifacts"][1],
                layer="deterministic",
            )
            _skip_arm_section_records(
                deterministic,
                kind="arm_result_rows",
                arm=arm,
                record_count=core_sections["row_evidence"][1],
                layer="deterministic",
            )
            _skip_arm_section_records(
                observational,
                kind="arm_latency_samples",
                arm=arm,
                record_count=observation_sections["samples"][1],
                layer="observational",
            )
            continue
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
        if (
            result.arm != arm
            or result.support_sha256 != header.get("support_sha256")
            or header.get("support_event_ids")
            != [row.support.event_id for row in result.row_evidence]
        ):
            raise ValueError("arm support binding differs")
        return NonAuthoritativeRescueArmResult(
            arm=arm,
            deterministic_result_sha256=expected_digest,
            result=result,
        )
    raise ValueError("rescue target arm was not found")


def iter_non_authoritative_rescue_arm_results(
    *,
    checkpoint_root: Path,
    limits: V5KaggleResourceGates,
) -> Iterator[NonAuthoritativeRescueArmResult]:
    """Restore and validate one arm at a time without materializing all four arms."""
    deterministic, observational, header, expected_order, expected_core_digests = (
        _open_arm_checkpoint(checkpoint_root=checkpoint_root, limits=limits)
    )

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
        expected_digest = expected_core_digests[index]
        if (
            type(expected_digest) is not str
            or core.get("deterministic_result_sha256") != expected_digest
        ):
            raise ValueError("arm deterministic result index binding differs")
        result = _restore_arm_result(core=core, observation=observation)
        if (
            result.arm != arm
            or result.support_sha256 != header.get("support_sha256")
            or header.get("support_event_ids")
            != [row.support.event_id for row in result.row_evidence]
        ):
            raise ValueError("arm support binding differs")
        item = NonAuthoritativeRescueArmResult(
            arm=arm,
            deterministic_result_sha256=expected_digest,
            result=result,
        )
        yield item
        del item, result, core, observation, core_sections, observation_sections
        gc.collect()

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


def build_non_authoritative_rescue_metric_documents(
    *,
    complete_metrics: tuple[V5CompleteArmMetrics, ...],
    controls: V5ExecutedControlSuite,
    readiness: V5ReadinessEvidence,
    deterministic_result_sha256: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    """Rebuild the exact metric documents without retaining full arm results.

    The returned documents intentionally match the frozen metric transformation.
    Callers must place them inside a separately labeled non-authoritative rescue
    envelope and must not publish them as an official Stage 70 checkpoint.
    """
    expected_order = tuple(arm.value for arm in _STAGED_ARMS)
    if tuple(item.arm for item in complete_metrics) != expected_order or len(
        deterministic_result_sha256
    ) != len(expected_order):
        raise ValueError("rescue metrics require exact ordered four arms")
    complete_documents = tuple(item.model_dump(mode="json") for item in complete_metrics)
    controls_document = controls.model_dump(mode="json")
    readiness_document = readiness.model_dump(mode="json")
    stable_controls, control_digests = _stable_controls(controls_document)
    core: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-metric-core/1",
        "arm_order": expected_order,
        "deterministic_result_sha256": deterministic_result_sha256,
        "complete_metrics": tuple(
            _stable_complete_metrics(
                document,
                deterministic_result_sha256=result_digest,
            )
            for document, result_digest in zip(
                complete_documents,
                deterministic_result_sha256,
                strict=True,
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
        dict[str, object],
        json.loads(_canonical_bytes(observation)),
    )


def build_non_authoritative_compact_arm_metric_documents(
    *,
    metric: V5CompleteArmMetrics,
    deterministic_result_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind one exact metric result without materializing its bootstrap draws as JSON."""
    latency_names = {"p50_latency_ms", "p95_latency_ms", "p99_latency_ms"}
    aggregate = {
        name: estimate.model_dump(mode="json") for name, estimate in metric.aggregate.items()
    }
    core: dict[str, object] = {
        "arm": metric.arm.value,
        "deterministic_result_sha256": deterministic_result_sha256,
        "support_sha256": metric.support_sha256,
        "aggregate": {
            name: estimate for name, estimate in aggregate.items() if name not in latency_names
        },
        "calibration_sha256": metric.calibration.calibration_sha256,
        "economics_sha256": metric.economics.economics_sha256,
        "family_sha256": [item.family_sha256 for item in metric.by_family],
        "bootstrap_sha256": metric.bootstrap.bootstrap_sha256,
    }
    core["deterministic_complete_metrics_sha256"] = _sha256(_canonical_bytes(core))
    observation: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-arm-metric/1",
        "arm": metric.arm.value,
        "arm_result_sha256": metric.arm_result_sha256,
        "support_sha256": metric.support_sha256,
        "complete_metrics_sha256": metric.complete_metrics_sha256,
        "aggregate": aggregate,
        "calibration_sha256": metric.calibration.calibration_sha256,
        "economics_sha256": metric.economics.economics_sha256,
        "families": tuple(
            {
                "family": item.family,
                "support_count": item.support_count,
                "campaign_count": item.campaign_count,
                "recall": item.recall.model_dump(mode="json"),
                "precision": item.precision.model_dump(mode="json"),
                "campaign_detection_rate": item.campaign_detection_rate.model_dump(mode="json"),
                "family_sha256": item.family_sha256,
            }
            for item in metric.by_family
        ),
        "bootstrap": {
            "seed": metric.bootstrap.seed,
            "replicates": metric.bootstrap.replicates,
            "confidence_level": metric.bootstrap.confidence_level,
            "interval_method": metric.bootstrap.interval_method,
            "resampling_unit": metric.bootstrap.resampling_unit,
            "stratification": metric.bootstrap.stratification,
            "strata": metric.bootstrap.strata,
            "intervals": tuple(
                interval.model_dump(mode="json") for interval in metric.bootstrap.intervals
            ),
            "bootstrap_sha256": metric.bootstrap.bootstrap_sha256,
        },
    }
    observation["compact_observation_sha256"] = _sha256(_canonical_bytes(observation))
    return cast(dict[str, object], json.loads(_canonical_bytes(core))), cast(
        dict[str, object],
        json.loads(_canonical_bytes(observation)),
    )


__all__ = [
    "NonAuthoritativeRescueArmResult",
    "build_non_authoritative_compact_arm_metric_documents",
    "build_non_authoritative_rescue_metric_documents",
    "iter_non_authoritative_rescue_arm_results",
    "load_non_authoritative_rescue_arm_result",
]
