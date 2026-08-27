"""Locked-development complete evidence payload contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar.evaluation.v5_controls import V5ExecutedControlSuite
from apar.evaluation.v5_evaluation import V5Arm, V5EvaluationResult
from apar.evaluation.v5_evidence_bundle import (
    _ARM_ORDER,
    V5PackedDocument,
    V5ReadinessEvidence,
    _collect_execution_artifacts,
    _compact_result,
    build_v5_readiness_evidence,
)
from apar.evaluation.v5_evidence_layers import (
    DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
    LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
    LOCKED_DETERMINISTIC_CORE_SCHEMA,
    build_deterministic_core_document,
    build_locked_deterministic_core_document,
    build_observational_latency_document,
)
from apar.evaluation.v5_evidence_protocol import V5EvidenceProtocol
from apar.evaluation.v5_kaggle_protocol import V5KaggleMode, V5KaggleSupportPlan
from apar.evaluation.v5_metrics import evaluate_v5_complete_result
from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


_PREDECESSOR_STAGES = (
    "00_authorize",
    "10_corpus",
    "20_features",
    "30_arms",
    "40_label_shuffle",
    "50_identity_rename",
    "51_future_causality",
    "52_equal_time_isolation",
    "53_feature_leakage",
    "60_single_class_controls",
    "70_metrics",
)


class V5CheckpointChainBinding(BaseModel):
    """Non-self-referential address of the complete Stage 00-70 chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-checkpoint-chain/1"]
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predecessor_stage_manifest_sha256: tuple[tuple[str, str], ...]
    predecessor_chain_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def chain_is_exact(self) -> Self:
        if (
            tuple(stage for stage, _digest_value in self.predecessor_stage_manifest_sha256)
            != _PREDECESSOR_STAGES
        ):
            raise ValueError("checkpoint chain requires exact Stage 00-70 order")
        digests = tuple(
            digest_value for _stage, digest_value in self.predecessor_stage_manifest_sha256
        )
        if len(set(digests)) != len(digests) or any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("checkpoint chain manifest digests are invalid")
        document = self.model_dump(mode="json", exclude={"predecessor_chain_root_sha256"})
        if self.predecessor_chain_root_sha256 != _digest(document):
            raise ValueError("checkpoint predecessor chain-root digest mismatch")
        return self


class V5LockedDeterministicCoreBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-locked-deterministic-core/2"]
    exclusion_schema: tuple[tuple[str, tuple[str, ...], str], ...]
    core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exclusions_are_exact(self) -> Self:
        if self.schema_version != LOCKED_DETERMINISTIC_CORE_SCHEMA:
            raise ValueError("locked deterministic-core schema differs")
        if self.exclusion_schema != LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA:
            raise ValueError("locked deterministic-core exclusion schema differs")
        return self


class V5LockedEvidencePayload(BaseModel):
    """Complete production evidence; legacy summary fields cannot satisfy it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-locked-development-payload/2"]
    run_binding: V5LockedEvidenceRunBinding
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_protocol: V5EvidenceProtocol
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_artifact_pool: tuple[V5PackedDocument, ...]
    arm_results: tuple[V5PackedDocument, ...]
    complete_metrics: tuple[V5PackedDocument, ...]
    controls: V5PackedDocument
    readiness: V5ReadinessEvidence
    deterministic_core: V5LockedDeterministicCoreBinding
    observational_latency: V5PackedDocument
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_shape_is_exact(self) -> Self:
        if not self.execution_artifact_pool or any(
            item.kind != "execution_artifact" for item in self.execution_artifact_pool
        ):
            raise ValueError("locked execution artifact pool is incomplete")
        if tuple(item.kind for item in self.arm_results) != ("arm_result",) * 4:
            raise ValueError("locked payload requires exact four arm results")
        if tuple(item.kind for item in self.complete_metrics) != ("complete_metrics",) * 4:
            raise ValueError("locked payload requires exact four complete metrics")
        if self.controls.kind != "executed_controls":
            raise ValueError("locked payload executed controls are missing")
        if self.observational_latency.kind != "observational_latency":
            raise ValueError("locked payload observational latency is missing")
        if self.catalog_sha256 != self.run_binding.catalog_sha256:
            raise ValueError("locked payload catalog binding differs")
        if (
            self.evidence_protocol.evidence_protocol_sha256
            != self.run_binding.evidence_protocol_sha256
            or self.evidence_protocol.base_protocol_sha256 != self.run_binding.base_protocol_sha256
            or self.evidence_protocol.arm_protocol_sha256 != self.run_binding.arm_protocol_sha256
            or self.evidence_protocol.implementation_sha256
            != self.run_binding.implementation_sha256
        ):
            raise ValueError("locked payload protocol/source binding differs")
        if self.payload_sha256 != _digest(self.model_dump(mode="json", exclude={"payload_sha256"})):
            raise ValueError("locked payload digest mismatch")
        return self


class V5StagedDeterministicCoreBinding(BaseModel):
    """Deterministic semantic address for either closed Kaggle run mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-kaggle-deterministic-core/1"]
    exclusion_schema: tuple[tuple[str, tuple[str, ...], str], ...]
    core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exclusions_are_frozen(self) -> Self:
        if self.exclusion_schema != DETERMINISTIC_CORE_EXCLUSION_SCHEMA:
            raise ValueError("staged deterministic-core exclusion schema differs")
        return self


class V5StagedEvidencePayload(BaseModel):
    """Complete staged evidence for capacity validation or locked execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-kaggle-staged-payload/1"]
    mode: V5KaggleMode
    profile: Literal["production"]
    development_test_seed: int
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_plan: V5KaggleSupportPlan
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_chain: V5CheckpointChainBinding
    evidence_protocol: V5EvidenceProtocol
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_artifact_pool: tuple[V5PackedDocument, ...]
    arm_results: tuple[V5PackedDocument, ...]
    complete_metrics: tuple[V5PackedDocument, ...]
    controls: V5PackedDocument
    readiness: V5ReadinessEvidence
    deterministic_core: V5StagedDeterministicCoreBinding
    observational_latency: V5PackedDocument
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_shape_is_exact(self) -> Self:
        expected_seed = {
            V5KaggleMode.CAPACITY_VALIDATION: 404,
            V5KaggleMode.LOCKED_SUCCESSOR: 2404,
        }[self.mode]
        if (
            self.development_test_seed != expected_seed
            or self.support_plan.mode is not self.mode
            or self.support_plan.profile != self.profile
        ):
            raise ValueError("staged payload mode/profile/seed/support binding differs")
        if self.checkpoint_chain.attempt_receipt_sha256 != self.attempt_receipt_sha256:
            raise ValueError("staged payload attempt/chain binding differs")
        if not self.execution_artifact_pool or any(
            item.kind != "execution_artifact" for item in self.execution_artifact_pool
        ):
            raise ValueError("staged execution artifact pool is incomplete")
        if tuple(item.kind for item in self.arm_results) != ("arm_result",) * 4:
            raise ValueError("staged payload requires exact four arm results")
        if tuple(item.kind for item in self.complete_metrics) != ("complete_metrics",) * 4:
            raise ValueError("staged payload requires exact four complete metrics")
        if (
            self.controls.kind != "executed_controls"
            or self.observational_latency.kind != "observational_latency"
        ):
            raise ValueError("staged payload control or latency evidence is missing")
        if self.payload_sha256 != _digest(self.model_dump(mode="json", exclude={"payload_sha256"})):
            raise ValueError("staged payload digest mismatch")
        return self


def _support_counts(rows: Sequence[object]) -> tuple[int, dict[str, int]]:
    legitimate = 0
    fraud: dict[str, int] = {}
    for item in rows:
        support: Any = getattr(item, "support", item)
        if support.label == 0:
            legitimate += 1
        else:
            fraud[support.family] = fraud.get(support.family, 0) + 1
    return legitimate, dict(sorted(fraud.items()))


def _validate_locked_support(
    *,
    results: tuple[V5EvaluationResult, ...],
    run_binding: V5LockedEvidenceRunBinding,
) -> None:
    plan = {item.partition: item for item in run_binding.support_plan.partitions}
    reference = results[0]
    if reference.arm_spec is None:
        raise ValueError("locked arm specification is missing")
    expected_development = plan["development_test"]
    legitimate, fraud = _support_counts(reference.row_evidence)
    if (
        legitimate != expected_development.legitimate_rows
        or fraud != dict(expected_development.fraud_rows_by_family)
        or len(reference.row_evidence) != expected_development.total_rows
    ):
        raise ValueError("locked development-test support differs from production plan")
    for partition in reference.arm_spec.training_partitions:
        expected = plan[partition.partition]
        legitimate, fraud = _support_counts(partition.support_records)
        if (
            legitimate != expected.legitimate_rows
            or fraud != dict(expected.fraud_rows_by_family)
            or len(partition.support_records) != expected.total_rows
        ):
            raise ValueError(f"locked {partition.partition} support differs from production plan")


def _validate_staged_support(
    *,
    results: tuple[V5EvaluationResult, ...],
    support_plan: V5KaggleSupportPlan,
) -> None:
    plan = {item.partition: item for item in support_plan.partitions}
    reference = results[0]
    if reference.arm_spec is None:
        raise ValueError("staged arm specification is missing")
    expected_development = plan["development_test"]
    legitimate, fraud = _support_counts(reference.row_evidence)
    if (
        legitimate != expected_development.legitimate_rows
        or fraud != dict(expected_development.fraud_rows_by_family)
        or len(reference.row_evidence) != expected_development.total_rows
    ):
        raise ValueError("staged development-test support differs from production plan")
    for partition in reference.arm_spec.training_partitions:
        expected = plan[partition.partition]
        legitimate, fraud = _support_counts(partition.support_records)
        if (
            legitimate != expected.legitimate_rows
            or fraud != dict(expected.fraud_rows_by_family)
            or len(partition.support_records) != expected.total_rows
        ):
            raise ValueError(f"staged {partition.partition} support differs from production plan")


def build_v5_staged_evidence_payload(
    *,
    mode: V5KaggleMode,
    run_binding_sha256: str,
    support_plan: V5KaggleSupportPlan,
    chain: V5CheckpointChainBinding,
    evidence_protocol: V5EvidenceProtocol,
    catalog_sha256: str,
    arm_results: Sequence[V5EvaluationResult],
    controls: V5ExecutedControlSuite,
) -> bytes:
    """Build the complete staged payload without executing a model or simulator."""
    results = tuple(arm_results)
    if tuple(result.arm for result in results) != tuple(arm.value for arm in _ARM_ORDER):
        raise ValueError("staged evidence requires exact ordered four arm results")
    if len(run_binding_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in run_binding_sha256
    ):
        raise ValueError("staged evidence run binding is invalid")
    expected_seed = {
        V5KaggleMode.CAPACITY_VALIDATION: 404,
        V5KaggleMode.LOCKED_SUCCESSOR: 2404,
    }[mode]
    if support_plan.mode is not mode:
        raise ValueError("staged evidence support mode differs")
    _validate_staged_support(results=results, support_plan=support_plan)
    complete = tuple(
        evaluate_v5_complete_result(result=result, protocol=evidence_protocol) for result in results
    )
    readiness = build_v5_readiness_evidence(
        metrics=complete[_ARM_ORDER.index(V5Arm.FULL_SENTINEL)],
        controls=controls,
    )
    artifacts = _collect_execution_artifacts(results)
    if len(artifacts) != support_plan.retained_execution_artifacts:
        raise ValueError("staged execution artifact count differs from support plan")
    packed_artifacts = tuple(
        V5PackedDocument.pack(
            kind="execution_artifact",
            document=artifact.model_dump(mode="json"),
            max_uncompressed_bytes=evidence_protocol.bounds.max_single_execution_bytes,
        )
        for artifact in artifacts
    )
    if sum(item.uncompressed_bytes for item in packed_artifacts) > (
        evidence_protocol.locked_artifact_storage.maximum_envelope_bytes
    ):
        raise ValueError("staged execution artifact pool exceeds production bound")
    compact_results = tuple(_compact_result(result) for result in results)
    complete_documents = tuple(item.model_dump(mode="json") for item in complete)
    controls_document = controls.model_dump(mode="json")
    readiness_document = readiness.model_dump(mode="json")
    packed_results = tuple(
        V5PackedDocument.pack(kind="arm_result", document=document) for document in compact_results
    )
    packed_complete = tuple(
        V5PackedDocument.pack(kind="complete_metrics", document=document)
        for document in complete_documents
    )
    packed_controls = V5PackedDocument.pack(kind="executed_controls", document=controls_document)
    if mode is V5KaggleMode.CAPACITY_VALIDATION:
        semantic_core = build_deterministic_core_document(
            safe_seed=expected_seed,
            evidence_protocol=evidence_protocol.model_dump(mode="json"),
            catalog_sha256=catalog_sha256,
            execution_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
            arm_results=list(compact_results),
            complete_metrics=list(complete_documents),
            controls=controls_document,
            readiness=readiness_document,
        )
    else:
        semantic_core = build_locked_deterministic_core_document(
            run_binding={
                "mode": "locked_development",
                "profile": "production",
                "development_test_seed": expected_seed,
                "kaggle_run_binding_sha256": run_binding_sha256,
                "support_plan_sha256": support_plan.support_plan_sha256,
            },
            evidence_protocol=evidence_protocol.model_dump(mode="json"),
            catalog_sha256=catalog_sha256,
            execution_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
            arm_results=list(compact_results),
            complete_metrics=list(complete_documents),
            controls=controls_document,
            readiness=readiness_document,
        )
    staged_core_document = {
        "schema_version": "apar-sentinel-v5-kaggle-deterministic-core/1",
        "exclusion_schema": DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        "mode": mode,
        "profile": "production",
        "development_test_seed": expected_seed,
        "run_binding_sha256": run_binding_sha256,
        "support_plan_sha256": support_plan.support_plan_sha256,
        "semantic_core": semantic_core,
    }
    deterministic_core = V5StagedDeterministicCoreBinding(
        schema_version="apar-sentinel-v5-kaggle-deterministic-core/1",
        exclusion_schema=DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        core_sha256=_digest(staged_core_document),
    )
    observational = build_observational_latency_document(
        deterministic_core_sha256_value=deterministic_core.core_sha256,
        arm_results=list(compact_results),
        complete_metrics=list(complete_documents),
        controls=controls_document,
        readiness=readiness_document,
    )
    packed_observational = V5PackedDocument.pack(
        kind="observational_latency", document=observational
    )
    values = {
        "schema_version": "apar-sentinel-v5-kaggle-staged-payload/1",
        "mode": mode,
        "profile": "production",
        "development_test_seed": expected_seed,
        "run_binding_sha256": run_binding_sha256,
        "support_plan": support_plan,
        "attempt_receipt_sha256": chain.attempt_receipt_sha256,
        "checkpoint_chain": chain,
        "evidence_protocol": evidence_protocol,
        "catalog_sha256": catalog_sha256,
        "execution_artifact_pool": packed_artifacts,
        "arm_results": packed_results,
        "complete_metrics": packed_complete,
        "controls": packed_controls,
        "readiness": readiness,
        "deterministic_core": deterministic_core,
        "observational_latency": packed_observational,
    }
    document = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in values.items()
    }
    values["payload_sha256"] = _digest(document)
    payload = V5StagedEvidencePayload.model_validate(values)
    serialized = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if len(serialized) > (evidence_protocol.locked_artifact_storage.maximum_envelope_bytes):
        raise ValueError("staged payload exceeds the maximum envelope")
    return serialized


def build_v5_locked_evidence_payload(
    *,
    run_binding: V5LockedEvidenceRunBinding | dict[str, object],
    attempt_receipt_sha256: str,
    evidence_protocol: V5EvidenceProtocol,
    catalog_sha256: str,
    arm_results: Sequence[V5EvaluationResult],
    controls: V5ExecutedControlSuite | None,
) -> bytes:
    """Build one complete locked payload without publishing or choosing a seed."""
    results = tuple(arm_results)
    if tuple(result.arm for result in results) != tuple(arm.value for arm in _ARM_ORDER):
        raise ValueError("locked evidence requires exact ordered four arm results")
    binding = V5LockedEvidenceRunBinding.model_validate(run_binding)
    if len(attempt_receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in attempt_receipt_sha256
    ):
        raise ValueError("locked attempt receipt digest is invalid")
    if controls is None:
        raise ValueError("locked evidence requires all seven executed controls")
    if (
        evidence_protocol.run_modes.locked_development.profile != "production"
        or evidence_protocol.run_modes.locked_development.development_test_seed != 2404
    ):
        raise ValueError("locked evidence protocol mode binding differs")
    if catalog_sha256 != binding.catalog_sha256:
        raise ValueError("locked evidence catalog binding differs")
    _validate_locked_support(results=results, run_binding=binding)
    complete = tuple(
        evaluate_v5_complete_result(result=result, protocol=evidence_protocol) for result in results
    )
    readiness = build_v5_readiness_evidence(
        metrics=complete[_ARM_ORDER.index(V5Arm.FULL_SENTINEL)],
        controls=controls,
    )
    artifacts = _collect_execution_artifacts(results)
    if len(artifacts) != binding.support_plan.retained_execution_artifacts:
        raise ValueError("locked execution artifact count differs from production plan")
    packed_artifacts = tuple(
        V5PackedDocument.pack(
            kind="execution_artifact",
            document=artifact.model_dump(mode="json"),
            max_uncompressed_bytes=evidence_protocol.bounds.max_single_execution_bytes,
        )
        for artifact in artifacts
    )
    if sum(item.uncompressed_bytes for item in packed_artifacts) > (
        evidence_protocol.locked_artifact_storage.maximum_envelope_bytes
    ):
        raise ValueError("locked execution artifact pool exceeds production bound")
    compact_results = tuple(_compact_result(result) for result in results)
    complete_documents = tuple(item.model_dump(mode="json") for item in complete)
    controls_document = controls.model_dump(mode="json")
    readiness_document = readiness.model_dump(mode="json")
    packed_results = tuple(
        V5PackedDocument.pack(kind="arm_result", document=document) for document in compact_results
    )
    packed_complete = tuple(
        V5PackedDocument.pack(kind="complete_metrics", document=document)
        for document in complete_documents
    )
    packed_controls = V5PackedDocument.pack(kind="executed_controls", document=controls_document)
    core_document = build_locked_deterministic_core_document(
        run_binding=binding.model_dump(mode="json"),
        evidence_protocol=evidence_protocol.model_dump(mode="json"),
        catalog_sha256=catalog_sha256,
        execution_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        arm_results=list(compact_results),
        complete_metrics=list(complete_documents),
        controls=controls_document,
        readiness=readiness_document,
    )
    deterministic_core = V5LockedDeterministicCoreBinding(
        schema_version=LOCKED_DETERMINISTIC_CORE_SCHEMA,
        exclusion_schema=LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        core_sha256=_digest(core_document),
    )
    observational = build_observational_latency_document(
        deterministic_core_sha256_value=deterministic_core.core_sha256,
        arm_results=list(compact_results),
        complete_metrics=list(complete_documents),
        controls=controls_document,
        readiness=readiness_document,
    )
    packed_observational = V5PackedDocument.pack(
        kind="observational_latency", document=observational
    )
    values = {
        "schema_version": "apar-sentinel-v5-locked-development-payload/2",
        "run_binding": binding,
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "evidence_protocol": evidence_protocol,
        "catalog_sha256": catalog_sha256,
        "execution_artifact_pool": packed_artifacts,
        "arm_results": packed_results,
        "complete_metrics": packed_complete,
        "controls": packed_controls,
        "readiness": readiness,
        "deterministic_core": deterministic_core,
        "observational_latency": packed_observational,
    }
    document = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in values.items()
    }
    values["payload_sha256"] = _digest(document)
    payload = V5LockedEvidencePayload.model_validate(values)
    serialized = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if len(serialized) > (evidence_protocol.locked_artifact_storage.maximum_envelope_bytes):
        raise ValueError("locked payload exceeds the maximum envelope")
    return serialized


__all__ = [
    "V5CheckpointChainBinding",
    "V5LockedDeterministicCoreBinding",
    "V5LockedEvidencePayload",
    "V5StagedDeterministicCoreBinding",
    "V5StagedEvidencePayload",
    "build_v5_locked_evidence_payload",
    "build_v5_staged_evidence_payload",
]
