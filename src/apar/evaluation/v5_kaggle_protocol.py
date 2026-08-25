"""Closed recovery protocol for staged private Kaggle Sentinel v5 execution."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_protocol import load_v5_development_protocol
from apar.evaluation.v5_run_mode import (
    V5PartitionSupportPlan,
    V5RunMode,
    build_v5_run_support_plan,
)


class V5KaggleMode(StrEnum):
    """The only two Kaggle execution modes accepted by the staged runner."""

    CAPACITY_VALIDATION = "kaggle_capacity_validation"
    LOCKED_SUCCESSOR = "kaggle_locked_successor"


class V5KaggleStage(StrEnum):
    """The immutable checkpoint order."""

    AUTHORIZE = "00_authorize"
    CORPUS = "10_corpus"
    FEATURES = "20_features"
    ARMS = "30_arms"
    LABEL_SHUFFLE = "40_label_shuffle"
    INVARIANCE_CONTROLS = "50_invariance_controls"
    SINGLE_CLASS_CONTROLS = "60_single_class_controls"
    METRICS = "70_metrics"
    FINALIZE = "80_finalize"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class V5KaggleRunSpec(_FrozenModel):
    mode: V5KaggleMode
    profile: Literal["production"]
    development_test_seed: int = Field(gt=0)
    repeatable: bool
    authorization_required: bool

    @model_validator(mode="after")
    def mode_is_closed(self) -> Self:
        expected = {
            V5KaggleMode.CAPACITY_VALIDATION: (404, True, False),
            V5KaggleMode.LOCKED_SUCCESSOR: (2404, False, True),
        }[self.mode]
        observed = (
            self.development_test_seed,
            self.repeatable,
            self.authorization_required,
        )
        if observed != expected:
            raise ValueError("Kaggle mode seed/repeatability/authorization differs")
        return self


class V5KaggleResourceGates(_FrozenModel):
    max_peak_rss_bytes: Literal[19327352832]
    max_stage_seconds: Literal[21600]
    max_stage_output_bytes: Literal[10000000000]
    max_checkpoint_chunk_bytes: Literal[67108864]
    max_checkpoint_chunks: Literal[160]

    @model_validator(mode="after")
    def chunk_capacity_covers_output_bound(self) -> Self:
        if (
            self.max_checkpoint_chunk_bytes * self.max_checkpoint_chunks
            < self.max_stage_output_bytes
        ):
            raise ValueError("checkpoint chunk bound cannot contain stage output")
        return self


class V5KaggleEnvironmentBinding(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-kaggle-environment/1"]
    provider: Literal["kaggle"]
    image: str = Field(min_length=1, max_length=256)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version: str = Field(pattern=r"^3\.12\.[0-9]+$")
    architecture: Literal["x86_64"]
    cpu_count: int = Field(gt=0, le=256)
    dependency_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notebook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    internet_enabled: Literal[False]
    accelerator: Literal["none"]
    file_fsync_supported: Literal[True]
    directory_fsync_supported: Literal[True]
    hardlink_no_replace_supported: Literal[True]
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def bind(
        cls,
        *,
        provider: Literal["kaggle"],
        image: str,
        image_sha256: str,
        python_version: str,
        architecture: Literal["x86_64"],
        cpu_count: int,
        dependency_manifest_sha256: str,
        source_archive_sha256: str,
        notebook_sha256: str,
        internet_enabled: Literal[False],
        accelerator: Literal["none"],
        file_fsync_supported: Literal[True],
        directory_fsync_supported: Literal[True],
        hardlink_no_replace_supported: Literal[True],
    ) -> V5KaggleEnvironmentBinding:
        values = {
            "schema_version": "apar-sentinel-v5-kaggle-environment/1",
            "provider": provider,
            "image": image,
            "image_sha256": image_sha256,
            "python_version": python_version,
            "architecture": architecture,
            "cpu_count": cpu_count,
            "dependency_manifest_sha256": dependency_manifest_sha256,
            "source_archive_sha256": source_archive_sha256,
            "notebook_sha256": notebook_sha256,
            "internet_enabled": internet_enabled,
            "accelerator": accelerator,
            "file_fsync_supported": file_fsync_supported,
            "directory_fsync_supported": directory_fsync_supported,
            "hardlink_no_replace_supported": hardlink_no_replace_supported,
        }
        values["environment_sha256"] = _canonical_digest(values)
        return cls.model_validate(values)

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        expected = _canonical_digest(
            self.model_dump(mode="json", exclude={"environment_sha256"})
        )
        if self.environment_sha256 != expected:
            raise ValueError("Kaggle environment digest differs")
        return self


class V5KaggleCheckpointProtocol(_FrozenModel):
    manifest_schema_version: Literal[
        "apar-sentinel-v5-kaggle-checkpoint-manifest/1"
    ]
    observation_schema_version: Literal["apar-sentinel-v5-kaggle-observation/1"]
    record_stream_schema_version: Literal[
        "apar-sentinel-v5-kaggle-record-stream/1"
    ]
    compression: Literal["gzip-zlib-level-9"]
    checkpoint_directory: Literal["apar-v5-checkpoint"]
    final_directory: Literal["apar-v5-final"]
    manifest_name: Literal["checkpoint.manifest.json"]
    observation_name: Literal["observational.json"]
    chunks_directory: Literal["chunks"]


class V5KagglePrivateInputs(_FrozenModel):
    """Exact private Kaggle identities and mounted artifact filenames."""

    owner_slug: Literal["dylanmoraes"]
    source_dataset_slug: Literal["apar-sentinel-v5-source3"]
    source_archive_name: Literal["apar-v5-source3.tar.gz"]
    source_manifest_name: Literal["source-manifest.json"]
    wheelhouse_dataset_slug: Literal[
        "apar-sentinel-v5-wheelhouse-py312-linux-x86-64"
    ]
    wheelhouse_manifest_name: Literal["wheelhouse-manifest.json"]
    safe_evidence_dataset_slug: Literal["apar-sentinel-v5-safe-evidence"]
    safe_evidence_name: Literal["safe-evidence.json"]
    safe_evidence_manifest_name: Literal["safe-evidence-manifest.json"]
    notebook_slug_prefix: Literal["apar-sentinel-v5"]


class V5KaggleExecutionManifest(_FrozenModel):
    """Closed private-input authority selecting capacity or locked execution."""

    schema_version: Literal["apar-sentinel-v5-kaggle-execution-input/1"]
    execution_mode: V5KaggleMode
    profile: Literal["production"]
    development_test_seed: int = Field(gt=0)
    authorization_required: bool
    successor_authorization_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    approved_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_name: Literal["safe-evidence.json"]
    artifact_size_bytes: int = Field(gt=0)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def closed_mode_and_digest_match(self) -> Self:
        expected = {
            V5KaggleMode.CAPACITY_VALIDATION: (404, False),
            V5KaggleMode.LOCKED_SUCCESSOR: (2404, True),
        }[self.execution_mode]
        if (self.development_test_seed, self.authorization_required) != expected:
            raise ValueError("execution manifest mode/seed/authorization differs")
        if self.authorization_required != (self.successor_authorization_sha256 is not None):
            raise ValueError("execution manifest successor authorization differs")
        if self.manifest_sha256 != _canonical_digest(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        ):
            raise ValueError("execution manifest self-digest differs")
        return self


class V5KaggleSuccessorOutputs(_FrozenModel):
    """Repository paths that must remain absent until successor execution."""

    attempt_receipt_path: Literal[
        "docs/experiments/defense-v5-kaggle-successor-attempt.json"
    ]
    checkpoint_directory_path: Literal[
        "docs/experiments/defense-v5-kaggle-successor-checkpoints"
    ]
    candidate_manifest_path: Literal[
        "docs/experiments/defense-v5-kaggle-development-candidate.manifest.json"
    ]
    candidate_chunks_path: Literal[
        "docs/experiments/defense-v5-kaggle-development-candidate.manifest.json.chunks"
    ]
    judge_summary_path: Literal[
        "docs/experiments/defense-v5-kaggle-development-summary.json"
    ]


class V5KaggleSupportPlan(_FrozenModel):
    mode: V5KaggleMode
    profile: Literal["production"]
    partitions: tuple[V5PartitionSupportPlan, ...]
    retained_execution_artifacts: int = Field(gt=0)
    retained_execution_payload_estimate_bytes: int = Field(gt=0)
    support_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        expected = _canonical_digest(
            self.model_dump(mode="json", exclude={"support_plan_sha256"})
        )
        if self.support_plan_sha256 != expected:
            raise ValueError("Kaggle support-plan digest differs")
        if tuple(item.partition for item in self.partitions) != (
            "train",
            "calibration",
            "threshold",
            "development_test",
        ):
            raise ValueError("Kaggle support-plan partition order differs")
        return self


class V5KaggleRecoveryBinding(_FrozenModel):
    attempt_receipt_path: Literal[
        "docs/experiments/defense-v5-locked-development-attempt.json"
    ]
    attempt_receipt_raw_sha256: Literal[
        "c9093272309605293f6377699df1810485901e0e3c5dfa9f81226ddea31151e8"
    ]
    attempt_receipt_self_sha256: Literal[
        "2cd207fdef0b808a8623843152195495d25d40b5d7903c5e71fd936611a09b93"
    ]
    abort_record_path: Literal[
        "docs/experiments/defense-v5-locked-development-abort.json"
    ]
    abort_record_sha256: Literal[
        "dc0743f1fe93356ea1e06af188d7a0e08cf46f0fcea02a674dbc1b2ec63d94d8"
    ]
    historical_result_path: Literal[
        "docs/experiments/defense-v5-development-result.json"
    ]
    historical_result_sha256: Literal[
        "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185"
    ]
    historical_safe_core_path: Literal[
        "config/defense/defense-v5-safe-core-freeze.json"
    ]
    historical_safe_core_file_sha256: Literal[
        "6c38f80c49826659e7484c56b24c797436fa050275252a39b6db36bf5d89f830"
    ]
    historical_safe_core_sha256: Literal[
        "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
    ]
    consumed_attempt_safe_core_sha256: Literal[
        "8cd3bba2cda47aa5b0d0a85fed4476eeeff787f3f0d2fcec973cc3e30a7b0435"
    ]
    observational_environment_sha256: Literal[
        "415902f1184ebe83deac0380e419a2eb6f11c3d7147c2c67fe38f76fe72ecc33"
    ]
    retry_permitted: Literal[False]


class V5KaggleSourceBindings(_FrozenModel):
    base_protocol_path: Literal["config/defense/defense-v5-development.json"]
    base_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_protocol_path: Literal["config/defense/defense-v5-evidence.json"]
    evidence_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_protocol_path: Literal["config/defense/defense-v5-arms.json"]
    arm_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_catalog_path: Literal["config/defense/feature-catalog-v5.json"]
    feature_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V5KaggleProtocol(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-kaggle-recovery-protocol/1"]
    protocol_id: Literal["apar-sentinel-v5-kaggle-staged-recovery"]
    stage_order: tuple[V5KaggleStage, ...]
    capacity: V5KaggleRunSpec
    locked: V5KaggleRunSpec
    resources: V5KaggleResourceGates
    checkpoint: V5KaggleCheckpointProtocol
    private_inputs: V5KagglePrivateInputs
    successor_outputs: V5KaggleSuccessorOutputs
    recovery: V5KaggleRecoveryBinding
    source_bindings: V5KaggleSourceBindings
    protocol_sha256: str = Field(default="", pattern=r"^[0-9a-f]{0}$|^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def stage_and_mode_contract_is_exact(self) -> Self:
        if self.stage_order != tuple(V5KaggleStage):
            raise ValueError("Kaggle stage order differs from frozen contract")
        if self.capacity.mode is not V5KaggleMode.CAPACITY_VALIDATION:
            raise ValueError("capacity slot has the wrong closed mode")
        if self.locked.mode is not V5KaggleMode.LOCKED_SUCCESSOR:
            raise ValueError("locked slot has the wrong closed mode")
        if self.capacity.development_test_seed == self.locked.development_test_seed:
            raise ValueError("capacity and locked modes share a seed")
        return self

    def run_binding_sha256(self, mode: V5KaggleMode) -> str:
        """Bind a closed run choice to the validated protocol and source facts."""
        selected = self.capacity if mode is V5KaggleMode.CAPACITY_VALIDATION else self.locked
        return _canonical_digest(
            {
                "schema_version": "apar-sentinel-v5-kaggle-run-binding/1",
                "protocol_sha256": self.protocol_sha256,
                "run": selected.model_dump(mode="json"),
                "source_bindings": self.source_bindings.model_dump(mode="json"),
                "recovery": self.recovery.model_dump(mode="json"),
            }
        )


class _StagePredecessor(Protocol):
    stage: V5KaggleStage


def _canonical_digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"bound evidence path is missing or not a file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable canonical JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def _verify_recovery(root: Path, binding: V5KaggleRecoveryBinding) -> None:
    receipt_path = root / binding.attempt_receipt_path
    if _file_digest(receipt_path) != binding.attempt_receipt_raw_sha256:
        raise ValueError("failed attempt receipt bytes differ")
    receipt = _read_object(receipt_path, description="failed attempt receipt")
    if receipt.get("receipt_sha256") != binding.attempt_receipt_self_sha256:
        raise ValueError("failed attempt receipt self-digest differs")
    if (
        receipt.get("approved_safe_deterministic_core_sha256")
        != binding.consumed_attempt_safe_core_sha256
    ):
        raise ValueError("failed attempt safe-core binding differs")
    if (
        receipt.get("approved_safe_observational_environment_sha256")
        != binding.observational_environment_sha256
    ):
        raise ValueError("failed attempt environment binding differs")

    abort = _read_object(
        root / binding.abort_record_path,
        description="failed attempt abort record",
    )
    claimed_abort_digest = abort.pop("record_sha256", None)
    if claimed_abort_digest != binding.abort_record_sha256:
        raise ValueError("failed attempt abort record digest binding differs")
    if _canonical_digest(abort) != binding.abort_record_sha256:
        raise ValueError("failed attempt abort record self-digest differs")
    expected_abort = {
        "attempt_receipt_path": binding.attempt_receipt_path,
        "attempt_receipt_raw_sha256": binding.attempt_receipt_raw_sha256,
        "attempt_receipt_self_sha256": binding.attempt_receipt_self_sha256,
        "historical_result_sha256": binding.historical_result_sha256,
        "retry_permitted": False,
        "candidate_manifest_published": False,
        "candidate_chunks_published": False,
        "judge_summary_published": False,
    }
    if any(abort.get(key) != value for key, value in expected_abort.items()):
        raise ValueError("failed attempt abort facts differ")

    if (
        _file_digest(root / binding.historical_result_path)
        != binding.historical_result_sha256
    ):
        raise ValueError("historical development result bytes differ")
    safe_path = root / binding.historical_safe_core_path
    if _file_digest(safe_path) != binding.historical_safe_core_file_sha256:
        raise ValueError("historical safe-core freeze bytes differ")
    safe = _read_object(safe_path, description="historical safe-core freeze")
    if safe.get("approved_deterministic_core_sha256") != (
        binding.historical_safe_core_sha256
    ):
        raise ValueError("historical safe-core digest differs")


def _verify_source_bindings(root: Path, bindings: V5KaggleSourceBindings) -> None:
    pairs = (
        (bindings.base_protocol_path, bindings.base_protocol_sha256),
        (bindings.evidence_protocol_path, bindings.evidence_protocol_sha256),
        (bindings.arm_protocol_path, bindings.arm_protocol_sha256),
        (bindings.feature_catalog_path, bindings.feature_catalog_sha256),
    )
    for relative, expected in pairs:
        if _file_digest(root / relative) != expected:
            raise ValueError(f"Kaggle source binding differs: {relative}")


def load_v5_kaggle_protocol(path: Path, *, root: Path) -> V5KaggleProtocol:
    """Load the strict recovery protocol and validate every preserved binding."""
    document = _read_object(path, description="Kaggle recovery protocol")
    parsed = V5KaggleProtocol.model_validate(document)
    _verify_recovery(root, parsed.recovery)
    _verify_source_bindings(root, parsed.source_bindings)
    protocol_digest = _canonical_digest(
        parsed.model_dump(mode="json", exclude={"protocol_sha256"})
    )
    return parsed.model_copy(update={"protocol_sha256": protocol_digest})


def build_v5_kaggle_support_plan(
    *, root: Path, protocol: V5KaggleProtocol, mode: V5KaggleMode
) -> V5KaggleSupportPlan:
    """Derive production-size retained support without executing population code."""
    if type(protocol) is not V5KaggleProtocol:
        raise TypeError("Kaggle protocol must be an exact V5KaggleProtocol")
    selected = (
        protocol.capacity
        if mode is V5KaggleMode.CAPACITY_VALIDATION
        else protocol.locked
    )
    if selected.profile != "production":
        raise ValueError("Kaggle support plan requires the production profile")
    development = load_v5_development_protocol(
        root / protocol.source_bindings.base_protocol_path
    )
    evidence = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path,
        root=root,
    )
    locked_plan = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    values: dict[str, object] = {
        "mode": mode,
        "profile": "production",
        "partitions": locked_plan.partitions,
        "retained_execution_artifacts": locked_plan.retained_execution_artifacts,
        "retained_execution_payload_estimate_bytes": (
            locked_plan.retained_execution_payload_estimate_bytes
        ),
    }
    digest_document = {
        "mode": mode,
        "profile": "production",
        "partitions": [
            item.model_dump(mode="json") for item in locked_plan.partitions
        ],
        "retained_execution_artifacts": locked_plan.retained_execution_artifacts,
        "retained_execution_payload_estimate_bytes": (
            locked_plan.retained_execution_payload_estimate_bytes
        ),
    }
    values["support_plan_sha256"] = _canonical_digest(digest_document)
    return V5KaggleSupportPlan.model_validate(values)


def load_v5_kaggle_execution_manifest(
    path: Path,
    *,
    safe_evidence: Path,
    approved_commit: str,
    protocol: V5KaggleProtocol,
) -> V5KaggleExecutionManifest:
    """Validate the private execution selector against source and safe bytes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("Kaggle execution manifest is missing or linked")
    document = _read_object(path, description="Kaggle execution manifest")
    if path.read_bytes() != _canonical_bytes(document) + b"\n":
        raise ValueError("Kaggle execution manifest is not canonical JSON")
    manifest = V5KaggleExecutionManifest.model_validate(document)
    if (
        safe_evidence.is_symlink()
        or not safe_evidence.is_file()
        or safe_evidence.stat().st_nlink != 1
    ):
        raise ValueError("approved safe evidence is missing or linked")
    expected_binding = protocol.run_binding_sha256(manifest.execution_mode)
    if (
        manifest.approved_commit != approved_commit
        or manifest.protocol_sha256 != protocol.protocol_sha256
        or manifest.run_binding_sha256 != expected_binding
        or manifest.artifact_size_bytes != safe_evidence.stat().st_size
        or manifest.artifact_sha256 != _file_digest(safe_evidence)
    ):
        raise ValueError("Kaggle execution manifest binding differs")
    return manifest


def resolve_next_v5_kaggle_stage(
    predecessor: _StagePredecessor | None,
) -> V5KaggleStage:
    """Infer the only admissible next stage from an already verified predecessor."""
    if predecessor is None:
        return V5KaggleStage.AUTHORIZE
    if type(predecessor.stage) is not V5KaggleStage:
        raise TypeError("predecessor stage must be an exact V5KaggleStage")
    stages = tuple(V5KaggleStage)
    index = stages.index(predecessor.stage)
    if index == len(stages) - 1:
        raise ValueError("final stage has no successor")
    return stages[index + 1]


__all__ = [
    "V5KaggleCheckpointProtocol",
    "V5KaggleEnvironmentBinding",
    "V5KaggleExecutionManifest",
    "V5KaggleMode",
    "V5KagglePrivateInputs",
    "V5KaggleProtocol",
    "V5KaggleRecoveryBinding",
    "V5KaggleResourceGates",
    "V5KaggleRunSpec",
    "V5KaggleSourceBindings",
    "V5KaggleStage",
    "V5KaggleSuccessorOutputs",
    "V5KaggleSupportPlan",
    "build_v5_kaggle_support_plan",
    "load_v5_kaggle_protocol",
    "load_v5_kaggle_execution_manifest",
    "resolve_next_v5_kaggle_stage",
]
