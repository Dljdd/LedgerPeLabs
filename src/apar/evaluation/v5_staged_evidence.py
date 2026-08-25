"""Typed execution boundaries for the Sentinel v5 staged evidence pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

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
from apar.evaluation.v5_metrics import V5CompleteArmMetrics, evaluate_v5_complete_result
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
    V5KaggleStage.INVARIANCE_CONTROLS: V5ControlGroup.INVARIANCE,
    V5KaggleStage.SINGLE_CLASS_CONTROLS: V5ControlGroup.SINGLE_CLASS,
}
_INVARIANCE_CONTROL_NAMES = {
    "identity_rename",
    "future_causality",
    "equal_time_isolation",
    "feature_leakage",
}
_WORKLOAD_CONTROL_NAMES = {"benign_only", "fraud_only_diagnostic"}


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


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
                "schema_version": "apar-sentinel-v5-kaggle-arms/1",
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
        yield V5CheckpointInput(
            kind="arm_result",
            key=result.arm,
            canonical_bytes=_canonical_bytes(core),
        )
        yield V5CheckpointInput(
            kind="arm_latency",
            key=result.arm,
            canonical_bytes=_canonical_bytes(observation),
            layer="observational",
        )


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
        header_record.kind != "arm_header"
        or header_record.key != "arms"
        or header.get("schema_version") != "apar-sentinel-v5-kaggle-arms/1"
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
            core_record = next(deterministic)
            latency_record = next(observational)
        except StopIteration as error:
            raise ValueError("arm result or latency evidence is missing") from error
        if (
            core_record.kind != "arm_result"
            or latency_record.kind != "arm_latency"
            or core_record.key != arm
            or latency_record.key != arm
        ):
            raise ValueError("arm result order differs")
        core = _read_json_record(core_record, label="arm deterministic result")
        observation = _read_json_record(latency_record, label="arm latency observation")
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
    """Build complete metrics from exact arm and three control checkpoints."""
    _validate_capability(capability, required_stage=V5KaggleStage.METRICS)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    corpus_manifest = read_v5_checkpoint_manifest(
        output_root=corpus_checkpoint_root, limits=protocol.resources
    )
    arm_manifest = read_v5_checkpoint_manifest(
        output_root=arm_checkpoint_root, limits=protocol.resources
    )
    control_roots = tuple(control_checkpoint_roots)
    if len(control_roots) != 3:
        raise ValueError("metric stage requires exact three control checkpoints")
    control_manifests = tuple(
        read_v5_checkpoint_manifest(output_root=path, limits=protocol.resources)
        for path in control_roots
    )
    expected_control_stages = (
        V5KaggleStage.LABEL_SHUFFLE,
        V5KaggleStage.INVARIANCE_CONTROLS,
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
            for previous, current in zip(control_manifests, control_manifests[1:], strict=True)
        )
        or any(
            item.run_binding_sha256 != capability.run_binding_sha256
            or item.attempt_receipt_sha256 != capability.attempt_receipt_sha256
            for item in manifests
        )
    ):
        raise PermissionError("metric checkpoint lineage differs")
    arm_results = load_v5_arm_checkpoint(
        checkpoint_root=arm_checkpoint_root, limits=protocol.resources
    )
    control_groups = tuple(
        load_v5_control_group_checkpoint(checkpoint_root=path, limits=protocol.resources)
        for path in control_roots
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path, root=root
    )
    evidence = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=control_groups,
        evidence_protocol=evidence_protocol,
    )
    core, observation = _metric_stage_core_and_observation(
        evidence=evidence, arm_results=arm_results
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


def load_v5_metric_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> V5MetricStageEvidence:
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
    return _restore_metric_stage_evidence(
        core=_read_json_record(deterministic[0], label="deterministic metric stage"),
        observation=_read_json_record(observational[0], label="observational metric stage"),
    )


def _build_checkpoint_chain_binding(
    manifests: Sequence[V5CheckpointManifest],
) -> V5CheckpointChainBinding:
    frozen = tuple(manifests)
    expected = tuple(V5KaggleStage)[:-1]
    if tuple(item.stage for item in frozen) != expected:
        raise ValueError("checkpoint chain requires exact Stage 00-70 manifests")
    if any(
        current.predecessor_manifest_sha256 != previous.manifest_sha256
        for previous, current in zip(frozen, frozen[1:], strict=True)
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
    if len(roots) != 8:
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
    arm_results = load_v5_arm_checkpoint(checkpoint_root=roots[3], limits=protocol.resources)
    control_groups = tuple(
        load_v5_control_group_checkpoint(checkpoint_root=roots[index], limits=protocol.resources)
        for index in (4, 5, 6)
    )
    retained_metrics = load_v5_metric_checkpoint(
        checkpoint_root=roots[7], limits=protocol.resources
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path, root=root
    )
    recomputed = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=control_groups,
        evidence_protocol=evidence_protocol,
    )
    if recomputed != retained_metrics:
        raise ValueError("finalization recomputation differs from Stage 70 evidence")
    support_plan = build_v5_kaggle_support_plan(root=root, protocol=protocol, mode=capability.mode)
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.source_bindings.feature_catalog_path
    )
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
) -> V5StagedEvidencePayload:
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
    claimed = core.pop("final_core_sha256", None)
    if type(claimed) is not str or _sha256(_canonical_bytes(core)) != claimed:
        raise ValueError("final deterministic core digest differs")
    payload = V5StagedEvidencePayload.model_validate_json(observational[0].canonical_bytes)
    if (
        core.get("mode") != payload.mode.value
        or core.get("run_binding_sha256") != payload.run_binding_sha256
        or core.get("support_plan_sha256") != payload.support_plan.support_plan_sha256
        or core.get("deterministic_core_sha256") != payload.deterministic_core.core_sha256
    ):
        raise ValueError("final deterministic/payload binding differs")
    return payload


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
