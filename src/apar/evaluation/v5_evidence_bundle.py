"""Bounded complete evidence envelope for Sentinel v5 development evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar.evaluation.v5_controls import V5ExecutedControlSuite
from apar.evaluation.v5_evaluation import V5Arm, V5EvaluationResult, V5ExecutionArtifact
from apar.evaluation.v5_evidence_layers import (
    DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
    DETERMINISTIC_CORE_SCHEMA,
    build_deterministic_core_document,
    build_observational_latency_document,
)
from apar.evaluation.v5_evidence_protocol import (
    V5EvidenceProtocol,
    V5MetricApplicability,
)
from apar.evaluation.v5_metrics import (
    V5BootstrapInterval,
    V5CompleteArmMetrics,
    V5MetricEstimate,
    evaluate_v5_complete_result,
)

_ARM_ORDER = (
    V5Arm.RULES_ONLY,
    V5Arm.ENSEMBLE_NO_GRAPH,
    V5Arm.ENSEMBLE_WITH_GRAPH,
    V5Arm.FULL_SENTINEL,
)
_PACKED_DOCUMENT_LIMIT = 536_870_912


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class V5PackedDocument(BaseModel):
    """One bounded immutable canonical document compressed before aggregation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "test",
        "execution_artifact",
        "arm_result",
        "complete_metrics",
        "executed_controls",
        "observational_latency",
    ]
    compression: Literal["zlib-9"]
    content_base64: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compressed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncompressed_bytes: int = Field(gt=0, le=_PACKED_DOCUMENT_LIMIT)
    compressed_bytes: int = Field(gt=0, le=_PACKED_DOCUMENT_LIMIT)
    packed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def pack(
        cls,
        *,
        kind: Literal[
            "test",
            "execution_artifact",
            "arm_result",
            "complete_metrics",
            "executed_controls",
            "observational_latency",
        ],
        document: object,
        max_uncompressed_bytes: int = _PACKED_DOCUMENT_LIMIT,
    ) -> V5PackedDocument:
        raw = _canonical_bytes(document)
        if len(raw) > max_uncompressed_bytes:
            raise ValueError(f"{kind} document exceeds its frozen expanded bound")
        compressed = zlib.compress(raw, level=9)
        values = {
            "kind": kind,
            "compression": "zlib-9",
            "content_base64": base64.b64encode(compressed).decode("ascii"),
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "uncompressed_bytes": len(raw),
            "compressed_bytes": len(compressed),
        }
        values["packed_sha256"] = _digest(values)
        return cls.model_validate(values)

    def content_bytes(self) -> bytes:
        try:
            compressed = base64.b64decode(self.content_base64, validate=True)
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(compressed, self.uncompressed_bytes + 1)
        except (ValueError, zlib.error) as error:
            raise ValueError("packed document compression is invalid") from error
        if (
            len(raw) > self.uncompressed_bytes
            or decompressor.unconsumed_tail
            or not decompressor.eof
        ):
            raise ValueError("packed document exceeds its declared expanded bound")
        return raw

    def document(self) -> dict[str, Any]:
        try:
            value = json.loads(self.content_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("packed document is not valid JSON") from error
        if type(value) is not dict or _canonical_bytes(value) != self.content_bytes():
            raise ValueError("packed document must be one canonical JSON object")
        return value

    @property
    def arm(self) -> str:
        value = self.document().get("arm")
        if self.kind != "arm_result" or type(value) is not str:
            raise AttributeError("packed document is not an arm result")
        return value

    @property
    def suite_sha256(self) -> str:
        value = self.document().get("suite_sha256")
        if self.kind != "executed_controls" or type(value) is not str:
            raise AttributeError("packed document is not an executed control suite")
        return value

    @model_validator(mode="after")
    def packed_content_is_bound(self) -> Self:
        try:
            compressed = base64.b64decode(self.content_base64, validate=True)
        except ValueError as error:
            raise ValueError("packed document base64 is invalid") from error
        raw = self.content_bytes()
        if len(raw) != self.uncompressed_bytes or len(compressed) != self.compressed_bytes:
            raise ValueError("packed document byte counts mismatch")
        if hashlib.sha256(raw).hexdigest() != self.content_sha256:
            raise ValueError("packed document content digest mismatch")
        if hashlib.sha256(compressed).hexdigest() != self.compressed_sha256:
            raise ValueError("packed document compressed digest mismatch")
        if self.packed_sha256 != _digest(self.model_dump(mode="json", exclude={"packed_sha256"})):
            raise ValueError("packed document digest mismatch")
        self.document()
        return self


def _collect_execution_artifacts(
    results: Sequence[V5EvaluationResult],
) -> tuple[V5ExecutionArtifact, ...]:
    artifacts: dict[str, V5ExecutionArtifact] = {}
    for result in results:
        collections = [result.execution_artifacts]
        if result.arm_spec is not None:
            collections.extend(
                partition.execution_artifacts for partition in result.arm_spec.training_partitions
            )
        for collection in collections:
            for artifact in collection:
                existing = artifacts.get(artifact.evidence_sha256)
                if existing is not None and existing != artifact:
                    raise ValueError("execution artifact content-address collision")
                artifacts[artifact.evidence_sha256] = artifact
    return tuple(artifacts[key] for key in sorted(artifacts))


def _compact_result(result: V5EvaluationResult) -> dict[str, Any]:
    document = result.model_dump(mode="json")
    artifacts = document.pop("execution_artifacts")
    document["execution_artifact_refs"] = [item["evidence_sha256"] for item in artifacts]
    arm_spec = document.get("arm_spec")
    if type(arm_spec) is dict:
        for partition in arm_spec.get("training_partitions", []):
            partition_artifacts = partition.pop("execution_artifacts")
            partition["execution_artifact_refs"] = [
                item["evidence_sha256"] for item in partition_artifacts
            ]
    return document


def _expand_result(
    document: Mapping[str, Any],
    artifacts: Mapping[str, V5ExecutionArtifact],
) -> dict[str, Any]:
    expanded = json.loads(_canonical_bytes(document))
    if type(expanded) is not dict:
        raise ValueError("retained arm result must be a canonical object")

    def resolve(refs: object) -> list[dict[str, Any]]:
        if type(refs) is not list or any(type(item) is not str for item in refs):
            raise ValueError("execution artifact references are invalid")
        try:
            return [artifacts[item].model_dump(mode="json") for item in refs]
        except KeyError as error:
            raise ValueError("execution artifact reference is missing") from error

    expanded["execution_artifacts"] = resolve(expanded.pop("execution_artifact_refs", None))
    arm_spec = expanded.get("arm_spec")
    if type(arm_spec) is dict:
        for partition in arm_spec.get("training_partitions", []):
            partition["execution_artifacts"] = resolve(
                partition.pop("execution_artifact_refs", None)
            )
    return cast(dict[str, Any], expanded)


class V5ReadinessGateEvidence(BaseModel):
    """One frozen point/interval gate with its exact source evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    family: str | None = None
    comparison: Literal["lower_bound_gte", "upper_bound_lte", "point_lte", "defined_interval"]
    target: float | None
    applicability: V5MetricApplicability
    point: float | None
    lower: float | None
    upper: float | None
    passed: bool
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def gate_is_consistent(self) -> Self:
        if self.applicability is not V5MetricApplicability.DEFINED:
            expected = False
        elif self.comparison == "lower_bound_gte":
            expected = (
                self.lower is not None and self.target is not None and self.lower >= self.target
            )
        elif self.comparison == "upper_bound_lte":
            expected = (
                self.upper is not None and self.target is not None and self.upper <= self.target
            )
        elif self.comparison == "point_lte":
            expected = (
                self.point is not None and self.target is not None and self.point <= self.target
            )
        else:
            expected = self.lower is not None and self.upper is not None
        if self.passed != expected:
            raise ValueError("readiness gate outcome disagrees with frozen comparison")
        if self.gate_sha256 != _digest(self.model_dump(mode="json", exclude={"gate_sha256"})):
            raise ValueError("readiness gate digest mismatch")
        return self


class V5ReadinessEvidence(BaseModel):
    """Final status derived only from mandatory gates and executed controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated_arm: Literal["full_sentinel"]
    gates: tuple[V5ReadinessGateEvidence, ...]
    qualifying_controls: tuple[tuple[str, bool, str], ...]
    status: Literal["ready", "not_ready"]
    readiness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def status_is_recomputed(self) -> Self:
        if not self.gates or not self.qualifying_controls:
            raise ValueError("readiness evidence is incomplete")
        expected = (
            "ready"
            if (
                all(gate.passed for gate in self.gates)
                and all(passed for _name, passed, _digest_value in self.qualifying_controls)
            )
            else "not_ready"
        )
        if self.status != expected:
            raise ValueError("readiness status disagrees with gates and controls")
        if self.readiness_sha256 != _digest(
            self.model_dump(mode="json", exclude={"readiness_sha256"})
        ):
            raise ValueError("readiness evidence digest mismatch")
        return self


class V5DeterministicCoreBinding(BaseModel):
    """Frozen exact exclusion schema and its canonical deterministic address."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-deterministic-core/1"]
    exclusion_schema: tuple[tuple[str, tuple[str, ...], str], ...]
    core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exclusion_contract_is_exact(self) -> Self:
        if self.schema_version != DETERMINISTIC_CORE_SCHEMA:
            raise ValueError("unknown deterministic core schema")
        if self.exclusion_schema != DETERMINISTIC_CORE_EXCLUSION_SCHEMA:
            raise ValueError("deterministic core exclusion schema differs")
        return self


class V5EvidencePayload(BaseModel):
    """Canonical uncompressed evidence document consumed by the verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.1.0"]
    safe_seed: Literal[404]
    evidence_protocol: V5EvidenceProtocol
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_artifact_pool: tuple[V5PackedDocument, ...]
    arm_results: tuple[V5PackedDocument, ...]
    complete_metrics: tuple[V5PackedDocument, ...]
    controls: V5PackedDocument
    readiness: V5ReadinessEvidence
    deterministic_core: V5DeterministicCoreBinding
    observational_latency: V5PackedDocument
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def payload_is_complete(self) -> Self:
        if not self.execution_artifact_pool or any(
            item.kind != "execution_artifact" for item in self.execution_artifact_pool
        ):
            raise ValueError("evidence payload execution artifact pool is incomplete")
        artifact_models = tuple(
            V5ExecutionArtifact.model_validate(item.document())
            for item in self.execution_artifact_pool
        )
        if tuple(item.evidence_sha256 for item in artifact_models) != tuple(
            sorted({item.evidence_sha256 for item in artifact_models})
        ):
            raise ValueError("execution artifact pool must be unique and canonical")
        artifact_map = {item.evidence_sha256: item for item in artifact_models}
        if any(item.kind != "arm_result" for item in self.arm_results):
            raise ValueError("arm result packed-document kinds differ")
        results = tuple(
            V5EvaluationResult.model_validate(_expand_result(item.document(), artifact_map))
            for item in self.arm_results
        )
        if any(item.kind != "complete_metrics" for item in self.complete_metrics):
            raise ValueError("complete metric packed-document kinds differ")
        metrics_documents = tuple(
            V5CompleteArmMetrics.model_validate(item.document()) for item in self.complete_metrics
        )
        if self.controls.kind != "executed_controls":
            raise ValueError("executed controls packed-document kind differs")
        control_suite = V5ExecutedControlSuite.model_validate(self.controls.document())
        if self.observational_latency.kind != "observational_latency":
            raise ValueError("observational latency packed-document kind differs")
        expected_arms = tuple(arm.value for arm in _ARM_ORDER)
        if tuple(item.arm for item in results) != expected_arms:
            raise ValueError("evidence payload requires exact ordered four arms")
        if tuple(item.arm.value for item in metrics_documents) != expected_arms:
            raise ValueError("complete metrics require exact ordered four arms")
        if len({item.support_sha256 for item in results}) != 1:
            raise ValueError("evidence arms do not share identical ordered support")
        if any(
            metrics.arm_result_sha256 != result.result_sha256
            or metrics.support_sha256 != result.support_sha256
            for metrics, result in zip(metrics_documents, results, strict=True)
        ):
            raise ValueError("complete metrics disagree with retained arm results")
        if control_suite.support_sha256 != _digest(
            tuple(row.support.event_id for row in results[0].row_evidence)
        ):
            raise ValueError("executed controls and arm results use different support")
        if (
            control_suite.evidence_protocol_sha256
            != self.evidence_protocol.evidence_protocol_sha256
            or control_suite.implementation_sha256 != self.evidence_protocol.implementation_sha256
        ):
            raise ValueError("control suite protocol binding mismatch")
        execution_documents = [item.document() for item in self.execution_artifact_pool]
        result_documents = [item.document() for item in self.arm_results]
        complete_documents = [item.document() for item in self.complete_metrics]
        controls_document = self.controls.document()
        readiness_document = self.readiness.model_dump(mode="json")
        core_document = build_deterministic_core_document(
            safe_seed=self.safe_seed,
            evidence_protocol=self.evidence_protocol.model_dump(mode="json"),
            catalog_sha256=self.catalog_sha256,
            execution_artifacts=execution_documents,
            arm_results=result_documents,
            complete_metrics=complete_documents,
            controls=controls_document,
            readiness=readiness_document,
        )
        if self.deterministic_core.core_sha256 != _digest(core_document):
            raise ValueError("deterministic core digest mismatch")
        expected_observational = build_observational_latency_document(
            deterministic_core_sha256_value=self.deterministic_core.core_sha256,
            arm_results=result_documents,
            complete_metrics=complete_documents,
            controls=controls_document,
            readiness=readiness_document,
        )
        if self.observational_latency.document() != expected_observational:
            raise ValueError("observational latency evidence mismatch")
        if self.payload_sha256 != _digest(self.model_dump(mode="json", exclude={"payload_sha256"})):
            raise ValueError("complete evidence payload digest mismatch")
        return self


class V5EvidenceEnvelope(BaseModel):
    """Deterministically compressed, content-addressed offline artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-evidence-envelope/2"]
    compression: Literal["zlib-9"]
    payload_base64: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compressed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncompressed_bytes: int = Field(gt=0, le=536_870_912)
    compressed_bytes: int = Field(gt=0, le=536_870_912)
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def payload_bytes(self) -> bytes:
        try:
            compressed = base64.b64decode(self.payload_base64, validate=True)
            payload = zlib.decompress(compressed)
        except (ValueError, zlib.error) as error:
            raise ValueError("evidence envelope payload is invalid") from error
        return payload

    def serialized_bytes(self) -> bytes:
        """Return the sole canonical on-disk envelope representation."""
        return _canonical_bytes(self)

    def payload(self) -> V5EvidencePayload:
        return V5EvidencePayload.model_validate_json(self.payload_bytes())

    @model_validator(mode="after")
    def envelope_is_bound(self) -> Self:
        try:
            compressed = base64.b64decode(self.payload_base64, validate=True)
        except ValueError as error:
            raise ValueError("evidence envelope base64 is invalid") from error
        payload = self.payload_bytes()
        if len(compressed) != self.compressed_bytes or len(payload) != self.uncompressed_bytes:
            raise ValueError("evidence envelope byte counts mismatch")
        if hashlib.sha256(compressed).hexdigest() != self.compressed_sha256:
            raise ValueError("compressed evidence digest mismatch")
        if hashlib.sha256(payload).hexdigest() != self.payload_sha256:
            raise ValueError("uncompressed evidence digest mismatch")
        document = self.model_dump(mode="json", exclude={"envelope_sha256"})
        if self.envelope_sha256 != _digest(document):
            raise ValueError("evidence envelope digest mismatch")
        self.payload()
        return self


class _V5ReadinessBootstrapSource(Protocol):
    @property
    def intervals(self) -> Sequence[V5BootstrapInterval]: ...


class _V5ReadinessFamilySource(Protocol):
    @property
    def family(self) -> str: ...


class _V5ReadinessMetricSource(Protocol):
    @property
    def arm(self) -> V5Arm: ...

    @property
    def by_family(self) -> Sequence[_V5ReadinessFamilySource]: ...

    @property
    def bootstrap(self) -> _V5ReadinessBootstrapSource: ...

    @property
    def aggregate(self) -> Mapping[str, V5MetricEstimate]: ...


def _interval(
    metrics: _V5ReadinessMetricSource, metric: str, family: str | None = None
) -> V5BootstrapInterval:
    matches = tuple(
        item
        for item in metrics.bootstrap.intervals
        if item.metric == metric and item.family == family
    )
    if len(matches) != 1:
        raise ValueError(f"required bootstrap interval is missing or duplicated: {metric}")
    return matches[0]


def _gate_from_interval(
    *,
    interval: V5BootstrapInterval,
    metric: str,
    comparison: Literal["lower_bound_gte", "upper_bound_lte", "defined_interval"],
    target: float | None,
    family: str | None = None,
) -> V5ReadinessGateEvidence:
    values = {
        "metric": metric,
        "family": family,
        "comparison": comparison,
        "target": target,
        "applicability": interval.applicability.value,
        "point": interval.point,
        "lower": interval.lower,
        "upper": interval.upper,
        "passed": (
            interval.applicability is V5MetricApplicability.DEFINED
            and (
                interval.lower is not None and target is not None and interval.lower >= target
                if comparison == "lower_bound_gte"
                else interval.upper is not None and target is not None and interval.upper <= target
                if comparison == "upper_bound_lte"
                else interval.lower is not None and interval.upper is not None
            )
        ),
        "source_sha256": interval.interval_sha256,
    }
    values["gate_sha256"] = _digest(values)
    return V5ReadinessGateEvidence.model_validate(values)


def _gate_from_point(*, metric: V5MetricEstimate, target: float) -> V5ReadinessGateEvidence:
    values = {
        "metric": metric.name,
        "family": None,
        "comparison": "point_lte",
        "target": target,
        "applicability": metric.applicability.value,
        "point": metric.value,
        "lower": None,
        "upper": None,
        "passed": (
            metric.applicability is V5MetricApplicability.DEFINED
            and metric.value is not None
            and metric.value <= target
        ),
        "source_sha256": metric.metric_sha256,
    }
    values["gate_sha256"] = _digest(values)
    return V5ReadinessGateEvidence.model_validate(values)


def _build_v5_readiness_evidence_from_source(
    *, metrics: _V5ReadinessMetricSource, controls: V5ExecutedControlSuite
) -> V5ReadinessEvidence:
    """Evaluate the frozen gates from a complete or draw-addressed metric source."""
    if metrics.arm is not V5Arm.FULL_SENTINEL:
        raise ValueError("readiness is evaluated only for full_sentinel")
    gates: list[V5ReadinessGateEvidence] = []
    for family in metrics.by_family:
        gates.append(
            _gate_from_interval(
                interval=_interval(metrics, "recall", family.family),
                metric="family_recall",
                family=family.family,
                comparison="lower_bound_gte",
                target=0.75,
            )
        )
    for metric, target in (
        ("false_decline_rate", 0.001),
        ("review_rate", 0.01),
        ("challenge_rate", 0.02),
    ):
        gates.append(
            _gate_from_interval(
                interval=_interval(metrics, metric),
                metric=("manual_review_rate" if metric == "review_rate" else metric),
                comparison="upper_bound_lte",
                target=target,
            )
        )
    gates.extend(
        (
            _gate_from_interval(
                interval=_interval(metrics, "captured_value_fraction"),
                metric="captured_value_fraction",
                comparison="lower_bound_gte",
                target=0.70,
            ),
            _gate_from_interval(
                interval=_interval(metrics, "expected_calibration_error"),
                metric="expected_calibration_error",
                comparison="upper_bound_lte",
                target=0.10,
            ),
            _gate_from_point(metric=metrics.aggregate["p95_latency_ms"], target=50.0),
            _gate_from_interval(
                interval=_interval(metrics, "campaign_detection_rate"),
                metric="campaign_detection_rate",
                comparison="defined_interval",
                target=None,
            ),
        )
    )
    qualifying = tuple(
        (control.name, control.passed, control.control_sha256)
        for control in controls.controls
        if control.qualifies_for_readiness
    )
    status = (
        "ready"
        if all(gate.passed for gate in gates) and all(item[1] for item in qualifying)
        else "not_ready"
    )
    values = {
        "evaluated_arm": V5Arm.FULL_SENTINEL.value,
        "gates": tuple(gates),
        "qualifying_controls": qualifying,
        "status": status,
    }
    values["readiness_sha256"] = _digest(
        {
            **values,
            "gates": [gate.model_dump(mode="json") for gate in gates],
        }
    )
    return V5ReadinessEvidence.model_validate(values)


def build_v5_readiness_evidence(
    *, metrics: V5CompleteArmMetrics, controls: V5ExecutedControlSuite
) -> V5ReadinessEvidence:
    """Evaluate the exact frozen full-Sentinel point and interval gates."""
    return _build_v5_readiness_evidence_from_source(metrics=metrics, controls=controls)


def build_v5_evidence_envelope(
    *,
    seed: int,
    evidence_protocol: V5EvidenceProtocol,
    catalog_sha256: str,
    arm_results: Sequence[V5EvaluationResult],
    controls: V5ExecutedControlSuite,
) -> V5EvidenceEnvelope:
    """Build one deterministic safe-seed artifact without publishing a result."""
    if seed != evidence_protocol.safe_development_test_seed or int(seed) == 2404:
        raise ValueError("evidence envelope builder requires safe seed 404")
    results = tuple(arm_results)
    if tuple(result.arm for result in results) != tuple(arm.value for arm in _ARM_ORDER):
        raise ValueError("evidence envelope requires exact ordered four arm results")
    complete = tuple(
        evaluate_v5_complete_result(result=result, protocol=evidence_protocol) for result in results
    )
    readiness = build_v5_readiness_evidence(
        metrics=complete[_ARM_ORDER.index(V5Arm.FULL_SENTINEL)],
        controls=controls,
    )
    artifacts = _collect_execution_artifacts(results)
    packed_artifacts = tuple(
        V5PackedDocument.pack(
            kind="execution_artifact",
            document=artifact.model_dump(mode="json"),
            max_uncompressed_bytes=evidence_protocol.bounds.max_single_execution_bytes,
        )
        for artifact in artifacts
    )
    aggregate_execution_bytes = sum(item.uncompressed_bytes for item in packed_artifacts)
    if aggregate_execution_bytes > evidence_protocol.bounds.max_aggregate_execution_bytes:
        raise ValueError("execution artifact pool exceeds frozen aggregate bound")
    compact_results = tuple(_compact_result(result) for result in results)
    complete_documents = tuple(metrics.model_dump(mode="json") for metrics in complete)
    controls_document = controls.model_dump(mode="json")
    readiness_document = readiness.model_dump(mode="json")
    packed_results = tuple(
        V5PackedDocument.pack(kind="arm_result", document=document)
        for document in compact_results
    )
    packed_complete = tuple(
        V5PackedDocument.pack(kind="complete_metrics", document=document)
        for document in complete_documents
    )
    packed_controls = V5PackedDocument.pack(
        kind="executed_controls", document=controls_document
    )
    core_document = build_deterministic_core_document(
        safe_seed=seed,
        evidence_protocol=evidence_protocol.model_dump(mode="json"),
        catalog_sha256=catalog_sha256,
        execution_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        arm_results=list(compact_results),
        complete_metrics=list(complete_documents),
        controls=controls_document,
        readiness=readiness_document,
    )
    deterministic_core = V5DeterministicCoreBinding(
        schema_version=DETERMINISTIC_CORE_SCHEMA,
        exclusion_schema=DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        core_sha256=_digest(core_document),
    )
    observational_document = build_observational_latency_document(
        deterministic_core_sha256_value=deterministic_core.core_sha256,
        arm_results=list(compact_results),
        complete_metrics=list(complete_documents),
        controls=controls_document,
        readiness=readiness_document,
    )
    packed_observational = V5PackedDocument.pack(
        kind="observational_latency", document=observational_document
    )
    values = {
        "schema_version": "1.1.0",
        "safe_seed": seed,
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
    values["payload_sha256"] = _digest(
        {
            key: (
                value.model_dump(mode="json")
                if isinstance(value, BaseModel)
                else [item.model_dump(mode="json") for item in value]
                if isinstance(value, tuple)
                else value
            )
            for key, value in values.items()
        }
    )
    payload = V5EvidencePayload.model_validate(values)
    raw = _canonical_bytes(payload)
    if len(raw) > evidence_protocol.bounds.max_serialized_evidence_bytes:
        raise ValueError("uncompressed evidence exceeds frozen serialized bound")
    compressed = zlib.compress(raw, level=9)
    if len(compressed) > evidence_protocol.bounds.max_serialized_evidence_bytes:
        raise ValueError("compressed evidence exceeds frozen serialized bound")
    envelope_values = {
        "schema_version": "apar-sentinel-v5-evidence-envelope/2",
        "compression": "zlib-9",
        "payload_base64": base64.b64encode(compressed).decode("ascii"),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    envelope_values["envelope_sha256"] = _digest(envelope_values)
    return V5EvidenceEnvelope.model_validate(envelope_values)


__all__ = [
    "V5DeterministicCoreBinding",
    "V5EvidenceEnvelope",
    "V5PackedDocument",
    "V5EvidencePayload",
    "V5ReadinessEvidence",
    "V5ReadinessGateEvidence",
    "build_v5_evidence_envelope",
    "build_v5_readiness_evidence",
]
