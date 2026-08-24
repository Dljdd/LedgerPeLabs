"""Actually executed, replayable controls for Sentinel v5 development evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Self
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from apar.defense.sentinel import SentinelAction, train_sentinel_defender
from apar.evaluation.v5_arms import (
    V5TrainedArmSet,
    score_v5_arm_set,
    train_v5_arm_set,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5ArmConfiguration,
    V5ArmScoreSet,
    build_v5_arm_support_rows,
    build_v5_execution_artifacts,
    build_v5_training_partition_evidence,
    derive_v5_trust_failures,
)
from apar.evaluation.v5_evidence_protocol import V5EvidenceProtocol, V5MetricApplicability
from apar.evaluation.v5_population import V5Corpus, V5DecisionRow, V5ExecutionManifest
from apar.evaluation.v5_protocol import V5DevelopmentProtocol
from apar.features.sentinel import (
    SentinelFeatureBatch,
    SentinelFeatureCatalog,
    build_sentinel_features,
)

_CONTROL_NAMES = (
    "label_shuffle",
    "identity_rename",
    "future_causality",
    "equal_time_isolation",
    "benign_only",
    "fraud_only_diagnostic",
    "feature_leakage",
)
_MAX_CONTROL_ROWS = 100_000
_MAX_CONTROL_ARTIFACTS = 4_096
_MAX_CONTROL_EVIDENCE_BYTES = 128 * 1024 * 1024


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


def _canonical_json(value: str, *, field_name: str) -> object:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} is not valid JSON") from error
    if value != json.dumps(document, sort_keys=True):
        raise ValueError(f"{field_name} must use canonical JSON")
    return document


class V5ControlMeasurement(BaseModel):
    """One explicit control quantity with denominator and support lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    applicability: V5MetricApplicability
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def values_match_applicability(self) -> Self:
        values = (self.before, self.after, self.delta, self.numerator, self.denominator)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("control measurement values must be finite")
        if self.applicability is V5MetricApplicability.DEFINED:
            if self.denominator is None or self.denominator <= 0:
                raise ValueError("defined control measurement requires a positive denominator")
            if self.numerator is None:
                raise ValueError("defined control measurement requires a numerator")
            if self.before is not None and self.after is not None:
                expected_delta = self.after - self.before
                if self.delta is None or not math.isclose(
                    self.delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("control measurement delta disagrees with before/after")
        elif self.applicability is V5MetricApplicability.UNDEFINED:
            if self.denominator != 0 or any(
                value is not None for value in (self.before, self.after, self.delta)
            ):
                raise ValueError("undefined measurement requires an explicit zero denominator")
        elif any(value is not None for value in values):
            raise ValueError("unavailable/not-applicable measurement cannot claim values")
        return self


class V5ExecutedControlResult(BaseModel):
    """Bounded evidence proving one named control was actually executed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal[
        "label_shuffle",
        "identity_rename",
        "future_causality",
        "equal_time_isolation",
        "benign_only",
        "fraud_only_diagnostic",
        "feature_leakage",
    ]
    executed: Literal[True]
    qualifies_for_readiness: bool
    spec_json: str = Field(max_length=65_536)
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_support_ids: tuple[str, ...] = Field(max_length=_MAX_CONTROL_ROWS)
    input_support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_artifact_ids: tuple[str, ...] = Field(max_length=_MAX_CONTROL_ARTIFACTS)
    input_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    permutation_seed: int | None = None
    executed_arm_spec_sha256: tuple[tuple[str, str], ...]
    measurements: tuple[V5ControlMeasurement, ...]
    criterion: str = Field(min_length=1, max_length=4_096)
    passed: bool
    row_evidence_json: str = Field(max_length=_MAX_CONTROL_EVIDENCE_BYTES)
    row_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("input_support_ids", "input_artifact_ids")
    @classmethod
    def identifiers_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("control input identifiers must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def executed_evidence_is_complete(self) -> Self:
        if self.executed is not True:
            raise ValueError("control must be actually executed")
        spec = _canonical_json(self.spec_json, field_name="control specification")
        rows = _canonical_json(self.row_evidence_json, field_name="control row evidence")
        if self.spec_sha256 != _canonical_digest(spec):
            raise ValueError("control specification digest mismatch")
        if self.input_support_sha256 != _canonical_digest(self.input_support_ids):
            raise ValueError("control support digest mismatch")
        if self.input_artifact_sha256 != _canonical_digest(self.input_artifact_ids):
            raise ValueError("control artifact digest mismatch")
        if self.row_evidence_sha256 != _canonical_digest(rows):
            raise ValueError("control row evidence digest mismatch")
        if not self.executed_arm_spec_sha256 or any(
            not arm or len(digest) != 64 for arm, digest in self.executed_arm_spec_sha256
        ):
            raise ValueError("control executed arm specification evidence is incomplete")
        if not self.measurements:
            raise ValueError("executed control must retain measured values")
        if any(
            measurement.support_sha256 != self.input_support_sha256
            for measurement in self.measurements
        ):
            raise ValueError("control measurement support disagrees with control input")
        document = self.model_dump(mode="json", exclude={"control_sha256"})
        if self.control_sha256 != _canonical_digest(document):
            raise ValueError("executed control digest mismatch")
        return self


@dataclass(frozen=True, slots=True)
class _ControlRuntime:
    trained: V5TrainedArmSet
    scores: V5ArmScoreSet
    development_rows: tuple[V5DecisionRow, ...]
    development_executions: tuple[V5ExecutionManifest, ...]
    development_batch: SentinelFeatureBatch
    development_matrix: NDArray[np.float64]
    train_matrix: NDArray[np.float64]
    train_labels: NDArray[np.int_]
    calibration_matrix: NDArray[np.float64]
    calibration_labels: NDArray[np.int_]
    threshold_matrix: NDArray[np.float64]
    threshold_labels: NDArray[np.int_]
    future_rows: tuple[V5DecisionRow, ...]
    future_executions: tuple[V5ExecutionManifest, ...]


def _feature_matrix(
    rows: tuple[V5DecisionRow, ...], catalog: SentinelFeatureCatalog
) -> tuple[NDArray[np.float64], SentinelFeatureBatch]:
    batch = build_sentinel_features(rows, catalog=catalog)
    by_event = {
        provenance.event_id: tuple(values)
        for provenance, values in zip(batch.provenance, batch.matrix, strict=True)
    }
    matrix = np.array([by_event[row.event_id] for row in rows], dtype=np.float64)
    return matrix, batch


def _training_partition(
    *,
    name: Literal["train", "calibration", "threshold"],
    rows: tuple[V5DecisionRow, ...],
    executions: tuple[V5ExecutionManifest, ...],
    catalog: SentinelFeatureCatalog,
) -> tuple[NDArray[np.float64], NDArray[np.int_], SentinelFeatureBatch, object]:
    matrix, batch = _feature_matrix(rows, catalog)
    labels = np.array([int(row.is_fraud) for row in rows], dtype=int)
    evidence = build_v5_training_partition_evidence(
        partition=name,
        event_ids=tuple(row.event_id for row in rows),
        labels=labels,
        support=build_v5_arm_support_rows(rows),
        feature_batch_sha256=batch.batch_sha256,
        feature_matrix=matrix,
        feature_names=catalog.feature_names,
        catalog_sha256=catalog.catalog_sha256,
        execution_manifests=executions,
        feature_batch_source_matrix=batch.matrix,
    )
    return matrix, labels, batch, evidence


def _build_runtime(
    *,
    protocol: V5DevelopmentProtocol,
    corpus: V5Corpus,
    catalog: SentinelFeatureCatalog,
    configuration: V5ArmConfiguration,
) -> _ControlRuntime:
    train = corpus.partitions["train"]
    calibration = corpus.partitions["calibration"]
    threshold = corpus.partitions["threshold"]
    development = corpus.partitions["development_test"]
    future = corpus.partitions["hardening_train"]
    x_train, y_train, _train_batch, train_evidence = _training_partition(
        name="train",
        rows=train.decisions,
        executions=train.executions,
        catalog=catalog,
    )
    x_cal, y_cal, _cal_batch, cal_evidence = _training_partition(
        name="calibration",
        rows=calibration.decisions,
        executions=calibration.executions,
        catalog=catalog,
    )
    x_threshold, y_threshold, _threshold_batch, threshold_evidence = _training_partition(
        name="threshold",
        rows=threshold.decisions,
        executions=threshold.executions,
        catalog=catalog,
    )
    trained = train_v5_arm_set(
        configuration=configuration,
        catalog=catalog,
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        bootstrap_seed=protocol.seeds.bootstrap,
        train_evidence=train_evidence,  # type: ignore[arg-type]
        calibration_evidence=cal_evidence,  # type: ignore[arg-type]
        threshold_evidence=threshold_evidence,  # type: ignore[arg-type]
    )
    x_development, development_batch = _feature_matrix(development.decisions, catalog)
    support = build_v5_arm_support_rows(development.decisions)
    artifacts = build_v5_execution_artifacts(development.executions)
    scores = score_v5_arm_set(
        trained=trained,
        catalog=catalog,
        features_matrix=x_development,
        support=support,
        execution_artifacts=artifacts,
        trust_failures=derive_v5_trust_failures(support, artifacts),
        feature_provenance=development_batch.provenance,
    )
    return _ControlRuntime(
        trained=trained,
        scores=scores,
        development_rows=development.decisions,
        development_executions=development.executions,
        development_batch=development_batch,
        development_matrix=x_development,
        train_matrix=x_train,
        train_labels=y_train,
        calibration_matrix=x_cal,
        calibration_labels=y_cal,
        threshold_matrix=x_threshold,
        threshold_labels=y_threshold,
        future_rows=future.decisions,
        future_executions=future.executions,
    )


def _score_rows(
    *,
    runtime: _ControlRuntime,
    rows: tuple[V5DecisionRow, ...],
    executions: tuple[V5ExecutionManifest, ...],
    catalog: SentinelFeatureCatalog,
    matrix: NDArray[np.float64] | None = None,
    provenance: tuple[object, ...] | None = None,
) -> V5ArmScoreSet:
    built_matrix, batch = _feature_matrix(rows, catalog)
    selected_matrix = built_matrix if matrix is None else matrix
    support = build_v5_arm_support_rows(rows)
    artifacts = build_v5_execution_artifacts(executions)
    return score_v5_arm_set(
        trained=runtime.trained,
        catalog=catalog,
        features_matrix=selected_matrix,
        support=support,
        execution_artifacts=artifacts,
        trust_failures=derive_v5_trust_failures(support, artifacts),
        feature_provenance=(
            batch.provenance if provenance is None else tuple(provenance)  # type: ignore[arg-type]
        ),
    )


def _score_signature(scores: V5ArmScoreSet) -> dict[str, tuple[tuple[float, str], ...]]:
    return {
        arm.value: tuple((row.probability, row.action.value) for row in score.rows)
        for arm, score in scores.by_arm.items()
    }


def _support_digest(ids: tuple[str, ...]) -> str:
    return _canonical_digest(ids)


def _measurement(
    *,
    name: str,
    support_ids: tuple[str, ...],
    before: float | None,
    after: float | None,
    numerator: float | None,
    denominator: float | None,
    applicability: V5MetricApplicability = V5MetricApplicability.DEFINED,
) -> V5ControlMeasurement:
    delta = after - before if before is not None and after is not None else None
    return V5ControlMeasurement(
        name=name,
        applicability=applicability,
        before=before,
        after=after,
        delta=delta,
        numerator=numerator,
        denominator=denominator,
        support_sha256=_support_digest(support_ids),
    )


def _control_result(
    *,
    name: str,
    qualifies_for_readiness: bool,
    spec: object,
    support_ids: tuple[str, ...],
    artifact_ids: tuple[str, ...],
    permutation_seed: int | None,
    arm_digests: tuple[tuple[str, str], ...],
    measurements: tuple[V5ControlMeasurement, ...],
    criterion: str,
    passed: bool,
    row_evidence: object,
    implementation_sha256: str,
) -> V5ExecutedControlResult:
    spec_json = json.dumps(spec, sort_keys=True)
    row_json = json.dumps(row_evidence, sort_keys=True)
    values = {
        "name": name,
        "executed": True,
        "qualifies_for_readiness": qualifies_for_readiness,
        "spec_json": spec_json,
        "spec_sha256": _canonical_digest(json.loads(spec_json)),
        "input_support_ids": support_ids,
        "input_support_sha256": _support_digest(support_ids),
        "input_artifact_ids": artifact_ids,
        "input_artifact_sha256": _canonical_digest(artifact_ids),
        "permutation_seed": permutation_seed,
        "executed_arm_spec_sha256": arm_digests,
        "measurements": measurements,
        "criterion": criterion,
        "passed": passed,
        "row_evidence_json": row_json,
        "row_evidence_sha256": _canonical_digest(json.loads(row_json)),
        "implementation_sha256": implementation_sha256,
    }
    digest_values = dict(values)
    digest_values["measurements"] = [item.model_dump(mode="json") for item in measurements]
    values["control_sha256"] = _canonical_digest(digest_values)
    return V5ExecutedControlResult.model_validate(values)


def _baseline_arm_digests(runtime: _ControlRuntime) -> tuple[tuple[str, str], ...]:
    return tuple(
        (arm.value, score.spec.spec_sha256) for arm, score in runtime.scores.by_arm.items()
    )


def _metric_pair(
    labels: NDArray[np.int_], probabilities: NDArray[np.float64]
) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


def _label_shuffle_control(
    *,
    runtime: _ControlRuntime,
    evidence_protocol: V5EvidenceProtocol,
    catalog: SentinelFeatureCatalog,
) -> V5ExecutedControlResult:
    seed = evidence_protocol.controls.label_shuffle.permutation_seed
    rng = np.random.default_rng(seed)
    permutations = (
        rng.permutation(len(runtime.train_labels)),
        rng.permutation(len(runtime.calibration_labels)),
        rng.permutation(len(runtime.threshold_labels)),
    )
    shuffled = (
        runtime.train_labels[permutations[0]],
        runtime.calibration_labels[permutations[1]],
        runtime.threshold_labels[permutations[2]],
    )
    labels = np.array([int(row.is_fraud) for row in runtime.development_rows], dtype=int)
    before_roc: list[float] = []
    before_pr: list[float] = []
    after_roc: list[float] = []
    after_pr: list[float] = []
    arm_digests: list[tuple[str, str]] = []
    row_evidence: list[dict[str, object]] = []
    templates = {item.spec.arm: item for item in runtime.trained.arms}
    for arm in (
        V5Arm.ENSEMBLE_NO_GRAPH,
        V5Arm.ENSEMBLE_WITH_GRAPH,
        V5Arm.FULL_SENTINEL,
    ):
        trained = templates[arm]
        assert trained.defender is not None
        indices = list(trained.feature_indices)
        defender = train_sentinel_defender(
            x_train=runtime.train_matrix[:, indices],
            y_train=shuffled[0],
            x_calibration=runtime.calibration_matrix[:, indices],
            y_calibration=shuffled[1],
            x_threshold=runtime.threshold_matrix[:, indices],
            y_threshold=shuffled[2],
            catboost_seeds=trained.spec.model_seeds,
            bootstrap_seed=seed,
            enable_novelty=trained.spec.novelty,
        )
        baseline_probability = np.array(
            [row.probability for row in runtime.scores.by_arm[arm].rows], dtype=float
        )
        shuffled_probability = np.array(
            [defender.predict_member_scores(row[indices])[2] for row in runtime.development_matrix],
            dtype=float,
        )
        baseline_metrics = _metric_pair(labels, baseline_probability)
        shuffled_metrics = _metric_pair(labels, shuffled_probability)
        before_roc.append(baseline_metrics[0])
        before_pr.append(baseline_metrics[1])
        after_roc.append(shuffled_metrics[0])
        after_pr.append(shuffled_metrics[1])
        model_hashes = tuple(
            hashlib.sha256(model._serialize_model()).hexdigest() for model in defender.model_members
        )
        arm_digest = _canonical_digest(
            {
                "arm": arm.value,
                "permutation_seed": seed,
                "model_hashes": model_hashes,
                "thresholds": defender.thresholds.model_dump(mode="json"),
                "indices": indices,
            }
        )
        arm_digests.append((arm.value, arm_digest))
        row_evidence.append(
            {
                "arm": arm.value,
                "baseline_roc_auc": baseline_metrics[0],
                "shuffled_roc_auc": shuffled_metrics[0],
                "baseline_pr_auc": baseline_metrics[1],
                "shuffled_pr_auc": shuffled_metrics[1],
                "baseline_probability_sha256": _canonical_digest(baseline_probability.tolist()),
                "shuffled_probability_sha256": _canonical_digest(shuffled_probability.tolist()),
                "baseline_probabilities": baseline_probability.tolist(),
                "shuffled_probabilities": shuffled_probability.tolist(),
                "control_arm_sha256": arm_digest,
            }
        )
    mean_before_roc = float(np.mean(before_roc))
    mean_after_roc = float(np.mean(after_roc))
    mean_before_pr = float(np.mean(before_pr))
    mean_after_pr = float(np.mean(after_pr))
    support_ids = tuple(row.event_id for row in runtime.development_rows)
    prevalence = float(labels.mean())
    criteria = evidence_protocol.controls.label_shuffle
    passed = all(
        shuffled_roc <= criteria.max_roc_auc
        and shuffled_pr <= prevalence + criteria.max_pr_auc_excess_over_prevalence
        and baseline_roc - shuffled_roc >= criteria.min_roc_auc_delta
        for baseline_roc, shuffled_roc, shuffled_pr in zip(
            before_roc, after_roc, after_pr, strict=True
        )
    )
    artifacts = build_v5_execution_artifacts(runtime.development_executions)
    measurements = (
        _measurement(
            name="roc_auc",
            support_ids=support_ids,
            before=mean_before_roc,
            after=mean_after_roc,
            numerator=float(len(labels)),
            denominator=float(len(labels)),
        ),
        _measurement(
            name="pr_auc",
            support_ids=support_ids,
            before=mean_before_pr,
            after=mean_after_pr,
            numerator=float(labels.sum()),
            denominator=float(len(labels)),
        ),
        _measurement(
            name="roc_auc_delta",
            support_ids=support_ids,
            before=mean_after_roc,
            after=mean_before_roc,
            numerator=mean_before_roc - mean_after_roc,
            denominator=1.0,
        ),
    )
    return _control_result(
        name="label_shuffle",
        qualifies_for_readiness=True,
        spec={"name": "label_shuffle", **criteria.model_dump(mode="json")},
        support_ids=support_ids,
        artifact_ids=tuple(item.evidence_sha256 for item in artifacts),
        permutation_seed=seed,
        arm_digests=tuple(arm_digests),
        measurements=measurements,
        criterion=(
            "each learned arm: shuffled ROC-AUC <= 0.70, shuffled PR-AUC <= "
            "prevalence + 0.20, and baseline minus shuffled ROC-AUC >= 0.05"
        ),
        passed=passed,
        row_evidence={
            "labels": labels.tolist(),
            "partition_labels": [
                runtime.train_labels.tolist(),
                runtime.calibration_labels.tolist(),
                runtime.threshold_labels.tolist(),
            ],
            "permuted_labels": [item.tolist() for item in shuffled],
            "permutation_indices": [item.tolist() for item in permutations],
            "permutation_indices_sha256": _canonical_digest(
                [item.tolist() for item in permutations]
            ),
            "prevalence": prevalence,
            "arms": row_evidence,
        },
        implementation_sha256=evidence_protocol.implementation_sha256,
    )


def _identity_projection(
    *,
    rows: tuple[V5DecisionRow, ...],
    executions: tuple[V5ExecutionManifest, ...],
    namespace: str,
) -> tuple[dict[str, str], dict[str, object]]:
    domains: dict[str, set[str]] = {
        "account": set(),
        "actor": set(),
        "authentication_evidence": set(),
        "campaign": set(),
        "command": set(),
        "counterparty": set(),
        "credential": set(),
        "device": set(),
        "event": set(),
        "evidence": set(),
        "merchant": set(),
        "payee": set(),
        "payment": set(),
        "request": set(),
    }
    relationships: list[tuple[str, ...]] = []
    for execution in executions:
        domains["campaign"].add(execution.campaign_id)
        domains["evidence"].update({execution.evidence_sha256, execution.artifact_sha256})
        domains["account"].update(execution.account_ids)
        domains["credential"].update(execution.credential_ids)
        domains["device"].update(execution.device_ids)
        domains["merchant"].update(execution.merchant_ids)
        domains["payee"].update(execution.payee_ids)
        domains["request"].update(execution.trust_request_ids)
        domains["authentication_evidence"].update(execution.authentication_evidence_ids)
        for link in execution.lineage:
            domains["command"].add(link.command_id)
            domains["event"].add(link.event_id)
            domains["payment"].add(link.payment_id)
            domains["actor"].add(link.actor_id)
            domains["counterparty"].add(link.counterparty_id)
            relationships.append(
                (
                    execution.evidence_sha256,
                    execution.campaign_id,
                    link.command_id,
                    link.event_id,
                    link.payment_id,
                    link.actor_id,
                    link.counterparty_id,
                )
            )
        for trust in execution.trust_records:
            domains["request"].add(trust.request_id)
            domains["evidence"].update(
                {
                    trust.receipt_hash,
                    trust.request_hash,
                    trust.signature_hash,
                }
            )
            if trust.authentication_evidence_id is not None:
                domains["authentication_evidence"].add(trust.authentication_evidence_id)
    for row in rows:
        domains["actor"].add(row.actor_id)
        domains["counterparty"].add(row.counterparty_id)
        domains["campaign"].add(row.campaign_id)
        domains["payment"].add(row.payment_id)
        domains["event"].update({row.event_id, row.source_event_id})
        domains["command"].add(row.source_command_id)
        domains["evidence"].add(row.execution_evidence_sha256)
    all_values = set().union(*domains.values())
    mapping = {
        value: str(uuid5(NAMESPACE_URL, f"{namespace}:{value}")) for value in sorted(all_values)
    }
    domain_evidence = {
        name: {
            "count": len(values),
            "original_sha256": _canonical_digest(sorted(values)),
            "renamed_sha256": _canonical_digest(sorted(mapping[value] for value in values)),
        }
        for name, values in sorted(domains.items())
    }
    renamed_relationships = [
        tuple(mapping[value] for value in relationship) for relationship in relationships
    ]
    return mapping, {
        "identity_domains": domain_evidence,
        "bijection": sorted(mapping.items()),
        "original_relationships": relationships,
        "renamed_relationships": renamed_relationships,
        "original_relationship_sha256": _canonical_digest(relationships),
        "renamed_relationship_sha256": _canonical_digest(renamed_relationships),
        "bijection_sha256": _canonical_digest(mapping),
        "bijection_size": len(mapping),
    }


def _renamed_rows(
    rows: tuple[V5DecisionRow, ...], mapping: dict[str, str]
) -> tuple[V5DecisionRow, ...]:
    renamed = tuple(
        row.model_copy(
            update={
                "actor_id": mapping[row.actor_id],
                "counterparty_id": mapping[row.counterparty_id],
                "campaign_id": mapping[row.campaign_id],
                "payment_id": mapping[row.payment_id],
                "event_id": mapping[row.event_id],
                "source_command_id": mapping[row.source_command_id],
                "source_event_id": mapping[row.source_event_id],
                "execution_evidence_sha256": mapping[row.execution_evidence_sha256],
            }
        )
        for row in rows
    )
    return renamed


def _aligned_matrix(
    *,
    source_rows: tuple[V5DecisionRow, ...],
    transformed_rows: tuple[V5DecisionRow, ...],
    event_mapping: dict[str, str],
    catalog: SentinelFeatureCatalog,
) -> NDArray[np.float64]:
    batch = build_sentinel_features(transformed_rows, catalog=catalog)
    by_event = {
        provenance.event_id: values
        for provenance, values in zip(batch.provenance, batch.matrix, strict=True)
    }
    return np.array([by_event[event_mapping[row.event_id]] for row in source_rows], dtype=float)


def _invariance_control(
    *,
    name: Literal["identity_rename", "future_causality", "equal_time_isolation", "feature_leakage"],
    runtime: _ControlRuntime,
    evidence_protocol: V5EvidenceProtocol,
    catalog: SentinelFeatureCatalog,
) -> V5ExecutedControlResult:
    rows = runtime.development_rows
    mapping: dict[str, str] = {row.event_id: row.event_id for row in rows}
    evidence_extra: dict[str, object] = {}
    auxiliary_artifact_ids: tuple[str, ...] = ()
    spec: BaseModel
    if name == "identity_rename":
        full_mapping, identity_evidence = _identity_projection(
            rows=rows,
            executions=runtime.development_executions,
            namespace=evidence_protocol.controls.identity_rename.namespace,
        )
        transformed = _renamed_rows(rows, full_mapping)
        mapping = {
            row.event_id: renamed.event_id for row, renamed in zip(rows, transformed, strict=True)
        }
        after_matrix = _aligned_matrix(
            source_rows=rows,
            transformed_rows=transformed,
            event_mapping=mapping,
            catalog=catalog,
        )
        evidence_extra.update(identity_evidence)
        spec = evidence_protocol.controls.identity_rename
    elif name == "future_causality":
        maximum = max(row.decision_at for row in rows)
        future = tuple(row for row in runtime.future_rows if row.decision_at > maximum)[:8]
        if len(future) != 8:
            raise ValueError("safe future-causality control lacks retained later events")
        retained_future_evidence = {
            execution.evidence_sha256 for execution in runtime.future_executions
        }
        if any(row.execution_evidence_sha256 not in retained_future_evidence for row in future):
            raise ValueError("future control row lacks retained execution evidence")
        if any(
            row.decision_at
            < maximum
            + timedelta(seconds=evidence_protocol.controls.future_causality.offset_seconds)
            for row in future
        ):
            raise ValueError("future control rows do not satisfy frozen time offset")
        selected_evidence_ids = {row.execution_evidence_sha256 for row in future}
        selected_future_executions = tuple(
            execution
            for execution in runtime.future_executions
            if execution.evidence_sha256 in selected_evidence_ids
        )
        future_artifacts = build_v5_execution_artifacts(selected_future_executions)
        if {item.evidence_sha256 for item in future_artifacts} != selected_evidence_ids:
            raise ValueError("future control artifacts do not cover inserted rows")
        auxiliary_artifact_ids = tuple(item.evidence_sha256 for item in future_artifacts)
        combined = (*rows, *future)
        batch = build_sentinel_features(combined, catalog=catalog)
        by_event = {
            item.event_id: values
            for item, values in zip(batch.provenance, batch.matrix, strict=True)
        }
        after_matrix = np.array([by_event[row.event_id] for row in rows], dtype=float)
        evidence_extra["inserted_event_ids"] = [row.event_id for row in future]
        evidence_extra["baseline_max_decision_at"] = maximum.isoformat()
        evidence_extra["inserted_events"] = [
            {
                "event_id": row.event_id,
                "decision_at": row.decision_at.isoformat(),
                "execution_evidence_sha256": row.execution_evidence_sha256,
            }
            for row in future
        ]
        evidence_extra["inserted_execution_evidence_sha256"] = sorted(
            {row.execution_evidence_sha256 for row in future}
        )
        evidence_extra["inserted_execution_artifacts"] = [
            item.model_dump(mode="json") for item in future_artifacts
        ]
        evidence_extra["future_rows_are_retained_execution_evidence"] = True
        spec = evidence_protocol.controls.future_causality
    elif name == "equal_time_isolation":
        by_time: dict[datetime, list[V5DecisionRow]] = {}
        for row in rows:
            by_time.setdefault(row.decision_at, []).append(row)
        peers = next(
            (
                tuple(items[: evidence_protocol.controls.equal_time.peer_count])
                for _timestamp, items in sorted(by_time.items(), key=lambda item: item[0])
                if len(items) >= evidence_protocol.controls.equal_time.peer_count
            ),
            (),
        )
        if len(peers) != evidence_protocol.controls.equal_time.peer_count:
            raise ValueError("safe equal-time control lacks a real peer cohort")
        peer_ids = {peer.event_id for peer in peers}
        permuted = list(rows)
        positions = [index for index, row in enumerate(rows) if row.event_id in peer_ids]
        for position, peer in zip(positions, reversed(peers), strict=True):
            permuted[position] = peer
        batch = build_sentinel_features(tuple(permuted), catalog=catalog)
        by_event = {
            item.event_id: values
            for item, values in zip(batch.provenance, batch.matrix, strict=True)
        }
        after_matrix = np.array([by_event[row.event_id] for row in rows], dtype=float)
        provenance = {item.event_id: item for item in batch.provenance}
        peers_do_not_observe_each_other = all(
            not (set(provenance[peer.event_id].source_event_ids) & peer_ids) for peer in peers
        )
        evidence_extra["peer_event_ids"] = [row.event_id for row in peers]
        evidence_extra["peer_decision_at"] = peers[0].decision_at.isoformat()
        evidence_extra["peer_source_event_ids"] = {
            peer.event_id: list(provenance[peer.event_id].source_event_ids) for peer in peers
        }
        evidence_extra["peers_do_not_observe_each_other"] = peers_do_not_observe_each_other
        spec = evidence_protocol.controls.equal_time
    else:
        transformed = tuple(
            row.model_copy(
                update={
                    "is_fraud": not row.is_fraud,
                    "family": f"mutated-family-{index}",
                    "campaign_id": str(uuid5(NAMESPACE_URL, f"leak-campaign:{row.campaign_id}")),
                    "lifecycle_state": "mutated-final-outcome",
                    "predictive_features": {
                        **row.predictive_features,
                        "split": float(index + 1),
                        "seed": float(index + 2),
                        "generator": float(index + 3),
                        "final_outcome": float(index + 4),
                    },
                }
            )
            for index, row in enumerate(rows)
        )
        mapping = {row.event_id: row.event_id for row in rows}
        after_matrix = _aligned_matrix(
            source_rows=rows,
            transformed_rows=transformed,
            event_mapping=mapping,
            catalog=catalog,
        )
        evidence_extra["mutated_fields"] = list(
            evidence_protocol.controls.feature_leakage.forbidden_fields
        )
        evidence_extra["mutated_rows"] = [
            {
                "event_id": row.event_id,
                "is_fraud": row.is_fraud,
                "family": row.family,
                "campaign_id": row.campaign_id,
                "lifecycle_state": row.lifecycle_state,
                "split": row.predictive_features["split"],
                "seed": row.predictive_features["seed"],
                "generator": row.predictive_features["generator"],
                "final_outcome": row.predictive_features["final_outcome"],
            }
            for row in transformed
        ]
        spec = evidence_protocol.controls.feature_leakage
    matrix_equal = np.array_equal(runtime.development_matrix, after_matrix)
    rescored = _score_rows(
        runtime=runtime,
        rows=rows,
        executions=runtime.development_executions,
        catalog=catalog,
        matrix=after_matrix,
        provenance=runtime.development_batch.provenance,
    )
    baseline_signature = _score_signature(runtime.scores)
    after_signature = _score_signature(rescored)
    score_equal = baseline_signature == after_signature
    support_ids = tuple(row.event_id for row in rows)
    artifacts = build_v5_execution_artifacts(runtime.development_executions)
    row_count = float(len(rows))
    measurements = (
        _measurement(
            name="numeric_feature_rows",
            support_ids=support_ids,
            before=row_count,
            after=row_count if matrix_equal else 0.0,
            numerator=row_count if matrix_equal else 0.0,
            denominator=row_count,
        ),
        _measurement(
            name="prediction_rows",
            support_ids=support_ids,
            before=row_count,
            after=row_count if score_equal else 0.0,
            numerator=row_count if score_equal else 0.0,
            denominator=row_count,
        ),
    )
    return _control_result(
        name=name,
        qualifies_for_readiness=True,
        spec={"name": name, **spec.model_dump(mode="json")},
        support_ids=support_ids,
        artifact_ids=(
            *(item.evidence_sha256 for item in artifacts),
            *auxiliary_artifact_ids,
        ),
        permutation_seed=None,
        arm_digests=_baseline_arm_digests(runtime),
        measurements=measurements,
        criterion="exact numeric-feature, probability, rule, and action invariance",
        passed=matrix_equal
        and score_equal
        and bool(evidence_extra.get("peers_do_not_observe_each_other", True)),
        row_evidence={
            "before_matrix": runtime.development_matrix.tolist(),
            "after_matrix": after_matrix.tolist(),
            "before_matrix_sha256": _canonical_digest(runtime.development_matrix.tolist()),
            "after_matrix_sha256": _canonical_digest(after_matrix.tolist()),
            "before_score_signature": baseline_signature,
            "after_score_signature": after_signature,
            "before_score_sha256": _canonical_digest(baseline_signature),
            "after_score_sha256": _canonical_digest(after_signature),
            **evidence_extra,
        },
        implementation_sha256=evidence_protocol.implementation_sha256,
    )


def _single_class_control(
    *,
    fraud: bool,
    runtime: _ControlRuntime,
    evidence_protocol: V5EvidenceProtocol,
    catalog: SentinelFeatureCatalog,
) -> V5ExecutedControlResult:
    selected_executions = tuple(
        execution
        for execution in runtime.development_executions
        if (execution.family != "legitimate") is fraud
    )
    evidence_ids = {execution.evidence_sha256 for execution in selected_executions}
    rows = tuple(
        row for row in runtime.development_rows if row.execution_evidence_sha256 in evidence_ids
    )
    scores = _score_rows(
        runtime=runtime,
        rows=rows,
        executions=selected_executions,
        catalog=catalog,
    )
    support_ids = tuple(row.event_id for row in rows)
    arm_evidence: dict[str, object] = {}
    arm_measurements: list[V5ControlMeasurement] = []
    arm_workload_passes: list[bool] = []
    spec: BaseModel
    for arm, score in scores.by_arm.items():
        actions = tuple(row.action for row in score.rows)
        probabilities = np.array([row.probability for row in score.rows], dtype=float)
        latencies = np.array([row.latency_ms for row in score.rows], dtype=float)
        summary: dict[str, object] = {
            "score_sha256": score.score_sha256,
            "rows": [
                {
                    "event_id": row.support.event_id,
                    "action": row.action.value,
                    "probability": row.probability,
                    "latency_ms": row.latency_ms,
                }
                for row in score.rows
            ],
        }
        if fraud:
            interventions = sum(action is not SentinelAction.APPROVE for action in actions)
            recall = interventions / len(rows)
            arm_measurements.extend(
                (
                    _measurement(
                        name=f"{arm.value}.recall",
                        support_ids=support_ids,
                        before=recall,
                        after=recall,
                        numerator=float(interventions),
                        denominator=float(len(rows)),
                    ),
                    _measurement(
                        name=f"{arm.value}.p95_latency_ms",
                        support_ids=support_ids,
                        before=float(np.percentile(latencies, 95)),
                        after=float(np.percentile(latencies, 95)),
                        numerator=float(np.percentile(latencies, 95)),
                        denominator=1.0,
                    ),
                )
            )
            summary.update(
                {
                    "recall_numerator": interventions,
                    "recall_denominator": len(rows),
                    "probability_quantiles": (
                        None
                        if arm is V5Arm.RULES_ONLY
                        else [
                            float(np.percentile(probabilities, percentile))
                            for percentile in (5, 50, 95)
                        ]
                    ),
                    "probability_semantics": (
                        "not_applicable" if arm is V5Arm.RULES_ONLY else "calibrated_probability"
                    ),
                }
            )
        else:
            declines = sum(action is SentinelAction.DECLINE_HOLD for action in actions)
            challenges = sum(action is SentinelAction.CHALLENGE for action in actions)
            reviews = sum(action is SentinelAction.REVIEW_HOLD for action in actions)
            false_decline_rate = declines / len(rows)
            challenge_rate = challenges / len(rows)
            review_rate = reviews / len(rows)
            arm_workload_passes.append(
                false_decline_rate <= evidence_protocol.controls.benign_only.false_decline_rate_max
                and challenge_rate <= evidence_protocol.controls.benign_only.challenge_rate_max
                and review_rate <= evidence_protocol.controls.benign_only.review_rate_max
            )
            arm_measurements.extend(
                (
                    _measurement(
                        name=f"{arm.value}.false_decline_rate",
                        support_ids=support_ids,
                        before=false_decline_rate,
                        after=false_decline_rate,
                        numerator=float(declines),
                        denominator=float(len(rows)),
                    ),
                    _measurement(
                        name=f"{arm.value}.challenge_rate",
                        support_ids=support_ids,
                        before=challenge_rate,
                        after=challenge_rate,
                        numerator=float(challenges),
                        denominator=float(len(rows)),
                    ),
                    _measurement(
                        name=f"{arm.value}.review_rate",
                        support_ids=support_ids,
                        before=review_rate,
                        after=review_rate,
                        numerator=float(reviews),
                        denominator=float(len(rows)),
                    ),
                    _measurement(
                        name=f"{arm.value}.p95_latency_ms",
                        support_ids=support_ids,
                        before=float(np.percentile(latencies, 95)),
                        after=float(np.percentile(latencies, 95)),
                        numerator=float(np.percentile(latencies, 95)),
                        denominator=1.0,
                    ),
                )
            )
            summary.update(
                {
                    "false_decline_numerator": declines,
                    "challenge_numerator": challenges,
                    "review_numerator": reviews,
                    "legitimate_denominator": len(rows),
                    "probability_quantiles": (
                        None
                        if arm is V5Arm.RULES_ONLY
                        else [
                            float(np.percentile(probabilities, percentile))
                            for percentile in (5, 50, 95)
                        ]
                    ),
                    "brier": (
                        None
                        if arm is V5Arm.RULES_ONLY
                        else float(np.mean(np.square(probabilities)))
                    ),
                    "calibration_semantics": (
                        "not_applicable" if arm is V5Arm.RULES_ONLY else "all_observed_labels_zero"
                    ),
                }
            )
        arm_evidence[arm.value] = summary
    full = scores.by_arm[V5Arm.FULL_SENTINEL]
    full_actions = tuple(row.action for row in full.rows)
    full_probabilities = np.array([row.probability for row in full.rows], dtype=float)
    full_latencies = np.array([row.latency_ms for row in full.rows], dtype=float)
    if fraud:
        interventions = sum(action is not SentinelAction.APPROVE for action in full_actions)
        measurements = (
            _measurement(
                name="recall",
                support_ids=support_ids,
                before=float(interventions / len(rows)),
                after=float(interventions / len(rows)),
                numerator=float(interventions),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="false_decline_rate",
                support_ids=support_ids,
                before=None,
                after=None,
                numerator=0.0,
                denominator=0.0,
                applicability=V5MetricApplicability.UNDEFINED,
            ),
            _measurement(
                name="mean_probability",
                support_ids=support_ids,
                before=float(full_probabilities.mean()),
                after=float(full_probabilities.mean()),
                numerator=float(full_probabilities.sum()),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="roc_auc",
                support_ids=support_ids,
                before=None,
                after=None,
                numerator=float(len(rows)),
                denominator=0.0,
                applicability=V5MetricApplicability.UNDEFINED,
            ),
            _measurement(
                name="pr_auc",
                support_ids=support_ids,
                before=None,
                after=None,
                numerator=float(len(rows)),
                denominator=0.0,
                applicability=V5MetricApplicability.UNDEFINED,
            ),
            *arm_measurements,
        )
        name = "fraud_only_diagnostic"
        qualifies = False
        passed = True
        criterion = "diagnostic executes over fraud-only support and is barred from readiness"
        spec = evidence_protocol.controls.fraud_only
    else:
        declines = sum(action is SentinelAction.DECLINE_HOLD for action in full_actions)
        challenges = sum(action is SentinelAction.CHALLENGE for action in full_actions)
        reviews = sum(action is SentinelAction.REVIEW_HOLD for action in full_actions)
        measurements = (
            _measurement(
                name="recall",
                support_ids=support_ids,
                before=None,
                after=None,
                numerator=0.0,
                denominator=0.0,
                applicability=V5MetricApplicability.UNDEFINED,
            ),
            _measurement(
                name="false_decline_rate",
                support_ids=support_ids,
                before=declines / len(rows),
                after=declines / len(rows),
                numerator=float(declines),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="challenge_rate",
                support_ids=support_ids,
                before=challenges / len(rows),
                after=challenges / len(rows),
                numerator=float(challenges),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="review_rate",
                support_ids=support_ids,
                before=reviews / len(rows),
                after=reviews / len(rows),
                numerator=float(reviews),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="mean_probability",
                support_ids=support_ids,
                before=float(full_probabilities.mean()),
                after=float(full_probabilities.mean()),
                numerator=float(full_probabilities.sum()),
                denominator=float(len(rows)),
            ),
            _measurement(
                name="p95_latency_ms",
                support_ids=support_ids,
                before=float(np.percentile(full_latencies, 95)),
                after=float(np.percentile(full_latencies, 95)),
                numerator=float(np.percentile(full_latencies, 95)),
                denominator=1.0,
            ),
            *arm_measurements,
        )
        spec = evidence_protocol.controls.benign_only
        name = "benign_only"
        qualifies = True
        passed = all(arm_workload_passes)
        criterion = "legitimate false-decline/challenge/review rates satisfy frozen gates"
    artifacts = build_v5_execution_artifacts(selected_executions)
    return _control_result(
        name=name,
        qualifies_for_readiness=qualifies,
        spec={"name": name, **spec.model_dump(mode="json")},
        support_ids=support_ids,
        artifact_ids=tuple(item.evidence_sha256 for item in artifacts),
        permutation_seed=None,
        arm_digests=tuple(
            (arm.value, score.spec.spec_sha256) for arm, score in scores.by_arm.items()
        ),
        measurements=measurements,
        criterion=criterion,
        passed=passed,
        row_evidence={
            "arms": arm_evidence,
            "full_score_sha256": full.score_sha256,
        },
        implementation_sha256=evidence_protocol.implementation_sha256,
    )


def execute_v5_controls(
    *,
    protocol: V5DevelopmentProtocol,
    evidence_protocol: V5EvidenceProtocol,
    corpus: V5Corpus,
    catalog: SentinelFeatureCatalog,
    configuration: V5ArmConfiguration,
) -> V5ExecutedControlSuite:
    """Execute every frozen control against real safe-seed evidence."""
    if protocol.seeds.development_test != evidence_protocol.safe_development_test_seed:
        raise ValueError("executed controls require the frozen safe development-test seed")
    if int(protocol.seeds.development_test) == int(evidence_protocol.locked_development_test_seed):
        raise ValueError("executed controls cannot use the locked development-test seed")
    runtime = _build_runtime(
        protocol=protocol,
        corpus=corpus,
        catalog=catalog,
        configuration=configuration,
    )
    controls = (
        _label_shuffle_control(
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _invariance_control(
            name="identity_rename",
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _invariance_control(
            name="future_causality",
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _invariance_control(
            name="equal_time_isolation",
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _single_class_control(
            fraud=False,
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _single_class_control(
            fraud=True,
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
        _invariance_control(
            name="feature_leakage",
            runtime=runtime,
            evidence_protocol=evidence_protocol,
            catalog=catalog,
        ),
    )
    support_ids = tuple(row.event_id for row in runtime.development_rows)
    values = {
        "controls": controls,
        "evidence_protocol_sha256": evidence_protocol.evidence_protocol_sha256,
        "support_sha256": _support_digest(support_ids),
        "implementation_sha256": evidence_protocol.implementation_sha256,
    }
    digest_values = dict(values)
    digest_values["controls"] = [item.model_dump(mode="json") for item in controls]
    values["suite_sha256"] = _canonical_digest(digest_values)
    return V5ExecutedControlSuite.model_validate(values)


class V5ExecutedControlSuite(BaseModel):
    """Exact ordered mandatory control evidence for one evaluation support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controls: tuple[V5ExecutedControlResult, ...]
    evidence_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        if tuple(control.name for control in self.controls) != _CONTROL_NAMES:
            raise ValueError("control suite requires the exact ordered controls")
        fraud_only = self.controls[_CONTROL_NAMES.index("fraud_only_diagnostic")]
        if fraud_only.qualifies_for_readiness:
            raise ValueError("fraud-only diagnostic cannot qualify for readiness")
        if any(
            control.implementation_sha256 != self.implementation_sha256 for control in self.controls
        ):
            raise ValueError("control suite contains mixed implementation evidence")
        document = self.model_dump(mode="json", exclude={"suite_sha256"})
        if self.suite_sha256 != _canonical_digest(document):
            raise ValueError("executed control suite digest mismatch")
        return self


__all__ = [
    "V5ControlMeasurement",
    "V5ExecutedControlResult",
    "V5ExecutedControlSuite",
    "execute_v5_controls",
]
