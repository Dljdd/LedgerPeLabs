"""Executable Sentinel v5 comparison-arm training and scoring."""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from apar.contracts.events import PaymentEvent
from apar.defense.sentinel import (
    SentinelAction,
    SentinelDefender,
    SentinelThresholds,
    route_sentinel_components,
    train_sentinel_defender,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5ArmConfiguration,
    V5ArmRowEvidence,
    V5ArmScore,
    V5ArmScoreSet,
    V5ArmSpecification,
    V5ArmSupportRow,
    V5CalibratorManifest,
    V5ExecutionArtifact,
    V5IsolationForestManifest,
    V5IsolationTreeManifest,
    V5SerializedModelArtifact,
    V5TrainingPartitionEvidence,
    derive_v5_trust_failures,
    replay_v5_rule_result,
    validate_v5_rule_feature_provenance,
)
from apar.features.sentinel import SentinelFeatureCatalog, SentinelFeatureProvenance

_RULE_CHALLENGE_THRESHOLD = 0.60
_RULE_DECLINE_THRESHOLD = 0.90


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bound_spec(
    template: V5ArmSpecification,
    *,
    thresholds: SentinelThresholds | None,
    training_partitions: tuple[V5TrainingPartitionEvidence, ...],
    defender: SentinelDefender | None,
) -> V5ArmSpecification:
    threshold_values: dict[str, float] = {}
    if template.rules:
        threshold_values.update(
            {
                "rules_challenge": _RULE_CHALLENGE_THRESHOLD,
                "rules_decline": _RULE_DECLINE_THRESHOLD,
            }
        )
    if thresholds is not None:
        threshold_values.update(
            {
                "model_challenge": thresholds.challenge_threshold,
                "model_decline": thresholds.decline_threshold,
                "model_review": thresholds.review_threshold,
            }
        )
        if template.disagreement:
            threshold_values["disagreement_review"] = (
                thresholds.disagreement_review_threshold
            )
        if template.novelty:
            threshold_values.update(
                {
                    "novelty_challenge": thresholds.novelty_challenge_threshold,
                    "novelty_review": thresholds.novelty_review_threshold,
                }
            )
    ordered_thresholds = tuple(sorted(threshold_values.items()))
    threshold_facts = {
        "source_partition": template.threshold_source_partition,
        "method": template.threshold_method,
        "threshold_ordered_rows_sha256": training_partitions[2].ordered_rows_sha256,
        "threshold_support_sha256": training_partitions[2].ordered_support_sha256,
        "threshold_feature_batch_sha256": training_partitions[2].feature_batch_sha256,
        "threshold_feature_matrix_sha256": training_partitions[2].feature_matrix_sha256,
        "threshold_values": ordered_thresholds,
    }
    calibrator_manifests = (
        tuple(_calibrator_manifest(calibrator) for calibrator in defender.calibrators)
        if defender is not None
        else ()
    )
    model_artifacts = (
        tuple(_model_artifact(model) for model in defender.model_members)
        if defender is not None
        else ()
    )
    novelty_manifest = (
        _isolation_forest_manifest(defender.iso_forest)
        if defender is not None and defender.iso_forest is not None
        else None
    )
    values = template.model_dump(mode="json", exclude={"spec_sha256"})
    values["threshold_digest"] = _digest(threshold_facts)
    values["threshold_values"] = ordered_thresholds
    values["execution_bound"] = True
    values["training_partitions"] = [
        item.model_dump(mode="json") for item in training_partitions
    ]
    values["model_artifact_sha256"] = [
        artifact.artifact_sha256 for artifact in model_artifacts
    ]
    values["model_artifacts"] = [
        artifact.model_dump(mode="json") for artifact in model_artifacts
    ]
    values["calibrator_artifact_sha256"] = [
        manifest.artifact_sha256 for manifest in calibrator_manifests
    ]
    values["calibrator_manifests"] = [
        manifest.model_dump(mode="json") for manifest in calibrator_manifests
    ]
    values["novelty_artifact_sha256"] = (
        novelty_manifest.artifact_sha256 if novelty_manifest is not None else None
    )
    values["novelty_manifest"] = (
        novelty_manifest.model_dump(mode="json")
        if novelty_manifest is not None
        else None
    )
    values["spec_sha256"] = _digest(values)
    bound_values = dict(values)
    bound_values["training_partitions"] = training_partitions
    bound_values["model_artifacts"] = model_artifacts
    bound_values["calibrator_manifests"] = calibrator_manifests
    bound_values["novelty_manifest"] = novelty_manifest
    return V5ArmSpecification.model_validate(bound_values)


def _model_artifact(model: object) -> V5SerializedModelArtifact:
    with TemporaryDirectory(prefix="apar-v5-catboost-") as directory:
        path = Path(directory) / "model.json"
        model.save_model(str(path), format="json")  # type: ignore[attr-defined]
        document = json.loads(path.read_bytes())
    if type(document) is not dict or type(document.get("model_info")) is not dict:
        raise TypeError("CatBoost JSON artifact must contain model_info")
    model_info = document["model_info"]
    volatile_fields = ("model_guid", "train_finish_time")
    if any(type(model_info.get(field)) is not str for field in volatile_fields):
        raise ValueError("CatBoost volatile metadata schema differs")
    for field in volatile_fields:
        model_info.pop(field)
    serialized = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return V5SerializedModelArtifact(
        serialization="catboost-json-canonical-v1",
        payload_base64=b64encode(serialized).decode("ascii"),
        artifact_sha256=hashlib.sha256(serialized).hexdigest(),
    )


def _isolation_forest_manifest(model: object) -> V5IsolationForestManifest:
    trees = tuple(
        V5IsolationTreeManifest(
            children_left=tuple(int(value) for value in estimator.tree_.children_left),
            children_right=tuple(int(value) for value in estimator.tree_.children_right),
            feature=tuple(int(value) for value in estimator.tree_.feature),
            threshold=tuple(float(value) for value in estimator.tree_.threshold),
            decision_path_lengths=tuple(
                float(value)
                for value in model._decision_path_lengths[index]  # type: ignore[attr-defined]
            ),
            average_path_lengths=tuple(
                float(value)
                for value in model._average_path_length_per_tree[index]  # type: ignore[attr-defined]
            ),
            estimator_features=tuple(
                int(value)
                for value in model.estimators_features_[index]  # type: ignore[attr-defined]
            ),
        )
        for index, estimator in enumerate(model.estimators_)  # type: ignore[attr-defined]
    )
    feature_count = int(model.n_features_in_)  # type: ignore[attr-defined]
    max_samples = int(model._max_samples)  # type: ignore[attr-defined]
    offset = float(model.offset_)  # type: ignore[attr-defined]
    digest_values = {
        "feature_count": feature_count,
        "max_samples": max_samples,
        "offset": offset,
        "trees": [tree.model_dump(mode="json") for tree in trees],
    }
    return V5IsolationForestManifest(
        feature_count=feature_count,
        max_samples=max_samples,
        offset=offset,
        trees=trees,
        artifact_sha256=_digest(
            {
                "serialization": "sklearn-isolation-forest-tree-arrays-v1",
                **digest_values,
            }
        ),
    )


def _calibrator_manifest(calibrator: object) -> V5CalibratorManifest:
    values = {
        "x_thresholds": calibrator.X_thresholds_.tolist(),  # type: ignore[attr-defined]
        "y_thresholds": calibrator.y_thresholds_.tolist(),  # type: ignore[attr-defined]
        "out_of_bounds": calibrator.out_of_bounds,  # type: ignore[attr-defined]
    }
    return V5CalibratorManifest(
        **values,
        artifact_sha256=_digest(values),
    )


def _validate_partition(
    name: str,
    matrix: NDArray[np.float64],
    labels: NDArray[np.int_],
    *,
    feature_count: int,
) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise ValueError(f"{name} matrix does not match the bound catalog")
    if len(matrix) != len(labels) or not len(matrix):
        raise ValueError(f"{name} matrix and labels must align and be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} matrix contains non-finite features")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{name} partition must contain both classes")


def _feature_indices(
    spec: V5ArmSpecification,
    catalog: SentinelFeatureCatalog,
) -> tuple[int, ...]:
    if spec.feature_names == catalog.feature_names:
        return tuple(range(len(catalog.feature_names)))
    if spec.model and not spec.graph:
        return tuple(
            index
            for index, group in enumerate(catalog.feature_groups)
            if group not in {"graph", "integrity"}
        )
    claimed: set[int] = set()
    indices: list[int] = []
    for name in spec.feature_names:
        for index, candidate in enumerate(catalog.feature_names):
            if candidate == name and index not in claimed:
                claimed.add(index)
                indices.append(index)
                break
        else:
            raise ValueError(f"arm feature {name} is absent from the bound catalog")
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class V5TrainedArm:
    spec: V5ArmSpecification
    feature_indices: tuple[int, ...]
    defender: SentinelDefender | None


@dataclass(frozen=True, slots=True)
class V5TrainedArmSet:
    arms: tuple[V5TrainedArm, ...]

    @property
    def by_arm(self) -> dict[V5Arm, V5TrainedArm]:
        return {trained.spec.arm: trained for trained in self.arms}


def train_v5_arm_set(
    *,
    configuration: V5ArmConfiguration,
    catalog: SentinelFeatureCatalog,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int_],
    x_calibration: NDArray[np.float64],
    y_calibration: NDArray[np.int_],
    x_threshold: NDArray[np.float64],
    y_threshold: NDArray[np.int_],
    bootstrap_seed: int,
    train_evidence: V5TrainingPartitionEvidence,
    calibration_evidence: V5TrainingPartitionEvidence,
    threshold_evidence: V5TrainingPartitionEvidence,
) -> V5TrainedArmSet:
    """Train every declared learned arm without evaluating development-test rows."""
    if type(configuration) is not V5ArmConfiguration:
        raise TypeError("configuration must be an exact V5ArmConfiguration")
    if type(catalog) is not SentinelFeatureCatalog:
        raise TypeError("catalog must be an exact SentinelFeatureCatalog")
    if any(
        template.catalog_sha256 != catalog.catalog_sha256
        for template in configuration.arms
    ):
        raise ValueError("arm specification catalog digest mismatch")
    if any(template.bootstrap_seed != bootstrap_seed for template in configuration.arms):
        raise ValueError("bootstrap seed disagrees with the frozen arm specification")
    training_partitions = (train_evidence, calibration_evidence, threshold_evidence)
    if any(
        evidence.feature_names != catalog.feature_names
        for evidence in training_partitions
    ):
        raise ValueError("training evidence feature semantics differ from full catalog")
    matrices = (x_train, x_calibration, x_threshold)
    labels_by_partition = (y_train, y_calibration, y_threshold)
    expected_names = ("train", "calibration", "threshold")
    for expected_name, evidence, matrix, labels in zip(
        expected_names,
        training_partitions,
        matrices,
        labels_by_partition,
        strict=True,
    ):
        if evidence.partition != expected_name:
            raise ValueError("training partition provenance is swapped")
        if evidence.catalog_sha256 != catalog.catalog_sha256:
            raise ValueError("training partition catalog provenance mismatch")
        if evidence.feature_matrix_sha256 != _digest(matrix.tolist()):
            raise ValueError("training partition feature matrix digest mismatch")
        if evidence.labels != tuple(int(value) for value in labels.tolist()):
            raise ValueError("training partition labels disagree with provenance")
    partition_event_sets = [set(item.ordered_event_ids) for item in training_partitions]
    if any(
        left & right
        for index, left in enumerate(partition_event_sets)
        for right in partition_event_sets[index + 1 :]
    ):
        raise ValueError("training partition event provenance overlaps")
    for name, matrix, labels in (
        ("train", x_train, y_train),
        ("calibration", x_calibration, y_calibration),
        ("threshold", x_threshold, y_threshold),
    ):
        _validate_partition(name, matrix, labels, feature_count=len(catalog.feature_names))

    trained_arms: list[V5TrainedArm] = []
    for template in configuration.arms:
        indices = _feature_indices(template, catalog)
        if not template.model:
            trained_arms.append(
                V5TrainedArm(
                    spec=_bound_spec(
                        template,
                        thresholds=None,
                        training_partitions=training_partitions,
                        defender=None,
                    ),
                    feature_indices=indices,
                    defender=None,
                )
            )
            continue
        defender = train_sentinel_defender(
            x_train=x_train[:, indices],
            y_train=y_train,
            x_calibration=x_calibration[:, indices],
            y_calibration=y_calibration,
            x_threshold=x_threshold[:, indices],
            y_threshold=y_threshold,
            catboost_seeds=template.model_seeds,
            bootstrap_seed=bootstrap_seed,
            enable_novelty=template.novelty,
        )
        trained_arms.append(
            V5TrainedArm(
                spec=_bound_spec(
                    template,
                    thresholds=defender.thresholds,
                    training_partitions=training_partitions,
                    defender=defender,
                ),
                feature_indices=indices,
                defender=defender,
            )
        )
    return V5TrainedArmSet(arms=tuple(trained_arms))


def _model_action(probability: float, thresholds: SentinelThresholds) -> SentinelAction:
    if probability >= thresholds.decline_threshold:
        return SentinelAction.DECLINE_HOLD
    if probability >= thresholds.review_threshold:
        return SentinelAction.REVIEW_HOLD
    if probability >= thresholds.challenge_threshold:
        return SentinelAction.CHALLENGE
    return SentinelAction.APPROVE


def _rule_action(score: float) -> SentinelAction:
    if score >= _RULE_DECLINE_THRESHOLD:
        return SentinelAction.DECLINE_HOLD
    if score >= _RULE_CHALLENGE_THRESHOLD:
        return SentinelAction.CHALLENGE
    return SentinelAction.APPROVE


def _more_severe(left: SentinelAction, right: SentinelAction) -> SentinelAction:
    return left if left.severity >= right.severity else right


def route_full_sentinel_components(
    *,
    probability: float,
    disagreement: float,
    novelty: float,
    thresholds: SentinelThresholds,
) -> tuple[SentinelAction, bool, bool]:
    """Replay full-model routing from explicit bounded component evidence."""
    return route_sentinel_components(
        probability=probability,
        disagreement=disagreement,
        novelty=novelty,
        thresholds=thresholds,
    )


def _threshold_trace(
    spec: V5ArmSpecification,
    defender: SentinelDefender | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if spec.rules:
        values.update(
            {
                "rules_challenge": _RULE_CHALLENGE_THRESHOLD,
                "rules_decline": _RULE_DECLINE_THRESHOLD,
            }
        )
    if defender is not None:
        thresholds = defender.thresholds
        values.update(
            {
                "model_challenge": thresholds.challenge_threshold,
                "model_review": thresholds.review_threshold,
                "model_decline": thresholds.decline_threshold,
            }
        )
        if spec.disagreement:
            values["disagreement_review"] = thresholds.disagreement_review_threshold
        if spec.novelty:
            values.update(
                {
                    "novelty_challenge": thresholds.novelty_challenge_threshold,
                    "novelty_review": thresholds.novelty_review_threshold,
                }
            )
    return values


def _score_one_arm(
    *,
    trained: V5TrainedArm,
    catalog: SentinelFeatureCatalog,
    features_matrix: NDArray[np.float64],
    support: tuple[V5ArmSupportRow, ...],
    execution_artifacts: tuple[V5ExecutionArtifact, ...],
    trust_failures: list[bool],
    feature_provenance: tuple[SentinelFeatureProvenance, ...],
) -> V5ArmScore:
    evidence: list[V5ArmRowEvidence] = []
    execution_manifests = {
        artifact.evidence_sha256: artifact.manifest()
        for artifact in execution_artifacts
    }
    execution_events = {
        record.event_id: PaymentEvent.model_validate_json(record.event_json)
        for manifest in execution_manifests.values()
        for record in manifest.event_records
    }
    for index, source_row in enumerate(features_matrix):
        start = time.perf_counter_ns()
        if trained.spec.rules:
            rule_result = replay_v5_rule_result(
                support=support[index],
                catalog_feature_names=catalog.feature_names,
                catalog_feature_values=source_row,
                catalog_sha256=catalog.catalog_sha256,
                manifests=execution_manifests,
                source_event_ids=feature_provenance[index].source_event_ids,
                max_source_available_at=(
                    feature_provenance[index].max_source_available_at
                ),
                resolved_events=execution_events,
            )
            rule_score = rule_result.score
            rule_components = tuple(
                sorted((hit.reason.value, hit.score) for hit in rule_result.hits)
            )
            rule_manifest_sha256 = rule_result.manifest_digest
            rule_vector_sha256 = rule_result.vector_digest
            rule_evidence_source_ids = tuple(
                sorted(
                    {
                        source_id
                        for hit in rule_result.hits
                        for source_id in hit.evidence_source_ids
                    }
                )
            )
        else:
            rule_score, rule_components = None, ()
            rule_manifest_sha256 = None
            rule_vector_sha256 = None
            rule_evidence_source_ids = ()
        trust_failure = trust_failures[index]
        subset = source_row[list(trained.feature_indices)]
        model_raw_scores: tuple[float, ...] = ()
        model_calibrated_scores: tuple[float, ...] = ()
        probability_action: SentinelAction | None = None
        model_action: SentinelAction | None = None
        rule_action = _rule_action(rule_score) if rule_score is not None else None
        trust_action = (
            SentinelAction.DECLINE_HOLD
            if trained.spec.trust and trust_failure
            else None
        )
        novelty_raw: float | None = None
        novelty_routed = False
        disagreement_routed = False
        if trained.defender is None:
            assert rule_score is not None
            assert rule_action is not None
            action = trust_action or rule_action
            probability = 1.0 if trained.spec.trust and trust_failure else rule_score
            novelty = None
            disagreement = None
        elif trained.spec.arm is V5Arm.FULL_SENTINEL:
            (
                model_raw_scores,
                model_calibrated_scores,
                probability,
                disagreement,
            ) = trained.defender.predict_member_scores(subset)
            probability_action = _model_action(probability, trained.defender.thresholds)
            novelty_raw, novelty = trained.defender.predict_novelty(subset)
            model_action, disagreement_routed, novelty_routed = route_full_sentinel_components(
                probability=probability,
                disagreement=disagreement,
                novelty=novelty,
                thresholds=trained.defender.thresholds,
            )
            assert rule_action is not None
            action = trust_action or _more_severe(model_action, rule_action)
        else:
            (
                model_raw_scores,
                model_calibrated_scores,
                probability,
                _raw_disagreement,
            ) = trained.defender.predict_member_scores(subset)
            probability_action = _model_action(probability, trained.defender.thresholds)
            model_action = probability_action
            action = model_action
            novelty = None
            disagreement = None
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        row_values = {
            "support": support[index],
            "catalog_feature_values": tuple(float(value) for value in source_row),
            "subset_feature_values": tuple(float(value) for value in subset),
            "catalog_feature_sha256": _digest(tuple(float(value) for value in source_row)),
            "subset_feature_sha256": _digest(tuple(float(value) for value in subset)),
            "model_raw_scores": model_raw_scores,
            "model_calibrated_scores": model_calibrated_scores,
            "threshold_trace": _threshold_trace(trained.spec, trained.defender),
            "rule_components": rule_components,
            "rule_manifest_sha256": rule_manifest_sha256,
            "rule_vector_sha256": rule_vector_sha256,
            "rule_source_event_ids": feature_provenance[index].source_event_ids,
            "rule_max_source_available_at": (
                feature_provenance[index].max_source_available_at
            ),
            "rule_evidence_source_ids": rule_evidence_source_ids,
            "action": action,
            "probability": probability,
            "probability_action": probability_action,
            "model_action": model_action,
            "rule_action": rule_action,
            "trust_action": trust_action,
            "rule_score": rule_score,
            "trust_routed": bool(trained.spec.trust and trust_failure),
            "novelty_score": novelty,
            "novelty_raw_score": novelty_raw,
            "novelty_overridden": False,
            "disagreement": disagreement,
            "novelty_routed": novelty_routed,
            "disagreement_routed": disagreement_routed,
            "latency_ms": latency_ms,
            "arm_spec_sha256": trained.spec.spec_sha256,
        }
        row_values["row_output_sha256"] = _digest(
            {
                **row_values,
                "support": support[index].model_dump(mode="json"),
                "rule_max_source_available_at": feature_provenance[
                    index
                ].model_dump(mode="json")["max_source_available_at"],
            }
        )
        evidence.append(V5ArmRowEvidence.model_validate(row_values))
    support_sha256 = _digest([row.model_dump(mode="json") for row in support])
    score_digest_values = {
        "spec": trained.spec.model_dump(mode="json"),
        "support_sha256": support_sha256,
        "execution_artifacts": [
            artifact.model_dump(mode="json") for artifact in execution_artifacts
        ],
        "rows": [row.model_dump(mode="json") for row in evidence],
    }
    return V5ArmScore(
        spec=trained.spec,
        support_sha256=support_sha256,
        execution_artifacts=execution_artifacts,
        rows=tuple(evidence),
        score_sha256=_digest(score_digest_values),
    )


def score_v5_arm_set(
    *,
    trained: V5TrainedArmSet,
    catalog: SentinelFeatureCatalog,
    features_matrix: NDArray[np.float64],
    support: tuple[V5ArmSupportRow, ...],
    execution_artifacts: tuple[V5ExecutionArtifact, ...],
    trust_failures: list[bool],
    feature_provenance: tuple[SentinelFeatureProvenance, ...],
) -> V5ArmScoreSet:
    """Score each arm independently over one immutable ordered support."""
    if any(
        arm.spec.catalog_sha256 != catalog.catalog_sha256 for arm in trained.arms
    ):
        raise ValueError("trained arm catalog digest mismatch")
    if features_matrix.ndim != 2 or features_matrix.shape[1] != len(catalog.feature_names):
        raise ValueError("evaluation matrix does not match the bound catalog")
    if not np.isfinite(features_matrix).all():
        raise ValueError("evaluation matrix contains non-finite features")
    lengths = {len(features_matrix), len(support), len(trust_failures)}
    if lengths != {len(features_matrix)} or not len(features_matrix):
        raise ValueError("evaluation features, support, and trust must align")
    if len({row.event_id for row in support}) != len(support):
        raise ValueError("evaluation support event IDs must be unique")
    evaluation_ids = {row.event_id for row in support}
    if any(
        evaluation_ids & set(partition.ordered_event_ids)
        for arm in trained.arms
        for partition in arm.spec.training_partitions
    ):
        raise ValueError("evaluation support overlaps a model-development partition")
    derived_trust_failures = derive_v5_trust_failures(support, execution_artifacts)
    if trust_failures != derived_trust_failures:
        raise ValueError("trust failures disagree with retained verifier evidence")
    feature_provenance = validate_v5_rule_feature_provenance(
        support=support,
        provenance=feature_provenance,
        catalog=catalog,
        artifacts=execution_artifacts,
    )
    scores = {
        arm.spec.arm: _score_one_arm(
            trained=arm,
            catalog=catalog,
            features_matrix=features_matrix,
            support=support,
            execution_artifacts=execution_artifacts,
            trust_failures=trust_failures,
            feature_provenance=feature_provenance,
        )
        for arm in trained.arms
    }
    return V5ArmScoreSet(by_arm=scores)


__all__ = [
    "V5TrainedArm",
    "V5TrainedArmSet",
    "score_v5_arm_set",
    "train_v5_arm_set",
    "route_full_sentinel_components",
]
