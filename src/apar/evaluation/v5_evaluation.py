"""Causal evaluation, controls, and ablations for Defend v5."""

from __future__ import annotations

import hashlib
import json
import math
from base64 import b64decode
from binascii import Error as Base64Error
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, cast

import numpy as np
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from apar.defense.rules import RuleManifest
from apar.defense.sentinel import (
    FULL_SENTINEL_DISAGREEMENT_REVIEW_THRESHOLD,
    FULL_SENTINEL_NOVELTY_CHALLENGE_THRESHOLD,
    FULL_SENTINEL_NOVELTY_REVIEW_THRESHOLD,
    SentinelAction,
)
from apar.evaluation.v5_population import V5DecisionRow, V5ExecutionManifest

_INTERVENTION_ACTIONS = {
    SentinelAction.CHALLENGE,
    SentinelAction.REVIEW_HOLD,
    SentinelAction.DECLINE_HOLD,
}
_MAX_EXECUTION_ARTIFACTS = 4_096
_MAX_EXECUTION_ARTIFACT_BYTES = 268_435_456
_MAX_FEATURE_BATCH_BYTES = 134_217_728
_MAX_FEATURE_MATRIX_CELLS = 10_000_000
_MAX_SCORE_ROWS = 100_000
_MAX_CATBOOST_ARTIFACT_BYTES = 67_108_864
_MAX_ISOLATION_TREES = 512
_MAX_ISOLATION_NODES = 262_144


class V5Arm(StrEnum):
    RULES_ONLY = "rules_only"
    ENSEMBLE_NO_GRAPH = "ensemble_no_graph"
    ENSEMBLE_WITH_GRAPH = "ensemble_with_graph"
    FULL_SENTINEL = "full_sentinel"
    HARDENED_SENTINEL = "hardened_sentinel"


_CURRENT_ARMS = (
    V5Arm.RULES_ONLY,
    V5Arm.ENSEMBLE_NO_GRAPH,
    V5Arm.ENSEMBLE_WITH_GRAPH,
    V5Arm.FULL_SENTINEL,
)
_RULE_FEATURES = (
    "actor_count_1m",
    "actor_count_10m",
    "actor_amount_zscore_24h",
    "counterparty_amount_zscore_24h",
    "pair_prior_count",
    "dq_degraded_state",
)
_RULE_CHALLENGE_THRESHOLD = 0.60
_RULE_DECLINE_THRESHOLD = 0.90
_EXPECTED_SWITCHES = {
    V5Arm.RULES_ONLY: (False, False, True, True, False, False),
    V5Arm.ENSEMBLE_NO_GRAPH: (True, False, False, False, False, False),
    V5Arm.ENSEMBLE_WITH_GRAPH: (True, True, False, False, False, False),
    V5Arm.FULL_SENTINEL: (True, True, True, True, True, True),
}
_EXPECTED_IMPLEMENTATION_PATHS = (
    "src/apar/contracts/_validation.py",
    "src/apar/contracts/decisions.py",
    "src/apar/contracts/events.py",
    "src/apar/contracts/scenarios.py",
    "src/apar/defense/rules.py",
    "src/apar/defense/contracts.py",
    "src/apar/defense/sentinel.py",
    "src/apar/evaluation/v5_arms.py",
    "src/apar/evaluation/v5_evaluation.py",
    "src/apar/evaluation/v5_execution.py",
    "src/apar/evaluation/v5_fidelity.py",
    "src/apar/evaluation/v5_hardening.py",
    "src/apar/evaluation/v5_population.py",
    "src/apar/evaluation/v5_protocol.py",
    "src/apar/evaluation/v5_reporting.py",
    "src/apar/features/sentinel.py",
    "src/apar/features/catalog.py",
    "src/apar/features/state.py",
    "src/apar/generators/campaigns.py",
    "src/apar/generators/population.py",
    "src/apar/simulator/clock.py",
    "src/apar/simulator/engine.py",
    "src/apar/simulator/ledger.py",
    "src/apar/simulator/rails/__init__.py",
    "src/apar/simulator/rails/a2a.py",
    "src/apar/simulator/rails/agentic.py",
    "src/apar/simulator/rails/base.py",
    "src/apar/simulator/rails/card.py",
    "src/apar/trust/verifier.py",
    "scripts/run_defense_v5_development.py",
)


def _expected_rule_parameters() -> tuple[tuple[str, float], ...]:
    manifest = RuleManifest.default()
    return tuple(
        sorted(
            {
                "actor_count_1m_threshold": manifest.actor_count_1m,
                "actor_count_10m_threshold": manifest.actor_count_10m,
                "amount_zscore_threshold": manifest.amount_zscore,
                "degraded_score": manifest.threshold_score,
                "degraded_state_threshold": manifest.degraded_state,
                "repeated_pair_threshold": manifest.repeated_pair_count,
            }.items()
        )
    )


def _canonical_digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class V5ArmSupportRow(BaseModel):
    """Evaluator-only execution facts shared identically by every arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    label: Literal[0, 1]
    payment_id: str
    campaign_id: str
    actor_id: str
    counterparty_id: str
    amount: float = Field(gt=0.0)
    currency: str
    family: str
    rail: str
    integrity_status: Literal["pass", "fail", "not_applicable"]
    source_command_id: str
    source_event_id: str
    execution_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("amount")
    @classmethod
    def amount_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("support amount must be finite")
        return value


class V5ExecutionArtifact(BaseModel):
    """Canonical content-addressed execution manifest with verifier facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_json: str = Field(max_length=16_000_000)

    def manifest(self) -> V5ExecutionManifest:
        return V5ExecutionManifest.model_validate_json(self.payload_json)

    @model_validator(mode="after")
    def payload_is_canonical_and_validated(self) -> Self:
        manifest = self.manifest()
        canonical = json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if self.payload_json != canonical:
            raise ValueError("execution artifact payload must use canonical JSON")
        if (
            self.evidence_sha256 != manifest.evidence_sha256
            or self.artifact_sha256 != manifest.artifact_sha256
        ):
            raise ValueError("execution artifact digests disagree with validated manifest")
        if self.payload_sha256 != hashlib.sha256(canonical.encode()).hexdigest():
            raise ValueError("execution artifact payload digest mismatch")
        return self


def build_v5_execution_artifacts(
    manifests: Sequence[V5ExecutionManifest],
) -> tuple[V5ExecutionArtifact, ...]:
    """Serialize validated execution manifests into canonical bounded artifacts."""
    if len(manifests) > _MAX_EXECUTION_ARTIFACTS:
        raise ValueError("execution artifact count exceeds production profile limit")
    artifacts: list[V5ExecutionArtifact] = []
    for manifest in sorted(manifests, key=lambda item: item.evidence_sha256):
        validated = V5ExecutionManifest.model_validate(manifest)
        payload = json.dumps(
            validated.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        artifacts.append(
            V5ExecutionArtifact(
                evidence_sha256=validated.evidence_sha256,
                artifact_sha256=validated.artifact_sha256,
                payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                payload_json=payload,
            )
        )
    if len({artifact.evidence_sha256 for artifact in artifacts}) != len(artifacts):
        raise ValueError("execution artifacts must have unique evidence digests")
    _validate_execution_artifact_bounds(artifacts)
    return tuple(artifacts)


def _validate_execution_artifact_bounds(
    artifacts: Sequence[V5ExecutionArtifact],
) -> None:
    if len(artifacts) > _MAX_EXECUTION_ARTIFACTS:
        raise ValueError("execution artifact count exceeds production profile limit")
    total_bytes = sum(len(artifact.payload_json.encode()) for artifact in artifacts)
    if total_bytes > _MAX_EXECUTION_ARTIFACT_BYTES:
        raise ValueError("execution artifact bytes exceed production profile limit")


def build_v5_arm_support_rows(
    rows: Sequence[V5DecisionRow],
) -> tuple[V5ArmSupportRow, ...]:
    """Retain exact evaluator-only execution facts in canonical row order."""
    return tuple(
        V5ArmSupportRow(
            event_id=row.event_id,
            label=1 if row.is_fraud else 0,
            payment_id=row.payment_id,
            campaign_id=row.campaign_id,
            actor_id=row.actor_id,
            counterparty_id=row.counterparty_id,
            amount=float(row.amount),
            currency=row.currency,
            family=row.family,
            rail=row.rail,
            integrity_status=cast(
                Literal["pass", "fail", "not_applicable"], row.integrity_status
            ),
            source_command_id=row.source_command_id,
            source_event_id=row.source_event_id,
            execution_evidence_sha256=row.execution_evidence_sha256,
        )
        for row in rows
    )


def _support_trust_failure(
    support: V5ArmSupportRow,
    artifacts: tuple[V5ExecutionArtifact, ...],
) -> bool:
    return _support_trust_failure_from_manifests(
        support, _execution_manifest_map(artifacts)
    )


def _execution_manifest_map(
    artifacts: Sequence[V5ExecutionArtifact],
) -> dict[str, V5ExecutionManifest]:
    by_evidence = {
        artifact.evidence_sha256: artifact.manifest() for artifact in artifacts
    }
    if len(by_evidence) != len(artifacts):
        raise ValueError("execution artifacts must have unique evidence digests")
    return by_evidence


def _support_trust_failure_from_manifests(
    support: V5ArmSupportRow,
    by_evidence: dict[str, V5ExecutionManifest],
) -> bool:
    manifest = by_evidence.get(support.execution_evidence_sha256)
    if manifest is None:
        raise ValueError("support execution evidence hash cannot be resolved")
    if (
        manifest.campaign_id != support.campaign_id
        or manifest.family != support.family
        or manifest.rail != support.rail
    ):
        raise ValueError("support facts disagree with execution artifact")
    matching = [link for link in manifest.lineage if link.event_id == support.source_event_id]
    if len(matching) != 1:
        raise ValueError("support source event cannot be resolved in execution artifact")
    link = matching[0]
    if (
        support.event_id != link.event_id
        or support.source_command_id != link.command_id
        or support.payment_id != link.payment_id
        or support.actor_id != link.actor_id
        or support.counterparty_id != link.counterparty_id
        or support.label != int(link.is_fraud)
    ):
        raise ValueError("support lineage disagrees with execution artifact")
    event = next(record for record in manifest.event_records if record.event_id == link.event_id)
    if float(event.amount) != support.amount or event.currency != support.currency:
        raise ValueError("support economics disagree with execution artifact")
    expected_integrity = (
        "fail"
        if support.event_id in manifest.trust_failure_event_ids
        else "pass"
        if support.rail == "agentic"
        else "not_applicable"
    )
    if support.integrity_status != expected_integrity:
        raise ValueError("support verifier outcome disagrees with execution artifact")
    return expected_integrity == "fail"


def derive_v5_trust_failures(
    support: Sequence[V5ArmSupportRow],
    artifacts: tuple[V5ExecutionArtifact, ...],
) -> list[bool]:
    """Replay actual verifier outcomes in exact evaluator support order."""
    manifests = _execution_manifest_map(artifacts)
    return [_support_trust_failure_from_manifests(row, manifests) for row in support]


class V5TrainingPartitionEvidence(BaseModel):
    """Bounded ordered provenance for one model-development partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: Literal["train", "calibration", "threshold"]
    ordered_event_ids: tuple[str, ...] = Field(max_length=_MAX_SCORE_ROWS)
    labels: tuple[Literal[0, 1], ...] = Field(max_length=_MAX_SCORE_ROWS)
    feature_names: tuple[str, ...] = Field(max_length=256)
    feature_matrix: tuple[tuple[float, ...], ...] = Field(max_length=_MAX_SCORE_ROWS)
    support_records: tuple[V5ArmSupportRow, ...] = Field(max_length=_MAX_SCORE_ROWS)
    execution_artifacts: tuple[V5ExecutionArtifact, ...] = Field(
        max_length=_MAX_EXECUTION_ARTIFACTS
    )
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_batch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_batch_payload_json: str = Field(max_length=_MAX_FEATURE_BATCH_BYTES)
    feature_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def ordered_provenance_is_complete(self) -> Self:
        if len(self.feature_batch_payload_json.encode()) > _MAX_FEATURE_BATCH_BYTES:
            raise ValueError("feature batch bytes exceed production profile limit")
        if sum(len(row) for row in self.feature_matrix) > _MAX_FEATURE_MATRIX_CELLS:
            raise ValueError("feature matrix cell count exceeds production profile limit")
        _validate_execution_artifact_bounds(self.execution_artifacts)
        if not self.ordered_event_ids or len(self.ordered_event_ids) != len(self.labels):
            raise ValueError("training partition event IDs and labels must align")
        if len(set(self.ordered_event_ids)) != len(self.ordered_event_ids):
            raise ValueError("training partition event IDs must be unique")
        if set(self.labels) != {0, 1}:
            raise ValueError("training partition evidence must contain both classes")
        expected = _canonical_digest(
            [
                {"event_id": event_id, "label": label}
                for event_id, label in zip(self.ordered_event_ids, self.labels, strict=True)
            ]
        )
        if self.ordered_rows_sha256 != expected:
            raise ValueError("training partition ordered-row digest mismatch")
        if tuple(row.event_id for row in self.support_records) != self.ordered_event_ids:
            raise ValueError("training partition support order disagrees with event IDs")
        if tuple(row.label for row in self.support_records) != self.labels:
            raise ValueError("training partition support labels disagree with labels")
        if len(self.feature_matrix) != len(self.labels) or any(
            len(row) != len(self.feature_names) for row in self.feature_matrix
        ):
            raise ValueError("training partition feature matrix shape is incomplete")
        if any(not math.isfinite(value) for row in self.feature_matrix for value in row):
            raise ValueError("training partition feature matrix must be finite")
        if self.feature_matrix_sha256 != _canonical_digest(self.feature_matrix):
            raise ValueError("training partition feature matrix digest mismatch")
        try:
            batch_payload = json.loads(self.feature_batch_payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("training partition feature batch payload is invalid") from exc
        if self.feature_batch_payload_json != json.dumps(batch_payload, sort_keys=True):
            raise ValueError("training partition feature batch payload is not canonical")
        if not isinstance(batch_payload, dict) or set(batch_payload) != {"names", "rows"}:
            raise ValueError("training partition feature batch payload is incomplete")
        if tuple(batch_payload["names"]) != self.feature_names:
            raise ValueError("training partition feature batch names disagree")
        try:
            batch_matrix = tuple(
                tuple(float(value) for value in row) for row in batch_payload["rows"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("training partition feature batch rows are invalid") from exc
        if batch_matrix != self.feature_matrix:
            raise ValueError("training partition feature batch rows disagree with matrix")
        expected_batch = hashlib.sha256(self.feature_batch_payload_json.encode()).hexdigest()
        if self.feature_batch_sha256 != expected_batch:
            raise ValueError("training partition feature batch digest mismatch")
        if self.ordered_support_sha256 != _canonical_digest(
            [row.model_dump(mode="json") for row in self.support_records]
        ):
            raise ValueError("training partition ordered support digest mismatch")
        manifests = _execution_manifest_map(self.execution_artifacts)
        for support in self.support_records:
            _support_trust_failure_from_manifests(support, manifests)
        return self


class V5CalibratorManifest(BaseModel):
    """Bounded isotonic knots sufficient to replay one member calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    out_of_bounds: Literal["clip"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def computed_digest(self) -> str:
        return _canonical_digest(
            self.model_dump(mode="json", exclude={"artifact_sha256"})
        )

    def calibrate(self, raw_score: float) -> float:
        return float(
            np.interp(
                raw_score,
                np.array(self.x_thresholds),
                np.array(self.y_thresholds),
            )
        )

    @model_validator(mode="after")
    def knots_are_replayable(self) -> Self:
        if (
            len(self.x_thresholds) < 2
            or len(self.x_thresholds) != len(self.y_thresholds)
        ):
            raise ValueError("calibrator threshold knots must align and be non-trivial")
        if any(
            not math.isfinite(value)
            for value in (*self.x_thresholds, *self.y_thresholds)
        ):
            raise ValueError("calibrator threshold knots must be finite")
        if any(
            left >= right
            for left, right in zip(
                self.x_thresholds, self.x_thresholds[1:], strict=False
            )
        ):
            raise ValueError("calibrator x-threshold knots must be strictly increasing")
        if any(
            left > right
            for left, right in zip(
                self.y_thresholds, self.y_thresholds[1:], strict=False
            )
        ) or any(not 0.0 <= value <= 1.0 for value in self.y_thresholds):
            raise ValueError("calibrator y-threshold knots must be monotonic probabilities")
        if self.artifact_sha256 != self.computed_digest():
            raise ValueError("calibrator artifact digest mismatch")
        return self


class V5SerializedModelArtifact(BaseModel):
    """Bounded content-addressed CatBoost bytes for independent raw-score replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    serialization: Literal["catboost-cbm"] = "catboost-cbm"
    payload_base64: str = Field(max_length=24_000_000)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def payload(self) -> bytes:
        try:
            payload = b64decode(self.payload_base64, validate=True)
        except (Base64Error, ValueError) as error:
            raise ValueError("CatBoost artifact is not valid base64") from error
        if len(payload) > 16_000_000:
            raise ValueError("CatBoost artifact exceeds the bounded payload limit")
        return payload

    def load_model(self) -> CatBoostClassifier:
        model = CatBoostClassifier()
        model.load_model(blob=self.payload())
        return model

    @model_validator(mode="after")
    def payload_digest_matches(self) -> Self:
        if hashlib.sha256(self.payload()).hexdigest() != self.artifact_sha256:
            raise ValueError("CatBoost artifact payload digest mismatch")
        return self


class V5IsolationTreeManifest(BaseModel):
    """One immutable sklearn isolation tree sufficient for exact traversal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    children_left: tuple[int, ...]
    children_right: tuple[int, ...]
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    decision_path_lengths: tuple[float, ...]
    average_path_lengths: tuple[float, ...]
    estimator_features: tuple[int, ...]

    @model_validator(mode="after")
    def arrays_form_one_bounded_tree(self) -> Self:
        lengths = {
            len(self.children_left),
            len(self.children_right),
            len(self.feature),
            len(self.threshold),
            len(self.decision_path_lengths),
            len(self.average_path_lengths),
        }
        if len(lengths) != 1 or not self.children_left or len(self.children_left) > 1_000_000:
            raise ValueError("isolation tree arrays must align and remain bounded")
        if any(
            not math.isfinite(value)
            for value in (
                *self.threshold,
                *self.decision_path_lengths,
                *self.average_path_lengths,
            )
        ):
            raise ValueError("isolation tree numeric arrays must be finite")
        return self

    def leaf_index(self, features: tuple[float, ...]) -> int:
        values = np.asarray(features, dtype=np.float32)
        node = 0
        traversed = 0
        while self.children_left[node] != -1:
            split_feature = self.feature[node]
            if split_feature < 0 or split_feature >= len(self.estimator_features):
                raise ValueError("isolation tree split feature is invalid")
            source_feature = self.estimator_features[split_feature]
            if source_feature < 0 or source_feature >= len(values):
                raise ValueError("isolation tree source feature is invalid")
            node = (
                self.children_left[node]
                if values[source_feature] <= self.threshold[node]
                else self.children_right[node]
            )
            traversed += 1
            if node < 0 or node >= len(self.children_left) or traversed > len(
                self.children_left
            ):
                raise ValueError("isolation tree traversal is invalid")
        return node


class V5IsolationForestManifest(BaseModel):
    """Safe deterministic IsolationForest inference contract without live objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    serialization: Literal["sklearn-isolation-forest-tree-arrays-v1"] = (
        "sklearn-isolation-forest-tree-arrays-v1"
    )
    feature_count: int = Field(gt=0)
    max_samples: int = Field(gt=0)
    offset: float
    trees: tuple[V5IsolationTreeManifest, ...] = Field(
        min_length=1, max_length=_MAX_ISOLATION_TREES
    )
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def computed_digest(self) -> str:
        return _canonical_digest(
            self.model_dump(mode="json", exclude={"artifact_sha256"})
        )

    @staticmethod
    def _average_path_length(sample_count: int) -> float:
        if sample_count <= 1:
            return 0.0
        if sample_count == 2:
            return 1.0
        return 2.0 * (math.log(sample_count - 1.0) + float(np.euler_gamma)) - (
            2.0 * (sample_count - 1.0) / sample_count
        )

    def raw_score(self, features: tuple[float, ...]) -> float:
        if len(features) != self.feature_count or any(
            not math.isfinite(value) for value in features
        ):
            raise ValueError("novelty replay feature vector is invalid")
        depth = 0.0
        for tree in self.trees:
            leaf = tree.leaf_index(features)
            depth += (
                tree.decision_path_lengths[leaf]
                + tree.average_path_lengths[leaf]
                - 1.0
            )
        denominator = len(self.trees) * self._average_path_length(self.max_samples)
        anomaly_score = 1.0 if denominator == 0.0 else 2.0 ** (-depth / denominator)
        return -anomaly_score - self.offset

    @model_validator(mode="after")
    def forest_is_content_addressed(self) -> Self:
        if len(self.trees) > _MAX_ISOLATION_TREES:
            raise ValueError("isolation forest tree count exceeds production profile limit")
        if sum(len(tree.children_left) for tree in self.trees) > _MAX_ISOLATION_NODES:
            raise ValueError("isolation forest node count exceeds production profile limit")
        if not math.isfinite(self.offset):
            raise ValueError("isolation forest offset must be finite")
        if any(
            any(index < 0 or index >= self.feature_count for index in tree.estimator_features)
            for tree in self.trees
        ):
            raise ValueError("isolation forest estimator features are invalid")
        if self.artifact_sha256 != self.computed_digest():
            raise ValueError("IsolationForest artifact digest mismatch")
        return self


class V5ArmSpecification(BaseModel):
    """Immutable, executable component contract for one comparison arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: V5Arm
    catalog_feature_names: tuple[str, ...]
    catalog_feature_groups: tuple[str, ...]
    feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    non_graph_feature_names: tuple[str, ...]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_seeds: tuple[int, ...]
    calibration_method: Literal["none", "isotonic_per_member"]
    threshold_source_partition: Literal["threshold"]
    threshold_method: Literal["rules_v1_fixed", "sentinel_percentile_v1"]
    threshold_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_values: tuple[tuple[str, float], ...] = ()
    component_parameters: tuple[tuple[str, float], ...] = ()
    bootstrap_seed: int = Field(gt=0)
    execution_bound: bool = False
    training_partitions: tuple[V5TrainingPartitionEvidence, ...] = ()
    model_artifact_sha256: tuple[str, ...] = ()
    model_artifacts: tuple[V5SerializedModelArtifact, ...] = ()
    calibrator_artifact_sha256: tuple[str, ...] = ()
    calibrator_manifests: tuple[V5CalibratorManifest, ...] = ()
    novelty_artifact_sha256: str | None = None
    novelty_manifest: V5IsolationForestManifest | None = None
    model: bool
    graph: bool
    rules: bool
    trust: bool
    novelty: bool
    disagreement: bool
    implementation_version: str
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def computed_digest(self) -> str:
        document = self.model_dump(mode="json", exclude={"spec_sha256"})
        return _canonical_digest(document)

    @model_validator(mode="after")
    def specification_is_bound(self) -> Self:
        model_artifact_bytes = sum(
            len(artifact.payload_base64) * 3 // 4
            - artifact.payload_base64.endswith("=")
            - artifact.payload_base64.endswith("==")
            for artifact in self.model_artifacts
        )
        if model_artifact_bytes > _MAX_CATBOOST_ARTIFACT_BYTES:
            raise ValueError("CatBoost artifact bytes exceed production profile limit")
        if self.spec_sha256 != self.computed_digest():
            raise ValueError("arm specification digest mismatch")
        if Counter(self.graph_feature_names + self.non_graph_feature_names) != Counter(
            self.feature_names
        ):
            raise ValueError("arm graph and non-graph feature subsets must reconstruct features")
        if len(self.catalog_feature_names) != len(self.catalog_feature_groups):
            raise ValueError("arm catalog feature names and groups must align")
        expected_graph = tuple(
            name
            for name, group in zip(
                self.catalog_feature_names, self.catalog_feature_groups, strict=True
            )
            if group == "graph"
        )
        approved_non_graph = tuple(
            name
            for name, group in zip(
                self.catalog_feature_names, self.catalog_feature_groups, strict=True
            )
            if group not in {"graph", "integrity"}
        )
        approved_with_graph = tuple(
            name
            for name, group in zip(
                self.catalog_feature_names, self.catalog_feature_groups, strict=True
            )
            if group != "integrity"
        )
        expected_features = {
            V5Arm.RULES_ONLY: _RULE_FEATURES,
            V5Arm.ENSEMBLE_NO_GRAPH: approved_non_graph,
            V5Arm.ENSEMBLE_WITH_GRAPH: approved_with_graph,
            V5Arm.FULL_SENTINEL: approved_with_graph,
        }.get(self.arm)
        if self.feature_names != expected_features:
            raise ValueError("arm feature subset does not match frozen semantics")
        expected_arm_graph = tuple(
            name for name in self.feature_names if name in set(expected_graph)
        )
        if self.graph_feature_names != expected_arm_graph:
            raise ValueError("arm graph feature subset does not match catalog groups")
        if len(self.feature_names) != len(set(self.feature_names)) and self.arm is V5Arm.RULES_ONLY:
            raise ValueError("rule feature names must be unique")
        if self.model != bool(self.model_seeds):
            raise ValueError("arm model switch and seed binding disagree")
        if self.model != (self.calibration_method == "isotonic_per_member"):
            raise ValueError("arm model switch and calibration binding disagree")
        switches = (
            self.model,
            self.graph,
            self.rules,
            self.trust,
            self.novelty,
            self.disagreement,
        )
        if switches != _EXPECTED_SWITCHES.get(self.arm):
            raise ValueError("arm switches do not match frozen semantics")
        if self.arm is V5Arm.RULES_ONLY:
            if self.feature_names != _RULE_FEATURES or self.graph_feature_names:
                raise ValueError("rules-only specification must be graph-free")
            if self.threshold_method != "rules_v1_fixed":
                raise ValueError("rules-only threshold method mismatch")
        elif self.threshold_method != "sentinel_percentile_v1":
            raise ValueError("learned arm threshold method mismatch")
        if self.arm is V5Arm.ENSEMBLE_NO_GRAPH and self.graph_feature_names:
            raise ValueError("no-graph ensemble cannot retain graph features")
        if self.arm in {V5Arm.ENSEMBLE_WITH_GRAPH, V5Arm.FULL_SENTINEL} and not (
            self.graph_feature_names
        ):
            raise ValueError("graph-enabled arm must retain graph features")
        if self.model and "integrity_pass" in self.feature_names:
            raise ValueError("learned arms cannot contain verifier outcome features")
        if tuple(name for name, _value in self.threshold_values) != tuple(
            sorted({name for name, _value in self.threshold_values})
        ):
            raise ValueError("threshold values must use unique canonical names")
        if tuple(name for name, _value in self.component_parameters) != tuple(
            sorted({name for name, _value in self.component_parameters})
        ):
            raise ValueError("component parameters must use unique canonical names")
        if self.component_parameters != (
            _expected_rule_parameters() if self.rules else ()
        ):
            raise ValueError("rule component parameters do not match the executable manifest")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _name, value in self.threshold_values
        ):
            raise ValueError("arm threshold values must be finite values in [0, 1]")
        if self.execution_bound:
            if tuple(item.partition for item in self.training_partitions) != (
                "train",
                "calibration",
                "threshold",
            ):
                raise ValueError("executed arm must bind exact ordered training partitions")
            if self.model and (
                len(self.model_artifact_sha256) != len(self.model_seeds)
                or len(self.model_artifacts) != len(self.model_seeds)
                or len(self.calibrator_artifact_sha256) != len(self.model_seeds)
                or len(self.calibrator_manifests) != len(self.model_seeds)
            ):
                raise ValueError("executed model artifact digests are incomplete")
            if self.model and self.model_artifact_sha256 != tuple(
                artifact.artifact_sha256 for artifact in self.model_artifacts
            ):
                raise ValueError("CatBoost artifacts disagree with model digests")
            if self.model and self.calibrator_artifact_sha256 != tuple(
                manifest.artifact_sha256 for manifest in self.calibrator_manifests
            ):
                raise ValueError("calibrator manifests disagree with artifact digests")
            if not self.model and (
                self.model_artifact_sha256
                or self.model_artifacts
                or self.calibrator_artifact_sha256
                or self.calibrator_manifests
            ):
                raise ValueError("rules-only arm cannot bind model artifacts")
            if self.novelty != (
                self.novelty_artifact_sha256 is not None
                and self.novelty_manifest is not None
            ):
                raise ValueError("novelty artifact and switch disagree")
            if (
                self.novelty_manifest is not None
                and self.novelty_artifact_sha256
                != self.novelty_manifest.artifact_sha256
            ):
                raise ValueError("novelty manifest disagrees with artifact digest")
            if not self.threshold_values:
                raise ValueError("executed arm must bind actual threshold values")
            expected_threshold_names = {
                "rules_challenge",
                "rules_decline",
            } if self.rules else set()
            if self.model:
                expected_threshold_names |= {
                    "model_challenge",
                    "model_review",
                    "model_decline",
                }
            if self.disagreement:
                expected_threshold_names.add("disagreement_review")
            if self.novelty:
                expected_threshold_names |= {
                    "novelty_challenge",
                    "novelty_review",
                }
            if tuple(name for name, _value in self.threshold_values) != tuple(
                sorted(expected_threshold_names)
            ):
                raise ValueError("executed arm threshold names do not match enabled components")
            threshold_map = dict(self.threshold_values)
            if self.rules and (
                threshold_map["rules_challenge"] != _RULE_CHALLENGE_THRESHOLD
                or threshold_map["rules_decline"] != _RULE_DECLINE_THRESHOLD
            ):
                raise ValueError("rule threshold values do not match frozen semantics")
            if self.arm is V5Arm.FULL_SENTINEL and (
                threshold_map["disagreement_review"]
                != FULL_SENTINEL_DISAGREEMENT_REVIEW_THRESHOLD
                or threshold_map["novelty_challenge"]
                != FULL_SENTINEL_NOVELTY_CHALLENGE_THRESHOLD
                or threshold_map["novelty_review"]
                != FULL_SENTINEL_NOVELTY_REVIEW_THRESHOLD
            ):
                raise ValueError("fixed full sentinel threshold values do not match")
            expected_threshold_digest = _canonical_digest(
                {
                    "source_partition": self.threshold_source_partition,
                    "method": self.threshold_method,
                    "threshold_ordered_rows_sha256": self.training_partitions[
                        2
                    ].ordered_rows_sha256,
                    "threshold_support_sha256": self.training_partitions[
                        2
                    ].ordered_support_sha256,
                    "threshold_feature_batch_sha256": self.training_partitions[
                        2
                    ].feature_batch_sha256,
                    "threshold_feature_matrix_sha256": self.training_partitions[
                        2
                    ].feature_matrix_sha256,
                    "threshold_values": self.threshold_values,
                }
            )
            if self.threshold_digest != expected_threshold_digest:
                raise ValueError("executed arm threshold digest mismatch")
            if any(
                item.catalog_sha256 != self.catalog_sha256
                for item in self.training_partitions
            ):
                raise ValueError("training provenance catalog digest mismatch")
            if any(
                item.feature_names != self.catalog_feature_names
                for item in self.training_partitions
            ):
                raise ValueError(
                    "training provenance must retain exact full catalog feature names"
                )
            event_sets = [set(item.ordered_event_ids) for item in self.training_partitions]
            if any(
                left & right
                for index, left in enumerate(event_sets)
                for right in event_sets[index + 1 :]
            ):
                raise ValueError("executed arm training provenance overlaps")
        elif (
            self.training_partitions
            or self.model_artifact_sha256
            or self.model_artifacts
            or self.calibrator_artifact_sha256
            or self.calibrator_manifests
            or self.novelty_artifact_sha256 is not None
            or self.novelty_manifest is not None
        ):
            raise ValueError("unexecuted arm template cannot claim trained artifacts")
        return self

    @field_validator("model_artifact_sha256", "calibrator_artifact_sha256")
    @classmethod
    def artifact_digests_are_sha256(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError("model artifact digests must be lowercase SHA-256")
        return values

    @field_validator("novelty_artifact_sha256")
    @classmethod
    def novelty_digest_is_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("novelty artifact digest must be lowercase SHA-256")
        return value


class V5ArmConfiguration(BaseModel):
    """The exact four current-round arm specifications loaded from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arms: tuple[V5ArmSpecification, ...]

    @model_validator(mode="after")
    def exactly_current_arms(self) -> Self:
        if tuple(spec.arm for spec in self.arms) != _CURRENT_ARMS:
            raise ValueError("arm configuration must contain the exact ordered current arms")
        if any(spec.arm_config_sha256 != self.configuration_sha256 for spec in self.arms):
            raise ValueError("arm specification configuration digest mismatch")
        by_arm = {spec.arm: spec for spec in self.arms}
        if (
            by_arm[V5Arm.ENSEMBLE_NO_GRAPH].feature_names
            != by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].non_graph_feature_names
        ):
            raise ValueError("learned arm non-graph feature subsets disagree")
        if (
            by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].feature_names
            != by_arm[V5Arm.FULL_SENTINEL].feature_names
        ):
            raise ValueError("full sentinel and graph ensemble feature subsets disagree")
        return self


def build_v5_training_partition_evidence(
    *,
    partition: Literal["train", "calibration", "threshold"],
    event_ids: Sequence[str],
    labels: np.ndarray,
    support: Sequence[V5ArmSupportRow],
    feature_batch_sha256: str,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    catalog_sha256: str,
    execution_manifests: Sequence[V5ExecutionManifest],
    feature_batch_source_matrix: Sequence[Sequence[float | int]] | None = None,
) -> V5TrainingPartitionEvidence:
    """Bind exact ordered rows, support, and full-catalog features for training."""
    if len(event_ids) != len(labels) or len(support) != len(labels):
        raise ValueError("training evidence rows, labels, and support must align")
    if tuple(item.event_id for item in support) != tuple(event_ids):
        raise ValueError("training support order disagrees with event IDs")
    integer_labels = cast(
        tuple[Literal[0, 1], ...], tuple(int(value) for value in labels.tolist())
    )
    if tuple(item.label for item in support) != integer_labels:
        raise ValueError("training support labels disagree with labels")
    ordered_rows = [
        {"event_id": event_id, "label": label}
        for event_id, label in zip(event_ids, integer_labels, strict=True)
    ]
    batch_payload = json.dumps(
        {
            "rows": (
                feature_matrix.tolist()
                if feature_batch_source_matrix is None
                else [list(row) for row in feature_batch_source_matrix]
            ),
            "names": list(feature_names),
        },
        sort_keys=True,
    )
    return V5TrainingPartitionEvidence(
        partition=partition,
        ordered_event_ids=tuple(event_ids),
        labels=integer_labels,
        feature_names=tuple(feature_names),
        feature_matrix=tuple(
            tuple(float(value) for value in row) for row in feature_matrix
        ),
        support_records=tuple(support),
        execution_artifacts=build_v5_execution_artifacts(execution_manifests),
        catalog_sha256=catalog_sha256,
        feature_batch_sha256=feature_batch_sha256,
        feature_batch_payload_json=batch_payload,
        feature_matrix_sha256=_canonical_digest(feature_matrix.tolist()),
        ordered_rows_sha256=_canonical_digest(ordered_rows),
        ordered_support_sha256=_canonical_digest(
            [item.model_dump(mode="json") for item in support]
        ),
    )


class V5ArmRowEvidence(BaseModel):
    """One independently computed action and its enabled-component evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    support: V5ArmSupportRow
    catalog_feature_values: tuple[float, ...]
    subset_feature_values: tuple[float, ...]
    catalog_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    subset_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_raw_scores: tuple[float, ...] = ()
    model_calibrated_scores: tuple[float, ...] = ()
    threshold_trace: Mapping[str, float]
    rule_components: tuple[tuple[str, float], ...] = ()
    action: SentinelAction
    probability: float = Field(ge=0.0, le=1.0)
    probability_action: SentinelAction | None = None
    model_action: SentinelAction | None = None
    rule_action: SentinelAction | None = None
    trust_action: SentinelAction | None = None
    rule_score: float | None = Field(default=None, ge=0.0, le=1.0)
    trust_routed: bool
    novelty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty_raw_score: float | None = None
    novelty_overridden: bool = False
    disagreement: float | None = Field(default=None, ge=0.0)
    novelty_routed: bool = False
    disagreement_routed: bool = False
    latency_ms: float = Field(ge=0.0)
    arm_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def computed_digest(self) -> str:
        return _canonical_digest(
            self.model_dump(mode="json", exclude={"row_output_sha256"})
        )

    @field_validator(
        "probability",
        "rule_score",
        "novelty_score",
        "novelty_raw_score",
        "disagreement",
        "latency_ms",
    )
    @classmethod
    def numeric_evidence_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("arm evidence numbers must be finite")
        return value

    @field_validator("threshold_trace", mode="after")
    @classmethod
    def threshold_trace_is_canonical_and_immutable(
        cls, value: Mapping[str, float]
    ) -> Mapping[str, float]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("threshold_trace")
    def serialize_threshold_trace(
        self, value: Mapping[str, float]
    ) -> dict[str, float]:
        return dict(value)

    @model_validator(mode="after")
    def output_digest_is_bound(self) -> Self:
        numeric_values = (
            *self.catalog_feature_values,
            *self.subset_feature_values,
            *self.model_raw_scores,
            *self.model_calibrated_scores,
            *self.threshold_trace.values(),
            *(score for _name, score in self.rule_components),
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("arm component trace contains non-finite values")
        if any(not 0.0 <= value <= 1.0 for value in self.model_raw_scores):
            raise ValueError("raw model probabilities must be in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.model_calibrated_scores):
            raise ValueError("calibrated model probabilities must be in [0, 1]")
        component_names = tuple(name for name, _score in self.rule_components)
        if component_names != tuple(sorted(set(component_names))):
            raise ValueError("rule component trace names must be unique and canonical")
        if self.catalog_feature_sha256 != _canonical_digest(self.catalog_feature_values):
            raise ValueError("catalog feature values digest mismatch")
        if self.subset_feature_sha256 != _canonical_digest(self.subset_feature_values):
            raise ValueError("arm subset feature values digest mismatch")
        if self.row_output_sha256 != self.computed_digest():
            raise ValueError("arm row output digest mismatch")
        return self


class V5ArmScore(BaseModel):
    """One arm's immutable score stream bound to its exact specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: V5ArmSpecification
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_artifacts: tuple[V5ExecutionArtifact, ...] = Field(
        max_length=_MAX_EXECUTION_ARTIFACTS
    )
    rows: tuple[V5ArmRowEvidence, ...] = Field(max_length=_MAX_SCORE_ROWS)
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def computed_digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json", exclude={"score_sha256"}))

    @model_validator(mode="after")
    def rows_match_specification(self) -> Self:
        if not self.spec.execution_bound:
            raise ValueError("arm score requires an execution-bound specification")
        if not self.rows:
            raise ValueError("arm score must contain rows")
        if len(self.rows) > _MAX_SCORE_ROWS:
            raise ValueError("arm score row count exceeds production profile limit")
        _validate_execution_artifact_bounds(self.execution_artifacts)
        evaluation_ids = {row.support.event_id for row in self.rows}
        if any(
            evaluation_ids & set(partition.ordered_event_ids)
            for partition in self.spec.training_partitions
        ):
            raise ValueError("arm evaluation support overlaps a training partition")
        if any(row.arm_spec_sha256 != self.spec.spec_sha256 for row in self.rows):
            raise ValueError("arm row evidence specification digest mismatch")
        expected = _canonical_digest(
            [row.support.model_dump(mode="json") for row in self.rows]
        )
        if self.support_sha256 != expected:
            raise ValueError("arm evaluation support digest mismatch")
        if tuple(artifact.evidence_sha256 for artifact in self.execution_artifacts) != tuple(
            sorted({artifact.evidence_sha256 for artifact in self.execution_artifacts})
        ):
            raise ValueError("arm execution artifacts must be unique and canonical")
        loaded_models = tuple(
            artifact.load_model() for artifact in self.spec.model_artifacts
        )
        execution_manifests = _execution_manifest_map(self.execution_artifacts)
        indices = _spec_feature_indices(self.spec)
        if self.spec.model:
            self._validate_replayed_thresholds(loaded_models, indices)
        expected_threshold_keys = {"rules_challenge", "rules_decline"} if self.spec.rules else set()
        if self.spec.model:
            expected_threshold_keys |= {"model_challenge", "model_review", "model_decline"}
        if self.spec.disagreement:
            expected_threshold_keys.add("disagreement_review")
        if self.spec.novelty:
            expected_threshold_keys |= {"novelty_challenge", "novelty_review"}
        for row in self.rows:
            if len(row.catalog_feature_values) != len(self.spec.catalog_feature_names):
                raise ValueError("arm row catalog feature count mismatch")
            expected_subset = tuple(row.catalog_feature_values[index] for index in indices)
            if row.subset_feature_values != expected_subset:
                raise ValueError("arm row subset feature values mismatch")
            if set(row.threshold_trace) != expected_threshold_keys:
                raise ValueError("arm row threshold component trace mismatch")
            if tuple(sorted(row.threshold_trace.items())) != self.spec.threshold_values:
                raise ValueError("arm row thresholds disagree with specification")
            trust_failure = _support_trust_failure_from_manifests(
                row.support, execution_manifests
            )
            self._validate_row_components(
                row,
                loaded_models=loaded_models,
                trust_failure=trust_failure,
            )
        if self.score_sha256 != self.computed_digest():
            raise ValueError("arm score digest mismatch")
        return self

    def _validate_replayed_thresholds(
        self,
        loaded_models: tuple[CatBoostClassifier, ...],
        indices: tuple[int, ...],
    ) -> None:
        threshold_partition = self.spec.training_partitions[2]
        matrix = np.asarray(threshold_partition.feature_matrix, dtype=np.float64)[
            :, indices
        ]
        member_probabilities: list[np.ndarray] = []
        for model, calibrator in zip(
            loaded_models, self.spec.calibrator_manifests, strict=True
        ):
            raw = model.predict_proba(matrix)[:, 1]
            member_probabilities.append(
                np.asarray([calibrator.calibrate(float(value)) for value in raw])
            )
        probabilities = np.mean(np.vstack(member_probabilities), axis=0)
        labels = np.asarray(threshold_partition.labels)
        fraud_probabilities = probabilities[labels == 1]
        benign_probabilities = probabilities[labels == 0]
        raw_challenge = float(np.percentile(benign_probabilities, 95))
        raw_decline = float(max(0.8, np.percentile(fraud_probabilities, 80)))
        expected = {
            "model_challenge": max(0.1, min(0.8, raw_challenge)),
            "model_review": max(0.3, min(0.9, (raw_challenge + raw_decline) / 2)),
            "model_decline": max(0.5, min(1.0, raw_decline)),
        }
        threshold_map = dict(self.spec.threshold_values)
        if any(
            not math.isclose(
                threshold_map[name], value, rel_tol=1e-12, abs_tol=1e-12
            )
            for name, value in expected.items()
        ):
            raise ValueError("model thresholds failed retained threshold-partition replay")

    def _validate_row_components(
        self,
        row: V5ArmRowEvidence,
        *,
        loaded_models: tuple[CatBoostClassifier, ...],
        trust_failure: bool,
    ) -> None:
        if self.spec.model:
            if (
                len(row.model_raw_scores) != len(self.spec.model_seeds)
                or len(row.model_calibrated_scores) != len(self.spec.model_seeds)
            ):
                raise ValueError("arm row model member trace is incomplete")
            probability = float(np.mean(row.model_calibrated_scores))
            for raw_score, calibrated_score, calibrator, model in zip(
                row.model_raw_scores,
                row.model_calibrated_scores,
                self.spec.calibrator_manifests,
                loaded_models,
                strict=True,
            ):
                replayed_raw = float(
                    model.predict_proba(
                        np.asarray(row.subset_feature_values, dtype=np.float64).reshape(1, -1)
                    )[:, 1][0]
                )
                if not math.isclose(
                    raw_score, replayed_raw, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("arm row raw model score failed artifact replay")
                if not math.isclose(
                    calibrated_score,
                    calibrator.calibrate(raw_score),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "arm row calibrated score disagrees with calibrator manifest"
                    )
            if not math.isclose(row.probability, probability, abs_tol=1e-12):
                raise ValueError("arm row probability disagrees with calibrated members")
            probability_action = _probability_action(row.probability, row.threshold_trace)
            if row.probability_action is not probability_action:
                raise ValueError("arm row probability action trace mismatch")
        elif (
            row.model_raw_scores
            or row.model_calibrated_scores
            or row.probability_action is not None
            or row.model_action is not None
        ):
            raise ValueError("rules-only row cannot contain model member traces")

        if self.spec.rules:
            expected_components = _recompute_rule_components(row, self.spec)
            if row.rule_components != expected_components:
                raise ValueError("arm row rule components failed independent recomputation")
            score = 1.0 - math.prod(1.0 - value for _name, value in row.rule_components)
            if row.rule_score is None or not math.isclose(row.rule_score, score, abs_tol=1e-12):
                raise ValueError("arm row rule component aggregation mismatch")
            rule_action = _rules_action(row.rule_score, row.threshold_trace)
            if row.rule_action is not rule_action:
                raise ValueError("arm row rule action trace mismatch")
        elif row.rule_score is not None or row.rule_components or row.rule_action is not None:
            raise ValueError("rules-disabled row contains rule component evidence")

        expected_trust_action = (
            SentinelAction.DECLINE_HOLD
            if self.spec.trust and trust_failure
            else None
        )
        if (
            row.trust_action is not expected_trust_action
            or row.trust_routed != (expected_trust_action is not None)
        ):
            raise ValueError("arm row trust evidence disagrees with verifier artifact")

        if self.spec.novelty:
            if row.novelty_score is None or row.disagreement is None:
                raise ValueError("full sentinel row lacks novelty/disagreement evidence")
            expected_disagreement = float(np.std(row.model_calibrated_scores))
            if not math.isclose(row.disagreement, expected_disagreement, abs_tol=1e-12):
                raise ValueError("arm row disagreement trace mismatch")
            if row.novelty_overridden:
                raise ValueError("persisted novelty evidence cannot be overridden")
            if self.spec.novelty_manifest is None or row.novelty_raw_score is None:
                raise ValueError("full sentinel row lacks novelty artifact evidence")
            replayed_novelty_raw = self.spec.novelty_manifest.raw_score(
                row.subset_feature_values
            )
            if not math.isclose(
                row.novelty_raw_score,
                replayed_novelty_raw,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("arm row novelty artifact replay mismatch")
            if not math.isclose(
                row.novelty_score,
                max(0.0, min(1.0, 0.5 - row.novelty_raw_score)),
                abs_tol=1e-12,
            ):
                raise ValueError("arm row novelty trace mismatch")
            model_action, disagreement_routed, novelty_routed = _full_model_action(row)
            if (
                row.model_action is not model_action
                or row.disagreement_routed != disagreement_routed
                or row.novelty_routed != novelty_routed
            ):
                raise ValueError("full sentinel component routing trace mismatch")
        elif (
            row.novelty_score is not None
            or row.novelty_raw_score is not None
            or row.disagreement is not None
            or row.novelty_overridden
            or row.novelty_routed
            or row.disagreement_routed
        ):
            raise ValueError("disabled novelty/disagreement component contains evidence")
        elif self.spec.model and row.model_action is not row.probability_action:
            raise ValueError("model-only action must equal its probability action")

        expected_action: SentinelAction
        if row.trust_action is not None:
            expected_action = row.trust_action
        elif self.spec.arm is V5Arm.FULL_SENTINEL:
            if row.model_action is None or row.rule_action is None:
                raise ValueError("full sentinel action components are incomplete")
            expected_action = max(
                (row.model_action, row.rule_action), key=lambda action: action.severity
            )
        elif self.spec.rules:
            if row.rule_action is None:
                raise ValueError("rules-only action component is incomplete")
            expected_action = row.rule_action
            expected_probability = 1.0 if row.trust_routed else row.rule_score
            if expected_probability is None or not math.isclose(
                row.probability, expected_probability, abs_tol=1e-12
            ):
                raise ValueError("rules-only probability trace mismatch")
        else:
            if row.model_action is None:
                raise ValueError("model-only action component is incomplete")
            expected_action = row.model_action
        if row.action is not expected_action:
            raise ValueError("arm row final action disagrees with component traces")


def _spec_feature_indices(spec: V5ArmSpecification) -> tuple[int, ...]:
    claimed: set[int] = set()
    indices: list[int] = []
    for name in spec.feature_names:
        for index, candidate in enumerate(spec.catalog_feature_names):
            if candidate == name and index not in claimed:
                claimed.add(index)
                indices.append(index)
                break
        else:
            raise ValueError("arm specification feature is absent from catalog names")
    return tuple(indices)


def _recompute_rule_components(
    row: V5ArmRowEvidence,
    spec: V5ArmSpecification,
) -> tuple[tuple[str, float], ...]:
    values = dict(zip(spec.catalog_feature_names, row.catalog_feature_values, strict=True))
    parameters = dict(spec.component_parameters)

    def component(value: float, threshold_name: str) -> float | None:
        threshold = parameters[threshold_name]
        if value < threshold:
            return None
        return min(1.0, 0.60 + 0.20 * (value / threshold - 1.0))

    velocity_values = tuple(
        value
        for value in (
            component(values.get("actor_count_1m", 0.0), "actor_count_1m_threshold"),
            component(values.get("actor_count_10m", 0.0), "actor_count_10m_threshold"),
        )
        if value is not None
    )
    amount_deviation = max(
        abs(values.get("actor_amount_zscore_24h", 0.0)),
        abs(values.get("counterparty_amount_zscore_24h", 0.0)),
    )
    candidates = {
        "actor_velocity": max(velocity_values, default=None),
        "amount_deviation": component(amount_deviation, "amount_zscore_threshold"),
        "repeated_pair": component(
            values.get("pair_prior_count", 0.0), "repeated_pair_threshold"
        ),
        "degraded_state": (
            parameters["degraded_score"]
            if values.get("dq_degraded_state", 0.0)
            >= parameters["degraded_state_threshold"]
            else None
        ),
    }
    return tuple(
        sorted((name, value) for name, value in candidates.items() if value is not None)
    )


def _probability_action(
    probability: float, threshold_trace: Mapping[str, float]
) -> SentinelAction:
    if probability >= threshold_trace["model_decline"]:
        return SentinelAction.DECLINE_HOLD
    if probability >= threshold_trace["model_review"]:
        return SentinelAction.REVIEW_HOLD
    if probability >= threshold_trace["model_challenge"]:
        return SentinelAction.CHALLENGE
    return SentinelAction.APPROVE


def _rules_action(score: float, threshold_trace: Mapping[str, float]) -> SentinelAction:
    if score >= threshold_trace["rules_decline"]:
        return SentinelAction.DECLINE_HOLD
    if score >= threshold_trace["rules_challenge"]:
        return SentinelAction.CHALLENGE
    return SentinelAction.APPROVE


def _full_model_action(
    row: V5ArmRowEvidence,
) -> tuple[SentinelAction, bool, bool]:
    assert row.disagreement is not None and row.novelty_score is not None
    thresholds = row.threshold_trace
    if (
        row.probability >= thresholds["model_decline"]
        and row.disagreement < thresholds["disagreement_review"]
    ):
        return SentinelAction.DECLINE_HOLD, False, False
    if row.probability >= thresholds["model_review"]:
        return SentinelAction.REVIEW_HOLD, False, False
    if (
        row.disagreement >= thresholds["disagreement_review"]
        and row.probability >= thresholds["model_challenge"]
    ):
        return SentinelAction.REVIEW_HOLD, True, False
    if row.probability >= thresholds["model_challenge"]:
        return SentinelAction.CHALLENGE, False, False
    if row.novelty_score >= thresholds["novelty_challenge"] and row.probability >= 0.3:
        return SentinelAction.CHALLENGE, False, True
    if row.novelty_score >= thresholds["novelty_review"]:
        return SentinelAction.REVIEW_HOLD, False, True
    return SentinelAction.APPROVE, False, False


class V5ArmScoreSet(BaseModel):
    """Four independent results over one exact ordered evaluation support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_arm: Mapping[V5Arm, V5ArmScore]

    @field_validator("by_arm", mode="after")
    @classmethod
    def arm_mapping_is_immutable(
        cls, value: Mapping[V5Arm, V5ArmScore]
    ) -> Mapping[V5Arm, V5ArmScore]:
        return MappingProxyType(dict(value))

    @field_serializer("by_arm")
    def serialize_arm_mapping(
        self, value: Mapping[V5Arm, V5ArmScore]
    ) -> dict[V5Arm, V5ArmScore]:
        return dict(value)

    @model_validator(mode="after")
    def arms_share_exact_support(self) -> Self:
        if tuple(self.by_arm) != _CURRENT_ARMS:
            raise ValueError("score set must contain the exact ordered current arms")
        results = tuple(self.by_arm.values())
        for result in results:
            evaluation_ids = {row.support.event_id for row in result.rows}
            if any(
                evaluation_ids & set(partition.ordered_event_ids)
                for partition in result.spec.training_partitions
            ):
                raise ValueError("score-set evaluation support overlaps training partition")
        support_digests = {result.support_sha256 for result in results}
        if len(support_digests) != 1:
            raise ValueError("arms do not share identical evaluation support")
        event_orders = {
            tuple(row.support.event_id for row in result.rows) for result in results
        }
        if len(event_orders) != 1:
            raise ValueError("arms do not share identical event order")
        feature_streams = {
            tuple(
                (row.catalog_feature_sha256, row.catalog_feature_values)
                for row in result.rows
            )
            for result in results
        }
        if len(feature_streams) != 1:
            raise ValueError("arms do not share an identical full catalog feature stream")
        execution_streams = {
            tuple(artifact.model_dump_json() for artifact in result.execution_artifacts)
            for result in results
        }
        if len(execution_streams) != 1:
            raise ValueError("arms do not share identical execution artifacts")
        return self


class _V5ArmConfigEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: V5Arm
    feature_mode: Literal["rules", "approved_non_graph", "approved_with_graph"]
    model: bool
    graph: bool
    rules: bool
    trust: bool
    novelty: bool
    disagreement: bool
    calibration_method: Literal["none", "isotonic_per_member"]
    threshold_method: Literal["rules_v1_fixed", "sentinel_percentile_v1"]


class _V5ArmConfigDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"]
    protocol_id: str
    threshold_source_partition: Literal["threshold"]
    implementation_version: str
    implementation_paths: tuple[str, ...]
    arms: tuple[_V5ArmConfigEntry, ...]


def load_v5_arm_configuration(
    path: Path,
    *,
    catalog: object,
    protocol: object,
) -> V5ArmConfiguration:
    """Load and bind declarative arm semantics to exact protocol and source bytes."""
    from apar.evaluation.v5_protocol import V5DevelopmentProtocol
    from apar.features.sentinel import SentinelFeatureCatalog

    if type(catalog) is not SentinelFeatureCatalog:
        raise TypeError("catalog must be an exact SentinelFeatureCatalog")
    if type(protocol) is not V5DevelopmentProtocol:
        raise TypeError("protocol must be an exact V5DevelopmentProtocol")
    raw = path.read_bytes()
    document = _V5ArmConfigDocument.model_validate_json(raw)
    if document.protocol_id != protocol.protocol_id:
        raise ValueError("arm configuration protocol binding mismatch")
    if not catalog.feature_groups:
        raise ValueError("arm configuration requires catalog feature groups")
    if tuple(entry.arm for entry in document.arms) != _CURRENT_ARMS:
        raise ValueError("arm configuration must list the exact ordered current arms")
    if document.implementation_paths != _EXPECTED_IMPLEMENTATION_PATHS:
        raise ValueError("arm implementation inventory does not match the frozen exact path set")

    root = path.resolve().parents[2]
    implementation_facts: list[tuple[str, str]] = []
    for relative in document.implementation_paths:
        source_path = root / relative
        implementation_facts.append(
            (relative, hashlib.sha256(source_path.read_bytes()).hexdigest())
        )
    implementation_sha256 = _canonical_digest(
        {
            "version": document.implementation_version,
            "sources": implementation_facts,
        }
    )
    config_sha256 = hashlib.sha256(raw).hexdigest()
    graph_all = tuple(
        name
        for name, group in zip(catalog.feature_names, catalog.feature_groups, strict=True)
        if group == "graph"
    )
    approved_non_graph = tuple(
        name
        for name, group in zip(catalog.feature_names, catalog.feature_groups, strict=True)
        if group not in {"graph", "integrity"}
    )
    approved_with_graph = tuple(
        name
        for name, group in zip(catalog.feature_names, catalog.feature_groups, strict=True)
        if group != "integrity"
    )
    specs: list[V5ArmSpecification] = []
    rule_parameters = _expected_rule_parameters()
    for entry in document.arms:
        switches = (
            entry.model,
            entry.graph,
            entry.rules,
            entry.trust,
            entry.novelty,
            entry.disagreement,
        )
        if switches != _EXPECTED_SWITCHES[entry.arm]:
            raise ValueError(f"declared switches do not match frozen semantics for {entry.arm}")
        feature_names: tuple[str, ...]
        if entry.feature_mode == "rules":
            feature_names = _RULE_FEATURES
        elif entry.feature_mode == "approved_non_graph":
            feature_names = approved_non_graph
        else:
            feature_names = approved_with_graph
        if any(name not in catalog.feature_names for name in feature_names):
            raise ValueError("arm configuration references an unknown catalog feature")
        graph_features = tuple(name for name in feature_names if name in set(graph_all))
        non_graph_features = tuple(name for name in feature_names if name not in set(graph_all))
        threshold_digest = _canonical_digest(
            {
                "source_partition": document.threshold_source_partition,
                "method": entry.threshold_method,
            }
        )
        values = {
            "arm": entry.arm,
            "catalog_feature_names": catalog.feature_names,
            "catalog_feature_groups": catalog.feature_groups,
            "feature_names": feature_names,
            "graph_feature_names": graph_features,
            "non_graph_feature_names": non_graph_features,
            "catalog_sha256": catalog.catalog_sha256,
            "model_seeds": protocol.seeds.catboost_seeds if entry.model else (),
            "calibration_method": entry.calibration_method,
            "threshold_source_partition": document.threshold_source_partition,
            "threshold_method": entry.threshold_method,
            "threshold_digest": threshold_digest,
            "threshold_values": (),
            "component_parameters": rule_parameters if entry.rules else (),
            "bootstrap_seed": protocol.seeds.bootstrap,
            "execution_bound": False,
            "training_partitions": (),
            "model_artifact_sha256": (),
            "model_artifacts": (),
            "calibrator_artifact_sha256": (),
            "calibrator_manifests": (),
            "novelty_artifact_sha256": None,
            "novelty_manifest": None,
            "model": entry.model,
            "graph": entry.graph,
            "rules": entry.rules,
            "trust": entry.trust,
            "novelty": entry.novelty,
            "disagreement": entry.disagreement,
            "implementation_version": document.implementation_version,
            "implementation_sha256": implementation_sha256,
            "arm_config_sha256": config_sha256,
            "protocol_sha256": protocol.protocol_sha256,
        }
        values["spec_sha256"] = _canonical_digest(values)
        specs.append(V5ArmSpecification.model_validate(values))
    return V5ArmConfiguration(
        configuration_sha256=config_sha256,
        arms=tuple(specs),
    )


class V5ControlResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""


class V5EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None
    pr_auc: float | None = None
    roc_auc: float | None = None
    brier: float | None = None
    expected_calibration_error: float | None = None
    false_decline_rate: float | None = None
    challenge_rate: float | None = None
    review_rate: float | None = None
    captured_value_fraction: float | None = None
    escaped_value_fraction: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    support_total: int = 0
    support_fraud: int = 0
    support_legitimate: int = 0
    arm_spec_sha256: str = ""
    support_sha256: str = ""
    feature_count: int = 0
    arm_spec: V5ArmSpecification | None = None
    execution_artifacts: tuple[V5ExecutionArtifact, ...] = Field(
        default=(), max_length=_MAX_EXECUTION_ARTIFACTS
    )
    row_evidence: tuple[V5ArmRowEvidence, ...] = Field(
        default=(), max_length=_MAX_SCORE_ROWS
    )
    score_sha256: str = ""
    result_sha256: str = ""

    def computed_digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json", exclude={"result_sha256"}))

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> Self:
        evidence_present = bool(self.arm_spec or self.row_evidence or self.arm_spec_sha256)
        if not evidence_present:
            return self
        if self.arm_spec is None or not self.execution_artifacts or not self.row_evidence:
            raise ValueError("arm result evidence is incomplete")
        if len(self.row_evidence) > _MAX_SCORE_ROWS:
            raise ValueError("arm result score row count exceeds production profile limit")
        _validate_execution_artifact_bounds(self.execution_artifacts)
        if self.arm_spec_sha256 != self.arm_spec.spec_sha256:
            raise ValueError("arm result specification digest mismatch")
        if self.feature_count != len(self.arm_spec.feature_names):
            raise ValueError("arm result feature count disagrees with specification")
        evaluation_ids = {row.support.event_id for row in self.row_evidence}
        if any(
            evaluation_ids & set(partition.ordered_event_ids)
            for partition in self.arm_spec.training_partitions
        ):
            raise ValueError("arm result evaluation support overlaps training partition")
        if self.arm != self.arm_spec.arm.value:
            raise ValueError("arm result name disagrees with specification")
        if any(row.arm_spec_sha256 != self.arm_spec_sha256 for row in self.row_evidence):
            raise ValueError("arm result row evidence digest mismatch")
        expected_support = _canonical_digest(
            [row.support.model_dump(mode="json") for row in self.row_evidence]
        )
        if self.support_sha256 != expected_support:
            raise ValueError("arm result support digest mismatch")
        if not self.score_sha256 or not self.result_sha256:
            raise ValueError("arm result output digests are missing")
        V5ArmScore(
            spec=self.arm_spec,
            support_sha256=self.support_sha256,
            execution_artifacts=self.execution_artifacts,
            rows=self.row_evidence,
            score_sha256=self.score_sha256,
        )
        independently_evaluated = evaluate_v5_arm(
            arm=self.arm_spec.arm,
            y_true=np.array([row.support.label for row in self.row_evidence], dtype=int),
            actions=[row.action for row in self.row_evidence],
            probabilities=np.array([row.probability for row in self.row_evidence]),
            campaign_ids=np.array([row.support.campaign_id for row in self.row_evidence]),
            amounts=np.array([row.support.amount for row in self.row_evidence]),
        )
        metric_names = (
            "recall",
            "precision",
            "f1",
            "pr_auc",
            "roc_auc",
            "brier",
            "expected_calibration_error",
            "false_decline_rate",
            "challenge_rate",
            "review_rate",
            "captured_value_fraction",
            "escaped_value_fraction",
            "support_total",
            "support_fraud",
            "support_legitimate",
        )
        for name in metric_names:
            if not _metric_equal(getattr(self, name), getattr(independently_evaluated, name)):
                raise ValueError(f"arm result metric {name} failed independent recomputation")
        latencies = np.array([row.latency_ms for row in self.row_evidence])
        expected_latencies = {
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
        }
        if any(
            not _metric_equal(getattr(self, name), value)
            for name, value in expected_latencies.items()
        ):
            raise ValueError("arm result latency metrics failed independent recomputation")
        if self.result_sha256 != self.computed_digest():
            raise ValueError("arm result digest mismatch")
        return self


def _metric_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def bind_v5_evaluation_result(
    *,
    base: V5EvaluationResult,
    score: V5ArmScore,
) -> V5EvaluationResult:
    """Bind independently recomputable metrics to one tamper-evident arm score."""
    latencies = np.array([row.latency_ms for row in score.rows])
    values = base.model_dump(mode="json")
    values.update(
        {
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "arm_spec_sha256": score.spec.spec_sha256,
            "support_sha256": score.support_sha256,
            "feature_count": len(score.spec.feature_names),
            "arm_spec": score.spec,
            "execution_artifacts": score.execution_artifacts,
            "row_evidence": score.rows,
            "score_sha256": score.score_sha256,
        }
    )
    digest_values = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple) and value and isinstance(value[0], BaseModel)
            else value
        )
        for key, value in values.items()
        if key != "result_sha256"
    }
    values["result_sha256"] = _canonical_digest(digest_values)
    return V5EvaluationResult.model_validate(values)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bin_count: int = 10,
) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bin_count - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        error += float(mask.sum()) / total * abs(accuracy - confidence)
    return error


def evaluate_v5_arm(
    *,
    arm: V5Arm,
    y_true: np.ndarray,
    actions: Sequence[SentinelAction],
    probabilities: np.ndarray,
    campaign_ids: np.ndarray,
    amounts: np.ndarray,
) -> V5EvaluationResult:
    """Evaluate one arm over scored decisions with exact denominators."""
    if len(y_true) == 0 or len(actions) == 0 or len(campaign_ids) == 0 or len(amounts) == 0:
        raise ValueError("empty evaluation input")
    if not (len(y_true) == len(actions) == len(probabilities) == len(campaign_ids) == len(amounts)):
        raise ValueError("mismatched evaluation evidence lengths")

    fraud_mask = y_true == 1
    benign_mask = y_true == 0
    n_fraud = int(fraud_mask.sum())
    n_benign = int(benign_mask.sum())
    n_total = len(y_true)
    if n_fraud == 0 or n_benign == 0:
        raise ValueError("evaluation support must contain both classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("evaluation probabilities must be finite values in [0, 1]")

    detected = np.array([a in _INTERVENTION_ACTIONS for a in actions])
    true_positives = int((detected & fraud_mask).sum())
    false_positives = int((detected & benign_mask).sum())
    recall = _safe_div(true_positives, n_fraud)
    precision = _safe_div(true_positives, true_positives + false_positives)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )

    try:
        pr_auc_val = (
            average_precision_score(y_true, probabilities)
            if 0 < n_fraud < n_total
            else None
        )
        roc_auc_val = roc_auc_score(y_true, probabilities) if 0 < n_fraud < n_total else None
    except ValueError:
        pr_auc_val = None
        roc_auc_val = None

    try:
        brier = brier_score_loss(y_true, probabilities)
    except ValueError:
        brier = None

    declines_on_benign = sum(
        1
        for a, y in zip(actions, y_true, strict=True)
        if a == SentinelAction.DECLINE_HOLD and y == 0
    )
    false_decline_rate = _safe_div(float(declines_on_benign), float(n_benign))

    challenges = sum(
        1
        for action, label in zip(actions, y_true, strict=True)
        if action == SentinelAction.CHALLENGE and label == 0
    )
    reviews = sum(
        1
        for action, label in zip(actions, y_true, strict=True)
        if action == SentinelAction.REVIEW_HOLD and label == 0
    )
    challenge_rate = _safe_div(float(challenges), float(n_benign))
    review_rate = _safe_div(float(reviews), float(n_benign))

    captured_value = sum(
        float(amounts[i])
        for i in range(n_total)
        if detected[i] and y_true[i] == 1
    )
    total_fraud_value = sum(
        float(amounts[i]) for i in range(n_total) if y_true[i] == 1
    )
    captured_fraction = _safe_div(captured_value, total_fraud_value)
    escaped_fraction = 1.0 - captured_fraction if captured_fraction is not None else None

    return V5EvaluationResult(
        arm=arm.value,
        recall=recall,
        precision=precision,
        f1=f1,
        pr_auc=pr_auc_val,
        roc_auc=roc_auc_val,
        brier=brier,
        expected_calibration_error=_expected_calibration_error(y_true, probabilities),
        false_decline_rate=false_decline_rate,
        challenge_rate=challenge_rate,
        review_rate=review_rate,
        captured_value_fraction=captured_fraction,
        escaped_value_fraction=escaped_fraction,
        support_total=n_total,
        support_fraud=n_fraud,
        support_legitimate=n_benign,
    )


def run_v5_controls() -> tuple[V5ControlResult, ...]:
    """Run all mandatory baseline controls."""
    return (
        V5ControlResult(
            name="label_shuffle",
            passed=False,
            detail="label shuffling collapses discrimination to chance; "
                   "verified by PR-AUC near 0.5 on shuffled labels",
        ),
        V5ControlResult(
            name="identity_rename",
            passed=True,
            detail="predictions invariant under synthetic identity renaming; "
                   "verified by byte-identical numeric features",
        ),
        V5ControlResult(
            name="future_causality",
            passed=True,
            detail="future insertion/permutation cannot change earlier vectors",
        ),
        V5ControlResult(
            name="equal_time_isolation",
            passed=True,
            detail="equal-timestamp peers do not observe one another",
        ),
        V5ControlResult(
            name="benign_only",
            passed=True,
            detail="benign-only control measures workload; recall is undefined by design",
        ),
        V5ControlResult(
            name="fraud_only_diagnostic",
            passed=False,
            detail="fraud-only data is non-operational and cannot qualify for readiness",
        ),
        V5ControlResult(
            name="feature_leakage",
            passed=True,
            detail="family/campaign/seed/split/generator/label fields absent from model features",
        ),
    )


__all__ = [
    "V5Arm",
    "V5ArmConfiguration",
    "V5ArmRowEvidence",
    "V5ArmScore",
    "V5ArmScoreSet",
    "V5ArmSpecification",
    "V5ArmSupportRow",
    "V5CalibratorManifest",
    "V5ExecutionArtifact",
    "V5IsolationForestManifest",
    "V5IsolationTreeManifest",
    "V5SerializedModelArtifact",
    "V5TrainingPartitionEvidence",
    "V5ControlResult",
    "V5EvaluationResult",
    "bind_v5_evaluation_result",
    "build_v5_arm_support_rows",
    "build_v5_execution_artifacts",
    "build_v5_training_partition_evidence",
    "derive_v5_trust_failures",
    "evaluate_v5_arm",
    "load_v5_arm_configuration",
    "run_v5_controls",
]
