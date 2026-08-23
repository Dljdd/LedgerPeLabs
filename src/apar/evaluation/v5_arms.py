"""Executable Sentinel v5 comparison-arm training and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apar.defense.rules import RuleManifest
from apar.defense.sentinel import (
    SentinelAction,
    SentinelDefender,
    SentinelThresholds,
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
)
from apar.features.sentinel import SentinelFeatureCatalog

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
) -> V5ArmSpecification:
    threshold_facts: dict[str, object] = {
        "source_partition": template.threshold_source_partition,
        "method": template.threshold_method,
    }
    if template.rules:
        threshold_facts["rules"] = {
            "challenge": _RULE_CHALLENGE_THRESHOLD,
            "decline": _RULE_DECLINE_THRESHOLD,
            "manifest": RuleManifest.default().model_dump(mode="json"),
        }
    if thresholds is not None:
        threshold_facts["model"] = thresholds.model_dump(mode="json")
    values = template.model_dump(mode="json", exclude={"spec_sha256"})
    values["threshold_digest"] = _digest(threshold_facts)
    values["spec_sha256"] = _digest(values)
    return V5ArmSpecification.model_validate(values)


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
                    spec=_bound_spec(template, thresholds=None),
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
        )
        trained_arms.append(
            V5TrainedArm(
                spec=_bound_spec(template, thresholds=defender.thresholds),
                feature_indices=indices,
                defender=defender,
            )
        )
    return V5TrainedArmSet(arms=tuple(trained_arms))


def _component_score(value: float, threshold: float) -> float | None:
    if value < threshold:
        return None
    return min(1.0, 0.60 + 0.20 * (value / threshold - 1.0))


def _rule_score(
    row: NDArray[np.float64],
    *,
    catalog: SentinelFeatureCatalog,
) -> float:
    values = {
        name: float(row[index])
        for index, name in enumerate(catalog.feature_names)
    }
    manifest = RuleManifest.default()
    actor_velocity = max(
        (
            score
            for score in (
                _component_score(values.get("actor_count_1m", 0.0), manifest.actor_count_1m),
                _component_score(values.get("actor_count_10m", 0.0), manifest.actor_count_10m),
            )
            if score is not None
        ),
        default=None,
    )
    amount_deviation = max(
        abs(values.get("actor_amount_zscore_24h", 0.0)),
        abs(values.get("counterparty_amount_zscore_24h", 0.0)),
    )
    components = (
        actor_velocity,
        _component_score(values.get("graph_counterparty_fanin", 0.0), manifest.counterparty_fanin),
        _component_score(values.get("graph_actor_fanout", 0.0), manifest.actor_fanout),
        _component_score(amount_deviation, manifest.amount_zscore),
        _component_score(values.get("graph_shared_neighbor_count", 0.0), manifest.shared_neighbors),
        _component_score(values.get("pair_prior_count", 0.0), manifest.repeated_pair_count),
        manifest.threshold_score
        if values.get("dq_degraded_state", 0.0) >= manifest.degraded_state
        else None,
    )
    scores = tuple(score for score in components if score is not None)
    return 1.0 - math.prod(1.0 - score for score in scores)


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


def _score_one_arm(
    *,
    trained: V5TrainedArm,
    catalog: SentinelFeatureCatalog,
    features_matrix: NDArray[np.float64],
    support: tuple[V5ArmSupportRow, ...],
    trust_failures: list[bool],
    novelty_scores: list[float] | None,
) -> V5ArmScore:
    evidence: list[V5ArmRowEvidence] = []
    for index, source_row in enumerate(features_matrix):
        start = time.perf_counter_ns()
        rule_score = _rule_score(source_row, catalog=catalog) if trained.spec.rules else None
        trust_failure = trust_failures[index]
        if trained.defender is None:
            assert rule_score is not None
            action = (
                SentinelAction.DECLINE_HOLD
                if trained.spec.trust and trust_failure
                else _rule_action(rule_score)
            )
            probability = 1.0 if trained.spec.trust and trust_failure else rule_score
            novelty = None
            disagreement = None
        elif trained.spec.arm is V5Arm.FULL_SENTINEL:
            vector = source_row[list(trained.feature_indices)]
            decision = trained.defender.decide(
                vector,
                novelty_score=(novelty_scores[index] if novelty_scores is not None else None),
                trust_failure=trust_failure,
            )
            action = decision.action
            if not trust_failure:
                assert rule_score is not None
                action = _more_severe(action, _rule_action(rule_score))
            probability = decision.ensemble_probability
            novelty = decision.novelty_score
            disagreement = decision.disagreement
        else:
            vector = source_row[list(trained.feature_indices)]
            probability, _raw_disagreement = trained.defender.predict_probability(vector)
            action = _model_action(probability, trained.defender.thresholds)
            novelty = None
            disagreement = None
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        evidence.append(
            V5ArmRowEvidence(
                support=support[index],
                action=action,
                probability=probability,
                rule_score=rule_score,
                trust_routed=bool(trained.spec.trust and trust_failure),
                novelty_score=novelty,
                disagreement=disagreement,
                latency_ms=latency_ms,
                arm_spec_sha256=trained.spec.spec_sha256,
            )
        )
    support_sha256 = _digest([row.model_dump(mode="json") for row in support])
    return V5ArmScore(
        spec=trained.spec,
        support_sha256=support_sha256,
        rows=tuple(evidence),
    )


def score_v5_arm_set(
    *,
    trained: V5TrainedArmSet,
    catalog: SentinelFeatureCatalog,
    features_matrix: NDArray[np.float64],
    support: tuple[V5ArmSupportRow, ...],
    trust_failures: list[bool],
    novelty_scores: list[float] | None = None,
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
    if novelty_scores is not None:
        lengths.add(len(novelty_scores))
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in novelty_scores):
            raise ValueError("novelty scores must be finite values in [0, 1]")
    if lengths != {len(features_matrix)} or not len(features_matrix):
        raise ValueError("evaluation features, support, trust, and novelty must align")
    if len({row.event_id for row in support}) != len(support):
        raise ValueError("evaluation support event IDs must be unique")
    scores = {
        arm.spec.arm: _score_one_arm(
            trained=arm,
            catalog=catalog,
            features_matrix=features_matrix,
            support=support,
            trust_failures=trust_failures,
            novelty_scores=novelty_scores,
        )
        for arm in trained.arms
    }
    return V5ArmScoreSet(by_arm=scores)


__all__ = [
    "V5TrainedArm",
    "V5TrainedArmSet",
    "score_v5_arm_set",
    "train_v5_arm_set",
]
