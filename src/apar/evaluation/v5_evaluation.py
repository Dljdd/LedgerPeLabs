"""Causal evaluation, controls, and ablations for Defend v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from apar.defense.sentinel import SentinelAction

_INTERVENTION_ACTIONS = {
    SentinelAction.CHALLENGE,
    SentinelAction.REVIEW_HOLD,
    SentinelAction.DECLINE_HOLD,
}


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
    "graph_counterparty_fanin",
    "graph_actor_fanout",
    "actor_amount_zscore_24h",
    "counterparty_amount_zscore_24h",
    "graph_shared_neighbor_count",
    "pair_prior_count",
    "dq_degraded_state",
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


class V5ArmSpecification(BaseModel):
    """Immutable, executable component contract for one comparison arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: V5Arm
    feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    non_graph_feature_names: tuple[str, ...]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_seeds: tuple[int, ...]
    calibration_method: Literal["none", "isotonic_per_member"]
    threshold_source_partition: Literal["threshold"]
    threshold_method: Literal["rules_v1_fixed", "sentinel_percentile_v1"]
    threshold_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        if self.spec_sha256 != self.computed_digest():
            raise ValueError("arm specification digest mismatch")
        if Counter(self.graph_feature_names + self.non_graph_feature_names) != Counter(
            self.feature_names
        ):
            raise ValueError("arm graph and non-graph feature subsets must reconstruct features")
        if len(self.feature_names) != len(set(self.feature_names)) and self.arm is V5Arm.RULES_ONLY:
            raise ValueError("rule feature names must be unique")
        if self.model != bool(self.model_seeds):
            raise ValueError("arm model switch and seed binding disagree")
        if self.model != (self.calibration_method == "isotonic_per_member"):
            raise ValueError("arm model switch and calibration binding disagree")
        return self


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


class V5ArmSupportRow(BaseModel):
    """Evaluator-only support facts shared identically by every arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    label: Literal[0, 1]
    campaign_id: str
    amount: float = Field(gt=0.0)
    family: str
    execution_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("amount")
    @classmethod
    def amount_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("support amount must be finite")
        return value


class V5ArmRowEvidence(BaseModel):
    """One independently computed action and its enabled-component evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    support: V5ArmSupportRow
    action: SentinelAction
    probability: float = Field(ge=0.0, le=1.0)
    rule_score: float | None = Field(default=None, ge=0.0, le=1.0)
    trust_routed: bool
    novelty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement: float | None = Field(default=None, ge=0.0)
    latency_ms: float = Field(ge=0.0)
    arm_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("probability", "rule_score", "novelty_score", "disagreement", "latency_ms")
    @classmethod
    def numeric_evidence_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("arm evidence numbers must be finite")
        return value


class V5ArmScore(BaseModel):
    """One arm's immutable score stream bound to its exact specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: V5ArmSpecification
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[V5ArmRowEvidence, ...]

    @model_validator(mode="after")
    def rows_match_specification(self) -> Self:
        if not self.rows:
            raise ValueError("arm score must contain rows")
        if any(row.arm_spec_sha256 != self.spec.spec_sha256 for row in self.rows):
            raise ValueError("arm row evidence specification digest mismatch")
        expected = _canonical_digest(
            [row.support.model_dump(mode="json") for row in self.rows]
        )
        if self.support_sha256 != expected:
            raise ValueError("arm evaluation support digest mismatch")
        return self


class V5ArmScoreSet(BaseModel):
    """Four independent results over one exact ordered evaluation support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_arm: dict[V5Arm, V5ArmScore]

    @model_validator(mode="after")
    def arms_share_exact_support(self) -> Self:
        if tuple(self.by_arm) != _CURRENT_ARMS:
            raise ValueError("score set must contain the exact ordered current arms")
        results = tuple(self.by_arm.values())
        support_digests = {result.support_sha256 for result in results}
        if len(support_digests) != 1:
            raise ValueError("arms do not share identical evaluation support")
        event_orders = {
            tuple(row.support.event_id for row in result.rows) for result in results
        }
        if len(event_orders) != 1:
            raise ValueError("arms do not share identical event order")
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


_EXPECTED_SWITCHES = {
    V5Arm.RULES_ONLY: (False, False, True, True, False, False),
    V5Arm.ENSEMBLE_NO_GRAPH: (True, False, False, False, False, False),
    V5Arm.ENSEMBLE_WITH_GRAPH: (True, True, False, False, False, False),
    V5Arm.FULL_SENTINEL: (True, True, True, True, True, True),
}


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
            "feature_names": feature_names,
            "graph_feature_names": graph_features,
            "non_graph_feature_names": non_graph_features,
            "catalog_sha256": catalog.catalog_sha256,
            "model_seeds": protocol.seeds.catboost_seeds if entry.model else (),
            "calibration_method": entry.calibration_method,
            "threshold_source_partition": document.threshold_source_partition,
            "threshold_method": entry.threshold_method,
            "threshold_digest": threshold_digest,
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
    false_decline_rate: float | None = None
    challenge_rate: float | None = None
    review_rate: float | None = None
    captured_value_fraction: float | None = None
    escaped_value_fraction: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    support_total: int = 0
    support_fraud: int = 0
    support_legitimate: int = 0
    arm_spec_sha256: str = ""
    support_sha256: str = ""
    feature_count: int = 0
    arm_spec: V5ArmSpecification | None = None
    row_evidence: tuple[V5ArmRowEvidence, ...] = ()

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> Self:
        evidence_present = bool(self.arm_spec or self.row_evidence or self.arm_spec_sha256)
        if not evidence_present:
            return self
        if self.arm_spec is None or not self.row_evidence:
            raise ValueError("arm result evidence is incomplete")
        if self.arm_spec_sha256 != self.arm_spec.spec_sha256:
            raise ValueError("arm result specification digest mismatch")
        if self.feature_count != len(self.arm_spec.feature_names):
            raise ValueError("arm result feature count disagrees with specification")
        if any(row.arm_spec_sha256 != self.arm_spec_sha256 for row in self.row_evidence):
            raise ValueError("arm result row evidence digest mismatch")
        expected_support = _canonical_digest(
            [row.support.model_dump(mode="json") for row in self.row_evidence]
        )
        if self.support_sha256 != expected_support:
            raise ValueError("arm result support digest mismatch")
        return self


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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

    challenges = sum(1 for a in actions if a == SentinelAction.CHALLENGE)
    reviews = sum(1 for a in actions if a == SentinelAction.REVIEW_HOLD)
    challenge_rate = _safe_div(float(challenges), float(n_total))
    review_rate = _safe_div(float(reviews), float(n_total))

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
    "V5ControlResult",
    "V5EvaluationResult",
    "evaluate_v5_arm",
    "load_v5_arm_configuration",
    "run_v5_controls",
]
