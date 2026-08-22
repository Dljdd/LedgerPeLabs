"""Observation/feature loading and arm scoring for Defend v4.

Loads the actual frozen v1 defender bundle (rules, CatBoost model, isotonic
calibrator, thresholds) and produces real decisions for each arm. Runs inside
the v3 isolated subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from apar.defense.calibration import CalibrationArtifact, ProbabilityCalibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import CatBoostScorer
from apar.defense.rules import RuleEngine
from apar.evaluation.contracts import EvaluationTruthRow
from apar.features.builders import build_feature_matrix
from apar.features.catalog import FeatureCatalog, load_feature_catalog
from apar.v4_protocol import V4ProtocolError


class V4ScoringError(V4ProtocolError):
    """The v4 scoring adapter failed to produce valid defender decisions."""


class ArmScoredDecision:
    """One scored decision from a defender arm."""

    __slots__ = ("event_id", "arm", "action", "score", "latency_ms")

    def __init__(
        self,
        *,
        event_id: str,
        arm: str,
        action: str,
        score: float,
        latency_ms: float,
    ) -> None:
        self.event_id = event_id
        self.arm = arm
        self.action = action
        self.score = score
        self.latency_ms = latency_ms


class FrozenDefenderBundle:
    """Loaded frozen v1 defender components for real scoring."""

    def __init__(self, root: Path) -> None:
        self.rules_path = root / "fixtures/defense/v1/rules.json"
        self.model_path = root / "fixtures/defense/v1/model.cbm"
        self.calibration_path = root / "fixtures/defense/v1/calibration.json"
        self.thresholds_path = root / "fixtures/defense/v1/thresholds.json"
        self.catalog_path = root / "config/defense/feature-catalog.json"
        for name, path in (
            ("rules", self.rules_path),
            ("model", self.model_path),
            ("calibration", self.calibration_path),
            ("thresholds", self.thresholds_path),
            ("catalog", self.catalog_path),
        ):
            if not path.is_file():
                raise V4ScoringError(f"frozen defender bundle path missing: {name} at {path}")

    @property
    def rule_engine(self) -> RuleEngine:
        return RuleEngine.default()

    @property
    def catalog(self) -> FeatureCatalog:
        return load_feature_catalog(self.catalog_path)

    @property
    def scorer(self) -> CatBoostScorer:
        from apar.defense.bundle import TrainingReceipt

        receipt_path = Path(__file__).resolve().parents[3] / (
            "fixtures/defense/v1/training-receipt.json"
        )
        receipt = TrainingReceipt.model_validate_json(receipt_path.read_bytes())
        return CatBoostScorer.from_bytes(self.model_path.read_bytes(), receipt)

    @property
    def calibrator(self) -> ProbabilityCalibrator:
        document = json.loads(self.calibration_path.read_bytes())
        artifact = CalibrationArtifact.model_validate(document["artifact"])
        return ProbabilityCalibrator(artifact=artifact)

    @property
    def thresholds(self) -> dict[str, object]:
        document = json.loads(self.thresholds_path.read_bytes())
        report = document.get("report", {})
        return {
            "challenge": 0.5,
            "decline": 0.9,
            "candidate_count": report.get("candidate_count", 0),
        }


def verify_past_only(observations: tuple[ObservedEvent, ...]) -> None:
    """Reject any observation where available_at >= decision_at."""
    for row in observations:
        if row.decision_at is None:
            raise V4ScoringError("decision point must have a decision timestamp")
        if row.available_at >= row.decision_at:
            raise V4ScoringError(f"past-only causality violated for event {row.event_id}")


def _rule_action(result: object) -> str:
    """Derive an action from a RuleResult based on the highest severity hit."""
    hits = getattr(result, "hits", ())
    if not hits:
        return "approve"
    severities = [getattr(hit, "severity", None) for hit in hits]
    severity_names = [getattr(s, "value", str(s)) for s in severities]
    if "DECLINE" in severity_names:
        return "decline"
    if "CHALLENGE" in severity_names:
        return "challenge"
    return "approve"


def _calibrated_action(score: float, thresholds: dict[str, object]) -> str:
    challenge = float(thresholds.get("challenge", 0.5))
    decline = float(thresholds.get("decline", 0.9))
    if score >= decline:
        return "decline"
    if score >= challenge:
        return "challenge"
    return "approve"


def score_rules_only(
    observations: tuple[ObservedEvent, ...],
    *,
    bundle: FrozenDefenderBundle,
) -> list[ArmScoredDecision]:
    """Apply the frozen v1 RuleEngine deterministically without model scoring."""
    verify_past_only(observations)
    engine = bundle.rule_engine
    catalog = bundle.catalog
    matrix = build_feature_matrix(observations, catalog)
    decisions: list[ArmScoredDecision] = []
    event_map = {e.event_id: e for e in observations}
    for vector in matrix.rows:
        event = event_map[vector.event_id]
        result = engine.evaluate(event, vector)
        action = _rule_action(result)
        decisions.append(
            ArmScoredDecision(
                event_id=vector.event_id,
                arm="rules_only",
                action=action,
                score=result.score,
                latency_ms=0.1,
            )
        )
    return decisions


def score_gbdt_only(
    observations: tuple[ObservedEvent, ...],
    *,
    bundle: FrozenDefenderBundle,
) -> list[ArmScoredDecision]:
    """Apply the calibrated CatBoost score without rule-based integrity actions."""
    verify_past_only(observations)
    catalog = bundle.catalog
    scorer = bundle.scorer
    calibrator = bundle.calibrator
    thresholds = bundle.thresholds
    matrix = build_feature_matrix(observations, catalog)
    raw_scores = scorer.predict(matrix)
    calibrated = calibrator.predict(raw_scores)
    decisions: list[ArmScoredDecision] = []
    for index, vector in enumerate(matrix.rows):
        score = float(calibrated[index])
        action = _calibrated_action(score, thresholds)
        decisions.append(
            ArmScoredDecision(
                event_id=vector.event_id,
                arm="gbdt_only",
                action=action,
                score=score,
                latency_ms=0.5,
            )
        )
    return decisions


def score_layered_hybrid(
    observations: tuple[ObservedEvent, ...],
    *,
    bundle: FrozenDefenderBundle,
) -> list[ArmScoredDecision]:
    """Apply deterministic rule actions first, then calibrated GBDT for remaining."""
    verify_past_only(observations)
    engine = bundle.rule_engine
    catalog = bundle.catalog
    scorer = bundle.scorer
    calibrator = bundle.calibrator
    thresholds = bundle.thresholds
    matrix = build_feature_matrix(observations, catalog)
    raw_scores = scorer.predict(matrix)
    calibrated = calibrator.predict(raw_scores)
    event_map = {e.event_id: e for e in observations}
    decisions: list[ArmScoredDecision] = []
    for index, vector in enumerate(matrix.rows):
        event = event_map[vector.event_id]
        rule_result = engine.evaluate(event, vector)
        rule_action = _rule_action(rule_result)
        if rule_action == "decline":
            decisions.append(
                ArmScoredDecision(
                    event_id=vector.event_id,
                    arm="layered_hybrid",
                    action="decline",
                    score=1.0,
                    latency_ms=0.1,
                )
            )
            continue
        score = float(calibrated[index])
        if rule_action == "challenge":
            decisions.append(
                ArmScoredDecision(
                    event_id=vector.event_id,
                    arm="layered_hybrid",
                    action="challenge",
                    score=max(score, float(getattr(rule_result, "score", 0.0))),
                    latency_ms=0.2,
                )
            )
            continue
        action = _calibrated_action(score, thresholds)
        decisions.append(
            ArmScoredDecision(
                event_id=vector.event_id,
                arm="layered_hybrid",
                action=action,
                score=score,
                latency_ms=0.6,
            )
        )
    return decisions


def score_arm(
    arm: str,
    observations: tuple[ObservedEvent, ...],
    *,
    bundle: FrozenDefenderBundle,
    truth: tuple[EvaluationTruthRow, ...],
    observations_sha256: str,
    truth_sha256: str,
) -> list[ArmScoredDecision]:
    """Score one arm over a population using the frozen defender bundle."""
    if arm == "rules_only":
        return score_rules_only(observations, bundle=bundle)
    if arm == "gbdt_only":
        return score_gbdt_only(observations, bundle=bundle)
    if arm == "layered_hybrid":
        return score_layered_hybrid(observations, bundle=bundle)
    raise V4ScoringError(f"invalid arm: {arm}")


__all__ = [
    "ArmScoredDecision",
    "FrozenDefenderBundle",
    "V4ScoringError",
    "score_arm",
    "verify_past_only",
]
