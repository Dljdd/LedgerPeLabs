"""Calibrated Sentinel ensemble and four-action policy for Defend v5."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from catboost import CatBoostClassifier
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression


class SentinelAction(StrEnum):
    APPROVE = "approve"
    CHALLENGE = "challenge"
    REVIEW_HOLD = "review_hold"
    DECLINE_HOLD = "decline_hold"

    @property
    def severity(self) -> int:
        return {
            SentinelAction.APPROVE: 0,
            SentinelAction.CHALLENGE: 1,
            SentinelAction.REVIEW_HOLD: 2,
            SentinelAction.DECLINE_HOLD: 3,
        }[self]


class SentinelDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SentinelAction
    ensemble_probability: float
    disagreement: float
    novelty_score: float
    trust_failure: bool


class SentinelThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_threshold: float = 0.5
    review_threshold: float = 0.7
    decline_threshold: float = 0.9
    disagreement_review_threshold: float = 0.15
    novelty_challenge_threshold: float = 0.7
    novelty_review_threshold: float = 0.9


class SentinelModelManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    catboost_seeds: tuple[int, ...]
    feature_count: int
    thresholds: SentinelThresholds


def _fit_calibrator(
    raw_scores: np.ndarray, labels: np.ndarray
) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores, labels)
    return calibrator


def train_sentinel_defender(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    x_threshold: np.ndarray,
    y_threshold: np.ndarray,
    catboost_seeds: tuple[int, ...],
    bootstrap_seed: int,
) -> SentinelDefender:
    """Train a three-seed calibrated CatBoost ensemble with novelty router."""
    calibration_labels = set(y_calibration.tolist())
    if calibration_labels != {0, 1}:
        raise ValueError(
            f"one-class calibration partition: labels present = {sorted(calibration_labels)}"
        )
    threshold_labels = set(y_threshold.tolist())
    if threshold_labels != {0, 1}:
        raise ValueError(
            f"one-class threshold partition: labels present = {sorted(threshold_labels)}"
        )
    members: list[CatBoostClassifier] = []
    calibrators: list[IsotonicRegression] = []
    for seed in catboost_seeds:
        model = CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
        raw_cal = model.predict_proba(x_calibration)[:, 1]
        calibrator = _fit_calibrator(raw_cal, y_calibration)
        members.append(model)
        calibrators.append(calibrator)

    benign_mask = y_train == 0
    benign_features = x_train[benign_mask]
    iso_forest = IsolationForest(random_state=bootstrap_seed, contamination=0.05)
    iso_forest.fit(benign_features)

    thresholds = _select_thresholds(
        members=members,
        calibrators=calibrators,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        iso_forest=iso_forest,
    )

    return SentinelDefender(
        model_members=members,
        calibrators=calibrators,
        iso_forest=iso_forest,
        thresholds=thresholds,
        manifest=SentinelModelManifest(
            catboost_seeds=catboost_seeds,
            feature_count=x_train.shape[1],
            thresholds=thresholds,
        ),
    )


def _ensemble_probability(
    features: np.ndarray,
    members: list[CatBoostClassifier],
    calibrators: list[IsotonicRegression],
) -> tuple[float, float]:
    calibrated_scores = []
    for model, calibrator in zip(members, calibrators, strict=True):
        raw = model.predict_proba(features.reshape(1, -1))[:, 1][0]
        calibrated = float(calibrator.predict([raw])[0])
        calibrated_scores.append(max(0.0, min(1.0, calibrated)))
    mean_prob = float(np.mean(calibrated_scores))
    disagreement = float(np.std(calibrated_scores))
    return mean_prob, disagreement


def _novelty_score(iso_forest: IsolationForest, features: np.ndarray) -> float:
    raw = iso_forest.decision_function(features.reshape(1, -1))[0]
    return float(max(0.0, min(1.0, 0.5 - raw)))


def _select_thresholds(
    *,
    members: list[CatBoostClassifier],
    calibrators: list[IsotonicRegression],
    x_threshold: np.ndarray,
    y_threshold: np.ndarray,
    iso_forest: IsolationForest,
) -> SentinelThresholds:
    """Select operating thresholds on the threshold partition only."""
    probs = []
    for i in range(len(x_threshold)):
        p, _ = _ensemble_probability(x_threshold[i], members, calibrators)
        probs.append(p)
    probs_arr = np.array(probs)
    fraud_mask = y_threshold == 1
    benign_mask = y_threshold == 0

    fraud_probs = probs_arr[fraud_mask] if fraud_mask.any() else np.array([0.9])
    benign_probs = probs_arr[benign_mask] if benign_mask.any() else np.array([0.1])

    challenge_t = float(np.percentile(benign_probs, 95))
    decline_t = float(max(0.8, np.percentile(fraud_probs, 80)))
    review_t = float((challenge_t + decline_t) / 2)

    return SentinelThresholds(
        challenge_threshold=max(0.1, min(0.8, challenge_t)),
        review_threshold=max(0.3, min(0.9, review_t)),
        decline_threshold=max(0.5, min(1.0, decline_t)),
    )


class SentinelDefender(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_members: list[CatBoostClassifier]
    calibrators: list[IsotonicRegression]
    iso_forest: IsolationForest
    thresholds: SentinelThresholds
    manifest: SentinelModelManifest

    def decide(
        self,
        features: np.ndarray,
        *,
        novelty_score: float | None = None,
        trust_failure: bool = False,
    ) -> SentinelDecision:
        if trust_failure:
            return SentinelDecision(
                action=SentinelAction.DECLINE_HOLD,
                ensemble_probability=1.0,
                disagreement=0.0,
                novelty_score=0.0,
                trust_failure=True,
            )

        prob, disagreement = _ensemble_probability(
            features, self.model_members, self.calibrators
        )
        novelty = (
            novelty_score
            if novelty_score is not None
            else _novelty_score(self.iso_forest, features)
        )
        t = self.thresholds

        if prob >= t.decline_threshold and disagreement < t.disagreement_review_threshold:
            action = SentinelAction.DECLINE_HOLD
        elif prob >= t.review_threshold or (
            disagreement >= t.disagreement_review_threshold and prob >= t.challenge_threshold
        ):
            action = SentinelAction.REVIEW_HOLD
        elif prob >= t.challenge_threshold or (
            novelty >= t.novelty_challenge_threshold and prob >= 0.3
        ):
            action = SentinelAction.CHALLENGE
        elif novelty >= t.novelty_review_threshold:
            action = SentinelAction.REVIEW_HOLD
        else:
            action = SentinelAction.APPROVE

        return SentinelDecision(
            action=action,
            ensemble_probability=prob,
            disagreement=disagreement,
            novelty_score=novelty,
            trust_failure=False,
        )

    def decide_batch(self, features_matrix: np.ndarray) -> list[SentinelDecision]:
        return [self.decide(row) for row in features_matrix]
