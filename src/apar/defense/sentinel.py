"""Calibrated Sentinel ensemble and four-action policy for Defend v5."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

FULL_SENTINEL_DISAGREEMENT_REVIEW_THRESHOLD = 0.15
FULL_SENTINEL_NOVELTY_CHALLENGE_THRESHOLD = 0.7
FULL_SENTINEL_NOVELTY_REVIEW_THRESHOLD = 0.9


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
    disagreement_review_threshold: float = FULL_SENTINEL_DISAGREEMENT_REVIEW_THRESHOLD
    novelty_challenge_threshold: float = FULL_SENTINEL_NOVELTY_CHALLENGE_THRESHOLD
    novelty_review_threshold: float = FULL_SENTINEL_NOVELTY_REVIEW_THRESHOLD

    @model_validator(mode="after")
    def routing_thresholds_are_frozen(self) -> SentinelThresholds:
        if (
            self.disagreement_review_threshold
            != FULL_SENTINEL_DISAGREEMENT_REVIEW_THRESHOLD
            or self.novelty_challenge_threshold
            != FULL_SENTINEL_NOVELTY_CHALLENGE_THRESHOLD
            or self.novelty_review_threshold
            != FULL_SENTINEL_NOVELTY_REVIEW_THRESHOLD
        ):
            raise ValueError("full sentinel routing thresholds are fixed")
        return self


class SentinelModelManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    catboost_seeds: tuple[int, ...]
    feature_count: int
    thresholds: SentinelThresholds


def route_sentinel_components(
    *,
    probability: float,
    disagreement: float,
    novelty: float,
    thresholds: SentinelThresholds,
) -> tuple[SentinelAction, bool, bool]:
    """Apply the frozen monotonic model, disagreement, and novelty policy."""
    values = (probability, disagreement, novelty)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("sentinel component inputs must be finite values in [0, 1]")
    if (
        probability >= thresholds.decline_threshold
        and disagreement < thresholds.disagreement_review_threshold
    ):
        return SentinelAction.DECLINE_HOLD, False, False
    if probability >= thresholds.review_threshold:
        return SentinelAction.REVIEW_HOLD, False, False
    if (
        disagreement >= thresholds.disagreement_review_threshold
        and probability >= thresholds.challenge_threshold
    ):
        return SentinelAction.REVIEW_HOLD, True, False
    if (
        novelty >= thresholds.novelty_review_threshold
        and probability >= 0.3
    ):
        return SentinelAction.REVIEW_HOLD, False, True
    if probability >= thresholds.challenge_threshold:
        return SentinelAction.CHALLENGE, False, False
    if novelty >= thresholds.novelty_challenge_threshold:
        return SentinelAction.CHALLENGE, False, True
    return SentinelAction.APPROVE, False, False


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
    enable_novelty: bool = True,
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

    iso_forest: IsolationForest | None = None
    if enable_novelty:
        benign_mask = y_train == 0
        benign_features = x_train[benign_mask]
        iso_forest = IsolationForest(random_state=bootstrap_seed, contamination=0.05)
        iso_forest.fit(benign_features)

    thresholds = _select_thresholds(
        members=members,
        calibrators=calibrators,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
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
    _raw, _calibrated_scores, mean_prob, disagreement = _ensemble_trace(
        features, members, calibrators
    )
    return mean_prob, disagreement


def _ensemble_trace(
    features: np.ndarray,
    members: list[CatBoostClassifier],
    calibrators: list[IsotonicRegression],
) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
    raw_scores: list[float] = []
    calibrated_scores = []
    for model, calibrator in zip(members, calibrators, strict=True):
        raw = float(model.predict_proba(features.reshape(1, -1))[:, 1][0])
        raw_scores.append(raw)
        calibrated = float(calibrator.predict([raw])[0])
        calibrated_scores.append(max(0.0, min(1.0, calibrated)))
    mean_prob = float(np.mean(calibrated_scores))
    disagreement = float(np.std(calibrated_scores))
    return tuple(raw_scores), tuple(calibrated_scores), mean_prob, disagreement


def _novelty_score(iso_forest: IsolationForest, features: np.ndarray) -> float:
    _raw, bounded = _novelty_trace(iso_forest, features)
    return bounded


def _novelty_trace(
    iso_forest: IsolationForest, features: np.ndarray
) -> tuple[float, float]:
    raw = float(iso_forest.decision_function(features.reshape(1, -1))[0])
    return raw, float(max(0.0, min(1.0, 0.5 - raw)))


def _select_thresholds(
    *,
    members: list[CatBoostClassifier],
    calibrators: list[IsotonicRegression],
    x_threshold: np.ndarray,
    y_threshold: np.ndarray,
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
    iso_forest: IsolationForest | None
    thresholds: SentinelThresholds
    manifest: SentinelModelManifest

    def predict_probability(self, features: np.ndarray) -> tuple[float, float]:
        """Return calibrated ensemble probability and member disagreement only."""
        return _ensemble_probability(features, self.model_members, self.calibrators)

    def predict_member_scores(
        self, features: np.ndarray
    ) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
        """Return every raw/calibrated member score plus aggregate evidence."""
        return _ensemble_trace(features, self.model_members, self.calibrators)

    def predict_novelty(self, features: np.ndarray) -> tuple[float, float]:
        """Return raw IsolationForest decision function and bounded novelty."""
        if self.iso_forest is None:
            raise ValueError("novelty model is disabled for this defender")
        return _novelty_trace(self.iso_forest, features)

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
            else (
                _novelty_score(self.iso_forest, features)
                if self.iso_forest is not None
                else 0.0
            )
        )
        action, _disagreement_routed, _novelty_routed = route_sentinel_components(
            probability=prob,
            disagreement=disagreement,
            novelty=novelty,
            thresholds=self.thresholds,
        )

        return SentinelDecision(
            action=action,
            ensemble_probability=prob,
            disagreement=disagreement,
            novelty_score=novelty,
            trust_failure=False,
        )

    def decide_batch(
        self,
        features_matrix: np.ndarray,
        *,
        trust_failures: list[bool] | None = None,
    ) -> list[SentinelDecision]:
        if trust_failures is not None and len(trust_failures) != len(features_matrix):
            raise ValueError(
                f"trust_failures length {len(trust_failures)} does not match "
                f"features_matrix length {len(features_matrix)}"
            )
        return [
            self.decide(
                row,
                trust_failure=trust_failures[i] if trust_failures else False,
            )
            for i, row in enumerate(features_matrix)
        ]
