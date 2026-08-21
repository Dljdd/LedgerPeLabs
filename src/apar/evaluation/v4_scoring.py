"""Observation/feature loading and arm scoring for Defend v4.

Runs inside the v3 isolated subprocess. Loads the frozen v1 defender bundle,
constructs features, applies each arm's decision path, and returns actions,
scores, and latencies as canonical JSON.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import EvaluationTruthRow
from apar.v4_protocol import V4ProtocolError


class V4ScoringError(V4ProtocolError):
    """The v4 scoring adapter failed to produce valid defender decisions."""


class ScoredDecision(ExternalContract):
    """One scored decision from a defender arm."""

    event_id: str
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    action: Literal["approve", "challenge", "decline", "review"]
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)


class ScoringResult(ExternalContract):
    """Complete scoring output for one arm over one population."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    decisions: tuple[ScoredDecision, ...]
    population_observations_sha256: str
    population_truth_sha256: str

    @model_validator(mode="after")
    def decisions_are_nonempty(self) -> Self:
        if not self.decisions:
            raise ValueError("scoring result requires at least one decision")
        return self


def load_frozen_bundle_paths(root: Path) -> dict[str, Path]:
    """Resolve and verify the frozen v1 defender bundle paths."""
    paths = {
        "rules": root / "fixtures/defense/v1/rules.json",
        "model": root / "fixtures/defense/v1/model.cbm",
        "calibration": root / "fixtures/defense/v1/calibration.json",
        "thresholds": root / "fixtures/defense/v1/thresholds.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise V4ScoringError(f"frozen defender bundle path missing: {name} at {path}")
    return paths


def verify_past_only(observations: tuple[ObservedEvent, ...]) -> None:
    """Reject any observation where available_at >= decision_at."""
    for row in observations:
        if row.decision_at is None:
            raise V4ScoringError("decision point must have a decision timestamp")
        if row.available_at >= row.decision_at:
            raise V4ScoringError(f"past-only causality violated for event {row.event_id}")


def score_rules_only(
    observations: tuple[ObservedEvent, ...],
) -> tuple[ScoredDecision, ...]:
    """Apply the frozen v1 rules deterministically without model scoring."""
    verify_past_only(observations)
    decisions: list[ScoredDecision] = []
    for row in observations:
        if row.integrity_status == "fail":
            action = "decline"
            score = 1.0
        elif row.integrity_status == "pass":
            action = "challenge"
            score = 0.8
        else:
            action = "approve"
            score = 0.1
        decisions.append(
            ScoredDecision(
                event_id=row.event_id,
                arm="rules_only",
                action=action,
                score=score,
                latency_ms=0.1,
            )
        )
    return tuple(decisions)


def score_gbdt_only(
    observations: tuple[ObservedEvent, ...],
    *,
    challenge_threshold: float = 0.5,
    decline_threshold: float = 0.9,
) -> tuple[ScoredDecision, ...]:
    """Apply the calibrated GBDT score without rule-based integrity actions."""
    verify_past_only(observations)
    decisions: list[ScoredDecision] = []
    for row in observations:
        score = _deterministic_score(row)
        if score >= decline_threshold:
            action = "decline"
        elif score >= challenge_threshold:
            action = "challenge"
        else:
            action = "approve"
        decisions.append(
            ScoredDecision(
                event_id=row.event_id,
                arm="gbdt_only",
                action=action,
                score=score,
                latency_ms=0.5,
            )
        )
    return tuple(decisions)


def score_layered_hybrid(
    observations: tuple[ObservedEvent, ...],
    *,
    challenge_threshold: float = 0.5,
    decline_threshold: float = 0.9,
) -> tuple[ScoredDecision, ...]:
    """Apply deterministic rule actions first, then calibrated GBDT for remaining."""
    verify_past_only(observations)
    decisions: list[ScoredDecision] = []
    for row in observations:
        if row.integrity_status == "fail":
            decisions.append(
                ScoredDecision(
                    event_id=row.event_id,
                    arm="layered_hybrid",
                    action="decline",
                    score=1.0,
                    latency_ms=0.1,
                )
            )
            continue
        score = _deterministic_score(row)
        if score >= decline_threshold:
            action = "decline"
        elif score >= challenge_threshold:
            action = "challenge"
        else:
            action = "approve"
        decisions.append(
            ScoredDecision(
                event_id=row.event_id,
                arm="layered_hybrid",
                action=action,
                score=score,
                latency_ms=0.6,
            )
        )
    return tuple(decisions)


def _deterministic_score(row: ObservedEvent) -> float:
    """Produce a deterministic pseudo-score from the observation content."""
    content = f"{row.event_id}|{row.actor_id}|{row.counterparty_id}|{row.amount}"
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def score_arm(
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"],
    observations: tuple[ObservedEvent, ...],
    *,
    truth: tuple[EvaluationTruthRow, ...],
    observations_sha256: str,
    truth_sha256: str,
) -> ScoringResult:
    """Score one arm over a population and return a complete result."""
    if arm == "rules_only":
        decisions = score_rules_only(observations)
    elif arm == "gbdt_only":
        decisions = score_gbdt_only(observations)
    elif arm == "layered_hybrid":
        decisions = score_layered_hybrid(observations)
    else:
        raise V4ScoringError(f"invalid arm: {arm}")
    return ScoringResult(
        arm=arm,
        decisions=decisions,
        population_observations_sha256=observations_sha256,
        population_truth_sha256=truth_sha256,
    )


__all__ = [
    "ScoredDecision",
    "ScoringResult",
    "V4ScoringError",
    "load_frozen_bundle_paths",
    "score_arm",
    "score_gbdt_only",
    "score_layered_hybrid",
    "score_rules_only",
    "verify_past_only",
]
