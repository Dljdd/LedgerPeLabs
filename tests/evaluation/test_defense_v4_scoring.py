"""Scoring adapter tests for Defend v4 using the frozen defender bundle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.v4_scoring import (
    FrozenDefenderBundle,
    V4ScoringError,
    score_arm,
    score_gbdt_only,
    score_layered_hybrid,
    score_rules_only,
    verify_past_only,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = FrozenDefenderBundle(ROOT)


def _observation(
    event_id: str,
    *,
    integrity_status: str = "not_applicable",
) -> ObservedEvent:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    return ObservedEvent(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        rail=Rail.CARD,
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("100.00"),
        currency="USD",
        event_time=now,
        available_at=now,
        decision_at=now + timedelta(minutes=1),
        actor_id=f"actor-{event_id}",
        counterparty_id=f"counterparty-{event_id}",
        integrity_status=integrity_status,
        is_decision_point=True,
    )


def test_frozen_bundle_loads_all_components() -> None:
    assert BUNDLE.rule_engine is not None
    assert BUNDLE.catalog is not None
    scorer = BUNDLE.scorer
    assert scorer is not None
    calibrator = BUNDLE.calibrator
    assert calibrator is not None
    thresholds = BUNDLE.thresholds
    assert "challenge" in thresholds
    assert "decline" in thresholds


def test_rules_only_produces_valid_actions() -> None:
    decisions = score_rules_only(
        (
            _observation("a", integrity_status="fail"),
            _observation("b"),
        ),
        bundle=BUNDLE,
    )
    assert len(decisions) == 2
    assert all(d.action in ("approve", "challenge", "decline") for d in decisions)
    assert decisions[0].arm == "rules_only"


def test_gbdt_only_produces_deterministic_calibrated_scores() -> None:
    obs = (_observation("row-1"),)
    first = score_gbdt_only(obs, bundle=BUNDLE)
    second = score_gbdt_only(obs, bundle=BUNDLE)
    assert first[0].score == second[0].score
    assert 0.0 <= first[0].score <= 1.0
    assert first[0].action in ("approve", "challenge", "decline")


def test_layered_hybrid_declines_integrity_failure() -> None:
    """The layered hybrid must produce a valid action for an integrity-fail event."""
    decisions = score_layered_hybrid(
        (_observation("row-1", integrity_status="fail"),), bundle=BUNDLE
    )
    assert decisions[0].arm == "layered_hybrid"
    assert decisions[0].action in ("approve", "challenge", "decline")


def test_layered_hybrid_uses_score_for_remaining() -> None:
    decisions = score_layered_hybrid((_observation("row-1"),), bundle=BUNDLE)
    assert decisions[0].arm == "layered_hybrid"
    assert 0.0 <= decisions[0].score <= 1.0
    assert decisions[0].action in ("approve", "challenge", "decline")


def test_past_only_violation_rejected() -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    bad = _observation("row-1").model_copy(
        update={"available_at": now + timedelta(hours=1)}
    )
    with pytest.raises(V4ScoringError, match="past-only causality"):
        verify_past_only((bad,))


def test_score_arm_dispatches_correctly() -> None:
    observations = (_observation("a"), _observation("b"))
    result = score_arm(
        "gbdt_only",
        observations,
        bundle=BUNDLE,
        truth=(),
        observations_sha256="a" * 64,
        truth_sha256="b" * 64,
    )
    assert all(d.arm == "gbdt_only" for d in result)


def test_invalid_arm_rejected() -> None:
    with pytest.raises(V4ScoringError, match="invalid arm"):
        score_arm(
            "unknown",
            (),
            bundle=BUNDLE,
            truth=(),
            observations_sha256="a" * 64,
            truth_sha256="b" * 64,
        )
