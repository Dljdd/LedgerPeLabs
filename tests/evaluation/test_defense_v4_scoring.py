"""Scoring adapter tests for Defend v4."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.v4_scoring import (
    V4ScoringError,
    score_arm,
    score_gbdt_only,
    score_layered_hybrid,
    score_rules_only,
    verify_past_only,
)


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


def test_rules_only_declines_integrity_failure() -> None:
    decisions = score_rules_only((_observation("row-1", integrity_status="fail"),))
    assert decisions[0].action == "decline"
    assert decisions[0].score == 1.0


def test_rules_only_challenges_agentic_pass() -> None:
    decisions = score_rules_only((_observation("row-1", integrity_status="pass"),))
    assert decisions[0].action == "challenge"


def test_rules_only_approves_non_agentic() -> None:
    decisions = score_rules_only((_observation("row-1"),))
    assert decisions[0].action == "approve"


def test_gbdt_only_produces_deterministic_scores() -> None:
    obs = _observation("row-1")
    first = score_gbdt_only((obs,))
    second = score_gbdt_only((obs,))
    assert first[0].score == second[0].score
    assert 0.0 <= first[0].score <= 1.0


def test_layered_hybrid_rule_precedence_overrides_score() -> None:
    decisions = score_layered_hybrid((_observation("row-1", integrity_status="fail"),))
    assert decisions[0].action == "decline"
    assert decisions[0].score == 1.0


def test_layered_hybrid_uses_score_for_remaining() -> None:
    decisions = score_layered_hybrid((_observation("row-1"),))
    assert decisions[0].arm == "layered_hybrid"
    assert 0.0 <= decisions[0].score <= 1.0


def test_past_only_violation_rejected() -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    bad = _observation("row-1").model_copy(update={"available_at": now + timedelta(hours=1)})
    with pytest.raises(V4ScoringError, match="past-only causality"):
        verify_past_only((bad,))


def test_score_arm_returns_complete_result() -> None:
    observations = (_observation("a"), _observation("b"))
    result = score_arm(
        "rules_only",
        observations,
        truth=(),
        observations_sha256="a" * 64,
        truth_sha256="b" * 64,
    )
    assert result.arm == "rules_only"
    assert len(result.decisions) == 2


def test_invalid_arm_rejected() -> None:
    with pytest.raises(V4ScoringError, match="invalid arm"):
        score_arm("unknown", (), truth=(), observations_sha256="a" * 64, truth_sha256="b" * 64)
