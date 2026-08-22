"""Strict causal feature pipeline regression tests."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from apar.evaluation.v5_population import V5DecisionRow
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features


def _row(event_id: str, at: datetime, amount: str = "100.00") -> V5DecisionRow:
    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id="campaign-x",
        family="card_testing_cnp",
        actor_id=f"actor-{event_id}",
        counterparty_id=f"cp-{event_id}",
        amount=Decimal(amount),
        decision_at=at,
        is_fraud=True,
        lifecycle_state="probe",
        source_command_id=f"cmd-{event_id}",
        source_event_id=event_id,
        predictive_features={"amount": float(amount), "rail_card": 1.0},
    )


BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestCausalFeatures:
    def test_future_insertion_does_not_change_earlier_vector(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        original = [_row("a", BASE)]
        with_future = original + [_row("b", BASE + timedelta(hours=1))]
        r1 = build_sentinel_features(original, catalog=catalog)
        r2 = build_sentinel_features(with_future, catalog=catalog)
        assert r1.matrix[0] == r2.matrix[0], "future event changed earlier vector"

    def test_equal_time_peer_insertion_does_not_change_vector(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        original = [_row("a", BASE)]
        with_peer = [_row("a", BASE), _row("b", BASE)]
        r1 = build_sentinel_features(original, catalog=catalog)
        r2 = build_sentinel_features(with_peer, catalog=catalog)
        assert r1.matrix[0] == r2.matrix[0], "equal-time peer changed vector"

    def test_identity_renaming_preserves_numeric_features(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        rows = [_row("a", BASE), _row("b", BASE + timedelta(minutes=5))]
        renamed = [
            r.model_copy(update={
                "actor_id": f"renamed-{r.actor_id}",
                "counterparty_id": f"renamed-{r.counterparty_id}",
                "event_id": f"renamed-{r.event_id}",
            })
            for r in rows
        ]
        r1 = build_sentinel_features(rows, catalog=catalog)
        r2 = build_sentinel_features(renamed, catalog=catalog)
        assert r1.matrix == r2.matrix, "identity rename changed numeric features"

    def test_non_finite_feature_fails(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        bad = _row("a", BASE).model_copy(
            update={"predictive_features": {"amount": float("nan")}}
        )
        with pytest.raises(ValueError):
            build_sentinel_features([bad], catalog=catalog)

    def test_no_forbidden_fields_in_catalog(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        forbidden = {"family", "campaign_id", "is_fraud", "seed", "split"}
        assert not (forbidden & set(catalog.feature_names))
