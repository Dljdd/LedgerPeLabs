"""Window boundary and causal semantics regression tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from apar.evaluation.v5_population import V5DecisionRow
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features


def _row(event_id: str, at: datetime, actor: str = "actor-1", cp: str = "cp-1") -> V5DecisionRow:
    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id="campaign-x",
        family="card_testing_cnp",
        actor_id=actor,
        counterparty_id=cp,
        amount=Decimal("100.00"),
        decision_at=at,
        is_fraud=True,
        lifecycle_state="probe",
        source_command_id=f"cmd-{event_id}",
        source_event_id=event_id,
        predictive_features={"amount": 100.0, "rail_card": 1.0},
    )


BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestWindowBoundaries:
    """Test exact window boundaries for actor_count_24h."""

    @pytest.mark.parametrize("offset_minutes,expected_count", [
        (24 * 60 - 1, 1),   # just below 24h: included
        (24 * 60, 1),       # exactly 24h: included (strictly < decision_at, within window)
        (24 * 60 + 1, 0),   # just above 24h: excluded
    ])
    def test_actor_count_24h_boundaries(self, offset_minutes: int, expected_count: int) -> None:
        catalog = SentinelFeatureCatalog.default()
        prior_event = _row("prior", BASE - timedelta(minutes=offset_minutes))
        current = _row("current", BASE)
        batch = build_sentinel_features([prior_event, current], catalog=catalog)
        idx = catalog.feature_names.index("actor_count_24h")
        actual = batch.matrix[1][idx]  # second row is "current"
        assert actual == float(expected_count), (
            f"offset={offset_minutes}m: expected {expected_count}, got {actual}"
        )

    def test_counterparty_count_24h_filters_to_previous_24h(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        old_cp = _row("old", BASE - timedelta(hours=25), actor="actor-2", cp="cp-shared")
        recent_cp = _row("recent", BASE - timedelta(hours=1), actor="actor-3", cp="cp-shared")
        current = _row("current", BASE, actor="actor-1", cp="cp-shared")
        batch = build_sentinel_features([old_cp, recent_cp, current], catalog=catalog)
        idx = catalog.feature_names.index("counterparty_count_24h")
        assert batch.matrix[2][idx] == 1.0, "counterparty_count_24h should only count events in previous 24h"

    def test_counterparty_distinct_actors_filters_to_24h(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        old = _row("old", BASE - timedelta(hours=25), actor="old-actor", cp="cp-shared")
        recent = _row("recent", BASE - timedelta(hours=1), actor="new-actor", cp="cp-shared")
        current = _row("current", BASE, actor="current-actor", cp="cp-shared")
        batch = build_sentinel_features([old, recent, current], catalog=catalog)
        idx = catalog.feature_names.index("counterparty_distinct_actors_24h")
        assert batch.matrix[2][idx] == 1.0, "should only count distinct actors from previous 24h"


class TestEqualTimeIsolation:
    def test_equal_time_events_do_not_observe_each_other(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        peer_a = _row("peer-a", BASE, actor="shared-actor", cp="cp-a")
        peer_b = _row("peer-b", BASE, actor="shared-actor", cp="cp-b")
        batch = build_sentinel_features([peer_a, peer_b], catalog=catalog)
        idx = catalog.feature_names.index("actor_count_5m")
        assert batch.matrix[0][idx] == 0.0, "equal-time peers must not observe one another"
