"""Future graph leakage regression tests for Sentinel v5 features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from apar.evaluation.v5_population import V5DecisionRow
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features


def _row(event_id: str, at: datetime, actor: str, cp: str, amount: str = "100.00") -> V5DecisionRow:
    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id="campaign-x",
        family="card_testing_cnp",
        actor_id=actor,
        counterparty_id=cp,
        amount=Decimal(amount),
        decision_at=at,
        is_fraud=True,
        lifecycle_state="probe",
        source_command_id=f"cmd-{event_id}",
        source_event_id=event_id,
        predictive_features={"amount": float(amount), "rail_card": 1.0},
    )


BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestFutureGraphLeakage:
    def test_population_builder_future_graph_leakage(self) -> None:
        """RED: _enrich_features must not use future events for graph stats."""
        first_event = V5DecisionRow(
            event_id="a", payment_id="payment-a",
            campaign_id="campaign-x", family="card_testing_cnp",
            actor_id="actor-1", counterparty_id="cp-1",
            amount=Decimal("100.00"), decision_at=BASE,
            is_fraud=True, lifecycle_state="probe",
            source_command_id="cmd-a", source_event_id="a",
            predictive_features={"amount": 100.0, "rail_card": 1.0},
        )
        future_events = [
            V5DecisionRow(
                event_id=f"f{i}", payment_id=f"payment-f{i}",
                campaign_id="campaign-x", family="card_testing_cnp",
                actor_id=f"actor-{i}", counterparty_id=f"cp-f{i}",
                amount=Decimal("100.00"),
                decision_at=BASE + timedelta(hours=i + 1),
                is_fraud=True, lifecycle_state="probe",
                source_command_id=f"cmd-f{i}", source_event_id=f"f{i}",
                predictive_features={"amount": 100.0, "rail_card": 1.0},
            )
            for i in range(1, 4)
        ]

        catalog = SentinelFeatureCatalog.default()
        alone = build_sentinel_features([first_event], catalog=catalog)
        with_future = build_sentinel_features(
            [first_event] + future_events, catalog=catalog
        )
        idx_component = catalog.feature_names.index("graph_component_size")
        assert alone.matrix[0][idx_component] == with_future.matrix[0][idx_component], (
            f"future events changed component_size: "
            f"{alone.matrix[0][idx_component]} -> {with_future.matrix[0][idx_component]}"
        )
    def test_future_events_cannot_change_component_size(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        first_event = _row("a", BASE, "actor-1", "cp-1")
        future_events = [
            _row("b", BASE + timedelta(hours=1), "actor-1", "cp-2"),
            _row("c", BASE + timedelta(hours=2), "actor-2", "cp-3"),
            _row("d", BASE + timedelta(hours=3), "actor-2", "cp-4"),
        ]
        alone = build_sentinel_features([first_event], catalog=catalog)
        with_future = build_sentinel_features([first_event, *future_events], catalog=catalog)

        idx_component = catalog.feature_names.index("graph_component_size")
        alone_component = alone.matrix[0][idx_component]
        with_future_component = with_future.matrix[0][idx_component]
        assert alone_component == with_future_component, (
            f"future events changed component_size: {alone_component} -> {with_future_component}"
        )

    def test_future_events_cannot_change_edge_density(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        first_event = _row("a", BASE, "actor-1", "cp-1")
        future_events = [
            _row("b", BASE + timedelta(hours=1), "actor-2", "cp-2"),
            _row("c", BASE + timedelta(hours=2), "actor-3", "cp-3"),
        ]
        alone = build_sentinel_features([first_event], catalog=catalog)
        with_future = build_sentinel_features([first_event, *future_events], catalog=catalog)
        idx_density = catalog.feature_names.index("graph_edge_density")
        assert alone.matrix[0][idx_density] == with_future.matrix[0][idx_density]

    def test_future_events_cannot_change_shared_neighbors(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        first_event = _row("a", BASE, "actor-1", "cp-1")
        future_shared = _row("b", BASE + timedelta(hours=1), "actor-2", "cp-1")
        alone = build_sentinel_features([first_event], catalog=catalog)
        with_shared = build_sentinel_features([first_event, future_shared], catalog=catalog)
        idx_shared = catalog.feature_names.index("graph_shared_neighbor_count")
        assert alone.matrix[0][idx_shared] == with_shared.matrix[0][idx_shared]
