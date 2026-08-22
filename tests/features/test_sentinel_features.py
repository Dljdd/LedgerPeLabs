"""Causal Sentinel feature projection tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from apar.evaluation.v5_population import V5DecisionRow
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features

ROOT = Path(__file__).resolve().parents[2]


def _row(event_id: str, amount: str = "100.00", hour: int = 12) -> V5DecisionRow:
    from datetime import UTC, datetime, timedelta

    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id="test-campaign",
        family="card_testing_cnp",
        actor_id=f"actor-{event_id}",
        counterparty_id=f"cp-{event_id}",
        amount=Decimal(amount),
        decision_at=datetime(2026, 1, 1, hour, tzinfo=UTC) + timedelta(minutes=hash(event_id) % 60),
        is_fraud=True,
        predictive_features={"amount": float(amount), "txn_hour_sin": 0.0},
    )


class TestSentinelFeatures:
    def test_catalog_loads_from_config(self) -> None:
        catalog = SentinelFeatureCatalog.from_config(
            ROOT / "config/defense/feature-catalog-v5.json"
        )
        assert len(catalog.feature_names) >= 7
        assert catalog.catalog_sha256 != ""

    def test_build_returns_matrix(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        batch = build_sentinel_features([_row("a"), _row("b")], catalog=catalog)
        assert len(batch.rows) == 2
        assert batch.matrix is not None

    def test_no_forbidden_fields(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        forbidden = {"family", "campaign_id", "is_fraud", "seed", "split"}
        assert not (forbidden & set(catalog.feature_names))

    def test_deterministic(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        rows = [_row("a"), _row("b")]
        b1 = build_sentinel_features(rows, catalog=catalog)
        b2 = build_sentinel_features(rows, catalog=catalog)
        assert b1.batch_sha256 == b2.batch_sha256
