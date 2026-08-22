"""Label-flip feature invariance regression test."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from apar.evaluation.v5_population import V5DecisionRow
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features


def _row(event_id: str, at: datetime, actor: str, cp: str, *, is_fraud: bool) -> V5DecisionRow:
    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id="campaign-x",
        family="card_testing_cnp",
        actor_id=actor,
        counterparty_id=cp,
        amount=Decimal("100.00"),
        decision_at=at,
        is_fraud=is_fraud,
        lifecycle_state="probe",
        source_command_id=f"cmd-{event_id}",
        source_event_id=event_id,
        predictive_features={"amount": 100.0, "rail_card": 1.0},
    )


BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestLabelFlipInvariance:
    def test_flipping_labels_preserves_all_features(self) -> None:
        catalog = SentinelFeatureCatalog.default()
        rows_original = [
            _row("a", BASE, "actor-1", "cp-1", is_fraud=True),
            _row("b", BASE + timedelta(minutes=1), "actor-2", "cp-2", is_fraud=False),
        ]
        rows_flipped = [
            rows_original[0].model_copy(update={"is_fraud": False}),
            rows_original[1].model_copy(update={"is_fraud": True}),
        ]
        r1 = build_sentinel_features(rows_original, catalog=catalog)
        r2 = build_sentinel_features(rows_flipped, catalog=catalog)
        assert r1.matrix == r2.matrix, "flipping labels changed feature vectors"
