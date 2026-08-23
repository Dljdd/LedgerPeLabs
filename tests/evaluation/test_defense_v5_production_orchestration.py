"""Production-profile orchestration regression tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from apar.evaluation.v5_population import V5DecisionRow, build_v5_corpus
from apar.evaluation.v5_protocol import (
    V5Family,
    V5Profile,
    load_v5_development_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")


class TestProductionOrchestration:
    """Prove the production build path reaches all six partition builders."""

    def test_production_build_reaches_all_partitions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RED: production build currently raises UnboundLocalError."""
        builder_calls: list[tuple[str, int]] = []
        campaign_calls: list[tuple[str, dict[str, int]]] = []

        original_benign_builder = __import__(
            "apar.evaluation.v5_population", fromlist=["_build_benign_partition"]
        )._build_benign_partition
        original_campaign_builder = __import__(
            "apar.evaluation.v5_population", fromlist=["_build_fraud_campaigns_for_partition"]
        )._build_fraud_campaigns_for_partition

        def fake_benign(partition_name: str, count: int, seed_value: int):
            builder_calls.append((partition_name, count))
            return []

        def fake_campaigns(
            partition_name: str,
            campaigns_per_family: dict[str, int],
            seed_value: int,
        ):
            campaign_calls.append((partition_name, campaigns_per_family))
            return []

        import apar.evaluation.v5_population as pop_module

        monkeypatch.setattr(pop_module, "_build_benign_partition", fake_benign)
        monkeypatch.setattr(
            pop_module, "_build_fraud_campaigns_for_partition", fake_campaigns
        )

        corpus = build_v5_corpus(PROTOCOL, profile=V5Profile.PRODUCTION)

        assert len(builder_calls) == 6, (
            f"expected 6 partition builders called, got {len(builder_calls)}: {builder_calls}"
        )
        dev_test_call = [c for c in builder_calls if c[0] == "development_test"]
        assert dev_test_call, "development_test partition was not built"
        assert dev_test_call[0][1] >= 50_000, (
            f"development_test legitimate count must be >= 50000, got {dev_test_call[0][1]}"
        )

        dev_test_campaigns = [c for c in campaign_calls if c[0] == "development_test"]
        assert dev_test_campaigns, "development_test campaigns were not built"
        family_counts = dev_test_campaigns[0][1]
        for family in V5Family:
            assert family_counts.get(family.value, 0) >= 100, (
                f"development_test {family.value} must have >= 100 campaigns"
            )
