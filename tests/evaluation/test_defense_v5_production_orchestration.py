"""Production-profile orchestration contract tests without production execution."""

from __future__ import annotations

from pathlib import Path

import apar.evaluation.v5_population as population_module
from apar.evaluation.v5_protocol import V5Family, load_v5_development_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(
    ROOT / "config/defense/defense-v5-development.json"
)


class TestProductionOrchestration:
    def test_hand_built_fraud_partition_api_is_retired(self) -> None:
        assert not hasattr(
            population_module,
            "_build_fraud_campaigns_for_partition",
        )

    def test_locked_production_counts_cover_every_partition_family(self) -> None:
        assert PROTOCOL.production_dev_test_legitimate >= 50_000
        for family in V5Family:
            assert (
                PROTOCOL.production_dev_test_campaigns_per_family[family.value]
                >= 100
            )
