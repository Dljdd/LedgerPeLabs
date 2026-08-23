"""Production development-test partition size regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_protocol import load_v5_development_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")


class TestProductionPopulationSize:
    def test_development_test_seed_is_1404(self) -> None:
        assert PROTOCOL.seeds.development_test == 1404

    def test_production_declares_partition_specific_sizes(self) -> None:
        assert hasattr(PROTOCOL, "production_dev_test_legitimate"), (
            "protocol must declare partition-specific population sizes"
        )

    def test_production_dev_test_legitimate_at_least_50000(self) -> None:
        if hasattr(PROTOCOL, "production_dev_test_legitimate"):
            assert PROTOCOL.production_dev_test_legitimate >= 50_000

    def test_production_dev_test_campaigns_per_family_at_least_100(self) -> None:
        if hasattr(PROTOCOL, "production_dev_test_campaigns_per_family"):
            counts = PROTOCOL.production_dev_test_campaigns_per_family
            assert all(v >= 100 for v in counts.values())
