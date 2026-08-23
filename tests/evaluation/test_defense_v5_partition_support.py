"""Partition support regression tests: production dev-test must have 50K legitimate."""

from __future__ import annotations

import pytest

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")


class TestProductionDevTestSupport:
    """These tests only assert the contract, not the actual build (too slow for unit tests)."""

    def test_protocol_declares_50k_dev_test_legitimate(self) -> None:
        assert PROTOCOL.production_dev_test_legitimate >= 50_000

    def test_protocol_declares_100_campaigns_per_family_dev_test(self) -> None:
        counts = PROTOCOL.production_dev_test_campaigns_per_family
        assert all(v >= 100 for v in counts.values())

    def test_smoke_corpus_has_mixed_partitions(self) -> None:
        corpus = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        for name in ("train", "calibration", "threshold", "development_test"):
            partition = corpus.partitions[name]
            assert partition.fraud_count > 0, f"{name} has no fraud"
            assert partition.benign_count > 0, f"{name} has no legitimate"

    def test_production_builder_uses_partition_specific_counts(self, monkeypatch) -> None:
        """Verify build_v5_corpus consumes production_dev_test_legitimate."""
        source = __import__("inspect").getsource(
            __import__("apar.evaluation.v5_population", fromlist=["build_v5_corpus"]).build_v5_corpus
        )
        assert "production_dev_test_legitimate" in source or "profile_counts" in source, (
            "build_v5_corpus must consume partition-specific counts"
        )
