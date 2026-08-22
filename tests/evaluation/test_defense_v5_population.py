"""Mixed, group-disjoint population tests for Defend v5."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_population import (
    V5Corpus,
    build_v5_corpus,
)
from apar.evaluation.v5_protocol import V5Family, V5Profile, load_v5_development_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")
ALL_FAMILIES = {f.value for f in V5Family}


class TestSmokeCorpus:
    @pytest.fixture(scope="class")
    def smoke_corpus(self) -> V5Corpus:
        return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)

    def test_legitimate_and_fraud_coexist(self, smoke_corpus: V5Corpus) -> None:
        for partition_name, partition in smoke_corpus.partitions.items():
            if partition_name in ("hardening_train", "adaptive_holdout"):
                continue
            fraud_count = sum(1 for row in partition.decisions if row.is_fraud)
            benign_count = sum(1 for row in partition.decisions if not row.is_fraud)
            assert fraud_count > 0, f"no fraud in {partition_name}"
            assert benign_count > 0, f"no legitimate in {partition_name}"

    def test_all_four_families_present(self, smoke_corpus: V5Corpus) -> None:
        all_families_in_fraud = {
            row.family
            for partition in smoke_corpus.partitions.values()
            for row in partition.decisions
            if row.is_fraud
        }
        assert all_families_in_fraud == ALL_FAMILIES

    def test_identity_disjoint_across_partitions(self, smoke_corpus: V5Corpus) -> None:
        actor_sets: dict[str, set[str]] = {}
        campaign_sets: dict[str, set[str]] = {}
        for name, partition in smoke_corpus.partitions.items():
            actor_sets[name] = {row.actor_id for row in partition.decisions}
            campaign_sets[name] = {row.campaign_id for row in partition.decisions}
        names = list(smoke_corpus.partitions.keys())
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert not (actor_sets[left] & actor_sets[right]), f"actor overlap {left}/{right}"
                assert not (
                    campaign_sets[left] & campaign_sets[right]
                ), f"campaign overlap {left}/{right}"

    def test_predictive_projection_omits_forbidden_fields(self, smoke_corpus: V5Corpus) -> None:
        forbidden = {"family", "campaign_id", "scenario_id", "seed", "split", "is_fraud"}
        for partition in smoke_corpus.partitions.values():
            for row in partition.decisions:
                predictive_keys = set(row.predictive_features.keys())
                assert not (forbidden & predictive_keys), f"leak: {forbidden & predictive_keys}"

    def test_smoke_is_marked(self, smoke_corpus: V5Corpus) -> None:
        assert smoke_corpus.profile == V5Profile.SMOKE
        assert not smoke_corpus.is_production

    def test_deterministic_digest(self) -> None:
        c1 = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        c2 = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        assert c1.corpus_sha256 == c2.corpus_sha256


class TestProductionBounds:
    def test_production_profile_has_sufficient_legitimate_decisions(self) -> None:
        assert PROTOCOL.production_profile.legitimate_decisions >= 50_000

    def test_production_has_100_campaigns_per_family(self) -> None:
        counts = PROTOCOL.production_profile.campaigns_per_family
        assert all(v >= 100 for v in counts.values())
