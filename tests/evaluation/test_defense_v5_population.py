"""Mixed, group-disjoint population tests for Defend v5."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_population import (
    V5Corpus,
    V5ExecutionManifest,
    V5PartitionCorpus,
    build_v5_corpus,
)
from apar.evaluation.v5_protocol import V5Family, V5Profile
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)
ALL_FAMILIES = {f.value for f in V5Family}


@pytest.fixture(scope="module")
def smoke_corpus() -> V5Corpus:
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestSmokeCorpus:
    def test_legitimate_rows_are_projected_from_real_execution_evidence(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        """Removing real legitimate execution must make this test fail."""
        for partition in smoke_corpus.partitions.values():
            manifests = {
                execution.evidence_sha256: execution
                for execution in partition.executions
            }
            legitimate = [
                row for row in partition.decisions if row.family == "legitimate"
            ]
            assert legitimate
            for row in legitimate:
                assert row.execution_evidence_sha256
                manifest = manifests[row.execution_evidence_sha256]
                assert manifest.family == "legitimate"
                assert row.source_command_id.startswith("sha256:")
                assert row.source_event_id == row.event_id

    def test_every_operational_rail_has_both_classes(self, smoke_corpus: V5Corpus) -> None:
        """Removing any executed legitimate rail must make this test fail."""
        expected_rails = {"card", "a2a", "agentic"}
        for partition_name in ("train", "calibration", "threshold", "development_test"):
            rows = smoke_corpus.partitions[partition_name].decisions
            for rail in expected_rails:
                labels = {row.is_fraud for row in rows if row.rail == rail}
                assert labels == {False, True}, f"{partition_name}/{rail} lacks a class"

    def test_legitimate_execution_manifest_retains_canonical_economic_and_trust_facts(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        """Dropping immutable ledger or verifier facts must make this test fail."""
        manifests = smoke_corpus.partitions["train"].executions
        legitimate = [manifest for manifest in manifests if manifest.family == "legitimate"]
        assert {manifest.rail for manifest in legitimate} == {"card", "a2a", "agentic"}
        for manifest in legitimate:
            assert manifest.event_records
            assert manifest.ledger_postings or manifest.rail == "agentic"
            if manifest.rail == "agentic":
                assert manifest.trust_records
                assert all(record.request_json for record in manifest.trust_records)
                assert all(record.mandate_json for record in manifest.trust_records)
                assert all(record.public_key_hex for record in manifest.trust_records)

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

    def test_every_campaign_row_is_backed_by_real_execution_manifest(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        for partition in smoke_corpus.partitions.values():
            manifests = {
                execution.evidence_sha256: execution
                for execution in partition.executions
            }
            assert {execution.family for execution in partition.executions} >= ALL_FAMILIES
            for row in partition.decisions:
                execution = manifests[row.execution_evidence_sha256]
                lineage = {
                    item.event_id: item for item in execution.lineage
                }[row.source_event_id]
                assert lineage.command_id == row.source_command_id
                assert lineage.payment_id == row.payment_id
                assert lineage.is_fraud is row.is_fraud

    def test_every_agentic_row_has_real_verifier_execution(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        for partition in smoke_corpus.partitions.values():
            for row in partition.decisions:
                if row.rail == "agentic":
                    assert row.execution_evidence_sha256
                    assert row.source_command_id
                    assert row.source_event_id

    def test_executed_traffic_has_no_trivial_provenance_or_rail_fingerprint(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        """Restoring hand-built benign rows or a label-coded rail must fail here."""
        rows = smoke_corpus.partitions["development_test"].decisions
        labels = {row.is_fraud for row in rows}
        assert labels == {False, True}
        assert {bool(row.execution_evidence_sha256) for row in rows} == {True}
        for rail in ("card", "a2a", "agentic"):
            assert {row.is_fraud for row in rows if row.rail == rail} == labels
        assert {row.source_command_id.split(":", 1)[0] for row in rows} == {"sha256"}
        assert all(row.campaign_id.count("-") == 4 for row in rows)
        for state in ("settled", "declined", "refunded", "returned", "recovered"):
            state_labels = {row.is_fraud for row in rows if row.lifecycle_state == state}
            assert state_labels != {True}, f"{state} is a fraud-only fingerprint"

    def test_identity_renaming_preserves_numeric_feature_matrix(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        """Introducing raw identity-derived numeric features must make this fail."""
        rows = smoke_corpus.partitions["train"].decisions
        identities = {row.actor_id for row in rows} | {
            row.counterparty_id for row in rows
        }
        renamed_identities = {
            value: f"identity-{index}"
            for index, value in enumerate(sorted(identities))
        }
        renamed = tuple(
            row.model_copy(
                update={
                    "actor_id": renamed_identities[row.actor_id],
                    "counterparty_id": renamed_identities[row.counterparty_id],
                }
            )
            for row in rows
        )
        catalog = SentinelFeatureCatalog.default()
        original = build_sentinel_features(rows, catalog=catalog)
        renamed_features = build_sentinel_features(renamed, catalog=catalog)
        assert renamed_features.matrix == original.matrix

    def test_partition_rejects_unbacked_campaign_row(
        self,
        smoke_corpus: V5Corpus,
    ) -> None:
        row = next(
            row
            for row in smoke_corpus.partitions["train"].decisions
            if row.family != "legitimate"
        )

        with pytest.raises(ValueError, match="real execution evidence"):
            empty_executions: tuple[V5ExecutionManifest, ...] = ()
            V5PartitionCorpus(
                partition_name="train",
                decisions=(row,),
                executions=empty_executions,
            )

    def test_identity_disjoint_across_partitions(self, smoke_corpus: V5Corpus) -> None:
        actor_sets: dict[str, set[str]] = {}
        counterparty_sets: dict[str, set[str]] = {}
        campaign_sets: dict[str, set[str]] = {}
        payment_sets: dict[str, set[str]] = {}
        for name, partition in smoke_corpus.partitions.items():
            actor_sets[name] = {row.actor_id for row in partition.decisions}
            counterparty_sets[name] = {
                row.counterparty_id for row in partition.decisions
            }
            campaign_sets[name] = {row.campaign_id for row in partition.decisions}
            payment_sets[name] = {row.payment_id for row in partition.decisions}
        names = list(smoke_corpus.partitions.keys())
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert not (actor_sets[left] & actor_sets[right]), f"actor overlap {left}/{right}"
                assert not (
                    campaign_sets[left] & campaign_sets[right]
                ), f"campaign overlap {left}/{right}"
                assert not (
                    counterparty_sets[left] & counterparty_sets[right]
                ), f"counterparty overlap {left}/{right}"
                assert not (
                    payment_sets[left] & payment_sets[right]
                ), f"payment overlap {left}/{right}"

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
