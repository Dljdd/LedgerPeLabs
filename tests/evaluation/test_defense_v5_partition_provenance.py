"""Partition-provenance regression tests for Defend v5."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)


@pytest.fixture(scope="module")
def smoke_corpus():
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestPartitionProvenance:
    def test_train_partition_exists_with_both_classes(self, smoke_corpus) -> None:
        train = smoke_corpus.partitions["train"]
        assert train.fraud_count > 0, "train has no fraud"
        assert train.benign_count > 0, "train has no legitimate"

    def test_calibration_partition_has_both_classes(self, smoke_corpus) -> None:
        cal = smoke_corpus.partitions["calibration"]
        assert cal.fraud_count > 0, "calibration has no fraud"
        assert cal.benign_count > 0, "calibration has no legitimate"

    def test_threshold_partition_has_both_classes(self, smoke_corpus) -> None:
        thr = smoke_corpus.partitions["threshold"]
        assert thr.fraud_count > 0, "threshold has no fraud"
        assert thr.benign_count > 0, "threshold has no legitimate"

    def test_development_test_is_untouched_by_other_partitions(self, smoke_corpus) -> None:
        dev_test_ids = {r.event_id for r in smoke_corpus.partitions["development_test"].decisions}
        for name in ("train", "calibration", "threshold"):
            other_ids = {r.event_id for r in smoke_corpus.partitions[name].decisions}
            overlap = dev_test_ids & other_ids
            assert not overlap, (
                "event_id contamination: "
                f"development_test ∩ {name} = {len(overlap)} rows"
            )

    def test_no_identity_crosses_partitions(self, smoke_corpus) -> None:
        identity_fields = ["actor_id", "counterparty_id", "campaign_id", "payment_id", "event_id"]
        partitions = list(smoke_corpus.partitions.values())
        for field in identity_fields:
            seen: dict[str, str] = {}
            for partition in partitions:
                for row in partition.decisions:
                    value = getattr(row, field)
                    if value in seen and seen[value] != partition.partition_name:
                        pytest.fail(
                            f"identity leak: {field}={value!r} appears in both "
                            f"{seen[value]!r} and {partition.partition_name!r}"
                        )
                    seen[value] = partition.partition_name
