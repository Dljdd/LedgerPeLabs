"""Generator-fingerprint and behavioral overlap regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_population import V5Corpus, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)

_NON_TRUST_FEATURES = [
    name for name in SentinelFeatureCatalog.default().feature_names
    if not name.startswith(("integrity_", "dq_"))
]


@pytest.fixture(scope="module")
def smoke_corpus() -> V5Corpus:
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestGeneratorFingerprint:
    def test_no_single_feature_perfectly_separates(self, smoke_corpus) -> None:
        catalog = SentinelFeatureCatalog.default()
        train = smoke_corpus.partitions["train"]
        rows = train.decisions
        batch = build_sentinel_features(rows, catalog=catalog)
        labels = [r.is_fraud for r in rows]
        for feature_name in _NON_TRUST_FEATURES:
            idx = catalog.feature_names.index(feature_name)
            values = [row[idx] for row in batch.matrix]
            fraud_values = sorted(
                value
                for value, label in zip(values, labels, strict=True)
                if label
            )
            benign_values = sorted(
                value
                for value, label in zip(values, labels, strict=True)
                if not label
            )
            if not fraud_values or not benign_values:
                continue
            # Perfect separation: max(benign) < min(fraud) or max(fraud) < min(benign).
            perfectly_separated = (
                benign_values[-1] < fraud_values[0]
                or fraud_values[-1] < benign_values[0]
            )
            assert not perfectly_separated, (
                f"feature '{feature_name}' perfectly separates legitimate from fraud "
                f"(benign max={benign_values[-1]}, fraud min={fraud_values[0]})"
            )
