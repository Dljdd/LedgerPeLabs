"""Generator-fingerprint and behavioral overlap regression tests."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")

_NON_TRUST_FEATURES = [
    name for name in SentinelFeatureCatalog.default().feature_names
    if not name.startswith(("integrity_", "dq_"))
]


class TestGeneratorFingerprint:
    @pytest.fixture(scope="class")
    def smoke_corpus(self):
        return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)

    def test_no_single_feature_perfectly_separates(self, smoke_corpus) -> None:
        catalog = SentinelFeatureCatalog.default()
        train = smoke_corpus.partitions["train"]
        rows = train.decisions
        batch = build_sentinel_features(rows, catalog=catalog)
        labels = [r.is_fraud for r in rows]
        n_fraud = sum(labels)
        n_benign = len(labels) - n_fraud

        for feature_name in _NON_TRUST_FEATURES:
            idx = catalog.feature_names.index(feature_name)
            values = [row[idx] for row in batch.matrix]
            fraud_values = sorted(v for v, y in zip(values, labels) if y)
            benign_values = sorted(v for v, y in zip(values, labels) if not y)
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
