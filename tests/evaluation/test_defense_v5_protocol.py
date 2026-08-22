"""Sealed Defend v5 development protocol and frozen prior-evidence isolation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apar.evaluation.v5_protocol import (
    V5Family,
    V5Partition,
    V5Profile,
    V5ReadinessTargets,
    load_v5_development_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "config/defense/defense-v5-development.json"


class TestV5Partitions:
    def test_all_six_partitions_exist(self) -> None:
        assert V5Partition.TRAIN.value == "train"
        assert V5Partition.CALIBRATION.value == "calibration"
        assert V5Partition.THRESHOLD.value == "threshold"
        assert V5Partition.DEVELOPMENT_TEST.value == "development_test"
        assert V5Partition.HARDENING_TRAIN.value == "hardening_train"
        assert V5Partition.ADAPTIVE_HOLDOUT.value == "adaptive_holdout"

    def test_partition_is_str_enum(self) -> None:
        assert isinstance(V5Partition.TRAIN, str)


class TestV5Profiles:
    def test_smoke_and_production_only(self) -> None:
        assert V5Profile.SMOKE.value == "smoke"
        assert V5Profile.PRODUCTION.value == "production"
        assert len(V5Profile) == 2


class TestV5Families:
    def test_exact_four_families(self) -> None:
        expected = {
            "agentic_intent_abuse",
            "app_scam_mule",
            "card_testing_cnp",
            "synthetic_merchant_refund",
        }
        actual = {member.value for member in V5Family}
        assert actual == expected


class TestV5ReadinessTargets:
    def test_exact_target_values(self) -> None:
        targets = V5ReadinessTargets()
        assert targets.family_recall_min == 0.75
        assert targets.false_decline_rate_max == 0.001
        assert targets.manual_review_rate_max == 0.01
        assert targets.challenge_rate_max == 0.02
        assert targets.captured_value_fraction_min == 0.70
        assert targets.expected_calibration_error_max == 0.10
        assert targets.p95_decision_latency_ms_max == 50.0


class TestProtocolLoader:
    def test_loads_canonical_config(self) -> None:
        protocol = load_v5_development_protocol(PROTOCOL_PATH)
        assert protocol.protocol_id == "apar-sentinel-v5-development"
        assert protocol.sealed_evaluation_allowed is False
        assert protocol.development_only is True

    def test_protocol_digest_is_deterministic(self) -> None:
        p1 = load_v5_development_protocol(PROTOCOL_PATH)
        p2 = load_v5_development_protocol(PROTOCOL_PATH)
        assert p1.protocol_sha256 == p2.protocol_sha256
        assert len(p1.protocol_sha256) == 64

    def test_rejects_unknown_field(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_bytes())
        document["unexpected"] = True
        path = Path("/tmp/bad-protocol.json")
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="unexpected"):
            load_v5_development_protocol(path)

    def test_rejects_nan_and_infinity(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_bytes())
        document["readiness"]["family_recall_min"] = float("nan")
        path = Path("/tmp/nan-protocol.json")
        path.write_text(json.dumps(document, allow_nan=True))
        with pytest.raises(ValueError):
            load_v5_development_protocol(path)

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_v5_development_protocol(Path("/nonexistent.json"))
