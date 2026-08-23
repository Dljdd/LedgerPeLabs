"""Reporting and readiness verifier tests for Defend v5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.evaluation.v5_reporting import V5DevelopmentResult, build_v5_development_result
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)


class TestReporting:
    def test_result_has_valid_status(self) -> None:
        corpus = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        result = build_v5_development_result(
            protocol=PROTOCOL, corpus=corpus
        )
        assert result.status in (
            "development_ready", "development_not_ready", "invalid_corpus", "smoke",
        )

    def test_forbidden_claims_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden status"):
            V5DevelopmentResult(status="winner")

    def test_result_serializes_to_json(self) -> None:
        corpus = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        result = build_v5_development_result(protocol=PROTOCOL, corpus=corpus)
        payload = result.model_dump_json()
        parsed = json.loads(payload)
        assert "status" in parsed
