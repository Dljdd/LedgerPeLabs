"""Behavioral fidelity audit tests for Defend v5."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_fidelity import (
    FidelityDimension,
    audit_v5_fidelity,
)
from apar.evaluation.v5_population import V5Corpus, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)


@pytest.fixture(scope="module")
def corpus() -> V5Corpus:
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestFidelityAudit:
    def test_audit_returns_all_four_dimensions(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        dimensions = {check.dimension for check in result.checks}
        assert dimensions == set(FidelityDimension)

    def test_valid_corpus_passes(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        assert result.overall_status == "pass"
        assert all(check.passed for check in result.checks)

    def test_checks_have_observed_reference_tolerance(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        for check in result.checks:
            assert hasattr(check, "observed")
            assert hasattr(check, "reference_min")
            assert hasattr(check, "reference_max")

    def test_economic_reconciliation_check_present(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        economic = [c for c in result.checks if c.dimension is FidelityDimension.ECONOMIC]
        assert len(economic) >= 1

    def test_temporal_lifecycle_check_present(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        temporal = [c for c in result.checks if c.dimension is FidelityDimension.TEMPORAL]
        assert len(temporal) >= 1

    def test_relational_graph_check_present(self, corpus: V5Corpus) -> None:
        result = audit_v5_fidelity(corpus)
        relational = [c for c in result.checks if c.dimension is FidelityDimension.RELATIONAL]
        assert len(relational) >= 1
