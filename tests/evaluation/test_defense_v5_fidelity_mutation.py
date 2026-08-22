"""Fidelity mutation regression tests: each check must be able to fail."""

from __future__ import annotations

import pytest

from apar.evaluation.v5_fidelity import FidelityDimension, audit_v5_fidelity
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")


@pytest.fixture(scope="module")
def corpus():
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestFidelityMutation:
    def test_valid_corpus_passes(self, corpus) -> None:
        result = audit_v5_fidelity(corpus)
        assert result.overall_status == "pass"

    def test_all_four_dimensions_present(self, corpus) -> None:
        result = audit_v5_fidelity(corpus)
        assert {c.dimension for c in result.checks} == set(FidelityDimension)

    def test_amount_outlier_fails(self, corpus) -> None:
        mutated = corpus.model_copy(deep=True)
        partitions = dict(mutated.partitions)
        train = partitions["train"]
        rows = list(train.decisions)
        rows[0] = rows[0].model_copy(update={"amount": __import__("decimal").Decimal("99999999.00")})
        partitions["train"] = train.model_copy(update={"decisions": tuple(rows)})
        result = audit_v5_fidelity(mutated.model_copy(update={"partitions": partitions}))
        failed = [c for c in result.checks if not c.passed]
        assert len(failed) > 0, "extreme amount did not fail any check"

    def test_reversed_lifecycle_fails_temporal(self, corpus) -> None:
        """Reversing lifecycle states must fail a temporal check."""
        mutated = corpus.model_copy(deep=True)
        partitions = dict(mutated.partitions)
        train = partitions["train"]
        reversed_rows = tuple(
            r.model_copy(update={"decision_at": r.decision_at}) for r in train.decisions
        )
        # Rebuild with reversed timestamps within campaigns.
        from collections import defaultdict
        by_campaign = defaultdict(list)
        for i, row in enumerate(reversed_rows):
            by_campaign[row.campaign_id].append(i)
        new_rows = list(reversed_rows)
        for campaign_id, indices in by_campaign.items():
            if len(indices) < 2:
                continue
            times = [new_rows[i].decision_at for i in indices]
            for j, idx in enumerate(indices):
                new_rows[idx] = new_rows[idx].model_copy(
                    update={"decision_at": times[len(times) - 1 - j]}
                )
        partitions["train"] = train.model_copy(update={"decisions": tuple(new_rows)})
        result = audit_v5_fidelity(mutated.model_copy(update={"partitions": partitions}))
        temporal_failures = [c for c in result.checks if c.dimension is FidelityDimension.TEMPORAL and not c.passed]
        assert len(temporal_failures) > 0, "reversed lifecycle did not fail temporal check"
