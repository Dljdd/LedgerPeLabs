"""Real campaign/rail/ledger replay regression tests for all four families."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Family, V5Profile
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_safe_v5_test_protocol(ROOT)
FAMILIES = {f.value for f in V5Family}


@pytest.fixture(scope="module")
def corpus():
    return build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)


class TestRailReplay:
    def test_every_fraud_row_has_lifecycle_state(self, corpus) -> None:
        for partition in corpus.partitions.values():
            for row in partition.decisions:
                if row.is_fraud:
                    assert hasattr(row, "lifecycle_state"), (
                        f"missing lifecycle_state on {row.event_id}"
                    )
                    assert row.lifecycle_state, f"empty lifecycle_state on {row.event_id}"

    def test_all_four_families_have_campaigns_in_train(self, corpus) -> None:
        train_fraud_families = {
            r.family for r in corpus.partitions["train"].decisions if r.is_fraud
        }
        assert train_fraud_families == FAMILIES

    def test_fraud_rows_derive_from_commands_not_labels(self, corpus) -> None:
        for partition in corpus.partitions.values():
            for row in partition.decisions:
                if row.is_fraud:
                    assert row.source_command_id, f"no source_command_id on {row.event_id}"
                    assert row.source_event_id, f"no source_event_id on {row.event_id}"

    def test_agentic_integrity_failures_exist(self, corpus) -> None:
        agentic_failures = [
            r
            for p in corpus.partitions.values()
            for r in p.decisions
            if r.is_fraud and r.family == "agentic_intent_abuse" and r.integrity_status == "fail"
        ]
        assert len(agentic_failures) > 0, "no agentic integrity failures found"
