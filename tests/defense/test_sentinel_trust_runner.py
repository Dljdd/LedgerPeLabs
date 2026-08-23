"""Trust runner integration regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from apar.contracts.decisions import ReasonCode
from apar.contracts.events import Rail
from apar.defense.sentinel import SentinelAction, train_sentinel_defender
from apar.evaluation.v5_execution import project_execution_evidence
from scripts.run_defense_v5_development import (
    _decide_with_trust,
    _derive_trust_failures,
)
from tests.evaluation.test_defense_v5_execution_projection import _evidence


def _make_defender():
    rng = np.random.RandomState(42)
    x_train = np.vstack([
        rng.normal(0.0, 0.5, (100, 7)),
        rng.normal(3.0, 0.5, (30, 7)),
    ])
    y_train = np.array([0] * 100 + [1] * 30)
    x_cal = np.vstack([x_train[:20], x_train[100:110]])
    y_cal = np.concatenate([y_train[:20], y_train[100:110]])
    x_thr = np.vstack([x_train[20:30], x_train[110:115]])
    y_thr = np.concatenate([y_train[20:30], y_train[110:115]])
    return train_sentinel_defender(
        x_train=x_train, y_train=y_train,
        x_calibration=x_cal, y_calibration=y_cal,
        x_threshold=x_thr, y_threshold=y_thr,
        catboost_seeds=(1, 2, 3), bootstrap_seed=99,
    )


class TestTrustRunnerIntegration:
    def test_decide_batch_with_trust_failures(self) -> None:
        """RED: decide_batch must accept per-row trust_failure sequence."""
        defender = _make_defender()
        features = np.zeros((3, 7))
        trust_failures = [True, False, False]

        decisions = defender.decide_batch(features, trust_failures=trust_failures)
        assert decisions[0].action == SentinelAction.DECLINE_HOLD
        assert (
            decisions[1].action != SentinelAction.DECLINE_HOLD
            or decisions[1].trust_failure is False
        )

    def test_decide_batch_rejects_mismatched_trust_length(self) -> None:
        defender = _make_defender()
        features = np.zeros((3, 7))
        with pytest.raises(ValueError, match="trust"):
            defender.decide_batch(features, trust_failures=[True])

    def test_real_invalid_agentic_evidence_declines_in_runner_order(self) -> None:
        evidence = _evidence("agentic_intent_abuse", Rail.AGENTIC)
        rows = project_execution_evidence(evidence)
        signature_event_id = next(
            record.event_id
            for record in evidence.trust_evidence
            if record.receipt.reason_code is ReasonCode.SIGNATURE_INVALID
        )
        invalid = next(row for row in rows if row.event_id == signature_event_id)
        valid = next(row for row in rows if row.integrity_status == "pass")
        non_agentic = project_execution_evidence(
            _evidence("card_testing_cnp", Rail.CARD)
        )[0]
        ordered_rows = (valid, non_agentic, invalid)
        defender = _make_defender()
        features = np.zeros((len(ordered_rows), 7))

        trust_failures = _derive_trust_failures(ordered_rows)
        decisions = _decide_with_trust(defender, features, ordered_rows)

        assert trust_failures == [False, False, True]
        assert decisions[2].action == SentinelAction.DECLINE_HOLD
        assert decisions[2].trust_failure is True

    def test_runner_rejects_unbacked_agentic_status(self) -> None:
        valid = project_execution_evidence(
            _evidence("agentic_intent_abuse", Rail.AGENTIC)
        )[0]
        unbacked = valid.model_copy(
            update={
                "execution_evidence_sha256": "",
                "source_command_id": "",
                "source_event_id": "",
            }
        )

        with pytest.raises(ValueError, match="real verifier execution"):
            _derive_trust_failures((unbacked,))
