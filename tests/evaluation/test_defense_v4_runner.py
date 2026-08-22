"""One-attempt confirmatory runner tests for Defend v4."""

from __future__ import annotations

import hashlib

import pytest

from apar.evaluation.v4_preexecution import verify_v4_preexecution
from apar.evaluation.v4_runner import (
    V4ExecutionInputs,
    V4RunnerError,
    create_v4_receipt,
    execute_v4_arms,
    finalize_v4_receipt,
    verify_v4_approval,
)
from apar.evaluation.v4_scoring import FrozenDefenderBundle
from apar.evaluation.contracts import EvaluationTruthRow
from decimal import Decimal
from apar.v4_protocol import V4GateValues
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = FrozenDefenderBundle(ROOT)

def _inputs(approval: str | None = None) -> V4ExecutionInputs:
    return V4ExecutionInputs(
        protocol_id="apar-defend-v4",
        execution_nonce="a" * 64,
        source_tree_sha256="b" * 64,
        config_manifest_sha256="c" * 64,
        defender_bundle_sha256="d" * 64,
        population_manifest_sha256="e" * 64,
        evaluator_key_id="f" * 64,
        approval_token=approval or hashlib.sha256(b"freeze").hexdigest(),
    )


def test_correct_approval_token_accepted() -> None:
    freeze = hashlib.sha256(b"freeze").hexdigest()
    verify_v4_approval(_inputs(freeze), expected_freeze_digest=freeze)


def test_wrong_approval_token_rejected() -> None:
    with pytest.raises(V4RunnerError, match="approval token"):
        verify_v4_approval(_inputs("0" * 64), expected_freeze_digest=hashlib.sha256(b"freeze").hexdigest())


def test_create_receipt_then_duplicate_rejected(tmp_path: Path) -> None:
    create_v4_receipt(_inputs(), directory=tmp_path)
    with pytest.raises(V4RunnerError, match="already consumed"):
        create_v4_receipt(_inputs(), directory=tmp_path)


def test_execute_arms_produces_gate_results(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal
    from apar.contracts.events import EventKind, Rail
    from apar.defense.contracts import ObservedEvent
    from apar.evaluation.contracts import Family

    observations = tuple(
        ObservedEvent(
            event_id=f"row-{index}",
            payment_id=f"payment-{index}",
            rail=Rail.CARD,
            event_type=EventKind.AUTHORIZATION,
            amount=Decimal("100.00"),
            currency="USD",
            event_time=datetime(2026, 8, 21, tzinfo=UTC),
            available_at=datetime(2026, 8, 21, tzinfo=UTC),
            decision_at=datetime(2026, 8, 21, tzinfo=UTC) + timedelta(minutes=1),
            actor_id=f"actor-{index}",
            counterparty_id=f"counterparty-{index}",
            integrity_status="not_applicable",
            is_decision_point=True,
        )
        for index in range(10)
    )
    truth = tuple(
        EvaluationTruthRow(
            event_id=f"row-{index}",
            payment_id=f"payment-{index}",
            campaign_id=f"campaign-{index}",
            family="card_testing_cnp",
            viewpoint="development",
            is_fraud=index % 2 == 0,
            label_source="population_truth",
            label_mature_at=datetime(2026, 8, 22, tzinfo=UTC),
            first_settlement_at=datetime(2026, 8, 22, tzinfo=UTC),
            net_settled_value=Decimal("100.00"),
            lifecycle_event_ids=(f"row-{index}",),
        )
        for index in range(10)
    )
    results = execute_v4_arms(
        _inputs(),
        observations=observations,
        truth=truth,
        observations_sha256="a" * 64,
        truth_sha256="b" * 64,
        gates=V4GateValues(),
        bundle=BUNDLE,
    )
    assert len(results) == 3
    assert all(not result.gate_outcome.passed for result in results)


def test_finalize_receipt_sets_terminal_status(tmp_path: Path) -> None:
    receipt = create_v4_receipt(_inputs(), directory=tmp_path)
    finalized = finalize_v4_receipt(receipt, directory=tmp_path, status="no_promotion")
    assert finalized.terminal_status == "no_promotion"
