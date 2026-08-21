"""One-attempt confirmatory runner tests."""

from __future__ import annotations

import hashlib

import pytest

from apar.evaluation.v3_isolation import build_isolation_manifest
from apar.evaluation.v3_receipt import has_receipt
from apar.evaluation.v3_runner import (
    ExecutionInputs,
    V3RunnerError,
    create_receipt,
    execute_arms,
    finalize_receipt,
    verify_approval,
)


def _inputs(approval: str | None = None) -> ExecutionInputs:
    return ExecutionInputs(
        protocol_id="apar-defend-v3",
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
    verify_approval(_inputs(freeze), expected_freeze_digest=freeze)


def test_wrong_approval_token_rejected() -> None:
    with pytest.raises(V3RunnerError, match="approval token"):
        verify_approval(_inputs("0" * 64), expected_freeze_digest=hashlib.sha256(b"freeze").hexdigest())


def test_create_receipt_then_duplicate_rejected(tmp_path) -> None:
    create_receipt(_inputs(), directory=tmp_path)
    assert has_receipt(directory=tmp_path)
    with pytest.raises(V3RunnerError, match="already consumed"):
        create_receipt(_inputs(), directory=tmp_path)


def test_execute_arms_returns_outcomes(tmp_path) -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3", timeout_seconds=15.0)
    outcomes = execute_arms(_inputs(), manifest=manifest)
    assert len(outcomes) == 3
    assert all(outcome.status in ("no_promotion", "failed") for outcome in outcomes)


def test_finalize_receipt_sets_terminal_status(tmp_path) -> None:
    receipt = create_receipt(_inputs(), directory=tmp_path)
    finalized = finalize_receipt(
        receipt, directory=tmp_path, outcome=__import__("apar.evaluation.v3_runner", fromlist=["ExecutionOutcome"]).ExecutionOutcome(status="no_promotion")
    )
    assert finalized.terminal_status == "no_promotion"
    assert finalized.completed_at is not None
