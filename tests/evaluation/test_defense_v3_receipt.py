"""Atomic receipt and one-attempt semantics tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apar.evaluation.v3_receipt import (
    ExecutionReceipt,
    V3ReceiptError,
    has_receipt,
    read_receipt,
    write_receipt_atomically,
)


def _receipt() -> ExecutionReceipt:
    return ExecutionReceipt(
        protocol_id="apar-defend-v3",
        execution_nonce="a" * 64,
        source_tree_sha256="b" * 64,
        config_manifest_sha256="c" * 64,
        defender_bundle_sha256="d" * 64,
        population_manifest_sha256="e" * 64,
        evaluator_key_id="f" * 64,
        started_at=datetime(2026, 8, 21, 12, tzinfo=UTC),
    )


def test_write_and_read_roundtrip(tmp_path) -> None:
    write_receipt_atomically(_receipt(), directory=tmp_path)
    assert has_receipt(directory=tmp_path)
    loaded = read_receipt(directory=tmp_path)
    assert loaded is not None
    assert loaded.protocol_id == "apar-defend-v3"


def test_absent_receipt_returns_none(tmp_path) -> None:
    assert not has_receipt(directory=tmp_path)
    assert read_receipt(directory=tmp_path) is None


def test_receipt_is_canonical_json(tmp_path) -> None:
    receipt = _receipt()
    write_receipt_atomically(receipt, directory=tmp_path)
    raw = (tmp_path / "execution-receipt.json").read_bytes()
    import json

    document = json.loads(raw)
    assert json.dumps(document, sort_keys=True, separators=(",", ":")).encode() == raw


def test_completed_at_must_not_precede_started_at() -> None:
    with pytest.raises(ValueError, match="completed_at must not precede"):
        ExecutionReceipt(
            protocol_id="apar-defend-v3",
            execution_nonce="a" * 64,
            source_tree_sha256="b" * 64,
            config_manifest_sha256="c" * 64,
            defender_bundle_sha256="d" * 64,
            population_manifest_sha256="e" * 64,
            evaluator_key_id="f" * 64,
            started_at=datetime(2026, 8, 21, 12, tzinfo=UTC),
            completed_at=datetime(2026, 8, 21, 11, tzinfo=UTC),
        )
