"""Completed signed evidence renderer tests for Defend v3."""

from __future__ import annotations

import base64
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apar.evaluation.v2_selection import V2GateOutcome
from apar.evaluation.v3_reporting import (
    DefenseV3RenderResult,
    V3GateReport,
    V3ArmScorecard,
    V3ReportingError,
    not_executed_result,
    render_v3_scorecard,
)


def _signer_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)


def _signer_identity() -> tuple[str, str]:
    key = _signer_key()
    public = key.public_key().public_bytes(
        __import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding"]).Encoding.Raw,
        __import__("cryptography.hazmat.primitives.serialization", fromlist=["PublicFormat"]).PublicFormat.Raw,
    )
    return public.hex(), key


def test_not_executed_lists_all_arms_without_hidden_values() -> None:
    result = not_executed_result(protocol_id="apar-defend-v3", execution_nonce="a" * 64)
    assert result.status == "not_executed"
    assert [arm.arm for arm in result.arms] == ["rules_only", "gbdt_only", "layered_hybrid"]
    assert all(arm.gate.outcome.codes == ("not_executed",) for arm in result.arms)


def test_promotion_eligible_rejects_failed_gate() -> None:
    arms = (
        V3ArmScorecard(arm="rules_only", status="promotion_eligible", gate=V3GateReport(arm="rules_only", outcome=V2GateOutcome(passed=True))),
        V3ArmScorecard(arm="gbdt_only", status="promotion_eligible", gate=V3GateReport(arm="gbdt_only", outcome=V2GateOutcome(passed=False, codes=("CONTROL_INVALID",)))),
        V3ArmScorecard(arm="layered_hybrid", status="promotion_eligible", gate=V3GateReport(arm="layered_hybrid", outcome=V2GateOutcome(passed=True))),
    )
    with pytest.raises(ValueError, match="cannot contain a failed gate"):
        DefenseV3RenderResult(
            status="promotion_eligible",
            protocol_id="apar-defend-v3",
            execution_nonce="a" * 64,
            synthetic_scope="Synthetic-only evaluation; not a real-world prevalence or external-validity claim.",
            arms=arms,
        )


def test_workload_csv_has_both_denominators() -> None:
    result = not_executed_result(protocol_id="apar-defend-v3", execution_nonce="a" * 64)
    signer_hex, key = _signer_identity()
    unsigned = {**result.model_dump(mode="json"), "signer_key_id": signer_hex}
    signature = base64.b64encode(key.sign(__import__("apar.runs.wire", fromlist=["canonical_json_bytes"]).canonical_json_bytes(unsigned))).decode("ascii")
    _, files = render_v3_scorecard(result, signer_key_id=signer_hex, signature_base64=signature)
    assert b"review_cases,reviewed_transactions,review_case_rate,review_transaction_rate" in files["defense-v3-workload.csv"].splitlines()[0]


def test_limitations_contain_synthetic_non_claim() -> None:
    result = not_executed_result(protocol_id="apar-defend-v3", execution_nonce="a" * 64)
    signer_hex, key = _signer_identity()
    unsigned = {**result.model_dump(mode="json"), "signer_key_id": signer_hex}
    signature = base64.b64encode(key.sign(__import__("apar.runs.wire", fromlist=["canonical_json_bytes"]).canonical_json_bytes(unsigned))).decode("ascii")
    _, files = render_v3_scorecard(result, signer_key_id=signer_hex, signature_base64=signature)
    text = files["defense-v3-limitations.md"].decode("utf-8")
    assert "Synthetic-only evaluation" in text
    assert "not a real-world prevalence" in text
