"""Signed scorecard publication tests for Defend v4."""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apar.evaluation.v2_selection import V2GateOutcome
from apar.evaluation.v4_gate_evaluation import ArmGateEvidence, ArmPromotionResult, evaluate_gates
from apar.evaluation.v4_publication import (
    DefenseV4RenderResult,
    V4PublicationError,
    from_gate_results,
    not_executed_result,
    render_v4_scorecard,
)
from apar.runs.wire import canonical_json_bytes
from apar.v4_protocol import V4GateValues


def _signer() -> tuple[str, Ed25519PrivateKey]:
    key = Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return public.hex(), key


def test_not_executed_lists_all_arms() -> None:
    result = not_executed_result(protocol_id="apar-defend-v4", execution_nonce="a" * 64)
    assert result.status == "not_executed"
    assert [arm.arm for arm in result.arms] == ["rules_only", "gbdt_only", "layered_hybrid"]


def test_promotion_eligible_rejects_failed_arm() -> None:
    with pytest.raises(ValueError, match="cannot contain a failed arm"):
        DefenseV4RenderResult(
            status="promotion_eligible",
            protocol_id="apar-defend-v4",
            execution_nonce="a" * 64,
            synthetic_scope="Synthetic-only evaluation; not a real-world prevalence or external-validity claim.",
            arms=(
                __import__("apar.evaluation.v4_publication", fromlist=["V4ArmScorecard"]).V4ArmScorecard(arm="rules_only", status="no_promotion", gate_codes=("FAMILY_COVERAGE",)),
                __import__("apar.evaluation.v4_publication", fromlist=["V4ArmScorecard"]).V4ArmScorecard(arm="gbdt_only", status="promotion_eligible"),
                __import__("apar.evaluation.v4_publication", fromlist=["V4ArmScorecard"]).V4ArmScorecard(arm="layered_hybrid", status="promotion_eligible"),
            ),
        )


def test_from_gate_results_produces_no_promotion_on_failure() -> None:
    gates = V4GateValues()
    results = tuple(
        evaluate_gates(
            ArmGateEvidence(arm=arm),
            gates=gates,
        )
        for arm in ("rules_only", "gbdt_only", "layered_hybrid")
    )
    result = from_gate_results(
        protocol_id="apar-defend-v4",
        execution_nonce="a" * 64,
        results=results,
    )
    assert result.status == "no_promotion"


def test_render_and_verify_signature() -> None:
    result = not_executed_result(protocol_id="apar-defend-v4", execution_nonce="a" * 64)
    signer_hex, key = _signer()
    unsigned = {**result.model_dump(mode="json"), "signer_key_id": signer_hex}
    signature = base64.b64encode(key.sign(canonical_json_bytes(unsigned))).decode("ascii")
    card, files = render_v4_scorecard(result, signer_key_id=signer_hex, signature_base64=signature)
    assert card.to_json() == files["defense-v4-scorecard.json"]
    assert b"Synthetic-only evaluation" in files["defense-v4-limitations.md"]


def test_workload_csv_has_both_denominators() -> None:
    result = not_executed_result(protocol_id="apar-defend-v4", execution_nonce="a" * 64)
    signer_hex, key = _signer()
    unsigned = {**result.model_dump(mode="json"), "signer_key_id": signer_hex}
    signature = base64.b64encode(key.sign(canonical_json_bytes(unsigned))).decode("ascii")
    _, files = render_v4_scorecard(result, signer_key_id=signer_hex, signature_base64=signature)
    assert b"review_cases,reviewed_transactions,review_case_rate,review_transaction_rate" in files["defense-v4-workload.csv"].splitlines()[0]
