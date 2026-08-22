"""One-attempt confirmatory runner for Defend v4.

Wires actual scoring, metric computation, gate evaluation, and scorecard
publication into the v3 execution boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from apar.defense.contracts import ObservedEvent
from apar.evaluation.v3_isolation import IsolationCapabilityManifest
from apar.evaluation.v3_receipt import ExecutionReceipt, has_receipt, write_receipt_atomically
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.v4_gate_evaluation import (
    ArmPromotionResult,
    evaluate_gates,
    project_gate_evidence,
)
from apar.evaluation.v4_scoring import FrozenDefenderBundle
from apar.evaluation.v4_scoring import ArmScoredDecision
from apar.evaluation.v4_scoring import score_arm
from apar.v4_protocol import V4GateValues, V4ProtocolError


class V4RunnerError(V4ProtocolError):
    """Confirmatory execution was refused or terminated by a fail-closed check."""


@dataclass(frozen=True, slots=True)
class V4ExecutionInputs:
    protocol_id: str
    execution_nonce: str
    source_tree_sha256: str
    config_manifest_sha256: str
    defender_bundle_sha256: str
    population_manifest_sha256: str
    evaluator_key_id: str
    approval_token: str


@dataclass(frozen=True, slots=True)
class V4ExecutionOutcome:
    status: str
    render_result: DefenseV4RenderResult | None = None
    reason: str | None = None


def verify_v4_approval(
    inputs: V4ExecutionInputs,
    *,
    expected_freeze_digest: str,
) -> None:
    if not hmac.compare_digest(inputs.approval_token, expected_freeze_digest):
        raise V4RunnerError("approval token does not match the sealed freeze digest")


def create_v4_receipt(
    inputs: V4ExecutionInputs,
    *,
    directory: Path,
) -> ExecutionReceipt:
    if has_receipt(directory=directory):
        raise V4RunnerError("confirmatory attempt already consumed")
    from datetime import UTC, datetime

    receipt = ExecutionReceipt(
        protocol_id=inputs.protocol_id,
        execution_nonce=inputs.execution_nonce,
        source_tree_sha256=inputs.source_tree_sha256,
        config_manifest_sha256=inputs.config_manifest_sha256,
        defender_bundle_sha256=inputs.defender_bundle_sha256,
        population_manifest_sha256=inputs.population_manifest_sha256,
        evaluator_key_id=inputs.evaluator_key_id,
        started_at=datetime.now(UTC),
        terminal_status="running",
    )
    write_receipt_atomically(receipt, directory=directory)
    return receipt


def execute_v4_arms(
    inputs: V4ExecutionInputs,
    *,
    observations: tuple[ObservedEvent, ...],
    truth: tuple[EvaluationTruthRow, ...],
    observations_sha256: str,
    truth_sha256: str,
    gates: V4GateValues,
    bundle: FrozenDefenderBundle,
) -> tuple[ArmPromotionResult, ...]:
    """Score all three arms, evaluate gates, and return promotion results."""
    results: list[ArmPromotionResult] = []
    for arm in ("rules_only", "gbdt_only", "layered_hybrid"):
        try:
            decisions = score_arm(
                arm,
                observations,
                bundle=bundle,
                truth=(),
                observations_sha256=observations_sha256,
                truth_sha256=truth_sha256,
            )
        except Exception as error:
            raise V4RunnerError(f"scoring failed for arm {arm}: {error}") from error

        evidence = project_gate_evidence(decisions, truth, arm=arm)
        result = evaluate_gates(evidence, gates=gates)
        results.append(result)
    return tuple(results)


def finalize_v4_receipt(
    receipt: ExecutionReceipt,
    *,
    directory: Path,
    status: str,
) -> ExecutionReceipt:
    from datetime import UTC, datetime
    from typing import Literal

    valid_statuses = ("no_promotion", "promotion_eligible", "failed")
    if status not in valid_statuses:
        raise V4RunnerError(f"invalid terminal status: {status}")
    finalized = receipt.model_copy(
        update={
            "completed_at": datetime.now(UTC),
            "terminal_status": status,
        }
    )
    write_receipt_atomically(finalized, directory=directory)
    return finalized


__all__ = [
    "V4ExecutionInputs",
    "V4ExecutionOutcome",
    "V4RunnerError",
    "create_v4_receipt",
    "execute_v4_arms",
    "finalize_v4_receipt",
    "verify_v4_approval",
]
