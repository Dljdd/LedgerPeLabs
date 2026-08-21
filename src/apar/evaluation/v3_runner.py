"""One-attempt confirmatory runner for Defend v3.

Refuses execution unless every pre-execution check passes and an explicit
approval token matches the sealed freeze digest. Atomically writes the signed
receipt before running. Any failure terminates the attempt as no_promotion.
"""

from __future__ import annotations

import hashlib
import json
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from apar.evaluation.v3_isolation import IsolationCapabilityManifest
from apar.evaluation.v3_receipt import ExecutionReceipt, V3ReceiptError, has_receipt, write_receipt_atomically
from apar.evaluation.v3_runtime import DefenderRequest, V3RuntimeError, run_defender_arm
from apar.v3_protocol import V3ProtocolError


class V3RunnerError(V3ProtocolError):
    """Confirmatory execution was refused or terminated by a fail-closed check."""


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    protocol_id: str
    execution_nonce: str
    source_tree_sha256: str
    config_manifest_sha256: str
    defender_bundle_sha256: str
    population_manifest_sha256: str
    evaluator_key_id: str
    approval_token: str


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: Literal["no_promotion", "promotion_eligible", "failed"]
    reason: str | None = None


def verify_approval(
    inputs: ExecutionInputs,
    *,
    expected_freeze_digest: str,
) -> None:
    """Require an explicit approval token matching the sealed freeze digest."""
    if not hmac.compare_digest(inputs.approval_token, expected_freeze_digest):
        raise V3RunnerError("approval token does not match the sealed freeze digest")
    if len(inputs.approval_token) != 64:
        raise V3RunnerError("approval token must be a SHA-256 digest")


def create_receipt(
    inputs: ExecutionInputs,
    *,
    directory: Path,
) -> ExecutionReceipt:
    """Create and atomically persist the execution receipt before running."""
    if has_receipt(directory=directory):
        raise V3RunnerError("confirmatory attempt already consumed")
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


def execute_arms(
    inputs: ExecutionInputs,
    *,
    manifest: IsolationCapabilityManifest,
    arms: tuple[str, ...] = ("rules_only", "gbdt_only", "layered_hybrid"),
) -> tuple[ExecutionOutcome, ...]:
    """Execute all three arms through the isolated runtime; any failure stops."""
    outcomes: list[ExecutionOutcome] = []
    for arm in arms:
        request = DefenderRequest(
            arm=arm,
            protocol_id=inputs.protocol_id,
            execution_nonce=inputs.execution_nonce,
            input_payload=json.dumps(
                {
                    "arm": arm,
                    "protocol_id": inputs.protocol_id,
                    "execution_nonce": inputs.execution_nonce,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        try:
            run_defender_arm(request, manifest=manifest)
        except V3RuntimeError as error:
            outcomes.append(ExecutionOutcome(status="failed", reason=str(error)))
            break
        outcomes.append(ExecutionOutcome(status="no_promotion"))
    return tuple(outcomes)


def finalize_receipt(
    receipt: ExecutionReceipt,
    *,
    directory: Path,
    outcome: ExecutionOutcome,
) -> ExecutionReceipt:
    """Update the receipt with a terminal status after the attempt completes."""
    finalized = receipt.model_copy(
        update={
            "completed_at": datetime.now(UTC),
            "terminal_status": outcome.status,
        }
    )
    write_receipt_atomically(finalized, directory=directory)
    return finalized


__all__ = [
    "ExecutionInputs",
    "ExecutionOutcome",
    "V3RunnerError",
    "create_receipt",
    "execute_arms",
    "finalize_receipt",
    "verify_approval",
]
