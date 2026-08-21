"""Signed scorecard generation for Defend v4."""

from __future__ import annotations

import base64
import binascii
import csv
import io
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.v2_selection import V2GateOutcome
from apar.evaluation.v4_gate_evaluation import ArmPromotionResult
from apar.runs.wire import canonical_json_bytes
from apar.v4_protocol import SYNTHETIC_NON_CLAIM, V4ProtocolError

_ARMS = ("rules_only", "gbdt_only", "layered_hybrid")


class V4PublicationError(V4ProtocolError):
    """A v4 scorecard is malformed, incomplete, or makes an unsupported claim."""


class V4ArmScorecard(ExternalContract):
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    gate_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_is_coherent(self) -> Self:
        from typing import Self as _Self
        if self.status == "promotion_eligible" and self.gate_codes:
            raise ValueError("promotion_eligible arm cannot carry failed gate codes")
        return self


class DefenseV4RenderResult(ExternalContract):
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    protocol_id: str
    execution_nonce: str
    synthetic_scope: str
    arms: tuple[V4ArmScorecard, V4ArmScorecard, V4ArmScorecard]

    @model_validator(mode="after")
    def arms_are_complete_and_stable(self) -> Self:
        if tuple(item.arm for item in self.arms) != _ARMS:
            raise ValueError("scorecard arms must be complete and in stable order")
        if self.status == "promotion_eligible" and any(
            item.gate_codes for item in self.arms
        ):
            raise ValueError("promotion_eligible scorecard cannot contain a failed arm")
        return self


class DefenseV4Scorecard(DefenseV4RenderResult):
    signer_key_id: str
    signature_base64: str

    @model_validator(mode="after")
    def signature_is_valid(self) -> Self:
        try:
            public_bytes = bytes.fromhex(self.signer_key_id)
            signature = base64.b64decode(self.signature_base64, validate=True)
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                signature, canonical_json_bytes(self.unsigned_document())
            )
        except (InvalidSignature, ValueError, TypeError, binascii.Error):
            raise V4PublicationError("scorecard signature is invalid") from None
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def to_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def not_executed_result(
    *,
    protocol_id: str,
    execution_nonce: str,
) -> DefenseV4RenderResult:
    return DefenseV4RenderResult(
        status="not_executed",
        protocol_id=protocol_id,
        execution_nonce=execution_nonce,
        synthetic_scope=SYNTHETIC_NON_CLAIM,
        arms=tuple(
            V4ArmScorecard(arm=arm, status="not_executed", gate_codes=("not_executed",))
            for arm in _ARMS
        ),
    )


def from_gate_results(
    *,
    protocol_id: str,
    execution_nonce: str,
    results: tuple[ArmPromotionResult, ...],
) -> DefenseV4RenderResult:
    """Build a render result from evaluated gate results."""
    if len(results) != 3:
        raise V4PublicationError("gate results must contain exactly three arms")
    arms = tuple(
        V4ArmScorecard(
            arm=result.arm,
            status="promotion_eligible" if result.gate_outcome.passed else "no_promotion",
            gate_codes=result.gate_outcome.codes,
        )
        for result in results
    )
    overall = "promotion_eligible" if all(r.gate_outcome.passed for r in results) else "no_promotion"
    return DefenseV4RenderResult(
        status=overall,
        protocol_id=protocol_id,
        execution_nonce=execution_nonce,
        synthetic_scope=SYNTHETIC_NON_CLAIM,
        arms=arms,
    )


def render_v4_scorecard(
    result: DefenseV4RenderResult,
    *,
    signer_key_id: str,
    signature_base64: str,
) -> tuple[DefenseV4Scorecard, dict[str, bytes]]:
    unsigned = {
        **_tupleize(result.model_dump(mode="json")),
        "signer_key_id": signer_key_id,
    }
    card = DefenseV4Scorecard.model_validate({**unsigned, "signature_base64": signature_base64})
    files = {
        "defense-v4-scorecard.json": card.to_json(),
        "defense-v4-arm-metrics.csv": _arm_metrics_csv(card),
        "defense-v4-workload.csv": _workload_csv(card),
        "defense-v4-limitations.md": _limitations_markdown(card),
    }
    return card, files


def _tupleize(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, dict):
        return {key: _tupleize(item) for key, item in value.items()}
    return value


def _arm_metrics_csv(card: DefenseV4Scorecard) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "arm", "status",
            "precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "brier", "fpr",
            "challenge_rate", "false_decline_rate", "review_case_rate",
            "false_interventions_per_10k", "captured_value_fraction", "escaped_value_fraction",
            "time_to_alert_p95_seconds", "p95_decision_latency_ms",
        )
    )
    for arm in card.arms:
        writer.writerow(
            (
                arm.arm, arm.status,
                "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "",
            )
        )
    return buffer.getvalue().encode("utf-8")


def _workload_csv(card: DefenseV4Scorecard) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ("arm", "operating_stratum", "day", "review_cases", "reviewed_transactions", "review_case_rate", "review_transaction_rate", "status")
    )
    for arm in card.arms:
        for stratum in ("low", "medium", "high"):
            writer.writerow((arm.arm, stratum, "not_evaluated", "", "", "", "", arm.status))
    return buffer.getvalue().encode("utf-8")


def _limitations_markdown(card: DefenseV4Scorecard) -> bytes:
    lines = [
        "# APAR Defend v4 Limitations",
        "",
        f"Status: `{card.status}`.",
        "",
        card.synthetic_scope,
        "",
        "Arm and gate rows are retained so unavailable evidence and failed gates remain visible.",
    ]
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "DefenseV4RenderResult",
    "DefenseV4Scorecard",
    "V4ArmScorecard",
    "V4PublicationError",
    "from_gate_results",
    "not_executed_result",
    "render_v4_scorecard",
]
