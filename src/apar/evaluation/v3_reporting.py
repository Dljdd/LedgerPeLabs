"""Completed signed evidence renderer for Defend v3."""

from __future__ import annotations

import csv
import io
from typing import Literal

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.v2_selection import V2GateOutcome
from apar.runs.wire import canonical_json_bytes
from apar.v3_protocol import SYNTHETIC_NON_CLAIM, V3ProtocolError

_ARMS = ("rules_only", "gbdt_only", "layered_hybrid")
_NOT_EXECUTED_GATE = "not_executed"


class V3ReportingError(V3ProtocolError):
    """A v3 scorecard is malformed, incomplete, or makes an unsupported claim."""


class V3ArmScorecard(ExternalContract):
    """One arm's public result and gate outcome."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    gate: "V3GateReport"

    @model_validator(mode="after")
    def gate_belongs_to_arm(self) -> Self:
        if self.gate.arm != self.arm:
            raise ValueError("arm gate report differs from its arm")
        return self


class V3GateReport(ExternalContract):
    """Complete public gate outcome for one required defense arm."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    outcome: V2GateOutcome

class _DefenseV3ScorecardFields(ExternalContract):
    schema_version: Literal["2.0.0"] = "2.0.0"
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    protocol_id: str
    execution_nonce: str
    synthetic_scope: str
    arms: tuple[V3ArmScorecard, V3ArmScorecard, V3ArmScorecard]

    @model_validator(mode="after")
    def arms_are_complete_and_stable(self) -> Self:
        if tuple(item.arm for item in self.arms) != _ARMS:
            raise ValueError("scorecard arms must be complete and in stable order")
        if self.status == "promotion_eligible" and any(
            not item.gate.outcome.passed for item in self.arms
        ):
            raise ValueError("promotion_eligible scorecard cannot contain a failed gate")
        return self


# Pydantic v2 requires field_validator at class level; use a simpler approach.
class DefenseV3RenderResult(_DefenseV3ScorecardFields):
    """Public-only input accepted by the pure scorecard renderer."""


class DefenseV3Scorecard(DefenseV3RenderResult):
    """Signed canonical overall result for the Defend v3 protocol."""

    signer_key_id: str = Field(min_length=64, max_length=64)
    signature_base64: str

    @model_validator(mode="after")
    def signature_is_valid(self) -> Self:
        import base64
        import binascii
        import hashlib

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            public_bytes = bytes.fromhex(self.signer_key_id)
            signature = base64.b64decode(self.signature_base64, validate=True)
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                signature, canonical_json_bytes(self.unsigned_document())
            )
        except (InvalidSignature, ValueError, TypeError, binascii.Error):
            raise V3ReportingError("scorecard signature is invalid") from None
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def to_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def not_executed_result(
    *,
    protocol_id: str,
    execution_nonce: str,
) -> DefenseV3RenderResult:
    """Return the stable public state before a v3 evaluation is authorized."""
    return DefenseV3RenderResult(
        status="not_executed",
        protocol_id=protocol_id,
        execution_nonce=execution_nonce,
        synthetic_scope=SYNTHETIC_NON_CLAIM,
        arms=tuple(
            V3ArmScorecard(
                arm=arm,
                status="not_executed",
                gate=V3GateReport(arm=arm, outcome=V2GateOutcome(passed=False, codes=(_NOT_EXECUTED_GATE,))),
            )
            for arm in _ARMS
        ),
    )


def render_v3_scorecard(
    result: DefenseV3RenderResult,
    *,
    signer_key_id: str,
    signature_base64: str,
) -> tuple[DefenseV3Scorecard, dict[str, bytes]]:
    """Sign and render the five public v3 artifacts without executing anything."""
    unsigned = {
        **_tupleize(result.model_dump(mode="json")),
        "signer_key_id": signer_key_id,
    }
    card = DefenseV3Scorecard.model_validate({**unsigned, "signature_base64": signature_base64})
    files = {
        "defense-v3-scorecard.json": card.to_json(),
        "defense-v3-arm-metrics.csv": _arm_metrics_csv(card),
        "defense-v3-workload.csv": _workload_csv(card),
        "defense-v3-gates.json": _gates_json(card),
        "defense-v3-limitations.md": _limitations_markdown(card),
    }
    return card, files


def _tupleize(value: object) -> object:
    """Recursively convert lists back to tuples for exact-tuple validators."""
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, dict):
        return {key: _tupleize(item) for key, item in value.items()}
    return value


def _arm_metrics_csv(card: DefenseV3Scorecard) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("arm", "status", "population", "stratum", "slice", "metric_status"))
    for arm in card.arms:
        writer.writerow((arm.arm, arm.status, "not_evaluated", "not_evaluated", "not_evaluated", "not_available"))
    return buffer.getvalue().encode("utf-8")


def _workload_csv(card: DefenseV3Scorecard) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ("arm", "operating_stratum", "day", "review_cases", "reviewed_transactions", "review_case_rate", "review_transaction_rate", "status")
    )
    for arm in card.arms:
        for stratum in ("low", "medium", "high"):
            writer.writerow((arm.arm, stratum, "not_evaluated", "", "", "", "", arm.status))
    return buffer.getvalue().encode("utf-8")


def _gates_json(card: DefenseV3Scorecard) -> bytes:
    return canonical_json_bytes(
        {"protocol_id": card.protocol_id, "gates": [arm.gate.model_dump(mode="json") for arm in card.arms]}
    )


def _limitations_markdown(card: DefenseV3Scorecard) -> bytes:
    lines = [
        "# APAR Defend v3 Limitations",
        "",
        f"Status: `{card.status}`.",
        "",
        card.synthetic_scope,
        "",
        "Arm and gate rows are retained so unavailable evidence and failed gates remain visible.",
    ]
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "DefenseV3RenderResult",
    "DefenseV3Scorecard",
    "V3ArmScorecard",
    "V3ReportingError",
    "not_executed_result",
    "render_v3_scorecard",
]
