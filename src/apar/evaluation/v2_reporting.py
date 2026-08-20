"""Canonical public scorecards for the unexecuted Defend v2 protocol.

This module renders supplied public status summaries only.  It does not build a
population, invoke an evaluator, or derive a metric.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
from pathlib import Path
from typing import Literal, Self, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.v2_preregistration import (
    SYNTHETIC_NON_CLAIM,
    ExecutionReceipt,
    SyntheticScope,
)
from apar.evaluation.v2_selection import V2GateOutcome
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

ArmName = Literal["rules_only", "gbdt_only", "layered_hybrid"]
_ARMS: tuple[ArmName, ArmName, ArmName] = (
    "rules_only",
    "gbdt_only",
    "layered_hybrid",
)
_STRATA = ("low", "medium", "high")
_HEX = frozenset("0123456789abcdef")
_NOT_EXECUTED: Literal["not_executed"] = "not_executed"
_NOT_EXECUTED_GATE = "NOT_EXECUTED"
_DEFAULT_PROTOCOL_DIGEST = "de91bbbe3f2a837da5145ff2a7fa767fd021f2ade6ef3655ec1ad4e503c6e46c"


class V2ReportingContractError(ValueError):
    """A public v2 scorecard is incomplete, noncanonical, or not trustworthy."""


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_signer_identity(key_id: str, public_key_base64: str) -> None:
    _digest(key_id, field="signer_key_id")
    if type(public_key_base64) is not str:
        raise ValueError("public signing key is invalid")
    try:
        public = base64.b64decode(public_key_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("public signing key is invalid") from error
    if len(public) != 32 or hashlib.sha256(public).hexdigest() != key_id:
        raise ValueError("public signing identity is inconsistent")


def _verify_signature(public_key_base64: str, document: object, signature_base64: str) -> bool:
    try:
        public = base64.b64decode(public_key_base64, validate=True)
        signature = base64.b64decode(signature_base64, validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical_json_bytes(document))
    except (InvalidSignature, TypeError, ValueError, binascii.Error):
        return False
    return True


class DefenseV2GateReport(ExternalContract):
    """Complete public gate outcome for one required defense arm."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    outcome: V2GateOutcome


class V2ArmScorecard(ExternalContract):
    """One comparator row, retained even when no evaluation has occurred."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    gate: DefenseV2GateReport

    @model_validator(mode="after")
    def gate_belongs_to_arm(self) -> Self:
        if self.gate.arm != self.arm:
            raise ValueError("arm gate report differs from its arm")
        return self


class _DefenseV2ScorecardFields(ExternalContract):
    """Shared status fields for an unsigned render request and signed scorecard."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    protocol_digest: str
    synthetic_scope: SyntheticScope
    arms: tuple[V2ArmScorecard, V2ArmScorecard, V2ArmScorecard]

    @field_validator("protocol_digest")
    @classmethod
    def protocol_digest_is_exact(cls, value: str) -> str:
        return _digest(value, field="protocol_digest")

    @field_validator("arms", mode="before")
    @classmethod
    def arms_are_an_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("scorecard arms must be an exact tuple")
        return value

    @model_validator(mode="after")
    def status_keeps_every_arm_and_gate_visible(self) -> Self:
        if tuple(item.arm for item in self.arms) != _ARMS:
            raise ValueError("scorecard arms must be complete and in stable order")
        if self.status == "promotion_eligible" and any(
            not item.gate.outcome.passed for item in self.arms
        ):
            raise ValueError("promotion_eligible scorecard cannot contain a failed gate")
        if self.status == _NOT_EXECUTED and any(
            item.status != _NOT_EXECUTED
            or item.gate.outcome.passed
            or item.gate.outcome.codes != (_NOT_EXECUTED_GATE,)
            for item in self.arms
        ):
            raise ValueError("not_executed scorecard must retain explicit unavailable gates")
        return self


class DefenseV2RenderResult(_DefenseV2ScorecardFields):
    """Public-only input accepted by the pure scorecard renderer."""


class DefenseV2Scorecard(_DefenseV2ScorecardFields):
    """Signed canonical overall result for the Defend v2 protocol."""

    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator("signer_key_id")
    @classmethod
    def signer_key_id_is_exact(cls, value: str) -> str:
        return _digest(value, field="signer_key_id")

    @model_validator(mode="after")
    def signature_is_valid(self) -> Self:
        _validate_signer_identity(self.signer_key_id, self.public_key_base64)
        if not _verify_signature(
            self.public_key_base64, self.unsigned_document(), self.signature_base64
        ):
            raise ValueError("scorecard signature is invalid")
        return self

    def unsigned_document(self) -> dict[str, object]:
        """Return exactly the document covered by the publication signature."""
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def to_json(self) -> bytes:
        """Return strict canonical JSON after semantic and signature revalidation."""
        try:
            checked = DefenseV2Scorecard.model_validate(
                self.model_dump(mode="python", warnings=False), strict=True
            )
            return canonical_json_bytes(checked.model_dump(mode="json"))
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise V2ReportingContractError("scorecard failed semantic revalidation") from error

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        """Load only an exact canonical, self-verifying public scorecard."""
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise V2ReportingContractError("scorecard must be a JSON object")
            card = cls.model_validate(_tupleize_scorecard_document(document))
            if card.to_json() != payload:
                raise V2ReportingContractError("scorecard JSON is not canonical")
            return card
        except (WireContractError, ValidationError, ValueError, TypeError) as error:
            if isinstance(error, V2ReportingContractError):
                raise
            raise V2ReportingContractError(str(error)) from error


def not_executed_result(
    *, protocol_digest: str = _DEFAULT_PROTOCOL_DIGEST
) -> DefenseV2RenderResult:
    """Return the stable public state before a v2 evaluation is authorized."""
    return DefenseV2RenderResult(
        status=_NOT_EXECUTED,
        protocol_digest=protocol_digest,
        synthetic_scope=SYNTHETIC_NON_CLAIM,
        arms=cast(
            tuple[V2ArmScorecard, V2ArmScorecard, V2ArmScorecard],
            tuple(
                V2ArmScorecard(
                    arm=arm,
                    status=_NOT_EXECUTED,
                    gate=DefenseV2GateReport(
                        arm=arm,
                        outcome=V2GateOutcome(passed=False, codes=(_NOT_EXECUTED_GATE,)),
                    ),
                )
                for arm in _ARMS
            ),
        ),
    )


def load_current_v2_scorecard(root: Path, *, fallback: DefenseV2Scorecard) -> DefenseV2Scorecard:
    """Read the current signed scorecard while reconciling durable execution receipts."""
    if not isinstance(root, Path) or type(fallback) is not DefenseV2Scorecard:
        raise V2ReportingContractError("current scorecard lookup requires exact inputs")
    if fallback.protocol_digest != _DEFAULT_PROTOCOL_DIGEST:
        raise V2ReportingContractError("fallback scorecard protocol digest is invalid")
    state_root = root / ".apar"
    scorecard_path = state_root / "defense-v2" / "defense-v2-scorecard.json"
    try:
        receipt_present = False
        if state_root.exists():
            if not state_root.is_dir():
                raise V2ReportingContractError("durable V2 state is not a directory")
            for path in state_root.rglob("*"):
                if not path.is_file() or path == scorecard_path:
                    continue
                try:
                    document = strict_json_loads(path.read_bytes())
                    receipt = ExecutionReceipt.model_validate(document)
                except (OSError, WireContractError, ValidationError, ValueError, TypeError):
                    continue
                if receipt.preregistration_id == "apar-defend-v2":
                    receipt_present = True
                    break

        current = (
            DefenseV2Scorecard.from_json(scorecard_path.read_bytes())
            if scorecard_path.is_file()
            else None
        )
        if current is not None and current.protocol_digest != _DEFAULT_PROTOCOL_DIGEST:
            raise V2ReportingContractError("current scorecard protocol digest is invalid")
        if receipt_present:
            if current is None or current.status == _NOT_EXECUTED:
                raise V2ReportingContractError(
                    "execution receipt requires a verified completed scorecard"
                )
            return current
        if current is not None:
            if current.status != _NOT_EXECUTED:
                raise V2ReportingContractError(
                    "completed scorecard requires a durable execution receipt"
                )
            return current
        return fallback
    except (OSError, ValidationError, ValueError, TypeError) as error:
        if isinstance(error, V2ReportingContractError):
            raise
        raise V2ReportingContractError("current V2 scorecard state is invalid") from error


def render_v2_scorecard(
    result: DefenseV2RenderResult, *, signer: RunSigningIdentity
) -> tuple[DefenseV2Scorecard, dict[str, bytes]]:
    """Sign and render the five public v2 artifacts without executing anything."""
    if type(result) is not DefenseV2RenderResult:
        raise V2ReportingContractError("scorecard result must be an exact public render result")
    if type(signer) is not RunSigningIdentity:
        raise V2ReportingContractError("scorecard signer must be an exact RunSigningIdentity")
    try:
        checked = DefenseV2RenderResult.model_validate(
            result.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise V2ReportingContractError(str(error)) from error
    unsigned = {
        **checked.model_dump(mode="json"),
        "signer_key_id": signer.key_id,
        "public_key_base64": signer.public_key_base64,
    }
    signature = signer.sign(unsigned)
    card = DefenseV2Scorecard.model_validate(
        _tupleize_scorecard_document({**unsigned, "signature_base64": signature})
    )
    files = {
        "defense-v2-scorecard.json": card.to_json(),
        "defense-v2-arm-metrics.csv": _arm_metrics_csv(card),
        "defense-v2-workload.csv": _workload_csv(card),
        "defense-v2-gates.json": _gates_json(card),
        "defense-v2-limitations.md": _limitations_markdown(card),
    }
    return card, files


def _arm_metrics_csv(card: DefenseV2Scorecard) -> bytes:
    return _csv_bytes(
        (
            "arm",
            "status",
            "population",
            "stratum",
            "slice",
            "metric_status",
        ),
        (
            (
                arm.arm,
                arm.status,
                "not_evaluated",
                "not_evaluated",
                "not_evaluated",
                "not_available",
            )
            for arm in card.arms
        ),
    )


def _workload_csv(card: DefenseV2Scorecard) -> bytes:
    return _csv_bytes(
        (
            "arm",
            "operating_stratum",
            "day",
            "review_cases",
            "reviewed_transactions",
            "review_case_rate",
            "review_transaction_rate",
            "status",
        ),
        (
            (arm.arm, stratum, "not_evaluated", "", "", "", "", arm.status)
            for arm in card.arms
            for stratum in _STRATA
        ),
    )


def _gates_json(card: DefenseV2Scorecard) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "2.0.0",
            "status": card.status,
            "gates": [arm.gate.model_dump(mode="json") for arm in card.arms],
        }
    )


def _limitations_markdown(card: DefenseV2Scorecard) -> bytes:
    return (
        "# Defend v2 limitations\n\n"
        f"{card.synthetic_scope}\n\n"
        "No evaluation has been executed for this scorecard state. The displayed "
        "arm and gate rows are retained so unavailable evidence and failed gates "
        "cannot be concealed.\n"
    ).encode()


def _csv_bytes(header: tuple[str, ...], rows: object) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)  # type: ignore[arg-type]
    return stream.getvalue().encode("utf-8")


def _tupleize_scorecard_document(document: dict[str, object]) -> dict[str, object]:
    """Convert JSON array shapes to the exact immutable contract collections."""
    prepared = dict(document)
    arms = prepared.get("arms")
    if type(arms) is list:
        prepared_arms = []
        for item in arms:
            arm = dict(item) if type(item) is dict else item
            if type(arm) is dict:
                gate = arm.get("gate")
                if type(gate) is dict:
                    prepared_gate = dict(gate)
                    outcome = prepared_gate.get("outcome")
                    if type(outcome) is dict:
                        prepared_outcome = dict(outcome)
                        if type(prepared_outcome.get("codes")) is list:
                            prepared_outcome["codes"] = tuple(prepared_outcome["codes"])
                        prepared_gate["outcome"] = prepared_outcome
                    arm["gate"] = prepared_gate
            prepared_arms.append(arm)
        prepared["arms"] = tuple(prepared_arms)
    return prepared


__all__ = [
    "DefenseV2GateReport",
    "DefenseV2RenderResult",
    "DefenseV2Scorecard",
    "load_current_v2_scorecard",
    "V2ArmScorecard",
    "V2ReportingContractError",
    "not_executed_result",
    "render_v2_scorecard",
]
