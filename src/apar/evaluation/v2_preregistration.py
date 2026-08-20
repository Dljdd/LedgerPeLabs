"""Canonical, signed admission contracts for the unexecuted Defend v2 protocol.

This module deliberately seals only intent.  It cannot generate a population or
invoke an evaluator; a later evaluator-owned path must present the resulting
admission and record its one permitted confirmatory receipt.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_protocol import SeedCommitment
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_HEX = frozenset("0123456789abcdef")
SyntheticScope = Literal[
    "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
]
_SEED_NAMES = ("operating_population", "campaign_injection")
SYNTHETIC_NON_CLAIM: SyntheticScope = (
    "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
)
TRUSTED_V2_PREREGISTRATION_ID = "apar-defend-v2"
TRUSTED_V2_EXECUTION_NONCE = "4b4a92405a8c84ae5035bcbc510e06e1729238ffe0e78f106515d78d3c63c98a"
TRUSTED_V2_EVALUATOR_KEY_ID = "de52c5b7d396405990b8df80875bad49a2e845b06a576e668fb1ea292a55ee7e"
TRUSTED_V2_EVALUATOR_PUBLIC_KEY_BASE64 = "tfkAlTGPPwRM2OQLWEJT2Gd3vmK/ByITNOX9IZnPp/c="


class V2PreregistrationError(ValueError):
    """Raised when a v2 preregistration is incomplete, noncanonical, or untrusted."""


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _public_key(key_id: str, public_key_base64: str) -> Ed25519PublicKey:
    _digest(key_id, field="evaluator_key_id")
    if type(public_key_base64) is not str:
        raise ValueError("evaluator_public_key_base64 is invalid")
    try:
        public = base64.b64decode(public_key_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("evaluator_public_key_base64 is invalid") from error
    if len(public) != 32 or hashlib.sha256(public).hexdigest() != key_id:
        raise ValueError("evaluator public key identity is inconsistent")
    return Ed25519PublicKey.from_public_bytes(public)


class V2Preregistration(ExternalContract):
    """Every immutable input to the one permitted synthetic v2 evaluation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    preregistration_id: str
    protocol_profile_sha256: str
    manifest_registry_sha256: str
    source_manifest_sha256: str
    feature_manifest_sha256: str
    candidate_grid_sha256: str
    population_manifest_sha256: str
    seed_commitments: tuple[SeedCommitment, ...]
    evaluator_key_id: str
    evaluator_public_key_base64: str
    evaluator_capability_sha256: str
    metrics_manifest_sha256: str
    bootstrap_manifest_sha256: str
    controls_manifest_sha256: str
    budget_manifest_sha256: str
    reporting_schema_sha256: str
    fidelity_validation_bundle_sha256: str | None = None
    synthetic_scope: SyntheticScope
    synthetic_scope_sha256: str
    execution_nonce: str
    maximum_confirmatory_attempts: Literal[1] = 1
    signature_base64: str

    @field_validator(
        "source_manifest_sha256",
        "protocol_profile_sha256",
        "manifest_registry_sha256",
        "feature_manifest_sha256",
        "candidate_grid_sha256",
        "population_manifest_sha256",
        "evaluator_key_id",
        "evaluator_capability_sha256",
        "metrics_manifest_sha256",
        "bootstrap_manifest_sha256",
        "controls_manifest_sha256",
        "budget_manifest_sha256",
        "reporting_schema_sha256",
        "synthetic_scope_sha256",
        "execution_nonce",
    )
    @classmethod
    def digest_fields_are_exact(cls, value: str, info: Any) -> str:
        return _digest(value, field=info.field_name)

    @field_validator("fidelity_validation_bundle_sha256")
    @classmethod
    def fidelity_bundle_is_pinned_if_present(cls, value: str | None) -> str | None:
        return None if value is None else _digest(value, field="fidelity_validation_bundle_sha256")

    @field_validator("preregistration_id")
    @classmethod
    def preregistration_id_is_nonempty(cls, value: str) -> str:
        if type(value) is not str or not value:
            raise ValueError("preregistration_id must be non-empty")
        return value

    @field_validator("seed_commitments")
    @classmethod
    def evaluator_seed_commitments_are_complete(
        cls, value: tuple[SeedCommitment, ...]
    ) -> tuple[SeedCommitment, ...]:
        if (
            type(value) is not tuple
            or tuple(item.name for item in value) != _SEED_NAMES
            or any(type(item) is not SeedCommitment for item in value)
        ):
            raise ValueError("seed_commitments must bind named evaluator-held seeds")
        return value

    @model_validator(mode="after")
    def manifest_bindings_are_closed(self) -> Self:
        _public_key(self.evaluator_key_id, self.evaluator_public_key_base64)
        expected_scope_digest = hashlib.sha256(
            canonical_json_bytes(self.synthetic_scope)
        ).hexdigest()
        if self.synthetic_scope_sha256 != expected_scope_digest:
            raise ValueError("synthetic_scope_sha256 does not bind the exact synthetic non-claim")
        return self

    @classmethod
    def model_validate(cls, obj: object, **kwargs: Any) -> Self:
        """Expose one typed boundary error rather than leaking Pydantic internals."""
        try:
            return super().model_validate(obj, **kwargs)
        except ValidationError as error:
            raise V2PreregistrationError(str(error)) from error

    def unsigned_document(self) -> dict[str, object]:
        """Return the exact canonical document covered by the evaluator signature."""
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def verify_signature(self) -> bool:
        """Verify the embedded Ed25519 signature using only its pinned public key."""
        try:
            signature = base64.b64decode(self.signature_base64, validate=True)
            _public_key(self.evaluator_key_id, self.evaluator_public_key_base64).verify(
                signature, canonical_json_bytes(self.unsigned_document())
            )
        except (InvalidSignature, ValueError, TypeError, binascii.Error):
            return False
        return True

    def verify_trusted_authority(self) -> bool:
        """Bind the sealed production preregistration to its committed authority."""
        return (
            self.preregistration_id == TRUSTED_V2_PREREGISTRATION_ID
            and self.execution_nonce == TRUSTED_V2_EXECUTION_NONCE
            and trusted_v2_evaluator_identity(
                self.evaluator_key_id, self.evaluator_public_key_base64
            )
        )

    def verify_manifest_bindings(self) -> bool:
        """Recheck all binding semantics after unsafe model copying or deserialization."""
        try:
            for field in (
                "protocol_profile_sha256",
                "manifest_registry_sha256",
                "source_manifest_sha256",
                "feature_manifest_sha256",
                "candidate_grid_sha256",
                "population_manifest_sha256",
                "evaluator_key_id",
                "evaluator_capability_sha256",
                "metrics_manifest_sha256",
                "bootstrap_manifest_sha256",
                "controls_manifest_sha256",
                "budget_manifest_sha256",
                "reporting_schema_sha256",
                "synthetic_scope_sha256",
                "execution_nonce",
            ):
                _digest(getattr(self, field), field=field)
            if self.fidelity_validation_bundle_sha256 is not None:
                _digest(
                    self.fidelity_validation_bundle_sha256,
                    field="fidelity_validation_bundle_sha256",
                )
            if (
                self.synthetic_scope != SYNTHETIC_NON_CLAIM
                or self.synthetic_scope_sha256
                != hashlib.sha256(canonical_json_bytes(self.synthetic_scope)).hexdigest()
                or tuple(item.name for item in self.seed_commitments) != _SEED_NAMES
                or any(type(item) is not SeedCommitment for item in self.seed_commitments)
                or any(
                    _digest(item.commitment_sha256, field="seed_commitments")
                    != item.commitment_sha256
                    for item in self.seed_commitments
                )
                or self.maximum_confirmatory_attempts != 1
            ):
                return False
            _public_key(self.evaluator_key_id, self.evaluator_public_key_base64)
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise V2PreregistrationError("preregistration must be a JSON object")
            preregistration = cls.model_validate(document)
            if preregistration.canonical_bytes() != payload:
                raise V2PreregistrationError("preregistration JSON is not canonical")
            return preregistration
        except (WireContractError, ValueError) as error:
            if isinstance(error, V2PreregistrationError):
                raise
            raise V2PreregistrationError(str(error)) from error


class ExecutionReceipt(ExternalContract):
    """Minimal durable record that a confirmatory v2 admission was consumed."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    preregistration_id: str
    execution_nonce: str


class ExecutionAdmission(ExternalContract):
    """The only result of evaluating whether a sealed execution may begin."""

    admitted: bool
    reason: Literal["maximum_confirmatory_attempts_exhausted", "invalid_preregistration"] | None = (
        None
    )
    execution_nonce: str | None = None

    @model_validator(mode="after")
    def decision_is_unambiguous(self) -> Self:
        if self.admitted != (self.reason is None and self.execution_nonce is not None):
            raise ValueError("execution admission state is inconsistent")
        return self

    @classmethod
    def denied(
        cls, reason: Literal["maximum_confirmatory_attempts_exhausted", "invalid_preregistration"]
    ) -> Self:
        return cls(admitted=False, reason=reason)

    @classmethod
    def admitted_once(cls, execution_nonce: str) -> Self:
        return cls(admitted=True, execution_nonce=execution_nonce)


def sign_v2_preregistration(
    payload: Mapping[str, object], *, signer: EvaluatorSigningIdentity
) -> V2Preregistration:
    """Bind a complete preregistration to the exact evaluator authority key.

    The function intentionally has no approval parameter.  Possession of a
    valid signature binds immutable intent; it does not execute anything.
    """
    if not EvaluatorSigningIdentity.is_exact(signer):
        raise V2PreregistrationError("signer must be the exact evaluator authority")
    if type(payload) is not dict:
        raise V2PreregistrationError("preregistration payload must be an exact object")
    unsigned = dict(payload)
    if "signature_base64" in unsigned:
        raise V2PreregistrationError("caller cannot supply preregistration signature")
    for field, expected in (
        ("evaluator_key_id", signer.key_id),
        ("evaluator_public_key_base64", signer.public_key_base64),
    ):
        supplied = unsigned.get(field)
        if supplied is not None and supplied != expected:
            raise V2PreregistrationError(f"{field} differs from evaluator signer")
        unsigned[field] = expected
    prepared = V2Preregistration.model_validate({**unsigned, "signature_base64": ""})
    signature = signer._sign(prepared.unsigned_document())
    return V2Preregistration.model_validate(
        {**prepared.unsigned_document(), "signature_base64": signature}
    )


def admit_v2_execution(
    preregistration: V2Preregistration,
    *,
    existing_receipts: Sequence[ExecutionReceipt],
) -> ExecutionAdmission:
    """Admit only the first confirmatory execution of a valid sealed contract."""
    if type(preregistration) is not V2Preregistration:
        return ExecutionAdmission.denied("invalid_preregistration")
    if not isinstance(existing_receipts, Sequence) or isinstance(
        existing_receipts, (str, bytes, bytearray, memoryview)
    ):
        return ExecutionAdmission.denied("invalid_preregistration")
    if len(existing_receipts) >= preregistration.maximum_confirmatory_attempts:
        return ExecutionAdmission.denied("maximum_confirmatory_attempts_exhausted")
    if (
        not preregistration.verify_signature()
        or not preregistration.verify_manifest_bindings()
        or not preregistration.verify_trusted_authority()
    ):
        return ExecutionAdmission.denied("invalid_preregistration")
    return ExecutionAdmission.admitted_once(preregistration.execution_nonce)


def trusted_v2_evaluator_identity(key_id: object, public_key_base64: object) -> bool:
    """Return whether a key pair is the committed V2 evaluator/publication authority."""
    return (
        key_id == TRUSTED_V2_EVALUATOR_KEY_ID
        and public_key_base64 == TRUSTED_V2_EVALUATOR_PUBLIC_KEY_BASE64
    )


__all__ = [
    "ExecutionAdmission",
    "ExecutionReceipt",
    "SYNTHETIC_NON_CLAIM",
    "SyntheticScope",
    "TRUSTED_V2_EVALUATOR_KEY_ID",
    "TRUSTED_V2_EVALUATOR_PUBLIC_KEY_BASE64",
    "TRUSTED_V2_EXECUTION_NONCE",
    "TRUSTED_V2_PREREGISTRATION_ID",
    "V2Preregistration",
    "V2PreregistrationError",
    "admit_v2_execution",
    "sign_v2_preregistration",
    "trusted_v2_evaluator_identity",
]
