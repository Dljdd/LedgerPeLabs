"""Evaluator-owned negative controls for the synthetic Defend v2 run.

Controls are deliberately small and fail closed.  They consume already
materialized actions, scores, and evaluator truth; they do not run a defender
or construct an evaluation population.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preexecution import (
    V2VerifiedAuthority,
    _verified_v2_preregistration,
)
from apar.evaluation.v2_preregistration import V2Preregistration
from apar.runs.wire import canonical_json_bytes


class V2ControlError(ValueError):
    """A mandatory control could not produce trusted evaluator evidence."""


class V2ControlBinding(ExternalContract):
    """Exact execution and candidate context covered by a control signature."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    preregistration_id: str = Field(min_length=1)
    execution_nonce: str
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    candidate_id: str = Field(min_length=1, max_length=256)
    input_digest: str
    evaluator_key_id: str
    evaluator_public_key_base64: str

    @model_validator(mode="after")
    def binding_is_closed(self) -> V2ControlBinding:
        _require_digest(self.execution_nonce, field="execution_nonce")
        _require_digest(self.input_digest, field="input_digest")
        _require_public_identity(self.evaluator_key_id, self.evaluator_public_key_base64)
        return self

    @classmethod
    def from_preregistration(
        cls,
        preregistration: V2Preregistration,
        *,
        arm: Literal["rules_only", "gbdt_only", "layered_hybrid"],
        candidate_id: str,
        input_digest: str,
    ) -> V2ControlBinding:
        if (
            type(preregistration) is not V2Preregistration
            or not preregistration.verify_signature()
            or not preregistration.verify_manifest_bindings()
        ):
            raise V2ControlError("control binding requires a signed preregistration")
        return cls(
            preregistration_id=preregistration.preregistration_id,
            execution_nonce=preregistration.execution_nonce,
            arm=arm,
            candidate_id=candidate_id,
            input_digest=input_digest,
            evaluator_key_id=preregistration.evaluator_key_id,
            evaluator_public_key_base64=preregistration.evaluator_public_key_base64,
        )


class V2ControlContext(ExternalContract):
    """Independent expected execution context for one exact candidate."""

    preregistration: V2Preregistration
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    candidate_id: str = Field(min_length=1, max_length=256)
    input_digest: str

    @model_validator(mode="after")
    def context_is_closed(self) -> V2ControlContext:
        _require_digest(self.input_digest, field="input_digest")
        if (
            type(self.preregistration) is not V2Preregistration
            or not self.preregistration.verify_signature()
            or not self.preregistration.verify_manifest_bindings()
        ):
            raise ValueError("control context requires an intact signed preregistration")
        return self

    @classmethod
    def from_preregistration(
        cls,
        preregistration: V2Preregistration,
        *,
        verified_authority: object,
        arm: Literal["rules_only", "gbdt_only", "layered_hybrid"],
        candidate_id: str,
        input_digest: str,
    ) -> V2ControlContext:
        context = cls(
            preregistration=preregistration,
            arm=arm,
            candidate_id=candidate_id,
            input_digest=input_digest,
        )
        if not context.matches_verified_authority(verified_authority):
            raise V2ControlError("control context does not match verified authority")
        return context

    def matches_verified_authority(self, authority: object) -> bool:
        """Check opaque verifier-issued trust and its exact evaluator identity."""
        try:
            sealed = _verified_v2_preregistration(authority)
            return (
                type(sealed) is V2Preregistration
                and self.preregistration.matches_sealed_preregistration(sealed)
                and self.preregistration.evaluator_key_id == sealed.evaluator_key_id
                and self.preregistration.evaluator_public_key_base64
                == sealed.evaluator_public_key_base64
                and self.preregistration.execution_nonce == sealed.execution_nonce
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def binding(self) -> V2ControlBinding:
        return V2ControlBinding.from_preregistration(
            self.preregistration,
            arm=self.arm,
            candidate_id=self.candidate_id,
            input_digest=self.input_digest,
        )


class ControlResult(ExternalContract):
    """Immutable signed result of one mandatory negative control."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    valid: bool
    kind: Literal["benign_only", "score_permutation"]
    reason: str | None = None
    intervention_count: int = 0
    true_positive_count: int = 0
    efficacy_auc: float | None = None
    binding: V2ControlBinding
    signature_base64: str

    @model_validator(mode="after")
    def _coherent(self) -> ControlResult:
        values = self
        if values.valid != (values.reason is None):
            raise ValueError("control validity and reason disagree")
        if values.true_positive_count < 0 or values.intervention_count < 0:
            raise ValueError("control counts must be nonnegative")
        if not values.verify_attestation():
            raise ValueError("control attestation is invalid")
        return values

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def verify_attestation(self, expected_binding: V2ControlBinding | None = None) -> bool:
        if expected_binding is not None and self.binding != expected_binding:
            return False
        try:
            public = base64.b64decode(self.binding.evaluator_public_key_base64, validate=True)
            signature = base64.b64decode(self.signature_base64, validate=True)
            Ed25519PublicKey.from_public_bytes(public).verify(
                signature, canonical_json_bytes(self.unsigned_document())
            )
        except (InvalidSignature, ValueError, TypeError, binascii.Error):
            return False
        return True


class ControlValidity(ExternalContract):
    """Both mandatory evaluator-attested controls required by V2 selection."""

    benign_only: ControlResult
    score_permutation: ControlResult

    @model_validator(mode="after")
    def required_control_kinds_are_exact(self) -> ControlValidity:
        if type(self.benign_only) is not ControlResult or self.benign_only.kind != "benign_only":
            raise ValueError("benign-only control result is missing")
        if (
            type(self.score_permutation) is not ControlResult
            or self.score_permutation.kind != "score_permutation"
        ):
            raise ValueError("score-permutation control result is missing")
        if self.benign_only.binding != self.score_permutation.binding:
            raise ValueError("mandatory controls must share one exact binding")
        return self

    @classmethod
    def attest(
        cls,
        *,
        benign_only: ControlResult,
        score_permutation: ControlResult,
    ) -> ControlValidity:
        """Bind the two exact producer-issued results into one selection gate."""
        if type(benign_only) is not ControlResult or type(score_permutation) is not ControlResult:
            raise TypeError("control validity requires exact ControlResult evidence")
        return cls(benign_only=benign_only, score_permutation=score_permutation)

    def valid_for(
        self,
        *,
        verified_authority: object,
        expected_context: V2ControlContext,
    ) -> bool:
        """Return true only when both attestations match the exact candidate context."""
        try:
            return (
                type(verified_authority) is V2VerifiedAuthority
                and type(expected_context) is V2ControlContext
                and type(self.benign_only) is ControlResult
                and type(self.score_permutation) is ControlResult
                and admit_control_result(
                    self.benign_only,
                    verified_authority=verified_authority,
                    expected_context=expected_context,
                ).valid
                and admit_control_result(
                    self.score_permutation,
                    verified_authority=verified_authority,
                    expected_context=expected_context,
                ).valid
            )
        except (AttributeError, TypeError, ValueError):
            return False


class ControlAdmission(ExternalContract):
    """Typed boundary between controls and v2 evaluation admission."""

    valid: bool
    status: Literal["admitted", "no_promotion"]
    reason: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> ControlAdmission:
        values = self
        if values.valid != (values.status == "admitted"):
            raise ValueError("admission status and validity disagree")
        if values.status == "admitted" and values.reason is not None:
            raise ValueError("admitted control cannot carry a reason")
        if values.status == "no_promotion" and not values.reason:
            raise ValueError("no-promotion admission requires a reason")
        return values


ControlEvaluator = Callable[[np.ndarray, tuple[EvaluationTruthRow, ...], tuple[str, ...]], bool]


def admit_control_result(
    control: ControlResult,
    *,
    verified_authority: object,
    expected_context: V2ControlContext,
) -> ControlAdmission:
    """Convert a control result into a load-bearing whole-run admission."""
    if type(control) is not ControlResult:
        return ControlAdmission(
            valid=False, status="no_promotion", reason="malformed_control_result"
        )
    try:
        checked = ControlResult.model_validate(control.model_dump())
        context = V2ControlContext.model_validate(expected_context.model_dump())
    except Exception:
        return ControlAdmission(
            valid=False, status="no_promotion", reason="invalid_control_attestation"
        )
    sealed_preregistration = _verified_v2_preregistration(verified_authority)
    if (
        type(sealed_preregistration) is not V2Preregistration
        or type(expected_context) is not V2ControlContext
        or not context.matches_verified_authority(verified_authority)
        or checked.binding != context.binding()
        or checked.binding.evaluator_key_id != sealed_preregistration.evaluator_key_id
        or checked.binding.evaluator_public_key_base64
        != sealed_preregistration.evaluator_public_key_base64
    ):
        return ControlAdmission(
            valid=False, status="no_promotion", reason="control_binding_mismatch"
        )
    if not checked.valid:
        return ControlAdmission(
            valid=False, status="no_promotion", reason=control.reason or "control_invalid"
        )
    return ControlAdmission(valid=True, status="admitted")


def run_benign_only_control(
    *,
    actions: Sequence[Action | object],
    truth: Sequence[EvaluationTruthRow],
    signer: EvaluatorSigningIdentity,
    binding: V2ControlBinding,
) -> ControlResult:
    """Verify that an all-benign operating control makes no fraud claim.

    Customer and analyst interventions (challenge and decline) are retained as
    evidence, even though true-positive count must be zero.
    """

    _require_bound_signer(signer, binding)
    try:
        rows = _validated_truth(truth)
        action_values = _validated_actions(actions, rows)
        if any(row.is_fraud for row in rows):
            return _attest_control_result(
                signer=signer,
                binding=binding,
                valid=False,
                kind="benign_only",
                reason="malformed_benign_control",
            )
        interventions = sum(action is not Action.APPROVE for action in action_values)
        return _attest_control_result(
            signer=signer,
            binding=binding,
            valid=True,
            kind="benign_only",
            intervention_count=interventions,
            true_positive_count=0,
        )
    except Exception:
        return _attest_control_result(
            signer=signer,
            binding=binding,
            valid=False,
            kind="benign_only",
            reason="malformed_benign_control",
        )


def run_score_permutation_control(
    *,
    scores: np.ndarray,
    truth: Sequence[EvaluationTruthRow],
    blocks: Sequence[str],
    seed: int,
    evaluator: ControlEvaluator | None = None,
    signer: EvaluatorSigningIdentity,
    binding: V2ControlBinding,
) -> ControlResult:
    """Permute score blocks and reject any apparently qualifying efficacy.

    A block is the indivisible time/case unit supplied by the evaluator.  Only
    complete blocks move; row order within each block is retained.
    """

    _require_bound_signer(signer, binding)

    def invalid(reason: str) -> ControlResult:
        return _attest_control_result(
            signer=signer,
            binding=binding,
            valid=False,
            kind="score_permutation",
            reason=reason,
        )

    try:
        rows = _validated_truth(truth)
        if evaluator is None:
            return invalid("evaluator_missing")
        if not callable(evaluator):
            return invalid("malformed_evaluator")
        values = _validated_scores(scores, len(rows))
        block_values = _validated_blocks(blocks, len(rows))
        if type(seed) is not int:
            raise ValueError("seed must be an exact integer")
        permutation = np.random.default_rng(seed).permutation(
            np.unique(np.asarray(block_values, dtype=object))
        )
        permuted = _permute_scores_by_block(values, block_values, permutation)
        try:
            qualified = evaluator(permuted.copy(), rows, block_values)
        except Exception:
            return invalid("evaluator_failed")
        if type(qualified) is not bool:
            return invalid("malformed_evaluator_result")
        if qualified:
            return invalid("permuted_scores_qualified")
        auc = _auc(permuted, np.asarray([row.is_fraud for row in rows], dtype=bool))
        return _attest_control_result(
            signer=signer,
            binding=binding,
            valid=True,
            kind="score_permutation",
            efficacy_auc=auc,
        )
    except Exception:
        return invalid("malformed_permutation_control")


def _require_bound_signer(signer: EvaluatorSigningIdentity, binding: V2ControlBinding) -> None:
    if (
        not EvaluatorSigningIdentity.is_exact(signer)
        or type(binding) is not V2ControlBinding
        or signer.key_id != binding.evaluator_key_id
        or signer.public_key_base64 != binding.evaluator_public_key_base64
    ):
        raise V2ControlError("control signer does not match the bound evaluator authority")


def _attest_control_result(
    *,
    signer: EvaluatorSigningIdentity,
    binding: V2ControlBinding,
    valid: bool,
    kind: Literal["benign_only", "score_permutation"],
    reason: str | None = None,
    intervention_count: int = 0,
    true_positive_count: int = 0,
    efficacy_auc: float | None = None,
) -> ControlResult:
    _require_bound_signer(signer, binding)
    unsigned: dict[str, object] = {
        "schema_version": "1.0.0",
        "valid": valid,
        "kind": kind,
        "reason": reason,
        "intervention_count": intervention_count,
        "true_positive_count": true_positive_count,
        "efficacy_auc": efficacy_auc,
        "binding": binding.model_dump(mode="json"),
    }
    return ControlResult.model_validate({**unsigned, "signature_base64": signer._sign(unsigned)})


def _require_digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_public_identity(key_id: object, public_key_base64: object) -> None:
    _require_digest(key_id, field="evaluator_key_id")
    if type(public_key_base64) is not str:
        raise ValueError("evaluator public key is invalid")
    try:
        public = base64.b64decode(public_key_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("evaluator public key is invalid") from error
    if len(public) != 32 or hashlib.sha256(public).hexdigest() != key_id:
        raise ValueError("evaluator public key identity is inconsistent")


def _validated_truth(truth: Sequence[EvaluationTruthRow]) -> tuple[EvaluationTruthRow, ...]:
    if type(truth) not in (tuple, list) or not truth:
        raise ValueError("truth must be a nonempty sequence")
    rows = tuple(truth)
    if any(type(row) is not EvaluationTruthRow for row in rows):
        raise TypeError("truth rows must be exact EvaluationTruthRow instances")
    if len({row.event_id for row in rows}) != len(rows):
        raise ValueError("truth event IDs must be unique")
    return rows


def _validated_actions(
    actions: Sequence[Action | object], rows: tuple[EvaluationTruthRow, ...]
) -> tuple[Action, ...]:
    if type(actions) not in (tuple, list) or len(actions) != len(rows):
        raise ValueError("actions must cover the truth operating universe")
    result: list[Action] = []
    for item in actions:
        if type(item) is Action:
            result.append(item)
        else:
            action = getattr(item, "action", None)
            if type(action) is not Action:
                raise TypeError("actions must contain exact Action values")
            result.append(action)
    return tuple(result)


def _validated_scores(scores: np.ndarray, size: int) -> np.ndarray:
    if not isinstance(scores, np.ndarray) or scores.ndim != 1 or scores.shape[0] != size:
        raise ValueError("scores must be a one-dimensional array matching truth")
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("scores must be numeric")
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("scores must be finite values in [0, 1]")
    return values


def _validated_blocks(blocks: Sequence[str], size: int) -> tuple[str, ...]:
    if type(blocks) not in (tuple, list) or len(blocks) != size:
        raise ValueError("blocks must cover the truth operating universe")
    values = tuple(blocks)
    if any(type(block) is not str or not block for block in values):
        raise ValueError("blocks must be nonempty strings")
    return values


def _permute_scores_by_block(
    scores: np.ndarray, blocks: tuple[str, ...], permutation: np.ndarray
) -> np.ndarray:
    unique = tuple(np.unique(np.asarray(blocks, dtype=object)))
    if len(unique) != len(permutation) or set(permutation.tolist()) != set(unique):
        raise ValueError("invalid block permutation")
    result = np.empty_like(scores)
    for destination, source in zip(unique, permutation, strict=True):
        destination_indices = np.flatnonzero(np.asarray(blocks, dtype=object) == destination)
        source_indices = np.flatnonzero(np.asarray(blocks, dtype=object) == source)
        if destination_indices.size != source_indices.size:
            raise ValueError("score permutation must preserve block sizes")
        result[destination_indices] = scores[source_indices]
    return result


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


__all__ = [
    "ControlAdmission",
    "ControlEvaluator",
    "ControlResult",
    "ControlValidity",
    "V2ControlBinding",
    "V2ControlContext",
    "V2ControlError",
    "admit_control_result",
    "run_benign_only_control",
    "run_score_permutation_control",
]
