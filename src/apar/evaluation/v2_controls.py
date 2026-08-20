"""Evaluator-owned negative controls for the synthetic Defend v2 run.

Controls are deliberately small and fail closed.  They consume already
materialized actions, scores, and evaluator truth; they do not run a defender
or construct an evaluation population.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preregistration import trusted_v2_evaluator_identity
from apar.runs.wire import canonical_json_bytes


class V2ControlError(ValueError):
    """A mandatory control could not produce trusted evaluator evidence."""


class ControlResult(ExternalContract):
    """Immutable signed result of one mandatory negative control."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    valid: bool
    kind: Literal["benign_only", "score_permutation"]
    reason: str | None = None
    intervention_count: int = 0
    true_positive_count: int = 0
    efficacy_auc: float | None = None
    evaluator_key_id: str
    evaluator_public_key_base64: str
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

    def verify_attestation(self) -> bool:
        if not trusted_v2_evaluator_identity(
            self.evaluator_key_id, self.evaluator_public_key_base64
        ):
            return False
        try:
            public = base64.b64decode(self.evaluator_public_key_base64, validate=True)
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

    @property
    def valid(self) -> bool:
        """Return true only while both results retain evaluator-owned attestations."""
        try:
            return (
                type(self.benign_only) is ControlResult
                and type(self.score_permutation) is ControlResult
                and admit_control_result(self.benign_only).valid
                and admit_control_result(self.score_permutation).valid
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


def admit_control_result(control: ControlResult) -> ControlAdmission:
    """Convert a control result into a load-bearing whole-run admission."""
    if type(control) is not ControlResult:
        return ControlAdmission(
            valid=False, status="no_promotion", reason="malformed_control_result"
        )
    try:
        checked = ControlResult.model_validate(control.model_dump())
    except Exception:
        return ControlAdmission(
            valid=False, status="no_promotion", reason="invalid_control_attestation"
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
) -> ControlResult:
    """Verify that an all-benign operating control makes no fraud claim.

    Customer and analyst interventions (challenge and decline) are retained as
    evidence, even though true-positive count must be zero.
    """

    _require_trusted_signer(signer)
    try:
        rows = _validated_truth(truth)
        action_values = _validated_actions(actions, rows)
        if any(row.is_fraud for row in rows):
            return _attest_control_result(
                signer=signer,
                valid=False,
                kind="benign_only",
                reason="malformed_benign_control",
            )
        interventions = sum(action is not Action.APPROVE for action in action_values)
        return _attest_control_result(
            signer=signer,
            valid=True,
            kind="benign_only",
            intervention_count=interventions,
            true_positive_count=0,
        )
    except Exception:
        return _attest_control_result(
            signer=signer,
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
) -> ControlResult:
    """Permute score blocks and reject any apparently qualifying efficacy.

    A block is the indivisible time/case unit supplied by the evaluator.  Only
    complete blocks move; row order within each block is retained.
    """

    _require_trusted_signer(signer)

    def invalid(reason: str) -> ControlResult:
        return _attest_control_result(
            signer=signer,
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
            valid=True,
            kind="score_permutation",
            efficacy_auc=auc,
        )
    except Exception:
        return invalid("malformed_permutation_control")


def _require_trusted_signer(signer: EvaluatorSigningIdentity) -> None:
    if not EvaluatorSigningIdentity.is_exact(signer) or not trusted_v2_evaluator_identity(
        signer.key_id, signer.public_key_base64
    ):
        raise V2ControlError("control signer is not the trusted evaluator authority")


def _attest_control_result(
    *,
    signer: EvaluatorSigningIdentity,
    valid: bool,
    kind: Literal["benign_only", "score_permutation"],
    reason: str | None = None,
    intervention_count: int = 0,
    true_positive_count: int = 0,
    efficacy_auc: float | None = None,
) -> ControlResult:
    _require_trusted_signer(signer)
    unsigned: dict[str, object] = {
        "schema_version": "1.0.0",
        "valid": valid,
        "kind": kind,
        "reason": reason,
        "intervention_count": intervention_count,
        "true_positive_count": true_positive_count,
        "efficacy_auc": efficacy_auc,
        "evaluator_key_id": signer.key_id,
        "evaluator_public_key_base64": signer.public_key_base64,
    }
    return ControlResult.model_validate({**unsigned, "signature_base64": signer._sign(unsigned)})


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
    "V2ControlError",
    "admit_control_result",
    "run_benign_only_control",
    "run_score_permutation_control",
]
