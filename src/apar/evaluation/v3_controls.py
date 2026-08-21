"""Mandatory negative control runner for Defend v3.

Adapts v2 control primitives to v3 bindings without weakening signature,
identity, block, or validity checks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_controls import (
    ControlResult,
    V2ControlBinding,
    V2ControlContext,
    V2ControlError,
    run_benign_only_control,
    run_score_permutation_control,
)
from apar.v3_protocol import V3ProtocolError


class V3ControlError(V3ProtocolError):
    """A mandatory v3 control is malformed, tampered with, or invalid."""


def run_benign_control(
    *,
    actions: Sequence[Action],
    truth: Sequence[EvaluationTruthRow],
    signer: EvaluatorSigningIdentity,
    binding: V2ControlBinding,
) -> ControlResult:
    """Verify that an all-benign population produces no fraud claim."""
    if any(type(item) is not Action for item in actions):
        raise V3ControlError("benign control actions must be exact Action values")
    if any(type(row) is not EvaluationTruthRow for row in truth):
        raise V3ControlError("benign control truth must be exact EvaluationTruthRow values")
    return run_benign_only_control(actions=actions, truth=truth, signer=signer, binding=binding)


def run_permutation_control(
    *,
    scores: np.ndarray,
    truth: Sequence[EvaluationTruthRow],
    blocks: Sequence[str],
    seed: int,
    evaluator: object,
    signer: EvaluatorSigningIdentity,
    binding: V2ControlBinding,
) -> ControlResult:
    """Run a block-preserving score-permutation control."""
    if type(scores) is not np.ndarray:
        raise V3ControlError("permutation control scores must be a NumPy array")
    if any(type(row) is not EvaluationTruthRow for row in truth):
        raise V3ControlError("permutation control truth must be exact EvaluationTruthRow values")
    if any(type(block) is not str or not block for block in blocks):
        raise V3ControlError("permutation control blocks must be nonempty strings")
    if type(seed) is not int:
        raise V3ControlError("permutation control seed must be an exact integer")
    if not callable(evaluator):
        raise V3ControlError("permutation control evaluator must be callable")
    return run_score_permutation_control(
        scores=scores,
        truth=truth,
        blocks=blocks,
        seed=seed,
        evaluator=evaluator,
        signer=signer,
        binding=binding,
    )


__all__ = [
    "ControlResult",
    "V2ControlBinding",
    "V2ControlContext",
    "V2ControlError",
    "V3ControlError",
    "run_benign_control",
    "run_permutation_control",
]
