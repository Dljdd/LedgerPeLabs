"""Evaluator-owned negative controls for the synthetic Defend v2 run.

Controls are deliberately small and fail closed.  They consume already
materialized actions, scores, and evaluator truth; they do not run a defender
or construct an evaluation population.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from apar.contracts.decisions import Action
from apar.evaluation.contracts import EvaluationTruthRow


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Closed result of one mandatory negative control."""

    valid: bool
    reason: str | None = None
    intervention_count: int = 0
    true_positive_count: int = 0
    efficacy_auc: float | None = None


@dataclass(frozen=True, slots=True)
class ControlAdmission:
    """Typed boundary between controls and v2 evaluation admission."""

    valid: bool
    status: Literal["admitted", "no_promotion"]
    reason: str | None = None


ControlEvaluator = Callable[[np.ndarray, tuple[EvaluationTruthRow, ...], tuple[str, ...]], bool]


def admit_control_result(control: ControlResult) -> ControlAdmission:
    """Convert a control result into a load-bearing whole-run admission."""
    if type(control) is not ControlResult:
        return ControlAdmission(False, "no_promotion", "malformed_control_result")
    if not control.valid:
        return ControlAdmission(False, "no_promotion", control.reason or "control_invalid")
    return ControlAdmission(True, "admitted")


def run_benign_only_control(
    *, actions: Sequence[Action | object],
    truth: Sequence[EvaluationTruthRow],
) -> ControlResult:
    """Verify that an all-benign operating control makes no fraud claim.

    Customer and analyst interventions (challenge and decline) are retained as
    evidence, even though true-positive count must be zero.
    """

    try:
        rows = _validated_truth(truth)
        action_values = _validated_actions(actions, rows)
        if any(row.is_fraud for row in rows):
            return ControlResult(False, "malformed_benign_control")
        interventions = sum(action is not Action.APPROVE for action in action_values)
        return ControlResult(
            valid=True,
            intervention_count=interventions,
            true_positive_count=0,
        )
    except (TypeError, ValueError):
        return ControlResult(False, "malformed_benign_control")


def run_score_permutation_control(
    *,
    scores: np.ndarray,
    truth: Sequence[EvaluationTruthRow],
    blocks: Sequence[str],
    seed: int,
    evaluator: ControlEvaluator | None = None,
) -> ControlResult:
    """Permute score blocks and reject any apparently qualifying efficacy.

    A block is the indivisible time/case unit supplied by the evaluator.  Only
    complete blocks move; row order within each block is retained.
    """

    try:
        rows = _validated_truth(truth)
        values = _validated_scores(scores, len(rows))
        block_values = _validated_blocks(blocks, len(rows))
        if type(seed) is not int:
            raise ValueError("seed must be an exact integer")
        permutation = np.random.default_rng(seed).permutation(
            np.unique(np.asarray(block_values, dtype=object))
        )
        permuted = _permute_scores_by_block(values, block_values, permutation)
        if evaluator is not None:
            if not callable(evaluator):
                raise TypeError("evaluator must be callable")
            if evaluator(permuted.copy(), rows, block_values):
                return ControlResult(False, "permuted_scores_qualified")
        auc = _auc(permuted, np.asarray([row.is_fraud for row in rows], dtype=bool))
        return ControlResult(True, efficacy_auc=auc)
    except (TypeError, ValueError):
        return ControlResult(False, "malformed_permutation_control")


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
    "admit_control_result",
    "run_benign_only_control",
    "run_score_permutation_control",
]
