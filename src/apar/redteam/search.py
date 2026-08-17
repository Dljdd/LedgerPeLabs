"""Evaluator-owned search orchestration and preregistered capability metrics."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.redteam.policies import (
    AttackCandidate,
    Feedback,
    ParameterBounds,
    Policy,
    VisibleTrial,
    visible_objective,
)

_POLICY_NAMES = ("adaptive", "cached_llm", "fixed", "random")


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _exact_non_negative_int(label: str, value: object, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _exact_finite_decimal(label: str, value: object, *, non_negative: bool = False) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    if not value.is_finite() or (non_negative and value < 0):
        raise ValueError(f"{label} must be finite")
    return value


class SearchResult(ExternalContract):
    """Complete proposal and coarse-feedback record for one matched search."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    seed: int
    policy_name: str
    proposals: tuple[AttackCandidate, ...]
    trials: tuple[VisibleTrial, ...]
    objective_values: tuple[Decimal, ...]
    winner: AttackCandidate | None
    proposal_budget: int
    query_budget: int
    logical_time_budget: int
    proposals_used: int
    queries_used: int
    logical_time_used: int

    @field_validator("seed", mode="before")
    @classmethod
    def seed_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("seed", value, maximum=2**63 - 1)

    @field_validator("policy_name", mode="before")
    @classmethod
    def policy_name_is_exact(cls, value: object) -> object:
        checked = _exact_text("policy_name", value)
        if checked not in {*_POLICY_NAMES, "llm"}:
            raise ValueError("policy_name is not a declared attacker policy")
        return checked

    @field_validator("proposals", mode="before")
    @classmethod
    def proposals_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not AttackCandidate for item in value):
            raise TypeError("proposals must be an exact tuple of exact AttackCandidate records")
        return value

    @field_validator("trials", mode="before")
    @classmethod
    def trials_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not VisibleTrial for item in value):
            raise TypeError("trials must be an exact tuple of exact VisibleTrial records")
        return value

    @field_validator("objective_values", mode="before")
    @classmethod
    def objectives_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not Decimal or not item.is_finite() for item in value
        ):
            raise TypeError("objective_values must be an exact tuple of finite Decimals")
        return value

    @field_validator("winner", mode="before")
    @classmethod
    def winner_is_exact(cls, value: object) -> object:
        if value is not None and type(value) is not AttackCandidate:
            raise TypeError("winner must be an exact AttackCandidate or None")
        return value

    @field_validator(
        "proposal_budget",
        "query_budget",
        "logical_time_budget",
        "proposals_used",
        "queries_used",
        "logical_time_used",
        mode="before",
    )
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("search budget/count", value, maximum=1000)

    @model_validator(mode="after")
    def record_is_consistent(self) -> Self:
        size = len(self.proposals)
        if len(self.trials) != size or len(self.objective_values) != size:
            raise ValueError("proposal, trial, and objective sequences must align")
        if tuple(trial.candidate for trial in self.trials) != self.proposals:
            raise ValueError("trial candidates must preserve proposal order")
        if tuple(trial.objective_value for trial in self.trials) != self.objective_values:
            raise ValueError("trial objectives must preserve objective order")
        if not (
            self.proposal_budget
            == self.query_budget
            == self.logical_time_budget
            == self.proposals_used
            == self.queries_used
            == self.logical_time_used
            == size
        ):
            raise ValueError("proposal, query, and logical-time budgets must match exactly")
        if size == 0 and self.winner is not None:
            raise ValueError("an empty search cannot have a winner")
        if size > 0 and self.winner not in self.proposals:
            raise ValueError("winner must be one of the proposed candidates")
        return self


class AdaptiveSearch:
    """Run a policy without passing or retaining the evaluator capability."""

    __slots__ = ("_bounds", "_disclose_realized_value", "_policy")

    def __init__(
        self,
        *,
        policy: Policy,
        bounds: ParameterBounds,
        disclose_realized_value: bool = False,
    ) -> None:
        if type(bounds) is not ParameterBounds:
            raise TypeError("bounds must be an exact ParameterBounds")
        if type(disclose_realized_value) is not bool:
            raise TypeError("disclose_realized_value must be an exact bool")
        propose = getattr(policy, "propose", None)
        if not callable(propose):
            raise TypeError("policy must expose propose")
        policy_name = getattr(policy, "policy_name", None)
        if policy_name not in {*_POLICY_NAMES, "llm"} or type(policy_name) is not str:
            raise TypeError("policy must expose an exact declared policy_name")
        self._policy = policy
        self._bounds = bounds
        self._disclose_realized_value = disclose_realized_value

    def search(
        self,
        seed: int,
        budget: int,
        evaluate: Callable[[AttackCandidate], Feedback],
    ) -> SearchResult:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TypeError("seed must be an exact integer in [0, 2**63)")
        checked_budget = _exact_non_negative_int("budget", budget, maximum=1000)
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        rng = np.random.default_rng(seed)
        proposals: list[AttackCandidate] = []
        trials: list[VisibleTrial] = []
        objectives: list[Decimal] = []
        for _ in range(checked_budget):
            candidate = self._policy.propose(tuple(trials), self._bounds, rng)
            if type(candidate) is not AttackCandidate:
                raise TypeError("policy must return an exact AttackCandidate")
            self._bounds.validate_params(candidate.params)
            proposals.append(candidate)
            try:
                returned = evaluate(candidate)
                if type(returned) is not Feedback:
                    raise TypeError("evaluate must return an exact Feedback")
                feedback = Feedback(
                    action=returned.action,
                    reason_family=returned.reason_family,
                    realized_value=(
                        returned.realized_value
                        if self._disclose_realized_value
                        else None
                    ),
                )
            except Exception:
                feedback = Feedback(
                    action=Action.DECLINE,
                    reason_family="evaluation_failure",
                    realized_value=None,
                )
            objective = visible_objective(feedback)
            objectives.append(objective)
            trials.append(
                VisibleTrial(
                    candidate=candidate,
                    feedback=feedback,
                    objective_value=objective,
                )
            )
        winner = (
            None
            if not trials
            else min(
                trials,
                key=lambda trial: (
                    -trial.objective_value,
                    trial.candidate.candidate_id,
                ),
            ).candidate
        )
        return SearchResult(
            seed=seed,
            policy_name=self._policy.policy_name,
            proposals=tuple(proposals),
            trials=tuple(trials),
            objective_values=tuple(objectives),
            winner=winner,
            proposal_budget=checked_budget,
            query_budget=checked_budget,
            logical_time_budget=checked_budget,
            proposals_used=len(proposals),
            queries_used=len(proposals),
            logical_time_used=len(proposals),
        )

class PrimaryOutcome(StrEnum):
    """Preregisterable family-level capability outcomes."""

    VALID_YIELD = "valid_yield"
    NET_SETTLED_VALUE = "net_settled_value"
    ADAPTATION_SPEED = "adaptation_speed"
    CAMPAIGN_SCALE = "campaign_scale"


class FamilyThreshold(ExternalContract):
    """A family outcome and fixed minimum delta declared before execution."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator("primary_outcome", mode="before")
    @classmethod
    def outcome_is_exact(cls, value: object) -> object:
        if type(value) is not PrimaryOutcome:
            raise TypeError("primary_outcome must be an exact PrimaryOutcome")
        return value

    @field_validator("minimum_delta", mode="before")
    @classmethod
    def delta_is_exact(cls, value: object) -> object:
        checked = _exact_finite_decimal("minimum_delta", value, non_negative=True)
        if checked <= 0:
            raise ValueError("minimum_delta must be strictly positive and measurable")
        return checked


class CapabilityPreregistration(ExternalContract):
    """Frozen seeds, budgets, outcomes, and thresholds created before a run."""

    seeds: tuple[int, ...]
    budget: int
    thresholds: tuple[FamilyThreshold, ...]

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_are_canonical(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise TypeError("seeds must be a non-empty exact tuple")
        checked = tuple(
            _exact_non_negative_int("seed", item, maximum=2**63 - 1) for item in value
        )
        if checked != tuple(sorted(set(checked))):
            raise ValueError("seeds must be unique and sorted")
        return checked

    @field_validator("budget", mode="before")
    @classmethod
    def budget_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("budget", value, maximum=1000)

    @field_validator("thresholds", mode="before")
    @classmethod
    def thresholds_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or not value or any(
            type(item) is not FamilyThreshold for item in value
        ):
            raise TypeError("thresholds must be a non-empty tuple of exact records")
        families = tuple(item.family for item in value)
        if families != tuple(sorted(set(families))):
            raise ValueError("family thresholds must be unique and sorted")
        return value


class PolicyMetrics(ExternalContract):
    """Observed aggregate outcomes for one family-policy cell."""

    proposal_count: int
    approved_count: int
    valid_yield: Decimal
    net_settled_value: Decimal
    adaptation_speed: Decimal
    campaign_scale: int

    @field_validator("proposal_count", "approved_count", "campaign_scale", mode="before")
    @classmethod
    def integer_metrics_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("integer metric", value)

    @field_validator(
        "valid_yield", "net_settled_value", "adaptation_speed", mode="before"
    )
    @classmethod
    def decimal_metrics_are_exact(cls, value: object) -> object:
        return _exact_finite_decimal("decimal metric", value, non_negative=True)


class FamilyCapabilityMetrics(ExternalContract):
    """Matched policy aggregates and observed support decision for one family."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal
    observed_delta: Decimal
    supported: bool
    fixed: PolicyMetrics
    random: PolicyMetrics
    adaptive: PolicyMetrics
    cached_llm: PolicyMetrics

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator("primary_outcome", mode="before")
    @classmethod
    def outcome_is_exact(cls, value: object) -> object:
        if type(value) is not PrimaryOutcome:
            raise TypeError("primary_outcome must be exact")
        return value

    @field_validator("minimum_delta", "observed_delta", mode="before")
    @classmethod
    def deltas_are_exact(cls, value: object) -> object:
        return _exact_finite_decimal("delta", value)

    @field_validator("supported", mode="before")
    @classmethod
    def supported_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("supported must be an exact bool")
        return value

    @field_validator("fixed", "random", "adaptive", "cached_llm", mode="before")
    @classmethod
    def policy_metrics_are_exact(cls, value: object) -> object:
        if type(value) is not PolicyMetrics:
            raise TypeError("policy metrics must be exact PolicyMetrics")
        return value


class CapabilityDeltaReport(ExternalContract):
    """Observed capability deltas with an honesty-locked adaptive claim."""

    family_metrics: tuple[FamilyCapabilityMetrics, ...]
    supported_family_count: int
    matched_budgets: bool
    adaptive_net_value: Decimal
    random_net_value: Decimal
    adaptive_claim: str

    @field_validator("family_metrics", mode="before")
    @classmethod
    def family_metrics_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not FamilyCapabilityMetrics for item in value
        ):
            raise TypeError("family_metrics must be an exact tuple of exact records")
        return value

    @field_validator("supported_family_count", mode="before")
    @classmethod
    def count_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("supported_family_count", value)

    @field_validator("matched_budgets", mode="before")
    @classmethod
    def matched_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("matched_budgets must be an exact bool")
        return value

    @field_validator("adaptive_net_value", "random_net_value", mode="before")
    @classmethod
    def net_values_are_exact(cls, value: object) -> object:
        return _exact_finite_decimal("net value", value, non_negative=True)

    @field_validator("adaptive_claim", mode="before")
    @classmethod
    def claim_is_exact(cls, value: object) -> object:
        if value not in {"supported", "not_supported"} or type(value) is not str:
            raise ValueError("adaptive_claim must be supported or not_supported")
        return value

    @model_validator(mode="after")
    def claims_follow_observations(self) -> Self:
        if self.supported_family_count != sum(
            metric.supported for metric in self.family_metrics
        ):
            raise ValueError("supported_family_count must be observed, not declared")
        expected = (
            "supported"
            if self.adaptive_net_value > self.random_net_value
            else "not_supported"
        )
        if self.adaptive_claim != expected:
            raise ValueError("adaptive_claim contradicts observed net value")
        return self


def _aggregate(results: tuple[SearchResult, ...]) -> PolicyMetrics:
    proposal_count = sum(len(result.trials) for result in results)
    approved = tuple(
        trial
        for result in results
        for trial in result.trials
        if trial.feedback.action is Action.APPROVE
    )
    net_value = sum(
        (trial.feedback.realized_value or Decimal(0) for trial in approved),
        Decimal(0),
    )
    with localcontext() as context:
        context.prec = 28
        valid_yield = (
            Decimal(0)
            if proposal_count == 0
            else Decimal(len(approved)) / Decimal(proposal_count)
        )
        speeds = []
        for result in results:
            first = next(
                (
                    index
                    for index, trial in enumerate(result.trials, start=1)
                    if trial.feedback.action is Action.APPROVE
                ),
                len(result.trials) + 1,
            )
            speeds.append(first)
        adaptation_speed = Decimal(sum(speeds)) / Decimal(len(speeds))
    scale = max((trial.candidate.params.payment_count for trial in approved), default=0)
    return PolicyMetrics(
        proposal_count=proposal_count,
        approved_count=len(approved),
        valid_yield=valid_yield,
        net_settled_value=net_value,
        adaptation_speed=adaptation_speed,
        campaign_scale=scale,
    )


def _observed_delta(
    outcome: PrimaryOutcome,
    adaptive: PolicyMetrics,
    random: PolicyMetrics,
) -> Decimal:
    if outcome is PrimaryOutcome.VALID_YIELD:
        return adaptive.valid_yield - random.valid_yield
    if outcome is PrimaryOutcome.NET_SETTLED_VALUE:
        return adaptive.net_settled_value - random.net_settled_value
    if outcome is PrimaryOutcome.ADAPTATION_SPEED:
        return random.adaptation_speed - adaptive.adaptation_speed
    return Decimal(adaptive.campaign_scale - random.campaign_scale)


def capability_delta_report(
    preregistration: CapabilityPreregistration,
    results: dict[str, dict[str, tuple[SearchResult, ...]]],
) -> CapabilityDeltaReport:
    """Compute only preregistered observed deltas from matched result cells."""
    if type(preregistration) is not CapabilityPreregistration:
        raise TypeError("preregistration must be exact")
    if type(results) is not dict:
        raise TypeError("results must be an exact mapping")
    threshold_by_family = {item.family: item for item in preregistration.thresholds}
    if set(results) != set(threshold_by_family):
        raise ValueError("results must contain exactly the preregistered families")
    family_metrics: list[FamilyCapabilityMetrics] = []
    for family in sorted(results):
        cells = results[family]
        if type(cells) is not dict or tuple(sorted(cells)) != _POLICY_NAMES:
            raise ValueError("each family needs fixed, random, adaptive, and cached_llm cells")
        aggregates: dict[str, PolicyMetrics] = {}
        for policy_name in _POLICY_NAMES:
            runs = cells[policy_name]
            if type(runs) is not tuple or len(runs) != len(preregistration.seeds) or any(
                type(item) is not SearchResult for item in runs
            ):
                raise TypeError("policy runs must match the preregistered seed count")
            for result in runs:
                if result.policy_name != policy_name:
                    raise ValueError("capability result was relabeled under another policy")
                if not (
                    result.proposal_budget
                    == result.query_budget
                    == result.logical_time_budget
                    == preregistration.budget
                ):
                    raise ValueError("capability comparison budgets are not matched")
            if tuple(result.seed for result in runs) != preregistration.seeds:
                raise ValueError("capability comparison seeds are not matched")
            aggregates[policy_name] = _aggregate(runs)
        threshold = threshold_by_family[family]
        delta = _observed_delta(
            threshold.primary_outcome,
            aggregates["adaptive"],
            aggregates["random"],
        )
        family_metrics.append(
            FamilyCapabilityMetrics(
                family=family,
                primary_outcome=threshold.primary_outcome,
                minimum_delta=threshold.minimum_delta,
                observed_delta=delta,
                supported=delta >= threshold.minimum_delta,
                fixed=aggregates["fixed"],
                random=aggregates["random"],
                adaptive=aggregates["adaptive"],
                cached_llm=aggregates["cached_llm"],
            )
        )
    adaptive_net = sum(
        (metric.adaptive.net_settled_value for metric in family_metrics), Decimal(0)
    )
    random_net = sum(
        (metric.random.net_settled_value for metric in family_metrics), Decimal(0)
    )
    return CapabilityDeltaReport(
        family_metrics=tuple(family_metrics),
        supported_family_count=sum(metric.supported for metric in family_metrics),
        matched_budgets=True,
        adaptive_net_value=adaptive_net,
        random_net_value=random_net,
        adaptive_claim="supported" if adaptive_net > random_net else "not_supported",
    )


__all__ = [
    "AdaptiveSearch",
    "CapabilityDeltaReport",
    "CapabilityPreregistration",
    "FamilyCapabilityMetrics",
    "FamilyThreshold",
    "PolicyMetrics",
    "PrimaryOutcome",
    "SearchResult",
    "capability_delta_report",
]
