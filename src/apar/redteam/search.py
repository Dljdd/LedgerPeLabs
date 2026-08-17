"""Evaluator-owned orchestration, provenance, deadlines, and capability metrics."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import ConfigDict, PrivateAttr, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.redteam.policies import (
    AttackCandidate,
    CandidateContractError,
    Feedback,
    ParameterBounds,
    Policy,
    VisibleTrial,
    normalize_internal_history,
    reconstruct_bounds,
    reconstruct_candidate,
    reconstruct_feedback,
    reconstruct_history,
    validate_candidate_lineage,
    visible_objective,
)

_POLICY_NAMES = ("adaptive", "cached_llm", "fixed", "random")
_POLICY_VERSIONS = {
    "adaptive": "1.0.0",
    "cached_llm": "1.0.0",
    "fixed": "1.0.0",
    "random": "1.0.0",
}
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _exact_digest(label: str, value: object) -> str:
    text = _exact_text(label, value)
    if len(text) != 64 or not set(text) <= _HEX:
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return text


def _exact_non_negative_int(
    label: str,
    value: object,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _exact_decimal(label: str, value: object, *, non_negative: bool = False) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    if not value.is_finite() or (non_negative and value < 0):
        raise ValueError(f"{label} must be finite")
    return value


def _set_seal(contract: ExternalContract, seal: str) -> None:
    private = contract.__pydantic_private__
    if private is None:
        raise RuntimeError("contract private storage is unavailable")
    private["_integrity_seal"] = seal


class DisclosureProfile(ExternalContract):
    """Immutable scenario-owned feedback disclosure configuration."""

    profile_id: str
    expose_realized_value: bool
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("profile_id", mode="before")
    @classmethod
    def profile_is_exact(cls, value: object) -> object:
        return _exact_text("profile_id", value)

    @field_validator("expose_realized_value", mode="before")
    @classmethod
    def exposure_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("expose_realized_value must be an exact bool")
        return value

    def model_post_init(self, _context: object) -> None:
        _set_seal(self, self.profile_digest)

    @property
    def profile_digest(self) -> str:
        return _digest(
            {
                "profile_id": self.profile_id,
                "expose_realized_value": self.expose_realized_value,
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not DisclosureProfile:
            raise CandidateContractError("disclosure profile subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("disclosure profile field set is not exact")
        if self._integrity_seal != self.profile_digest:
            raise CandidateContractError("disclosure profile integrity seal changed")


def _reconstruct_disclosure(value: DisclosureProfile) -> DisclosureProfile:
    if type(value) is not DisclosureProfile:
        raise CandidateContractError("disclosure profile must be exact")
    value.assert_pristine()
    return DisclosureProfile(
        profile_id=value.profile_id,
        expose_realized_value=value.expose_realized_value,
    )


class EvaluationContract(ExternalContract):
    """Digest-only evaluator provenance, never passed to an attacker policy."""

    family: str
    bounds_digest: str
    hidden_template_digest: str
    background_digest: str
    population_digest: str
    evaluator_digest: str
    defender_digest: str
    disclosure_profile: DisclosureProfile
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator(
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        mode="before",
    )
    @classmethod
    def digests_are_exact(cls, value: object) -> object:
        return _exact_digest("evaluation provenance digest", value)

    @field_validator("disclosure_profile", mode="before")
    @classmethod
    def disclosure_is_exact(cls, value: object) -> object:
        if type(value) is not DisclosureProfile:
            raise TypeError("disclosure_profile must be exact")
        value.assert_pristine()
        return value

    def model_post_init(self, _context: object) -> None:
        _set_seal(self, self.contract_digest)

    @property
    def disclosure_profile_digest(self) -> str:
        return self.disclosure_profile.profile_digest

    @property
    def contract_digest(self) -> str:
        return _digest(
            {
                "family": self.family,
                "bounds_digest": self.bounds_digest,
                "hidden_template_digest": self.hidden_template_digest,
                "background_digest": self.background_digest,
                "population_digest": self.population_digest,
                "evaluator_digest": self.evaluator_digest,
                "defender_digest": self.defender_digest,
                "disclosure_profile_digest": self.disclosure_profile_digest,
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not EvaluationContract:
            raise CandidateContractError("evaluation contract subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("evaluation contract field set is not exact")
        self.disclosure_profile.assert_pristine()
        if self._integrity_seal != self.contract_digest:
            raise CandidateContractError("evaluation contract integrity seal changed")


def reconstruct_evaluation_contract(value: EvaluationContract) -> EvaluationContract:
    if type(value) is not EvaluationContract:
        raise CandidateContractError("evaluation contract must be exact")
    value.assert_pristine()
    return EvaluationContract(
        family=value.family,
        bounds_digest=value.bounds_digest,
        hidden_template_digest=value.hidden_template_digest,
        background_digest=value.background_digest,
        population_digest=value.population_digest,
        evaluator_digest=value.evaluator_digest,
        defender_digest=value.defender_digest,
        disclosure_profile=_reconstruct_disclosure(value.disclosure_profile),
    )


class SearchResult(ExternalContract):
    """Complete result bound to exact policy, environment, disclosure, and budgets."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    family: str
    bounds_digest: str
    hidden_template_digest: str
    background_digest: str
    population_digest: str
    evaluator_digest: str
    defender_digest: str
    disclosure_profile_digest: str
    evaluation_contract_digest: str
    policy_name: str
    policy_version: str
    seed: int
    proposals: tuple[AttackCandidate, ...]
    trials: tuple[VisibleTrial, ...]
    objective_values: tuple[Decimal, ...]
    winner: AttackCandidate | None
    proposal_budget: int
    query_budget: int
    logical_time_budget: int
    wall_time_budget_ms: int
    proposals_used: int
    queries_used: int
    logical_time_used: int
    wall_time_elapsed_ms: int
    wall_time_exhausted: bool
    wall_time_overrun_ms: int

    @field_validator("family", "policy_name", "policy_version", mode="before")
    @classmethod
    def text_is_exact(cls, value: object) -> object:
        return _exact_text("result text", value)

    @field_validator(
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "disclosure_profile_digest",
        "evaluation_contract_digest",
        mode="before",
    )
    @classmethod
    def result_digests_are_exact(cls, value: object) -> object:
        return _exact_digest("result provenance digest", value)

    @field_validator("seed", mode="before")
    @classmethod
    def seed_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("seed", value, maximum=2**63 - 1)

    @field_validator("proposals", mode="before")
    @classmethod
    def proposals_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not AttackCandidate for item in value):
            raise TypeError("proposals must be an exact tuple of candidates")
        return value

    @field_validator("trials", mode="before")
    @classmethod
    def trials_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not VisibleTrial for item in value):
            raise TypeError("trials must be an exact tuple of visible trials")
        return value

    @field_validator("objective_values", mode="before")
    @classmethod
    def objectives_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not Decimal or not item.is_finite() for item in value
        ):
            raise TypeError("objective values must be exact finite Decimals")
        return value

    @field_validator("winner", mode="before")
    @classmethod
    def winner_is_exact(cls, value: object) -> object:
        if value is not None and type(value) is not AttackCandidate:
            raise TypeError("winner must be an exact candidate or None")
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
    def search_counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("search count", value, maximum=1000)

    @field_validator(
        "wall_time_budget_ms",
        "wall_time_elapsed_ms",
        "wall_time_overrun_ms",
        mode="before",
    )
    @classmethod
    def wall_counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("wall time", value, maximum=3_600_000)

    @field_validator("wall_time_exhausted", mode="before")
    @classmethod
    def exhausted_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("wall_time_exhausted must be an exact bool")
        return value

    @model_validator(mode="after")
    def result_is_consistent(self) -> Self:
        size = len(self.proposals)
        if len(self.trials) != size or len(self.objective_values) != size:
            raise ValueError("proposal, trial, and objective sequences must align")
        if tuple(trial.candidate for trial in self.trials) != self.proposals:
            raise ValueError("trial candidates must preserve proposal order")
        if tuple(trial.objective_value for trial in self.trials) != self.objective_values:
            raise ValueError("trial objectives must preserve objective order")
        if not (
            self.proposal_budget == self.query_budget == self.logical_time_budget
            and self.proposals_used == self.queries_used == self.logical_time_used == size
            and size <= self.proposal_budget
        ):
            raise ValueError("proposal, query, and logical-time accounting must match")
        if not self.wall_time_exhausted and size != self.proposal_budget:
            raise ValueError("a non-exhausted search must use its complete discrete budget")
        expected_overrun = max(0, self.wall_time_elapsed_ms - self.wall_time_budget_ms)
        if self.wall_time_overrun_ms != expected_overrun:
            raise ValueError("wall time overrun must be derived from elapsed time")
        if (size == 0) != (self.winner is None):
            raise ValueError("winner presence must match non-empty proposals")
        if self.winner is not None and self.winner not in self.proposals:
            raise ValueError("winner must be a proposed candidate")
        reconstruct_history(self.trials)
        return self


class AdaptiveSearch:
    """Run a policy without passing or retaining the evaluator callable."""

    __slots__ = (
        "_bounds",
        "_clock_ns",
        "_evaluation_contract",
        "_policy",
        "_policy_name",
        "_policy_version",
    )

    def __init__(
        self,
        *,
        policy: Policy,
        bounds: ParameterBounds,
        evaluation_contract: EvaluationContract,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        public_bounds = reconstruct_bounds(bounds)
        contract = reconstruct_evaluation_contract(evaluation_contract)
        if contract.family != public_bounds.family:
            raise ValueError("evaluation contract family does not match public bounds")
        if contract.bounds_digest != public_bounds.bounds_digest:
            raise ValueError("evaluation contract does not bind these public bounds")
        if not callable(getattr(policy, "propose", None)):
            raise TypeError("policy must expose propose")
        policy_name = _exact_text("policy_name", getattr(policy, "policy_name", None))
        policy_version = _exact_text("policy_version", getattr(policy, "policy_version", None))
        if policy_name not in {*_POLICY_NAMES, "llm"}:
            raise ValueError("policy_name is undeclared")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._policy = policy
        self._policy_name = policy_name
        self._policy_version = policy_version
        self._bounds = public_bounds
        self._evaluation_contract = contract
        self._clock_ns = clock_ns

    def _clock(self, previous: int | None = None) -> int:
        value = self._clock_ns()
        if type(value) is not int or value < 0:
            raise TypeError("monotonic clock must return an exact non-negative integer")
        if previous is not None and value < previous:
            raise RuntimeError("monotonic clock moved backwards")
        return value

    def search(
        self,
        seed: int,
        budget: int,
        wall_time_budget_ms: int,
        evaluate: Callable[[AttackCandidate], Feedback],
    ) -> SearchResult:
        checked_seed = _exact_non_negative_int("seed", seed, maximum=2**63 - 1)
        checked_budget = _exact_non_negative_int("budget", budget, maximum=1000)
        checked_wall = _exact_non_negative_int(
            "wall_time_budget_ms", wall_time_budget_ms, maximum=3_600_000
        )
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        rng = np.random.default_rng(checked_seed)
        proposals: list[AttackCandidate] = []
        trials: list[VisibleTrial] = []
        objectives: list[Decimal] = []
        start = self._clock()
        latest = start
        deadline = start + checked_wall * 1_000_000
        exhausted = False
        for _ in range(checked_budget):
            latest = self._clock(latest)
            if latest >= deadline:
                exhausted = True
                break
            visible_history = normalize_internal_history(tuple(trials))
            public_bounds = self._bounds
            candidate = self._policy.propose(visible_history, public_bounds, rng)
            latest = self._clock(latest)
            if latest >= deadline:
                exhausted = True
                break
            checked_candidate = validate_candidate_lineage(candidate, visible_history)
            public_bounds.validate_vector(checked_candidate.params)
            try:
                returned = reconstruct_feedback(evaluate(checked_candidate))
                feedback = Feedback(
                    action=returned.action,
                    reason_family=returned.reason_family,
                    realized_value=(
                        returned.realized_value
                        if self._evaluation_contract.disclosure_profile.expose_realized_value
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
            proposal = reconstruct_candidate(checked_candidate)
            trial = VisibleTrial(
                candidate=proposal,
                feedback=feedback,
                objective_value=objective,
            )
            proposals.append(proposal)
            trials.append(trial)
            objectives.append(objective)
            latest = self._clock(latest)
            if latest >= deadline and len(proposals) < checked_budget:
                exhausted = True
                break
        end = self._clock(latest)
        elapsed_ns = end - start
        elapsed_ms = elapsed_ns // 1_000_000
        if elapsed_ns > checked_wall * 1_000_000 and elapsed_ms == checked_wall:
            elapsed_ms += 1
        if checked_budget > len(proposals) and end >= deadline:
            exhausted = True
        winner = (
            None
            if not trials
            else min(
                trials,
                key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id),
            ).candidate
        )
        contract = self._evaluation_contract
        return SearchResult(
            family=contract.family,
            bounds_digest=contract.bounds_digest,
            hidden_template_digest=contract.hidden_template_digest,
            background_digest=contract.background_digest,
            population_digest=contract.population_digest,
            evaluator_digest=contract.evaluator_digest,
            defender_digest=contract.defender_digest,
            disclosure_profile_digest=contract.disclosure_profile_digest,
            evaluation_contract_digest=contract.contract_digest,
            policy_name=self._policy_name,
            policy_version=self._policy_version,
            seed=checked_seed,
            proposals=tuple(proposals),
            trials=tuple(trials),
            objective_values=tuple(objectives),
            winner=winner,
            proposal_budget=checked_budget,
            query_budget=checked_budget,
            logical_time_budget=checked_budget,
            wall_time_budget_ms=checked_wall,
            proposals_used=len(proposals),
            queries_used=len(proposals),
            logical_time_used=len(proposals),
            wall_time_elapsed_ms=elapsed_ms,
            wall_time_exhausted=exhausted,
            wall_time_overrun_ms=max(0, elapsed_ms - checked_wall),
        )


class PrimaryOutcome(StrEnum):
    VALID_YIELD = "valid_yield"
    NET_SETTLED_VALUE = "net_settled_value"
    ADAPTATION_SPEED = "adaptation_speed"
    CAMPAIGN_SCALE = "campaign_scale"


class FamilyThreshold(ExternalContract):
    """Preregistered family outcome, threshold, and exact evaluator provenance."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal
    evaluation_contract: EvaluationContract

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

    @field_validator("minimum_delta", mode="before")
    @classmethod
    def delta_is_exact(cls, value: object) -> object:
        checked = _exact_decimal("minimum_delta", value, non_negative=True)
        if checked <= 0:
            raise ValueError("minimum_delta must be strictly positive")
        return checked

    @field_validator("evaluation_contract", mode="before")
    @classmethod
    def contract_is_exact(cls, value: object) -> object:
        if type(value) is not EvaluationContract:
            raise TypeError("evaluation_contract must be exact")
        return reconstruct_evaluation_contract(value)

    @model_validator(mode="after")
    def family_matches_contract(self) -> Self:
        if self.family != self.evaluation_contract.family:
            raise ValueError("threshold family must match evaluation contract")
        return self


class CapabilityPreregistration(ExternalContract):
    seeds: tuple[int, ...]
    budget: int
    wall_time_budget_ms: int
    thresholds: tuple[FamilyThreshold, ...]

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise TypeError("seeds must be a non-empty exact tuple")
        checked = tuple(_exact_non_negative_int("seed", item, maximum=2**63 - 1) for item in value)
        if checked != tuple(sorted(set(checked))):
            raise ValueError("seeds must be unique and sorted")
        return checked

    @field_validator("budget", mode="before")
    @classmethod
    def budget_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("budget", value, maximum=1000)

    @field_validator("wall_time_budget_ms", mode="before")
    @classmethod
    def wall_budget_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("wall time budget", value, maximum=3_600_000)

    @field_validator("thresholds", mode="before")
    @classmethod
    def thresholds_are_exact(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or not value
            or any(type(item) is not FamilyThreshold for item in value)
        ):
            raise TypeError("thresholds must be a non-empty exact tuple")
        families = tuple(item.family for item in value)
        if families != tuple(sorted(set(families))):
            raise ValueError("threshold families must be unique and sorted")
        return value


class PolicyMetrics(ExternalContract):
    """Observed policy metrics with yield derived rather than caller supplied."""

    proposal_count: int
    approved_count: int
    net_settled_value: Decimal
    adaptation_speed: Decimal
    campaign_scale: int

    @field_validator("proposal_count", "approved_count", "campaign_scale", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("metric count", value)

    @field_validator("net_settled_value", "adaptation_speed", mode="before")
    @classmethod
    def decimals_are_exact(cls, value: object) -> object:
        return _exact_decimal("metric Decimal", value, non_negative=True)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> Self:
        if self.approved_count > self.proposal_count:
            raise ValueError("approved_count cannot exceed proposal_count")
        if self.campaign_scale != self.approved_count:
            raise ValueError("campaign_scale must equal observed approved campaigns")
        return self

    @property
    def valid_yield(self) -> Decimal:
        if self.proposal_count == 0:
            return Decimal(0)
        with localcontext() as context:
            context.prec = 28
            return Decimal(self.approved_count) / Decimal(self.proposal_count)


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


class FamilyCapabilityMetrics(ExternalContract):
    """Matched policy aggregates with delta and support derived internally."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal
    evaluation_contract_digest: str
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
            raise TypeError("primary outcome must be exact")
        return value

    @field_validator("minimum_delta", mode="before")
    @classmethod
    def minimum_is_exact(cls, value: object) -> object:
        return _exact_decimal("minimum_delta", value, non_negative=True)

    @field_validator("evaluation_contract_digest", mode="before")
    @classmethod
    def contract_digest_is_exact(cls, value: object) -> object:
        return _exact_digest("evaluation contract digest", value)

    @field_validator("fixed", "random", "adaptive", "cached_llm", mode="before")
    @classmethod
    def metrics_are_exact(cls, value: object) -> object:
        if type(value) is not PolicyMetrics:
            raise TypeError("policy metrics must be exact")
        return value

    @property
    def observed_delta(self) -> Decimal:
        return _observed_delta(self.primary_outcome, self.adaptive, self.random)

    @property
    def supported(self) -> bool:
        return self.observed_delta >= self.minimum_delta


class CapabilityDeltaReport(ExternalContract):
    """Capability report whose counts and adaptive claim cannot be relabeled."""

    family_metrics: tuple[FamilyCapabilityMetrics, ...]
    matched_budgets: bool

    @field_validator("family_metrics", mode="before")
    @classmethod
    def metrics_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not FamilyCapabilityMetrics for item in value
        ):
            raise TypeError("family_metrics must be an exact tuple")
        return value

    @field_validator("matched_budgets", mode="before")
    @classmethod
    def matched_is_true(cls, value: object) -> object:
        if value is not True or type(value) is not bool:
            raise ValueError("matched_budgets must be observed true")
        return value

    @property
    def supported_family_count(self) -> int:
        return sum(metric.supported for metric in self.family_metrics)

    @property
    def adaptive_net_value(self) -> Decimal:
        return sum(
            (metric.adaptive.net_settled_value for metric in self.family_metrics),
            Decimal(0),
        )

    @property
    def random_net_value(self) -> Decimal:
        return sum(
            (metric.random.net_settled_value for metric in self.family_metrics),
            Decimal(0),
        )

    @property
    def adaptive_claim(self) -> str:
        return "supported" if self.adaptive_net_value > self.random_net_value else "not_supported"


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
    first_successes = tuple(
        next(
            (
                index
                for index, trial in enumerate(result.trials, start=1)
                if trial.feedback.action is Action.APPROVE
            ),
            len(result.trials) + 1,
        )
        for result in results
    )
    with localcontext() as context:
        context.prec = 28
        speed = Decimal(sum(first_successes)) / Decimal(len(first_successes))
    return PolicyMetrics(
        proposal_count=proposal_count,
        approved_count=len(approved),
        net_settled_value=net_value,
        adaptation_speed=speed,
        campaign_scale=len(approved),
    )


def capability_delta_report(
    preregistration: CapabilityPreregistration,
    results: dict[str, dict[str, tuple[SearchResult, ...]]],
) -> CapabilityDeltaReport:
    """Compute only preregistered outcomes from exact matched provenance cells."""
    if type(preregistration) is not CapabilityPreregistration:
        raise TypeError("preregistration must be exact")
    if type(results) is not dict or any(type(key) is not str for key in results):
        raise TypeError("results must have exact string family keys")
    thresholds = {item.family: item for item in preregistration.thresholds}
    if set(results) != set(thresholds):
        raise ValueError("results must contain exactly preregistered families")
    metrics: list[FamilyCapabilityMetrics] = []
    for family in sorted(results):
        cells = results[family]
        if type(cells) is not dict or any(type(key) is not str for key in cells):
            raise TypeError("policy cells must have exact string keys")
        if tuple(sorted(cells)) != _POLICY_NAMES:
            raise ValueError("each family needs fixed, random, adaptive, and cached_llm")
        threshold = thresholds[family]
        contract = threshold.evaluation_contract
        aggregates: dict[str, PolicyMetrics] = {}
        for policy_name in _POLICY_NAMES:
            runs = cells[policy_name]
            if (
                type(runs) is not tuple
                or len(runs) != len(preregistration.seeds)
                or any(type(result) is not SearchResult for result in runs)
            ):
                raise TypeError("policy runs must match preregistered seeds")
            if tuple(result.seed for result in runs) != preregistration.seeds:
                raise ValueError("capability comparison seeds are not matched")
            for result in runs:
                if result.policy_name != policy_name:
                    raise ValueError("capability result was relabeled under another policy")
                if result.policy_version != _POLICY_VERSIONS[policy_name]:
                    raise ValueError("capability result policy version does not match")
                if result.family != family:
                    raise ValueError("capability result family was swapped")
                result_provenance = (
                    result.bounds_digest,
                    result.hidden_template_digest,
                    result.background_digest,
                    result.population_digest,
                    result.evaluator_digest,
                    result.defender_digest,
                    result.disclosure_profile_digest,
                    result.evaluation_contract_digest,
                )
                contract_provenance = (
                    contract.bounds_digest,
                    contract.hidden_template_digest,
                    contract.background_digest,
                    contract.population_digest,
                    contract.evaluator_digest,
                    contract.defender_digest,
                    contract.disclosure_profile_digest,
                    contract.contract_digest,
                )
                if result_provenance != contract_provenance:
                    raise ValueError("capability result evaluator provenance does not match")
                if not (
                    result.proposal_budget
                    == result.query_budget
                    == result.logical_time_budget
                    == preregistration.budget
                    and result.wall_time_budget_ms == preregistration.wall_time_budget_ms
                ):
                    raise ValueError("capability comparison budgets are not matched")
            aggregates[policy_name] = _aggregate(runs)
        metrics.append(
            FamilyCapabilityMetrics(
                family=family,
                primary_outcome=threshold.primary_outcome,
                minimum_delta=threshold.minimum_delta,
                evaluation_contract_digest=contract.contract_digest,
                fixed=aggregates["fixed"],
                random=aggregates["random"],
                adaptive=aggregates["adaptive"],
                cached_llm=aggregates["cached_llm"],
            )
        )
    return CapabilityDeltaReport(family_metrics=tuple(metrics), matched_budgets=True)


__all__ = [
    "AdaptiveSearch",
    "CapabilityDeltaReport",
    "CapabilityPreregistration",
    "DisclosureProfile",
    "EvaluationContract",
    "FamilyCapabilityMetrics",
    "FamilyThreshold",
    "PolicyMetrics",
    "PrimaryOutcome",
    "SearchResult",
    "capability_delta_report",
    "reconstruct_evaluation_contract",
]
