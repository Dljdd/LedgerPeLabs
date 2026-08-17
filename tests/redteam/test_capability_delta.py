"""Preregistered capability comparison and honest-claim contracts."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.generators import CampaignGenerator, GenerationConstraintError, PopulationGenerator
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    CapabilityPreregistration,
    FamilyThreshold,
    Feedback,
    FixedPolicy,
    LLMPlannerPolicy,
    ParameterBounds,
    PrimaryOutcome,
    RandomPolicy,
    capability_delta_report,
)
from tests.factories import make_scenario_config, make_threat_card
from tests.redteam.conftest import campaign_params


def _runs(bounds, policy, *, seed: int, budget: int):  # type: ignore[no-untyped-def]
    def evaluate(candidate):  # type: ignore[no-untyped-def]
        approved = candidate.params.retry_intensity == 1
        return Feedback(
            action=Action.APPROVE if approved else Action.DECLINE,
            reason_family="approved" if approved else "velocity",
            realized_value=Decimal("60.00") if approved else Decimal("0.00"),
        )

    return AdaptiveSearch(
        policy=policy,
        bounds=bounds,
        disclose_realized_value=True,
    ).search(seed=seed, budget=budget, evaluate=evaluate)


def _cached_simple_run(bounds, *, seed: int, budget: int):  # type: ignore[no-untyped-def]
    online = LLMPlannerPolicy(_DefaultPlannerClient(bounds.defaults_document()))
    _runs(bounds, online, seed=seed, budget=budget)
    cached = LLMPlannerPolicy(
        _NoNetworkPlannerClient(bounds.defaults_document()),
        replay_cache=online.export_replay_cache(),
        require_cached_replay=True,
    )
    return _runs(bounds, cached, seed=seed, budget=budget)


def test_capability_report_uses_observed_thresholds_and_matched_budgets(card_bounds) -> None:  # type: ignore[no-untyped-def]
    preregistration = CapabilityPreregistration(
        seeds=(4, 9),
        budget=12,
        thresholds=(
            FamilyThreshold(
                family="card_a",
                primary_outcome=PrimaryOutcome.NET_SETTLED_VALUE,
                minimum_delta=Decimal("1"),
            ),
            FamilyThreshold(
                family="card_b",
                primary_outcome=PrimaryOutcome.VALID_YIELD,
                minimum_delta=Decimal("0.01"),
            ),
        ),
    )
    results = {
        family: {
            "fixed": tuple(
                _runs(card_bounds, FixedPolicy(), seed=seed, budget=12)
                for seed in preregistration.seeds
            ),
            "random": tuple(
                _runs(card_bounds, RandomPolicy(), seed=seed, budget=12)
                for seed in preregistration.seeds
            ),
            "adaptive": tuple(
                _runs(card_bounds, AdaptiveTournamentPolicy(), seed=seed, budget=12)
                for seed in preregistration.seeds
            ),
            "cached_llm": tuple(
                _cached_simple_run(card_bounds, seed=seed, budget=12)
                for seed in preregistration.seeds
            ),
        }
        for family in ("card_a", "card_b")
    }
    report = capability_delta_report(preregistration, results)
    assert report.matched_budgets is True
    assert report.supported_family_count >= 2, report
    assert report.adaptive_claim == (
        "supported"
        if report.adaptive_net_value > report.random_net_value
        else "not_supported"
    )


def test_no_delta_is_reported_not_supported(card_bounds) -> None:  # type: ignore[no-untyped-def]
    preregistration = CapabilityPreregistration(
        seeds=(1,),
        budget=1,
        thresholds=(
            FamilyThreshold(
                family="negative_control",
                primary_outcome=PrimaryOutcome.NET_SETTLED_VALUE,
                minimum_delta=Decimal("0.01"),
            ),
        ),
    )
    report = capability_delta_report(
        preregistration,
        {
            "negative_control": {
                "fixed": (_runs(card_bounds, FixedPolicy(), seed=1, budget=1),),
                "random": (_runs(card_bounds, RandomPolicy(), seed=1, budget=1),),
                "adaptive": (
                    _runs(card_bounds, AdaptiveTournamentPolicy(), seed=1, budget=1),
                ),
                "cached_llm": (
                    _cached_simple_run(card_bounds, seed=1, budget=1),
                ),
            }
        },
    )
    assert report.supported_family_count == 0
    assert report.adaptive_claim == "not_supported"


def test_capability_report_rejects_unmatched_budgets(card_bounds) -> None:  # type: ignore[no-untyped-def]
    preregistration = CapabilityPreregistration(
        seeds=(1,),
        budget=2,
        thresholds=(
            FamilyThreshold(
                family="negative_control",
                primary_outcome=PrimaryOutcome.VALID_YIELD,
                minimum_delta=Decimal("0.01"),
            ),
        ),
    )
    wrong_budget = _runs(card_bounds, FixedPolicy(), seed=1, budget=1)
    results = {
        "negative_control": {
            "fixed": (wrong_budget,),
            "random": (_runs(card_bounds, RandomPolicy(), seed=1, budget=1),),
            "adaptive": (
                _runs(card_bounds, AdaptiveTournamentPolicy(), seed=1, budget=1),
            ),
            "cached_llm": (
                _cached_simple_run(card_bounds, seed=1, budget=1),
            ),
        }
    }
    with pytest.raises(ValueError, match="budgets are not matched"):
        capability_delta_report(preregistration, results)


def test_capability_threshold_must_be_measurable_and_results_cannot_be_relabelled(
    card_bounds,
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="strictly positive"):
        FamilyThreshold(
            family="negative_control",
            primary_outcome=PrimaryOutcome.VALID_YIELD,
            minimum_delta=Decimal("0"),
        )
    preregistration = CapabilityPreregistration(
        seeds=(1,),
        budget=1,
        thresholds=(
            FamilyThreshold(
                family="negative_control",
                primary_outcome=PrimaryOutcome.VALID_YIELD,
                minimum_delta=Decimal("0.01"),
            ),
        ),
    )
    fixed = _runs(card_bounds, FixedPolicy(), seed=1, budget=1)
    with pytest.raises(ValueError, match="relabeled"):
        capability_delta_report(
            preregistration,
            {
                "negative_control": {
                    "fixed": (fixed,),
                    "random": (fixed,),
                    "adaptive": (
                        _runs(
                            card_bounds,
                            AdaptiveTournamentPolicy(),
                            seed=1,
                            budget=1,
                        ),
                    ),
                    "cached_llm": (
                        _cached_simple_run(card_bounds, seed=1, budget=1),
                    ),
                }
            },
        )


class _DefaultPlannerClient:
    provider = "fixture"
    model_id = "cached-default-v1"

    def __init__(self, defaults: dict[str, object]) -> None:
        self._defaults = defaults
        self.calls = 0

    def complete(self, request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        history = request["history"]
        assert type(history) is list
        return {
            "output": {
                "params": self._defaults,
                "parent_id": None,
                "generation": len(history),
            },
            "latency_ms": 1,
            "input_tokens": 1,
            "output_tokens": 1,
        }


class _NoNetworkPlannerClient(_DefaultPlannerClient):
    def complete(self, request: dict[str, object]) -> dict[str, object]:
        raise AssertionError("cached planner attempted a network call")


def _population(seed: int):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.A2A,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
    )
    bundle = compile_scenario(
        make_threat_card(rails=[Rail.A2A], default_config=config), config
    )
    return PopulationGenerator(seed=seed).generate(bundle)


def _real_search(family, population, bounds, policy, seed, budget):  # type: ignore[no-untyped-def]
    def evaluate(candidate):  # type: ignore[no-untyped-def]
        try:
            CampaignGenerator(seed=seed).generate(family, population, candidate.params)
        except GenerationConstraintError:
            return Feedback(
                action=Action.DECLINE,
                reason_family="invalid_candidate",
                realized_value=None,
            )
        if family == "card_testing_cnp":
            approved = candidate.params.retry_intensity == 1
            reason = "velocity"
        else:
            approved = candidate.params.cash_out_fraction == min(
                bounds.domain("cash_out_fraction").values
            )
            reason = "amount"
        return Feedback(
            action=Action.APPROVE if approved else Action.DECLINE,
            reason_family="approved" if approved else reason,
            realized_value=Decimal("25.00") if approved else Decimal("0.00"),
        )

    return AdaptiveSearch(
        policy=policy,
        bounds=bounds,
        disclose_realized_value=True,
    ).search(seed=seed, budget=budget, evaluate=evaluate)


def test_two_families_show_preregistered_delta_on_real_task5_feasibility() -> None:
    seeds = (4, 9)
    budget = 8
    preregistration = CapabilityPreregistration(
        seeds=seeds,
        budget=budget,
        thresholds=(
            FamilyThreshold(
                family="app_scam_mule",
                primary_outcome=PrimaryOutcome.VALID_YIELD,
                minimum_delta=Decimal("0.10"),
            ),
            FamilyThreshold(
                family="card_testing_cnp",
                primary_outcome=PrimaryOutcome.VALID_YIELD,
                minimum_delta=Decimal("0.10"),
            ),
        ),
    )
    population = _population(960)
    results = {}
    for family in ("app_scam_mule", "card_testing_cnp"):
        bounds = ParameterBounds.for_campaign(family, campaign_params(family))
        cells = {
            "fixed": tuple(
                _real_search(family, population, bounds, FixedPolicy(), seed, budget)
                for seed in seeds
            ),
            "random": tuple(
                _real_search(family, population, bounds, RandomPolicy(), seed, budget)
                for seed in seeds
            ),
            "adaptive": tuple(
                _real_search(
                    family,
                    population,
                    bounds,
                    AdaptiveTournamentPolicy(),
                    seed,
                    budget,
                )
                for seed in seeds
            ),
        }
        cached_runs = []
        for seed in seeds:
            online_client = _DefaultPlannerClient(bounds.defaults_document())
            online = LLMPlannerPolicy(online_client)
            _real_search(family, population, bounds, online, seed, budget)
            offline = LLMPlannerPolicy(
                _NoNetworkPlannerClient(bounds.defaults_document()),
                replay_cache=online.export_replay_cache(),
                require_cached_replay=True,
            )
            cached_runs.append(
                _real_search(family, population, bounds, offline, seed, budget)
            )
        cells["cached_llm"] = tuple(cached_runs)
        results[family] = cells

    report = capability_delta_report(preregistration, results)
    assert report.matched_budgets is True
    assert report.supported_family_count >= 2, [
        (
            metric.family,
            metric.random.valid_yield,
            metric.adaptive.valid_yield,
            metric.observed_delta,
        )
        for metric in report.family_metrics
    ]
    assert all(metric.supported for metric in report.family_metrics)
    invalid = [
        (family_name, policy_name, trial.candidate.params)
        for family_name, family in results.items()
        for policy_name, runs in family.items()
        for result in runs
        for trial in result.trials
        if trial.feedback.reason_family == "invalid_candidate"
    ]
    assert invalid == []


def test_all_four_public_bounds_emit_real_task5_feasible_candidates() -> None:
    population = _population(961)
    for family in (
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
        "agentic_intent_abuse",
    ):
        params = campaign_params(family)
        if family == "agentic_intent_abuse":
            with localcontext() as context:
                context.prec = 28
                rate = Decimal(24) / Decimal(26)
            params = replace(
                params,
                payment_count=26,
                target_illicit_rate=rate,
                agentic_attack_mix=rate,
            )
        bounds = ParameterBounds.for_campaign(family, params)
        for seed in range(4):
            candidate = RandomPolicy().propose((), bounds, np.random.default_rng(seed))
            commands = CampaignGenerator(seed=seed).generate(
                family, population, candidate.params
            )
            assert commands
