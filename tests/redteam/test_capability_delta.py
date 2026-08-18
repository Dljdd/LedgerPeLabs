"""Preregistered, replay-backed capability comparison and honest claims."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from apar.generators import Population
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AttackCandidate,
    CapabilityPreregistration,
    FamilyCapabilityMetrics,
    FamilyThreshold,
    FixedPolicy,
    LLMPlannerPolicy,
    PolicyMetrics,
    PrimaryOutcome,
    RandomPolicy,
    RunGroupCapability,
    SearchAuthority,
    SearchResult,
    capability_delta_report,
)
from apar.redteam.benchmark import (
    CampaignBenchmark,
    DefenderRuleSet,
    default_defender_rules,
)
from tests.redteam.conftest import campaign_benchmark, campaign_params

_SEEDS = (4, 9)
_BUDGET = 8
_WALL_BUDGET_MS = 60_000


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
        parent_id = None
        if history:
            latest = history[-1]
            assert type(latest) is dict
            parent_id = latest["candidate_id"]
        return {
            "output": {
                "params": self._defaults,
                "parent_id": parent_id,
                "generation": len(history),
            },
            "latency_ms": 1,
            "input_tokens": 1,
            "output_tokens": 1,
        }


class _NoNetworkPlannerClient(_DefaultPlannerClient):
    def complete(self, request: dict[str, object]) -> dict[str, object]:
        raise AssertionError("cached planner attempted a network call")


@pytest.fixture(scope="module")
def app_benchmark(benchmark_population: Population) -> CampaignBenchmark:
    return campaign_benchmark(
        "app_scam_mule",
        benchmark_population,
        expose_realized_value=True,
    )


@pytest.fixture(scope="module")
def value_card_benchmark(benchmark_population: Population) -> CampaignBenchmark:
    return campaign_benchmark(
        "card_testing_cnp",
        benchmark_population,
        expose_realized_value=True,
    )


@dataclass(frozen=True)
class _Experiment:
    authority: SearchAuthority
    run_group: RunGroupCapability
    benchmarks: dict[str, CampaignBenchmark]
    evaluators: dict[str, object]
    policies: dict[str, object]
    preregistration: CapabilityPreregistration


def _experiment(
    benchmarks: tuple[CampaignBenchmark, ...],
    *,
    seeds: tuple[int, ...] = _SEEDS,
    budget: int = _BUDGET,
    minimum_delta: Decimal = Decimal("0.10"),
) -> _Experiment:
    authority = SearchAuthority()
    run_group = authority.issue_run_group("capability-delta-test")
    by_family = {
        benchmark.evaluation_contract.family: benchmark for benchmark in benchmarks
    }
    evaluators = {
        family: benchmark.issue_evaluator_capability(authority)
        for family, benchmark in by_family.items()
    }
    policy_objects = {
        "fixed": FixedPolicy(),
        "random": RandomPolicy(),
        "adaptive": AdaptiveTournamentPolicy(),
        # Cached replay mechanics are tested independently below and in the LLM suite.
        # The comparison fixture registers a deterministic no-network planner baseline.
        "cached_llm": FixedPolicy(),
    }
    policies = {
        name: authority.register_policy(policy, name=name, version="test-v1")
        for name, policy in policy_objects.items()
    }
    thresholds = tuple(
        FamilyThreshold(
            family=family,
            primary_outcome=PrimaryOutcome.VALID_YIELD,
            minimum_delta=minimum_delta,
            evaluation_contract=evaluators[family].evaluation_contract,  # type: ignore[attr-defined]
            evaluator_capability_id=evaluators[family].capability_id,  # type: ignore[attr-defined]
            evaluator_code_digest=evaluators[family].evaluator_code_digest,  # type: ignore[attr-defined]
        )
        for family in sorted(by_family)
    )
    preregistration = authority.issue_preregistration(
        run_group=run_group,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=_WALL_BUDGET_MS,
        thresholds=thresholds,
        policies=tuple(policies[name] for name in sorted(policies)),  # type: ignore[arg-type]
    )
    return _Experiment(
        authority=authority,
        run_group=run_group,
        benchmarks=by_family,
        evaluators=evaluators,
        policies=policies,
        preregistration=preregistration,
    )


def _run(
    experiment: _Experiment,
    family: str,
    policy_name: str,
    *,
    seed: int,
    budget: int = _BUDGET,
    wall_budget_ms: int = _WALL_BUDGET_MS,
) -> SearchResult:
    return AdaptiveSearch(
        evaluator_capability=experiment.evaluators[family],  # type: ignore[arg-type]
        policy_capability=experiment.policies[policy_name],  # type: ignore[arg-type]
        run_group=experiment.run_group,
    ).search(
        seed=seed,
        budget=budget,
        wall_time_budget_ms=wall_budget_ms,
    )


def _cached_run(
    benchmark: CampaignBenchmark,
    *,
    seed: int,
    budget: int = _BUDGET,
    wall_budget_ms: int = _WALL_BUDGET_MS,
) -> SearchResult:
    authority = SearchAuthority()
    evaluator = benchmark.issue_evaluator_capability(authority)
    group = authority.issue_run_group("cached-replay-test")
    defaults = benchmark.public_bounds.defaults_document()
    online_client = _DefaultPlannerClient(defaults)
    online = LLMPlannerPolicy(online_client)
    online_capability = authority.register_policy(
        online, name="online_llm", version="test-v1"
    )
    AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=online_capability,
        run_group=group,
    ).search(
        seed=seed, budget=budget, wall_time_budget_ms=wall_budget_ms
    )
    assert online_client.calls == budget
    offline_client = _NoNetworkPlannerClient(defaults)
    cached = LLMPlannerPolicy(
        offline_client,
        replay_cache=online.export_replay_cache(),
        require_cached_replay=True,
    )
    cached_capability = authority.register_policy(
        cached, name="cached_llm", version="test-v1"
    )
    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=cached_capability,
        run_group=group,
    ).search(
        seed=seed, budget=budget, wall_time_budget_ms=wall_budget_ms
    )
    assert offline_client.calls == 0
    assert all(record.call_status == "cache_success" for record in cached.take_audit_records())
    return result


def _matched_results(
    experiment: _Experiment,
) -> dict[str, dict[str, tuple[SearchResult, ...]]]:
    results: dict[str, dict[str, tuple[SearchResult, ...]]] = {}
    for family in experiment.benchmarks:
        results[family] = {
            name: tuple(
                _run(
                    experiment,
                    family,
                    name,
                    seed=seed,
                    budget=experiment.preregistration.budget,
                    wall_budget_ms=experiment.preregistration.wall_time_budget_ms,
                )
                for seed in experiment.preregistration.seeds
            )
            for name in ("fixed", "random", "adaptive", "cached_llm")
        }
    return results


def test_replay_backed_capability_report_is_honest_and_matched(
    app_benchmark: CampaignBenchmark,
    value_card_benchmark: CampaignBenchmark,
) -> None:
    # Thresholds and exact evaluator contracts are frozen before any policy trial.
    benchmarks = (app_benchmark, value_card_benchmark)
    experiment = _experiment(benchmarks)
    results = _matched_results(experiment)

    report = capability_delta_report(
        experiment.preregistration, results, authority=experiment.authority
    )

    assert report.matched_budgets is True
    assert report.supported_family_count == sum(
        metric.observed_delta >= metric.minimum_delta for metric in report.family_metrics
    )
    assert report.adaptive_claim == (
        "supported"
        if report.adaptive_net_value > report.random_net_value
        else "not_supported"
    )
    assert all(
        trial.feedback.reason_family != "invalid_candidate"
        for family_cells in results.values()
        for runs in family_cells.values()
        for result in runs
        for trial in result.trials
    )


def test_minimum_agentic_space_is_a_negative_no_delta_control(
    benchmark_population: Population,
) -> None:
    benchmark = campaign_benchmark(
        "agentic_intent_abuse",
        benchmark_population,
        expose_realized_value=True,
    )
    experiment = _experiment(
        (benchmark,),
        seeds=(1, 2),
        budget=2,
        minimum_delta=Decimal("0.01"),
    )
    results = _matched_results(experiment)

    report = capability_delta_report(
        experiment.preregistration, results, authority=experiment.authority
    )

    assert benchmark.public_bounds.names == ()
    assert report.supported_family_count == 0
    assert report.family_metrics[0].observed_delta == 0
    assert report.adaptive_claim == "not_supported"


def test_defender_rule_permutation_cannot_change_evaluation(
    benchmark_population: Population,
    app_benchmark: CampaignBenchmark,
) -> None:
    declared = default_defender_rules()
    permuted = DefenderRuleSet(
        version=declared.version,
        rules=tuple(reversed(declared.rules)),
    )
    alternate = CampaignBenchmark(
        family="app_scam_mule",
        population=benchmark_population,
        hidden_template=campaign_params("app_scam_mule"),
        defender=permuted,
        disclosure_profile=app_benchmark.evaluation_contract.disclosure_profile,
        generator_seed=960,
    )

    assert permuted.defender_digest == declared.defender_digest
    assert alternate.evaluation_contract.contract_digest == (
        app_benchmark.evaluation_contract.contract_digest
    )
    for vector in app_benchmark.public_bounds.feasible_vectors:
        candidate = AttackCandidate(params=vector, parent_id=None, generation=0)
        assert alternate.evaluate_with_observation(candidate) == (
            app_benchmark.evaluate_with_observation(candidate)
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("policy_name", "random", "relabeled"),
        ("policy_version", "forged-v1", "policy version"),
        ("family", "app_scam_mule", "family was swapped"),
        ("bounds_digest", "0" * 64, "evaluator provenance"),
        ("hidden_template_digest", "0" * 64, "evaluator provenance"),
        ("background_digest", "0" * 64, "evaluator provenance"),
        ("population_digest", "0" * 64, "evaluator provenance"),
        ("evaluator_digest", "0" * 64, "evaluator provenance"),
        ("defender_digest", "0" * 64, "evaluator provenance"),
        ("disclosure_profile_digest", "0" * 64, "evaluator provenance"),
    ),
)
def test_capability_report_rejects_each_cross_provenance_cell(
    value_card_benchmark: CampaignBenchmark,
    field: str,
    replacement: str,
    message: str,
) -> None:
    experiment = _experiment(
        (value_card_benchmark,), seeds=(1,), budget=1
    )
    fixed = _run(experiment, "card_testing_cnp", "fixed", seed=1, budget=1)
    tampered = fixed.model_copy(update={field: replacement})
    results = _matched_results(experiment)
    results["card_testing_cnp"]["fixed"] = (tampered,)

    with pytest.raises(ValueError, match="issued|authentic|pristine"):
        capability_delta_report(
            experiment.preregistration, results, authority=experiment.authority
        )


def test_capability_report_rejects_unmatched_discrete_and_wall_budgets(
    value_card_benchmark: CampaignBenchmark,
) -> None:
    experiment = _experiment(
        (value_card_benchmark,), seeds=(1,), budget=2
    )
    wrong = _run(
        experiment,
        "card_testing_cnp",
        "fixed",
        seed=1,
        budget=1,
        wall_budget_ms=_WALL_BUDGET_MS - 1,
    )
    results = _matched_results(experiment)
    results["card_testing_cnp"]["fixed"] = (wrong,)

    with pytest.raises(ValueError, match="budgets are not matched"):
        capability_delta_report(
            experiment.preregistration, results, authority=experiment.authority
        )


def test_metric_counts_and_derived_fields_cannot_be_forged() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        PolicyMetrics(
            proposal_count=1,
            approved_count=2,
            net_settled_value=Decimal("0.00"),
            adaptation_speed=Decimal(1),
            campaign_scale=2,
        )
    with pytest.raises(ValueError, match="campaign_scale"):
        PolicyMetrics(
            proposal_count=2,
            approved_count=1,
            net_settled_value=Decimal("0.00"),
            adaptation_speed=Decimal(1),
            campaign_scale=2,
        )


def test_net_settled_value_rate_is_derived_as_relative_improvement() -> None:
    def metrics(value: str) -> PolicyMetrics:
        return PolicyMetrics(
            proposal_count=2,
            approved_count=1,
            net_settled_value=Decimal(value),
            adaptation_speed=Decimal(1),
            campaign_scale=1,
        )

    family = FamilyCapabilityMetrics(
        family="app_scam_mule",
        primary_outcome=PrimaryOutcome.NET_SETTLED_VALUE_RATE,
        minimum_delta=Decimal("0.10"),
        evaluation_contract_digest="a" * 64,
        fixed=metrics("100.00"),
        random=metrics("100.00"),
        adaptive=metrics("110.00"),
        cached_llm=metrics("100.00"),
    )

    assert family.observed_delta == Decimal("0.10")
    assert family.supported is True
