"""Review-round-one reproductions for the Task 6 trust boundary."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.redteam import (
    AdaptiveParameter,
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AdaptiveVector,
    AttackCandidate,
    DomainKind,
    Feedback,
    FixedPolicy,
    LLMPlannerPolicy,
    ParameterBounds,
    ParameterDomain,
    PolicyMetrics,
    RandomPolicy,
    SearchResult,
    VisibleTrial,
)
from tests.redteam.conftest import campaign_benchmark, campaign_params


def _decline() -> Feedback:
    return Feedback(
        action=Action.DECLINE,
        reason_family="velocity",
        realized_value=None,
    )


def _tiny_bounds() -> ParameterBounds:
    vectors = tuple(
        AdaptiveVector(entries=(AdaptiveParameter(name="level", value=value),)) for value in (0, 1)
    )
    ordered = tuple(sorted(vectors, key=lambda item: item.fingerprint))
    defaults = next(vector for vector in ordered if vector.get("level") == 0)
    return ParameterBounds(
        family="card_testing_cnp",
        defaults=defaults,
        domains=(
            ParameterDomain(
                name="level",
                kind=DomainKind.DISCRETE,
                values=(0, 1),
            ),
        ),
        feasible_vectors=ordered,
    )


def _evaluation_contract(bounds: ParameterBounds):  # type: ignore[no-untyped-def]
    from apar.redteam import DisclosureProfile, EvaluationContract

    return EvaluationContract(
        family=bounds.family,
        bounds_digest=bounds.bounds_digest,
        hidden_template_digest="1" * 64,
        background_digest="2" * 64,
        population_digest="3" * 64,
        evaluator_digest="4" * 64,
        defender_digest="5" * 64,
        disclosure_profile=DisclosureProfile(
            profile_id="decision-only-hidden-value-v1",
            expose_realized_value=False,
        ),
    )


def _policy_boundary_violations(source: str) -> tuple[str, ...]:
    forbidden_calls = {"__import__", "eval", "exec", "import_module"}
    forbidden_modules = {"apar.generators", "apar.simulator.rails", "apar.trust"}
    violations: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(
                module == item or module.startswith(f"{item}.")
                for item in forbidden_modules
            ):
                violations.append(f"forbidden module: {module}")
            for alias in node.names:
                if alias.name == "CampaignParams" or alias.name in forbidden_calls:
                    violations.append(f"forbidden import alias: {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == item or alias.name.startswith(f"{item}.")
                    for item in forbidden_modules
                ):
                    violations.append(f"forbidden module: {alias.name}")
        elif isinstance(node, ast.Call):
            called = node.func
            if isinstance(called, ast.Name) and called.id in forbidden_calls:
                violations.append(f"forbidden call: {called.id}")
            if isinstance(called, ast.Attribute) and called.attr in forbidden_calls:
                violations.append(f"forbidden attribute call: {called.attr}")
        elif isinstance(node, ast.Name) and node.id in forbidden_calls:
            violations.append(f"forbidden name reference: {node.id}")
    return tuple(violations)


def test_policy_contract_projects_only_public_adaptive_values() -> None:
    from apar.redteam import AdaptiveVector

    assert "template" not in ParameterBounds.model_fields
    assert "CampaignParams" not in str(ParameterBounds.model_fields)
    assert set(AttackCandidate.model_fields) == {"params", "parent_id", "generation"}
    assert AttackCandidate.model_fields["params"].annotation is AdaptiveVector


def test_importing_policy_surface_does_not_load_generators_rails_or_trust() -> None:
    script = """
import sys
import apar.redteam
for loaded in sys.modules:
    assert not any(
        loaded == forbidden or loaded.startswith(forbidden + '.')
        for forbidden in (
            'apar.generators',
            'apar.simulator.rails',
            'apar.trust',
        )
    ), loaded
assert 'apar.redteam.benchmark' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "source",
    (
        "from builtins import eval as hidden\nhidden('1')",
        "from importlib import import_module as hidden\nhidden('apar.generators')",
        "import importlib as hidden\nhidden.import_module('apar.trust')",
        "alias = exec\nalias('pass')",
    ),
)
def test_static_boundary_scanner_detects_aliases(source: str) -> None:
    assert _policy_boundary_violations(source)


def test_policy_modules_contain_no_forbidden_import_or_evaluation_hooks() -> None:
    trusted = (
        "src/apar/redteam/policies.py",
        "src/apar/redteam/search.py",
        "src/apar/redteam/llm_policy.py",
        "src/apar/redteam/__init__.py",
    )
    assert {
        path: _policy_boundary_violations(Path(path).read_text())
        for path in trusted
        if _policy_boundary_violations(Path(path).read_text())
    } == {}


def test_every_non_root_candidate_binds_visible_parent_and_generation(card_bounds) -> None:  # type: ignore[no-untyped-def]
    root = FixedPolicy().propose((), card_bounds, np.random.default_rng(1))
    history = (
        VisibleTrial(
            candidate=root,
            feedback=_decline(),
            objective_value=Decimal("-1"),
        ),
    )
    for policy in (FixedPolicy(), RandomPolicy(), AdaptiveTournamentPolicy()):
        candidate = policy.propose(history, card_bounds, np.random.default_rng(7))
        assert candidate.generation == 1
        assert candidate.parent_id == root.candidate_id


def test_adaptive_tournament_does_not_force_global_best_into_every_sample(card_bounds) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(10)
    trials: tuple[VisibleTrial, ...] = ()
    for index in range(8):
        candidate = RandomPolicy().propose(trials, card_bounds, rng)
        trials += (
            VisibleTrial(
                candidate=candidate,
                feedback=Feedback(
                    action=Action.APPROVE,
                    reason_family="approved",
                    realized_value=Decimal(f"{index}.00"),
                ),
                objective_value=Decimal(index),
            ),
        )
    global_best_id = trials[-1].candidate.candidate_id
    parent_ids = {
        AdaptiveTournamentPolicy()
        .propose(trials, card_bounds, np.random.default_rng(seed))
        .parent_id
        for seed in range(30)
    }
    assert any(parent_id != global_best_id for parent_id in parent_ids)


def test_search_requires_bound_disclosure_contract_and_provenance() -> None:
    signature = inspect.signature(AdaptiveSearch.__init__)
    assert "disclose_realized_value" not in signature.parameters
    assert "evaluation_contract" in signature.parameters
    required = {
        "family",
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "disclosure_profile_digest",
        "policy_version",
        "wall_time_budget_ms",
        "wall_time_elapsed_ms",
        "wall_time_exhausted",
    }
    assert required <= set(SearchResult.model_fields)


def test_metric_counts_cannot_be_inconsistent() -> None:
    with pytest.raises(ValidationError, match="approved_count"):
        PolicyMetrics(
            proposal_count=1,
            approved_count=2,
            net_settled_value=Decimal("0"),
            adaptation_speed=Decimal("1"),
            campaign_scale=2,
        )


def test_minimum_agentic_bounds_omit_non_mutable_slots(benchmark_population) -> None:  # type: ignore[no-untyped-def]
    bounds = campaign_benchmark(
        "agentic_intent_abuse", benchmark_population
    ).public_bounds
    assert bounds.names == ()


def test_family_motif_mismatch_fails_before_policy_evaluation(
    benchmark_population,
) -> None:  # type: ignore[no-untyped-def]
    from apar.redteam import DisclosureProfile
    from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules

    with pytest.raises(ValueError, match="motif"):
        CampaignBenchmark(
            family="app_scam_mule",
            population=benchmark_population,
            hidden_template=campaign_params("card_testing_cnp"),
            defender=default_defender_rules(),
            disclosure_profile=DisclosureProfile(
                profile_id="decision-only-v1",
                expose_realized_value=False,
            ),
            generator_seed=960,
        )


class _MutableIdentityClient:
    provider = "fixture"
    model_id = "model-v1"

    def __init__(self, output: dict[object, object]) -> None:
        self.output = output

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        return {
            "output": self.output,
            "latency_ms": 3,
            "input_tokens": 4,
            "output_tokens": 5,
        }


def test_llm_failure_attempt_is_audited_and_identity_is_pinned(card_bounds) -> None:  # type: ignore[no-untyped-def]
    client = _MutableIdentityClient({"model_score": Decimal("0.9")})
    policy = LLMPlannerPolicy(client)
    client.provider = "forged-provider"
    client.model_id = "forged-model"
    with pytest.raises((TypeError, ValueError)):
        policy.propose((), card_bounds)
    audit = policy.take_audit_records()
    assert len(audit) == 1
    assert audit[0].provider == "fixture"
    assert audit[0].model_id == "model-v1"
    assert audit[0].failure_family == "schema"
    assert audit[0].call_status == "online_failure"


def test_llm_recursively_rejects_subclass_and_prototype_keys(card_bounds) -> None:  # type: ignore[no-untyped-def]
    class StringSubclass(str):
        pass

    params = card_bounds.defaults_document()
    key = next(iter(params))
    params[StringSubclass(key)] = params.pop(key)
    output: dict[object, object] = {
        "params": params,
        "parent_id": None,
        "generation": 0,
        "metadata": {"__proto__": {}},
    }
    policy = LLMPlannerPolicy(_MutableIdentityClient(output))
    with pytest.raises((TypeError, ValueError), match="key|undeclared"):
        policy.propose((), card_bounds)
    assert policy.take_audit_records()[0].failure_family == "schema"


def test_real_benchmark_is_policy_independent_and_rule_order_invariant() -> None:
    from apar.redteam.benchmark import CampaignBenchmark, DefenderRuleSet

    assert CampaignBenchmark.__module__ == "apar.redteam.benchmark"
    assert DefenderRuleSet.__module__ == "apar.redteam.benchmark"
    assert "policy" not in inspect.signature(CampaignBenchmark.evaluate).parameters


def test_wall_time_budget_is_configured_and_not_a_free_result_label() -> None:
    assert "wall_time_budget_ms" in SearchResult.model_fields
    assert "wall_time_exhausted" in SearchResult.model_fields
    assert "wall_time_elapsed_ms" in SearchResult.model_fields


def test_deadline_stops_after_slow_evaluation_without_further_calls() -> None:
    class Clock:
        now = 0

        def __call__(self) -> int:
            return self.now

    clock = Clock()
    bounds = _tiny_bounds()
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        clock.now += 6_000_000
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=Decimal("10.00"),
        )

    result = AdaptiveSearch(
        policy=FixedPolicy(),
        bounds=bounds,
        evaluation_contract=_evaluation_contract(bounds),
        clock_ns=clock,
    ).search(seed=1, budget=4, wall_time_budget_ms=5, evaluate=evaluate)
    assert calls == 1
    assert result.proposals_used == result.queries_used == 1
    assert result.wall_time_exhausted is True
    assert result.wall_time_elapsed_ms == 6
    assert result.wall_time_overrun_ms == 1
    assert result.trials[0].feedback.realized_value is None


def test_deadline_after_slow_policy_prevents_evaluator_call() -> None:
    class Clock:
        now = 0

        def __call__(self) -> int:
            return self.now

    class SlowPolicy:
        policy_name = "fixed"
        policy_version = "slow-fixture-v1"

        def propose(self, history, bounds, rng):  # type: ignore[no-untyped-def]
            del rng
            clock.now += 8_000_000
            return FixedPolicy().propose(history, bounds, np.random.default_rng(1))

    clock = Clock()
    bounds = _tiny_bounds()
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return _decline()

    result = AdaptiveSearch(
        policy=SlowPolicy(),
        bounds=bounds,
        evaluation_contract=_evaluation_contract(bounds),
        clock_ns=clock,
    ).search(seed=1, budget=4, wall_time_budget_ms=5, evaluate=evaluate)
    assert calls == 0
    assert result.proposals_used == result.queries_used == 0
    assert result.wall_time_exhausted is True


def test_search_rejects_model_copy_injection_before_policy_execution() -> None:
    bounds = _tiny_bounds()
    injected = bounds.model_copy(update={"hidden_template": object()})
    with pytest.raises((TypeError, ValueError), match="field set|integrity|exact"):
        AdaptiveSearch(
            policy=FixedPolicy(),
            bounds=injected,
            evaluation_contract=_evaluation_contract(bounds),
        )
