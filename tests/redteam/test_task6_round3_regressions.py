"""Task 6 round-three callable, runner, and v3 freeze regressions."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AdaptiveVector,
    AttackCandidate,
    DomainKind,
    Feedback,
    ParameterBounds,
    ParameterDomain,
    RandomPolicy,
    SearchAuthority,
)
from tests.redteam.test_policies import _trial
from tests.redteam.test_round1_regressions import (
    _decline,
    _evaluation_contract,
    _RegisteredEvaluator,
    _tiny_bounds,
)

ROOT = Path(__file__).resolve().parents[2]
V3_PREREGISTRATION = ROOT / "docs/experiments/task6-v3-holdout-preregistration.json"
V3_RESULT = ROOT / "docs/experiments/task6-v3-holdout-result.json"


class _OriginalPolicy:
    def propose(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
        return AttackCandidate(
            params=bounds.defaults,
            parent_id=None if not history else history[-1].candidate.candidate_id,
            generation=len(history),
        )


def _replacement(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
    changed = next(
        vector
        for vector in bounds.feasible_vectors
        if vector.fingerprint != bounds.defaults.fingerprint
    )
    return AttackCandidate(
        params=changed,
        parent_id=None if not history else history[-1].candidate.candidate_id,
        generation=len(history),
    )


def _registered(policy: object):  # type: ignore[no-untyped-def]
    authority = SearchAuthority()
    bounds = _tiny_bounds()
    evaluator = _RegisteredEvaluator(lambda _candidate: _decline())
    evaluator_capability = authority.register_evaluator(
        owner=evaluator,
        bounds=bounds,
        evaluation_contract=_evaluation_contract(bounds),
        evaluate=evaluator.evaluate,
        dependency_digest="b" * 64,
    )
    policy_capability = authority.register_policy(
        policy,  # type: ignore[arg-type]
        name="fixed",
        version="callable-binding-v1",
    )
    return authority, evaluator_capability, policy_capability


def test_registered_instance_callable_cannot_be_replaced_after_registration() -> None:
    policy = _OriginalPolicy()
    authority, evaluator, registered = _registered(policy)
    policy.propose = MethodType(_replacement, policy)  # type: ignore[method-assign]

    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=registered,
        run_group=authority.issue_run_group("instance-substitution"),
    ).search(seed=1, budget=1, wall_time_budget_ms=1_000)

    assert result.proposals[0].params == evaluator.bounds.defaults


def test_registered_class_callable_cannot_be_replaced_after_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _OriginalPolicy()
    authority, evaluator, registered = _registered(policy)
    monkeypatch.setattr(_OriginalPolicy, "propose", _replacement)

    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=registered,
        run_group=authority.issue_run_group("class-substitution"),
    ).search(seed=1, budget=1, wall_time_budget_ms=1_000)

    assert result.proposals[0].params == evaluator.bounds.defaults


def test_registered_callable_slot_tampering_fails_before_policy_execution() -> None:
    policy = _OriginalPolicy()
    authority, evaluator, registered = _registered(policy)
    object.__setattr__(registered, "_propose", MethodType(_replacement, policy))

    with pytest.raises(ValueError, match="callable|implementation|issued"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("callable-slot-tampering"),
        )


def test_v3_random_and_adaptive_share_the_default_baseline() -> None:
    bounds = _tiny_bounds()
    random = RandomPolicy().propose((), bounds, np.random.default_rng(8))
    adaptive = AdaptiveTournamentPolicy().propose((), bounds, np.random.default_rng(8))

    assert random.params == bounds.defaults
    assert adaptive.params == bounds.defaults
    assert random.parent_id is adaptive.parent_id is None
    assert random.generation == adaptive.generation == 0


def _frontier_bounds() -> ParameterBounds:
    vectors = tuple(
        AdaptiveVector.from_mapping({"alpha": alpha, "beta": beta})
        for alpha in (0, 1, 2)
        for beta in ("low", "mid", "high")
    )
    default = next(
        vector
        for vector in vectors
        if vector.get("alpha") == 1 and vector.get("beta") == "mid"
    )
    return ParameterBounds(
        family="card_testing_cnp",
        defaults=default,
        domains=(
            ParameterDomain(name="alpha", kind=DomainKind.DISCRETE, values=(0, 1, 2)),
            ParameterDomain(
                name="beta",
                kind=DomainKind.CATEGORICAL,
                values=("low", "mid", "high"),
            ),
        ),
        feasible_vectors=tuple(sorted(vectors, key=lambda vector: vector.fingerprint)),
    )


def test_v3_frontier_covers_unique_default_boundaries_before_broader_search() -> None:
    bounds = _frontier_bounds()
    policy = AdaptiveTournamentPolicy()
    history = ()
    proposals: list[AttackCandidate] = []
    for generation in range(5):
        candidate = policy.propose(history, bounds, np.random.default_rng(91))
        proposals.append(candidate)
        history += (
            _trial(
                candidate,
                Feedback(
                    action=Action.DECLINE,
                    reason_family="velocity",
                    realized_value=None,
                ),
            ),
        )
        assert candidate.generation == generation

    assert proposals[0].params == bounds.defaults
    assert len({candidate.params.fingerprint for candidate in proposals}) == 5
    for candidate in proposals[1:]:
        assert bounds.changed_field_count(bounds.defaults, candidate.params) == 1
        changed_name = next(
            name
            for name in bounds.names
            if candidate.params.get(name) != bounds.defaults.get(name)
        )
        domain = bounds.domain(changed_name)
        assert candidate.params.get(changed_name) in {domain.values[0], domain.values[-1]}


def test_v3_policy_ast_contains_no_family_or_task5_specific_strategy() -> None:
    tree = ast.parse(inspect.getsource(AdaptiveTournamentPolicy))
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is str
    }
    forbidden = {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
        "retry_intensity",
        "mule_fanout",
        "cash_out_fraction",
        "device_reuse_rate",
        "merchant_concentration",
        "declined_probe_count",
        "distinct_payees",
        "503",
        "607",
        "0.10",
    }

    assert string_literals.isdisjoint(forbidden)
    assert AdaptiveTournamentPolicy.policy_version == "3.0.0"


def test_v3_runner_is_directly_reproducible_and_verify_only_executes_no_trial() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_task6_holdout.py"), "--verify-only"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "v3" in completed.stdout
    assert "no holdout trial executed" in completed.stdout
    assert not V3_RESULT.exists()


def test_production_experiment_reconstructs_the_frozen_task6_benchmark() -> None:
    from apar.redteam.task6_experiment import build_task6_experiment

    experiment = build_task6_experiment(ROOT)

    assert experiment.population_digest == (
        "a89e4cadd335500a87a8c4bf5fa5c68612ac931d07b0659352a200f175981c5a"
    )
    assert {
        family: benchmark.evaluation_contract.contract_digest
        for family, benchmark in experiment.benchmarks.items()
    } == {
        "app_scam_mule": (
            "4124bc88991f2a4a064d607ba964b4bf825bb390736181710a7ea26ab5197f58"
        ),
        "card_testing_cnp": (
            "937aba3d6c8698a34bb71cd9ba10dfa0d8c9c6d37b06647ad97d1da43cf66f57"
        ),
    }


def test_v3_preregistration_freezes_the_complete_confirmatory_contract() -> None:
    artifact = json.loads(V3_PREREGISTRATION.read_text(encoding="utf-8"))

    assert artifact["holdout"] == {
        "seeds": [503, 607, 709, 811, 907, 1009, 1103, 1201],
        "budget": 24,
        "wall_time_budget_ms": 120000,
        "maximum_additional_confirmatory_attempts": 1,
        "result_path": "docs/experiments/task6-v3-holdout-result.json",
    }
    assert artifact["fairness"]["shared_default_first_proposal"] is True
    assert artifact["network"]["allowed_calls"] == 0
    assert artifact["families"]["app_scam_mule"]["primary_outcome"] == (
        "net_settled_value_rate"
    )
    assert artifact["families"]["app_scam_mule"]["minimum_delta"] == "0.10"
    assert artifact["families"]["card_testing_cnp"]["primary_outcome"] == "valid_yield"
    assert artifact["families"]["card_testing_cnp"]["minimum_delta"] == "0.10"
    assert artifact["uncertainty"]["role"] == "descriptive_only"
    assert artifact["if_v3_fails"] == (
        "No further confirmatory holdout will be opened; later work is exploratory "
        "or Task 7 evaluation."
    )
    assert artifact["negative_control"]["support_expected"] is False
    assert artifact["environment"]["python_version"]
    assert artifact["environment"]["pyproject_sha256"]
    assert artifact["environment"]["lock_file"] is None
    for required_source in (
        "src/apar/redteam/policies.py",
        "src/apar/redteam/llm_policy.py",
        "src/apar/redteam/search.py",
        "src/apar/redteam/benchmark.py",
        "src/apar/redteam/task6_experiment.py",
        "scripts/run_task6_holdout.py",
        "pyproject.toml",
    ):
        assert required_source in artifact["source_files"]
    assert not V3_RESULT.exists()
