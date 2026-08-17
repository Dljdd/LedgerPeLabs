"""Matched-budget decision-only search and policy isolation tests."""

from __future__ import annotations

import ast
import inspect
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AttackCandidate,
    Feedback,
    FixedPolicy,
    ParameterBounds,
    RandomPolicy,
)
from tests.redteam.conftest import campaign_params


@pytest.mark.parametrize("budget", [0, 1, 12])
@pytest.mark.parametrize("policy", [FixedPolicy(), RandomPolicy(), AdaptiveTournamentPolicy()])
def test_search_has_exact_matched_budgets(card_bounds, budget: int, policy: object) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=Decimal("10.00"),
        )

    result = AdaptiveSearch(policy=policy, bounds=card_bounds).search(  # type: ignore[arg-type]
        seed=5,
        budget=budget,
        evaluate=evaluate,
    )
    assert len(result.proposals) == len(result.trials) == len(result.objective_values) == budget
    assert result.proposals_used == result.queries_used == result.logical_time_used == budget
    assert result.proposal_budget == result.query_budget == result.logical_time_budget == budget
    assert calls == budget
    assert result.seed == 5
    assert result.policy_name in {"fixed", "random", "adaptive"}
    assert all(trial.feedback.realized_value is None for trial in result.trials)


def test_maximum_budget_is_accounted_exactly(card_bounds) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.DECLINE,
            reason_family="policy",
            realized_value=None,
        )

    result = AdaptiveSearch(policy=FixedPolicy(), bounds=card_bounds).search(
        seed=5,
        budget=1000,
        evaluate=evaluate,
    )
    assert calls == 1000
    assert result.proposals_used == result.queries_used == result.logical_time_used == 1000
    assert result.proposal_budget == result.query_budget == result.logical_time_budget == 1000


def test_search_discloses_value_only_when_explicitly_enabled(card_bounds) -> None:  # type: ignore[no-untyped-def]
    def evaluate(_candidate: AttackCandidate) -> Feedback:
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=Decimal("12.34"),
        )

    hidden = AdaptiveSearch(policy=FixedPolicy(), bounds=card_bounds).search(
        seed=1, budget=1, evaluate=evaluate
    )
    visible = AdaptiveSearch(
        policy=FixedPolicy(),
        bounds=card_bounds,
        disclose_realized_value=True,
    ).search(seed=1, budget=1, evaluate=evaluate)
    assert hidden.trials[0].feedback.realized_value is None
    assert hidden.objective_values[0] == Decimal("0")
    assert visible.trials[0].feedback.realized_value == Decimal("12.34")
    assert visible.objective_values[0] == Decimal("12.34")


def test_search_fails_closed_and_accounts_for_evaluation_exceptions(card_bounds) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def broken(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        raise RuntimeError("hidden oracle reason and score=0.987")

    result = AdaptiveSearch(policy=RandomPolicy(), bounds=card_bounds).search(
        seed=8, budget=4, evaluate=broken
    )
    assert calls == 4
    assert result.queries_used == 4
    assert {trial.feedback.reason_family for trial in result.trials} == {
        "evaluation_failure"
    }
    assert "0.987" not in repr(result)


def test_visible_invalid_rejections_still_consume_one_query_each(card_bounds) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def reject(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.DECLINE,
            reason_family="invalid_candidate",
            realized_value=None,
        )

    result = AdaptiveSearch(policy=RandomPolicy(), bounds=card_bounds).search(
        seed=2, budget=5, evaluate=reject
    )
    assert calls == 5
    assert result.proposals_used == result.queries_used == result.logical_time_used == 5
    assert all(
        trial.feedback.reason_family == "invalid_candidate" for trial in result.trials
    )


def test_duplicate_candidates_are_evaluated_and_tie_break_stably(card_bounds) -> None:  # type: ignore[no-untyped-def]
    result = AdaptiveSearch(policy=FixedPolicy(), bounds=card_bounds).search(
        seed=1,
        budget=3,
        evaluate=lambda _candidate: Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=None,
        ),
    )
    assert len({candidate.fingerprint for candidate in result.proposals}) == 1
    assert result.queries_used == 3
    assert result.winner is not None
    assert result.winner.candidate_id == min(
        candidate.candidate_id for candidate in result.proposals
    )


def test_policy_never_receives_or_retains_evaluator_capability(card_bounds) -> None:  # type: ignore[no-untyped-def]
    class EvaluatorSentinel:
        audit_records = object()
        labels = object()
        mutation_reasons = object()
        verifier_fixture = object()

        def __call__(self, _candidate: AttackCandidate) -> Feedback:
            return Feedback(
                action=Action.DECLINE,
                reason_family="policy",
                realized_value=None,
            )

    sentinel = EvaluatorSentinel()
    policy = AdaptiveTournamentPolicy()
    AdaptiveSearch(policy=policy, bounds=card_bounds).search(
        seed=9, budget=3, evaluate=sentinel
    )
    assert vars(policy) == {}
    assert all(
        "evaluate" not in inspect.signature(method).parameters
        for method in [policy.propose]
    )
    assert not any(value is sentinel for value in vars(policy).values())


def test_policy_import_boundary_excludes_evaluator_and_trust_internals() -> None:
    root = Path("src/apar/redteam")
    forbidden_modules = {
        "apar.trust",
        "apar.evaluation_hidden",
        "apar.simulator.rails",
    }
    forbidden_names = {
        "_CampaignEvaluator",
        "CampaignEvidence",
        "AgenticFixture",
        "ReasonCode",
        "TrustVerifier",
    }
    violations: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    module == item or module.startswith(f"{item}.")
                    for item in forbidden_modules
                ):
                    violations.append(f"{path}:{module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        violations.append(f"{path}:{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == item or alias.name.startswith(f"{item}.")
                        for item in forbidden_modules
                    ):
                        violations.append(f"{path}:{alias.name}")
    assert violations == []


def test_search_is_reproducible_and_global_rng_is_unchanged(card_bounds) -> None:  # type: ignore[no-untyped-def]
    def evaluate(candidate: AttackCandidate) -> Feedback:
        approved = candidate.params.retry_intensity >= 4
        return Feedback(
            action=Action.APPROVE if approved else Action.DECLINE,
            reason_family="approved" if approved else "velocity",
            realized_value=None,
        )

    state = np.random.get_state()
    first = AdaptiveSearch(policy=AdaptiveTournamentPolicy(), bounds=card_bounds).search(
        seed=44, budget=10, evaluate=evaluate
    )
    second = AdaptiveSearch(policy=AdaptiveTournamentPolicy(), bounds=card_bounds).search(
        seed=44, budget=10, evaluate=evaluate
    )
    after = np.random.get_state()
    assert first == second
    assert all(
        (left == right).all() if hasattr(left, "all") else left == right
        for left, right in zip(state, after, strict=True)
    )


def test_minimum_agentic_matrix_runs_matched_adaptive_budget_without_fake_aliases() -> None:
    bounds = ParameterBounds.for_campaign(
        "agentic_intent_abuse", campaign_params("agentic_intent_abuse")
    )
    result = AdaptiveSearch(
        policy=AdaptiveTournamentPolicy(), bounds=bounds
    ).search(
        seed=7,
        budget=3,
        evaluate=lambda _candidate: Feedback(
            action=Action.DECLINE,
            reason_family="integrity",
            realized_value=None,
        ),
    )
    assert result.proposals_used == result.queries_used == 3
    assert {candidate.fingerprint for candidate in result.proposals} == {
        AttackCandidate(
            params=bounds.template, parent_id=None, generation=0
        ).fingerprint
    }
    assert all(candidate.params == bounds.template for candidate in result.proposals)
