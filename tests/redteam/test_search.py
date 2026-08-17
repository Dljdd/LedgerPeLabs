"""Matched-budget search, disclosure, provenance, and isolation behavior."""

from __future__ import annotations

import inspect
from decimal import Decimal

import numpy as np
import pytest

from apar.contracts.decisions import Action
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AttackCandidate,
    CandidateContractError,
    Feedback,
    FixedPolicy,
    RandomPolicy,
)
from apar.redteam.benchmark import CampaignBenchmark
from tests.redteam.conftest import campaign_benchmark


class StaticClock:
    def __call__(self) -> int:
        return 0


def _search(policy, benchmark: CampaignBenchmark) -> AdaptiveSearch:  # type: ignore[no-untyped-def]
    return AdaptiveSearch(
        policy=policy,
        bounds=benchmark.public_bounds,
        evaluation_contract=benchmark.evaluation_contract,
        clock_ns=StaticClock(),
    )


@pytest.mark.parametrize("budget", [0, 1, 12])
@pytest.mark.parametrize(
    "policy",
    [FixedPolicy(), RandomPolicy(), AdaptiveTournamentPolicy()],
)
def test_search_has_exact_matched_discrete_budgets(
    card_benchmark: CampaignBenchmark,
    budget: int,
    policy: object,
) -> None:
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=Decimal("10.00"),
        )

    result = _search(policy, card_benchmark).search(  # type: ignore[arg-type]
        seed=5,
        budget=budget,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    assert len(result.proposals) == len(result.trials) == budget
    assert result.proposals_used == result.queries_used == result.logical_time_used == budget
    assert result.proposal_budget == result.query_budget == result.logical_time_budget == budget
    assert result.wall_time_budget_ms == 1000
    assert result.wall_time_exhausted is False
    assert calls == budget
    assert result.seed == 5
    assert all(trial.feedback.realized_value is None for trial in result.trials)


def test_maximum_budget_is_accounted_exactly(card_benchmark: CampaignBenchmark) -> None:
    calls = 0

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.DECLINE,
            reason_family="policy",
            realized_value=None,
        )

    result = _search(FixedPolicy(), card_benchmark).search(
        seed=5,
        budget=1000,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    assert calls == 1000
    assert result.proposals_used == result.queries_used == 1000


def test_disclosure_is_bound_to_profile_not_free_search_boolean(
    benchmark_population,
) -> None:  # type: ignore[no-untyped-def]
    hidden = campaign_benchmark(
        "card_testing_cnp",
        benchmark_population,
        expose_realized_value=False,
    )
    visible = campaign_benchmark(
        "card_testing_cnp",
        benchmark_population,
        expose_realized_value=True,
    )

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=Decimal("12.34"),
        )

    hidden_result = _search(FixedPolicy(), hidden).search(
        seed=1,
        budget=1,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    visible_result = _search(FixedPolicy(), visible).search(
        seed=1,
        budget=1,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    assert hidden_result.trials[0].feedback.realized_value is None
    assert visible_result.trials[0].feedback.realized_value == Decimal("12.34")
    assert hidden_result.disclosure_profile_digest != (visible_result.disclosure_profile_digest)


def test_search_fails_closed_and_charges_evaluation_exceptions(
    card_benchmark: CampaignBenchmark,
) -> None:
    calls = 0

    def broken(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        raise RuntimeError("hidden oracle reason and score=0.987")

    result = _search(RandomPolicy(), card_benchmark).search(
        seed=8,
        budget=4,
        wall_time_budget_ms=1000,
        evaluate=broken,
    )
    assert calls == result.queries_used == 4
    assert {trial.feedback.reason_family for trial in result.trials} == {"evaluation_failure"}
    assert "0.987" not in repr(result)


def test_visible_invalid_rejections_consume_one_query_each(
    card_benchmark: CampaignBenchmark,
) -> None:
    calls = 0

    def reject(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.DECLINE,
            reason_family="invalid_candidate",
            realized_value=None,
        )

    result = _search(RandomPolicy(), card_benchmark).search(
        seed=2,
        budget=5,
        wall_time_budget_ms=1000,
        evaluate=reject,
    )
    assert calls == result.proposals_used == result.queries_used == 5


def test_duplicate_vectors_are_evaluated_and_winner_tie_breaks_stably(
    card_benchmark: CampaignBenchmark,
) -> None:
    result = _search(FixedPolicy(), card_benchmark).search(
        seed=1,
        budget=3,
        wall_time_budget_ms=1000,
        evaluate=lambda _candidate: Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=None,
        ),
    )
    assert len({candidate.fingerprint for candidate in result.proposals}) == 1
    assert len({candidate.candidate_id for candidate in result.proposals}) == 3
    assert result.winner is not None
    assert result.winner.candidate_id == min(
        candidate.candidate_id for candidate in result.proposals
    )


def test_policy_never_receives_or_retains_evaluator_capability(
    card_benchmark: CampaignBenchmark,
) -> None:
    class EvaluatorSentinel:
        audit_records = object()
        labels = object()
        mutation_reasons = object()
        verifier_fixture = object()
        dependencies = object()

        def __call__(self, _candidate: AttackCandidate) -> Feedback:
            return Feedback(
                action=Action.DECLINE,
                reason_family="policy",
                realized_value=None,
            )

    sentinel = EvaluatorSentinel()
    policy = AdaptiveTournamentPolicy()
    _search(policy, card_benchmark).search(
        seed=9,
        budget=3,
        wall_time_budget_ms=1000,
        evaluate=sentinel,
    )
    assert vars(policy) == {}
    assert "evaluate" not in inspect.signature(policy.propose).parameters
    assert not any(value is sentinel for value in vars(policy).values())


def test_search_is_reproducible_and_global_rng_is_unchanged(
    card_benchmark: CampaignBenchmark,
) -> None:
    def evaluate(_candidate: AttackCandidate) -> Feedback:
        return Feedback(
            action=Action.DECLINE,
            reason_family="velocity",
            realized_value=None,
        )

    state = np.random.get_state()
    first = _search(AdaptiveTournamentPolicy(), card_benchmark).search(
        seed=44,
        budget=10,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    second = _search(AdaptiveTournamentPolicy(), card_benchmark).search(
        seed=44,
        budget=10,
        wall_time_budget_ms=1000,
        evaluate=evaluate,
    )
    after = np.random.get_state()
    assert first == second
    assert all(
        (left == right).all() if hasattr(left, "all") else left == right
        for left, right in zip(state, after, strict=True)
    )


def test_search_result_binds_complete_environment_and_policy_provenance(
    card_benchmark: CampaignBenchmark,
) -> None:
    result = _search(FixedPolicy(), card_benchmark).search(
        seed=3,
        budget=1,
        wall_time_budget_ms=1000,
        evaluate=card_benchmark.evaluate,
    )
    contract = card_benchmark.evaluation_contract
    assert result.family == contract.family
    assert result.bounds_digest == contract.bounds_digest
    assert result.hidden_template_digest == contract.hidden_template_digest
    assert result.background_digest == contract.background_digest
    assert result.population_digest == contract.population_digest
    assert result.evaluator_digest == contract.evaluator_digest
    assert result.defender_digest == contract.defender_digest
    assert result.evaluation_contract_digest == contract.contract_digest
    assert (result.policy_name, result.policy_version) == ("fixed", "1.0.0")


def test_search_rejects_candidate_model_copy_and_does_not_evaluate(
    card_benchmark: CampaignBenchmark,
) -> None:
    calls = 0

    class InjectedPolicy:
        policy_name = "fixed"
        policy_version = "injected-v1"

        def propose(self, history, bounds, rng):  # type: ignore[no-untyped-def]
            candidate = FixedPolicy().propose(history, bounds, rng)
            return candidate.model_copy(update={"generation": candidate.generation + 1})

    def evaluate(_candidate: AttackCandidate) -> Feedback:
        nonlocal calls
        calls += 1
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=None,
        )

    with pytest.raises(CandidateContractError, match="integrity|generation"):
        _search(InjectedPolicy(), card_benchmark).search(
            seed=1,
            budget=1,
            wall_time_budget_ms=1000,
            evaluate=evaluate,
        )
    assert calls == 0


def test_minimum_agentic_matrix_preserves_matched_budget_without_fake_fields(
    benchmark_population,
) -> None:  # type: ignore[no-untyped-def]
    benchmark = campaign_benchmark("agentic_intent_abuse", benchmark_population)
    result = _search(AdaptiveTournamentPolicy(), benchmark).search(
        seed=7,
        budget=3,
        wall_time_budget_ms=1000,
        evaluate=lambda _candidate: Feedback(
            action=Action.DECLINE,
            reason_family="integrity",
            realized_value=None,
        ),
    )
    assert result.proposals_used == result.queries_used == 3
    assert all(candidate.params.names == () for candidate in result.proposals)
    assert len({candidate.fingerprint for candidate in result.proposals}) == 1
