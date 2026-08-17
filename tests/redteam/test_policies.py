"""Sanitized public contracts, feasible bounds, and non-LLM policy behavior."""

from __future__ import annotations

import inspect
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.generators import Population
from apar.redteam import (
    PUBLIC_REASON_FAMILIES,
    AdaptiveParameter,
    AdaptiveTournamentPolicy,
    AdaptiveVector,
    AttackCandidate,
    CandidateContractError,
    DisclosureProfile,
    DomainKind,
    Feedback,
    FixedPolicy,
    ParameterBounds,
    ParameterDomain,
    RandomPolicy,
    VisibleTrial,
)
from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules
from tests.redteam.conftest import campaign_benchmark, campaign_params


def _trial(candidate: AttackCandidate, feedback: Feedback) -> VisibleTrial:
    penalty = {
        Action.APPROVE: Decimal(0),
        Action.CHALLENGE: Decimal("-0.25"),
        Action.DECLINE: Decimal("-1"),
    }[feedback.action]
    objective = (
        feedback.realized_value if feedback.realized_value is not None else Decimal(0)
    ) + penalty
    return VisibleTrial(
        candidate=candidate,
        feedback=feedback,
        objective_value=objective,
    )


def test_feedback_contract_is_exact_closed_and_immutable() -> None:
    assert set(Feedback.model_fields) == {"action", "reason_family", "realized_value"}
    feedback = Feedback(
        action=Action.APPROVE,
        reason_family="approved",
        realized_value=None,
    )
    with pytest.raises(ValidationError):
        Feedback.model_validate(
            {
                "action": "approve",
                "reason_family": "approved",
                "realized_value": None,
                "model_score": 0.9,
            }
        )
    with pytest.raises(ValidationError):
        Feedback(
            action=Action.APPROVE,
            reason_family="hidden_rule_42",
            realized_value=None,
        )
    with pytest.raises(ValidationError):
        feedback.reason_family = "other"  # type: ignore[misc]
    assert "invalid_candidate" in PUBLIC_REASON_FAMILIES


def test_visible_trial_objective_cannot_smuggle_a_score(card_bounds) -> None:  # type: ignore[no-untyped-def]
    candidate = AttackCandidate(
        params=card_bounds.defaults,
        parent_id=None,
        generation=0,
    )
    feedback = Feedback(
        action=Action.APPROVE,
        reason_family="approved",
        realized_value=None,
    )
    with pytest.raises(ValidationError, match="public feedback"):
        VisibleTrial(
            candidate=candidate,
            feedback=feedback,
            objective_value=Decimal("0.987"),
        )


@pytest.mark.parametrize(
    "value",
    [Decimal("1E+999999"), Decimal("12.340"), Decimal("NaN"), Decimal("Infinity")],
)
def test_feedback_money_rejects_noncanonical_numeric_attacks(value: Decimal) -> None:
    with pytest.raises(ValidationError):
        Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=value,
        )


def test_candidate_identity_contains_only_sanitized_adaptive_vector(card_bounds) -> None:  # type: ignore[no-untyped-def]
    first = AttackCandidate(params=card_bounds.defaults, parent_id=None, generation=0)
    changed_vector = next(
        vector
        for vector in card_bounds.feasible_vectors
        if vector.fingerprint != card_bounds.defaults.fingerprint
    )
    changed = AttackCandidate(params=changed_vector, parent_id=None, generation=0)
    assert first.params.names == card_bounds.names
    assert "campaign_id" not in repr(first)
    assert "expected_motif" not in repr(first)
    assert "seed=" not in repr(first)
    assert changed.fingerprint != first.fingerprint
    assert changed.candidate_id != first.candidate_id


def test_parameter_bounds_have_no_hidden_template_and_are_integrity_sealed(card_bounds) -> None:  # type: ignore[no-untyped-def]
    assert set(ParameterBounds.model_fields) == {
        "family",
        "defaults",
        "domains",
        "feasible_vectors",
    }
    schema = repr(card_bounds.schema_document())
    for forbidden in (
        "campaign_id",
        "expected_motif",
        "target_value_total",
        "seed",
        "model_score",
    ):
        assert forbidden not in schema
    injected = card_bounds.model_copy(update={"hidden_template": object()})
    with pytest.raises(CandidateContractError, match="field set"):
        injected.assert_pristine()


def test_vector_and_domain_reject_subclasses_nonfinite_and_no_op_aliases() -> None:
    class EvilInt(int):
        pass

    with pytest.raises((TypeError, ValidationError)):
        AdaptiveParameter(name="retry_intensity", value=EvilInt(2))
    with pytest.raises(ValidationError):
        AdaptiveParameter(name="merchant_concentration", value=Decimal("NaN"))
    with pytest.raises(ValidationError, match="aliases"):
        ParameterDomain(
            name="merchant_concentration",
            kind=DomainKind.LINEAR,
            values=(Decimal("0.7"), Decimal("0.70")),
        )


def test_fixed_random_and_adaptive_are_deterministic_and_keep_global_rng(card_bounds) -> None:  # type: ignore[no-untyped-def]
    state_before = np.random.get_state()
    fixed = FixedPolicy().propose((), card_bounds, np.random.default_rng(7))
    random_first = RandomPolicy().propose((), card_bounds, np.random.default_rng(7))
    random_second = RandomPolicy().propose((), card_bounds, np.random.default_rng(7))
    history = (
        _trial(
            fixed,
            Feedback(
                action=Action.DECLINE,
                reason_family="velocity",
                realized_value=None,
            ),
        ),
    )
    adaptive_first = AdaptiveTournamentPolicy().propose(
        history,
        card_bounds,
        np.random.default_rng(9),
    )
    adaptive_second = AdaptiveTournamentPolicy().propose(
        history,
        card_bounds,
        np.random.default_rng(9),
    )
    state_after = np.random.get_state()
    assert fixed.params == card_bounds.defaults
    assert random_first == random_second
    assert adaptive_first == adaptive_second
    assert adaptive_first.parent_id == fixed.candidate_id
    assert 1 <= card_bounds.changed_field_count(fixed.params, adaptive_first.params) <= 3
    assert all(
        (left == right).all() if hasattr(left, "all") else left == right
        for left, right in zip(state_before, state_after, strict=True)
    )


def test_adaptive_selection_is_invariant_to_history_order(card_bounds) -> None:  # type: ignore[no-untyped-def]
    history: tuple[VisibleTrial, ...] = ()
    rng = np.random.default_rng(31)
    for index in range(5):
        candidate = RandomPolicy().propose(history, card_bounds, rng)
        history += (
            _trial(
                candidate,
                Feedback(
                    action=Action.APPROVE,
                    reason_family="approved",
                    realized_value=Decimal(f"{index}.00"),
                ),
            ),
        )
    first = AdaptiveTournamentPolicy().propose(
        history,
        card_bounds,
        np.random.default_rng(42),
    )
    second = AdaptiveTournamentPolicy().propose(
        tuple(reversed(history)),
        card_bounds,
        np.random.default_rng(42),
    )
    assert first == second


def _generic_bounds() -> ParameterBounds:
    vectors = tuple(
        AdaptiveVector.from_mapping({"alpha": value, "beta": "steady"})
        for value in (0, 1, 2)
    )
    return ParameterBounds(
        family="card_testing_cnp",
        defaults=vectors[0],
        domains=(
            ParameterDomain(name="alpha", kind=DomainKind.DISCRETE, values=(0, 1, 2)),
            ParameterDomain(
                name="beta", kind=DomainKind.CATEGORICAL, values=("steady",)
            ),
        ),
        feasible_vectors=tuple(sorted(vectors, key=lambda vector: vector.fingerprint)),
    )


def test_adaptive_bandit_is_family_agnostic_and_explores_unseen_direction() -> None:
    bounds = _generic_bounds()
    root = AttackCandidate(params=bounds.defaults, parent_id=None, generation=0)
    failed = AttackCandidate(
        params=next(vector for vector in bounds.feasible_vectors if vector.get("alpha") == 1),
        parent_id=root.candidate_id,
        generation=1,
    )
    history = (
        _trial(
            root,
            Feedback(
                action=Action.APPROVE,
                reason_family="approved",
                realized_value=Decimal("1.00"),
            ),
        ),
        _trial(
            failed,
            Feedback(
                action=Action.DECLINE,
                reason_family="velocity",
                realized_value=None,
            ),
        ),
    )

    proposal = AdaptiveTournamentPolicy().propose(
        history, bounds, np.random.default_rng(17)
    )

    assert proposal.parent_id == root.candidate_id
    assert proposal.params.get("alpha") == 2
    source = inspect.getsource(AdaptiveTournamentPolicy)
    assert "retry_intensity" not in source
    assert "mule_fanout" not in source
    assert "cash_out_fraction" not in source


def test_adaptive_bandit_uses_only_public_feedback_context(card_bounds) -> None:  # type: ignore[no-untyped-def]
    source = inspect.getsource(AdaptiveTournamentPolicy)
    for forbidden in (
        "model_score",
        "threshold",
        "hidden_template",
        "evaluator",
        "gradient",
        "expected_motif",
    ):
        assert forbidden not in source


@pytest.mark.parametrize(
    "family",
    [
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
        "agentic_intent_abuse",
    ],
)
def test_every_advertised_value_is_in_a_preflighted_task5_vector(
    family: str,
    benchmark_population: Population,
) -> None:
    benchmark = campaign_benchmark(family, benchmark_population)
    bounds = benchmark.public_bounds
    for domain in bounds.domains:
        for value in domain.values:
            vector = next(
                candidate
                for candidate in bounds.feasible_vectors
                if type(candidate.get(domain.name)) is type(value)
                and candidate.get(domain.name) == value
            )
            feedback, observation = benchmark.evaluate_with_observation(
                AttackCandidate(params=vector, parent_id=None, generation=0)
            )
            assert feedback.reason_family != "invalid_candidate"
            assert observation.fresh_replay_succeeded is True
            assert observation.ledger_conserved is True


def test_non_feasible_constraint_interaction_rejects_before_evaluator(
    benchmark_population: Population,
) -> None:
    benchmark = campaign_benchmark("app_scam_mule", benchmark_population)
    bounds = benchmark.public_bounds
    values = {entry.name: entry.value for entry in bounds.defaults.entries}
    values["cash_out_strategy"] = "burst"
    values["cash_out_delay_seconds"] = max(
        value for value in bounds.domain("cash_out_delay_seconds").values if type(value) is int
    )
    incompatible = AdaptiveVector.from_mapping(values)
    with pytest.raises(CandidateContractError, match="feasible"):
        bounds.validate_vector(incompatible)


def test_minimum_agentic_matrix_omits_fixed_slots_and_larger_matrix_exposes_real_slot(
    benchmark_population: Population,
) -> None:
    minimum = campaign_benchmark("agentic_intent_abuse", benchmark_population).public_bounds
    assert minimum.names == ()
    assert len(minimum.feasible_vectors) == 1

    base = campaign_params("agentic_intent_abuse")
    with localcontext() as context:
        context.prec = 28
        rate = Decimal(24) / Decimal(26)
    larger_template = replace(
        base,
        payment_count=26,
        target_illicit_rate=rate,
        agentic_attack_mix=rate,
    )
    larger = CampaignBenchmark(
        family="agentic_intent_abuse",
        population=benchmark_population,
        hidden_template=larger_template,
        defender=default_defender_rules(),
        disclosure_profile=DisclosureProfile(
            profile_id="artifact-decision-only-v1",
            expose_realized_value=False,
        ),
        generator_seed=960,
    ).public_bounds
    assert larger.has_non_no_op_mutation is True
    assert larger.names == ("agentic_mutations",)


def test_policy_cannot_retain_mutable_aliases(card_bounds) -> None:  # type: ignore[no-untyped-def]
    original_digest = card_bounds.bounds_digest
    serialized = card_bounds.document()
    serialized["family"] = "app_scam_mule"
    assert card_bounds.family == "card_testing_cnp"
    assert card_bounds.bounds_digest == original_digest
