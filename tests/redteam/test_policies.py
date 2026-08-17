"""Public contracts and bounded non-LLM policy behavior."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.generators import CampaignParameterError
from apar.redteam import (
    PUBLIC_REASON_FAMILIES,
    AdaptiveTournamentPolicy,
    AttackCandidate,
    DomainKind,
    Feedback,
    FixedPolicy,
    ParameterBounds,
    ParameterDomain,
    RandomPolicy,
    VisibleTrial,
)
from tests.redteam.conftest import campaign_params


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
        Feedback(action=Action.APPROVE, reason_family="hidden_rule_42", realized_value=None)
    with pytest.raises(ValidationError):
        feedback.reason_family = "other"  # type: ignore[misc]
    assert "invalid_candidate" in PUBLIC_REASON_FAMILIES


def test_visible_trial_objective_cannot_smuggle_a_score() -> None:
    candidate = AttackCandidate(
        params=campaign_params(), parent_id=None, generation=0
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
    assert set(VisibleTrial.model_fields) == {
        "candidate",
        "feedback",
        "objective_value",
    }


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


def test_candidate_has_stable_canonical_identity() -> None:
    params = campaign_params()
    first = AttackCandidate(params=params, parent_id=None, generation=0)
    second = AttackCandidate(params=params, parent_id=None, generation=0)
    changed = AttackCandidate(
        params=replace(params, retry_intensity=3),
        parent_id=None,
        generation=0,
    )
    assert first.candidate_id == second.candidate_id
    assert first.fingerprint == second.fingerprint
    assert changed.candidate_id != first.candidate_id
    assert changed.fingerprint != first.fingerprint
    assert len(first.candidate_id) == len(first.fingerprint) == 64


def test_parameter_bounds_declare_only_family_adaptive_fields(card_bounds: ParameterBounds) -> None:
    assert card_bounds.names == (
        "device_reuse_rate",
        "merchant_concentration",
        "retry_intensity",
    )
    assert "model_score" not in card_bounds.schema_document()
    assert "campaign_id" not in card_bounds.schema_document()
    assert {domain.kind.value for domain in card_bounds.domains} == {
        "discrete",
        "linear",
    }
    with pytest.raises(CampaignParameterError, match="non-adaptive"):
        card_bounds.with_updates(card_bounds.template, {"target_value_total": Decimal("600")})


def test_bounds_reject_subclasses_and_nonfinite_values(card_bounds: ParameterBounds) -> None:
    class EvilInt(int):
        pass

    with pytest.raises(CampaignParameterError):
        card_bounds.with_updates(card_bounds.template, {"retry_intensity": EvilInt(2)})
    with pytest.raises(CampaignParameterError):
        card_bounds.with_updates(
            card_bounds.template,
            {"merchant_concentration": Decimal("NaN")},
        )


def test_parameter_domain_rejects_numerically_equal_no_op_aliases() -> None:
    with pytest.raises(ValidationError, match="aliases"):
        ParameterDomain(
            name="merchant_concentration",
            kind=DomainKind.LINEAR,
            values=(Decimal("0.7"), Decimal("0.70")),
        )


def test_fixed_random_and_adaptive_are_deterministic_and_do_not_touch_global_rng(
    card_bounds: ParameterBounds,
) -> None:
    state_before = np.random.get_state()
    fixed = FixedPolicy().propose((), card_bounds, np.random.default_rng(7))
    random_first = RandomPolicy().propose((), card_bounds, np.random.default_rng(7))
    random_second = RandomPolicy().propose((), card_bounds, np.random.default_rng(7))
    feedback = Feedback(
        action=Action.DECLINE,
        reason_family="velocity",
        realized_value=None,
    )
    trial = VisibleTrial(candidate=fixed, feedback=feedback, objective_value=Decimal("-1"))
    adaptive_first = AdaptiveTournamentPolicy().propose(
        (trial,), card_bounds, np.random.default_rng(9)
    )
    adaptive_second = AdaptiveTournamentPolicy().propose(
        (trial,), card_bounds, np.random.default_rng(9)
    )
    state_after = np.random.get_state()

    assert fixed.params == card_bounds.template
    assert random_first == random_second
    assert adaptive_first == adaptive_second
    assert 1 <= card_bounds.changed_field_count(fixed.params, adaptive_first.params) <= 3
    assert all(
        (left == right).all() if hasattr(left, "all") else left == right
        for left, right in zip(state_before, state_after, strict=True)
    )


def test_adaptive_selection_is_invariant_to_history_order(
    card_bounds: ParameterBounds,
) -> None:
    rng = np.random.default_rng(31)
    candidates = tuple(RandomPolicy().propose((), card_bounds, rng) for _ in range(4))
    trials = tuple(
        VisibleTrial(
            candidate=candidate,
            feedback=Feedback(
                action=Action.APPROVE,
                reason_family="approved",
                realized_value=Decimal(f"{index}.00"),
            ),
            objective_value=Decimal(index),
        )
        for index, candidate in enumerate(candidates)
    )
    first = AdaptiveTournamentPolicy().propose(
        trials, card_bounds, np.random.default_rng(42)
    )
    second = AdaptiveTournamentPolicy().propose(
        tuple(reversed(trials)), card_bounds, np.random.default_rng(42)
    )
    assert first == second


@pytest.mark.parametrize(
    "family",
    ["app_scam_mule", "card_testing_cnp", "synthetic_merchant_refund"],
)
def test_random_and_adaptive_emit_only_canonical_family_candidates(family: str) -> None:
    bounds = ParameterBounds.for_campaign(family, campaign_params(family))
    history: tuple[VisibleTrial, ...] = ()
    for generation in range(16):
        candidate = RandomPolicy().propose(history, bounds, np.random.default_rng(generation))
        bounds.validate_params(candidate.params)
        feedback = Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=None,
        )
        history += (
            VisibleTrial(
                candidate=candidate,
                feedback=feedback,
                objective_value=Decimal(0),
            ),
        )
        adaptive = AdaptiveTournamentPolicy().propose(
            history, bounds, np.random.default_rng(generation + 100)
        )
        bounds.validate_params(adaptive.params)
        parent = next(
            trial.candidate
            for trial in history
            if trial.candidate.candidate_id == adaptive.parent_id
        )
        assert 1 <= bounds.changed_field_count(parent.params, adaptive.params) <= 3


def test_larger_agentic_matrix_exposes_only_real_mutation_slots() -> None:
    base = campaign_params("agentic_intent_abuse")
    with localcontext() as context:
        context.prec = 28
        rate = Decimal(24) / Decimal(26)
    params = replace(
        base,
        payment_count=26,
        target_illicit_rate=rate,
        agentic_attack_mix=rate,
    )
    bounds = ParameterBounds.for_campaign("agentic_intent_abuse", params)
    mutation_domain = bounds.domain("agentic_mutations")
    assert bounds.has_non_no_op_mutation is True
    assert len(mutation_domain.values) > 1
    history: tuple[VisibleTrial, ...] = ()
    for seed in range(12):
        candidate = RandomPolicy().propose(history, bounds, np.random.default_rng(seed))
        bounds.validate_params(candidate.params)
        assert candidate.params.agentic_mutations != ("identity",)
        trial = VisibleTrial(
            candidate=candidate,
            feedback=Feedback(
                action=Action.DECLINE,
                reason_family="integrity",
                realized_value=None,
            ),
            objective_value=Decimal("-1"),
        )
        history += (trial,)
    adaptive = AdaptiveTournamentPolicy().propose(
        history, bounds, np.random.default_rng(30)
    )
    parent = next(
        trial.candidate for trial in history if trial.candidate.candidate_id == adaptive.parent_id
    )
    assert 1 <= bounds.changed_field_count(parent.params, adaptive.params) <= 3
    bounds.validate_params(adaptive.params)
