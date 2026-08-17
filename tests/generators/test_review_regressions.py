"""Independent-review regressions for executable campaign acceptance."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.decisions import ReasonCode
from apar.contracts.events import Rail
from apar.generators.campaigns import (
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    CampaignGenerator,
    CampaignParameterError,
    GenerationConstraintError,
    _CampaignEvaluator,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import PopulationGenerator
from tests.factories import make_scenario_config, make_threat_card

_MOTIFS = {
    "app_scam_mule": APP_SCAM_MULE_MOTIF,
    "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
    "synthetic_merchant_refund": "card:authorize>clear>settle>refund|dispute>chargeback>recovery",
    "agentic_intent_abuse": "agentic:valid_control>delegated_binding_mutations>nonce_replay",
}


def _bundle(seed: int = 260_816):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.A2A,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
    )
    return compile_scenario(make_threat_card(rails=[Rail.A2A], default_config=config), config)


def _params(family: str, **updates: object):  # type: ignore[no-untyped-def]
    from apar.generators.campaigns import CampaignParams

    values: dict[str, object] = {
        "campaign_id": "00000000-0000-4000-8000-000000000901",
        "seed": 260_816,
        "payment_count": 10,
        "target_illicit_rate": Decimal("0.70"),
        "class_rate_tolerance": Decimal("0.05"),
        "target_value_total": Decimal("500.00"),
        "value_tolerance": Decimal("0.01"),
        "min_amount": Decimal("10.00"),
        "max_amount": Decimal("90.00"),
        "currency": "USD",
        "duration_hours": 12,
        "query_budget": 40,
        "min_delay_seconds": 1,
        "max_delay_seconds": 300,
        "expected_motif": _MOTIFS[family],
    }
    values.update(updates)
    return CampaignParams(**values)


def test_generator_retains_no_public_or_private_audit_result() -> None:
    population = PopulationGenerator(seed=1).generate(_bundle(1))
    generator = CampaignGenerator(seed=1)
    generator.generate("card_testing_cnp", population, _params("card_testing_cnp", seed=1))

    assert not hasattr(generator, "last_evidence")
    assert not hasattr(generator, "_last_evidence")


def test_seed_one_small_app_is_never_accepted_unless_real_rail_replay_succeeds() -> None:
    population = PopulationGenerator(seed=1).generate(_bundle(1))
    params = _params(
        "app_scam_mule",
        seed=1,
        payment_count=3,
        target_value_total=Decimal("150.00"),
        target_illicit_rate=Decimal("1.00"),
        class_rate_tolerance=Decimal("0.00"),
    )

    try:
        commands, audit = _CampaignEvaluator(seed=1).generate(
            "app_scam_mule", population, params
        )
    except GenerationConstraintError as error:
        assert error.attempts == 100
    else:
        assert commands
        assert audit.replay_succeeded is True
        assert audit.ledger_conserved is True


def test_requested_class_rate_changes_concrete_card_behavior() -> None:
    population = PopulationGenerator(seed=2).generate(_bundle(2))
    benign = _params(
        "card_testing_cnp",
        seed=2,
        target_illicit_rate=Decimal("0.00"),
        class_rate_tolerance=Decimal("0.00"),
    )
    moderate = replace(
        benign,
        target_illicit_rate=Decimal("0.50"),
        class_rate_tolerance=Decimal("0.00"),
    )
    illicit = replace(
        moderate,
        target_illicit_rate=Decimal("1.00"),
        class_rate_tolerance=Decimal("0.00"),
    )

    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=2).generate("card_testing_cnp", population, benign)
    benign_commands, benign_audit = _CampaignEvaluator(seed=2).generate(
        "card_testing_cnp", population, moderate
    )
    illicit_commands, illicit_audit = _CampaignEvaluator(seed=2).generate(
        "card_testing_cnp", population, illicit
    )

    assert campaign_bytes(benign_commands) != campaign_bytes(illicit_commands)
    assert benign_audit.illicit_rate == Decimal("0.5")
    assert illicit_audit.illicit_rate == Decimal("1")
    assert benign_audit.class_labels != illicit_audit.class_labels


def test_audit_values_are_recomputed_from_commands_and_reject_mutation() -> None:
    population = PopulationGenerator(seed=3).generate(_bundle(3))
    params = _params("synthetic_merchant_refund", seed=3)
    commands, audit = _CampaignEvaluator(seed=3).generate(
        "synthetic_merchant_refund", population, params
    )

    assert audit.attempted_value == sum(
        (
            command.payload["amount"]
            for command in commands
            if command.name in {"card.authorize", "card.decline"}
        ),
        Decimal("0.00"),
    )
    assert audit.unique_attempted_value == audit.attempted_value
    assert audit.settled_value <= audit.attempted_value
    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=3).validate(
            "synthetic_merchant_refund",
            commands[:-1],
            population,
            params,
        )


def test_schedule_honors_population_and_parameter_horizons() -> None:
    population = PopulationGenerator(seed=4).generate(_bundle(4))
    one_hour = replace(population, horizon_end=population.generated_at + timedelta(hours=1))
    params = _params("card_testing_cnp", seed=4, duration_hours=12)

    _, audit = _CampaignEvaluator(seed=4).generate("card_testing_cnp", one_hour, params)

    assert max(audit.schedule) <= one_hour.horizon_end


def test_deep_motifs_reject_reduced_lifecycle_fragments() -> None:
    population = PopulationGenerator(seed=5).generate(_bundle(5))
    app, _ = _CampaignEvaluator(seed=5).generate(
        "app_scam_mule", population, _params("app_scam_mule", seed=5)
    )
    card, _ = _CampaignEvaluator(seed=5).generate(
        "card_testing_cnp", population, _params("card_testing_cnp", seed=5)
    )

    with pytest.raises(ValueError):
        motif_signature(tuple(command for command in app if command.name != "a2a.post"))
    with pytest.raises(ValueError):
        motif_signature(tuple(command for command in card if command.name != "card.settle"))
    assert motif_signature(app) == APP_SCAM_MULE_MOTIF
    assert motif_signature(card) == CARD_TESTING_CNP_MOTIF


def test_campaign_parameter_caps_and_adaptive_dimensions_are_explicit() -> None:
    params = _params("app_scam_mule")

    assert params.merchant_concentration == Decimal("0.70")
    assert params.device_reuse_rate == Decimal("0.60")
    assert params.mule_count >= 2
    assert params.mule_layers >= 1
    assert params.mule_fanout >= 1
    assert params.cash_out_fraction <= Decimal("1")
    assert params.cash_out_strategy in {"staged", "burst", "delayed"}
    assert params.recovery_probability <= Decimal("1")
    with pytest.raises(CampaignParameterError) as caught:
        _params("app_scam_mule", payment_count=10_000_000)
    assert caught.value.code == "CAMPAIGN_PARAMETER_INVALID"


def test_adaptive_dimensions_change_concrete_family_behavior() -> None:
    population = PopulationGenerator(seed=61).generate(_bundle(61))

    app_params = _params("app_scam_mule", seed=61)
    app_low, _ = _CampaignEvaluator(seed=61).generate(
        "app_scam_mule",
        population,
        replace(app_params, cash_out_fraction=Decimal("0.20")),
    )
    app_high, _ = _CampaignEvaluator(seed=61).generate(
        "app_scam_mule",
        population,
        replace(app_params, cash_out_fraction=Decimal("0.30")),
    )
    low_cash = sum(
        (
            command.payload["amount"]
            for command in app_low
            if command.name == "a2a.initiate"
            and command.payload["actor_id"]
            in {entity.entity_id for entity in population.by_role("mule")}
            and command.payload["counterparty_id"]
            in {entity.entity_id for entity in population.by_role("attacker")}
        ),
        Decimal("0.00"),
    )
    high_cash = sum(
        (
            command.payload["amount"]
            for command in app_high
            if command.name == "a2a.initiate"
            and command.payload["actor_id"]
            in {entity.entity_id for entity in population.by_role("mule")}
            and command.payload["counterparty_id"]
            in {entity.entity_id for entity in population.by_role("attacker")}
        ),
        Decimal("0.00"),
    )
    assert low_cash == Decimal("100.00")
    assert high_cash == Decimal("150.00")

    card_params = _params("card_testing_cnp", seed=61)
    low_retry, _ = _CampaignEvaluator(seed=61).generate(
        "card_testing_cnp",
        population,
        replace(card_params, retry_intensity=1, merchant_concentration=Decimal("1.00")),
    )
    high_retry, _ = _CampaignEvaluator(seed=61).generate(
        "card_testing_cnp",
        population,
        replace(card_params, retry_intensity=5, merchant_concentration=Decimal("0.00")),
    )
    assert sum(command.name == "card.decline" for command in low_retry) == 1
    assert sum(command.name == "card.decline" for command in high_retry) == 5
    assert campaign_bytes(low_retry) != campaign_bytes(high_retry)

    merchant_params = _params("synthetic_merchant_refund", seed=61)
    low_recovery, _ = _CampaignEvaluator(seed=61).generate(
        "synthetic_merchant_refund",
        population,
        replace(merchant_params, recovery_probability=Decimal("0.15")),
    )
    high_recovery, _ = _CampaignEvaluator(seed=61).generate(
        "synthetic_merchant_refund",
        population,
        replace(merchant_params, recovery_probability=Decimal("0.80")),
    )
    assert sum(command.name == "card.recover" for command in low_recovery) == 2
    assert sum(command.name == "card.recover" for command in high_recovery) == 6


def test_population_models_executable_benign_shift_activities() -> None:
    first = PopulationGenerator(seed=6).generate(_bundle(6))
    second = PopulationGenerator(seed=7).generate(_bundle(7))

    assert first.benign_activities
    assert first.benign_commands
    assert {activity.shift for activity in first.benign_activities} == {
        "new_merchant",
        "shared_beneficiary",
        "shared_device",
        "travel",
    }
    assert first.benign_activities != second.benign_activities
    assert all(
        activity.actor_id in {entity.entity_id for entity in first.entities}
        for activity in first.benign_activities
    )


def test_agentic_matrix_is_observed_from_task4_not_declared_metadata() -> None:
    population = PopulationGenerator(seed=8).generate(_bundle(8))
    params = _params(
        "agentic_intent_abuse",
        seed=8,
        payment_count=25,
        target_illicit_rate=Decimal("0.92"),
        class_rate_tolerance=Decimal("0.01"),
        target_value_total=Decimal("500.00"),
        min_amount=Decimal("10.00"),
        max_amount=Decimal("30.00"),
    )

    commands, audit = _CampaignEvaluator(seed=8).generate(
        "agentic_intent_abuse", population, params
    )

    expected = {
        ReasonCode.AGENT_IDENTITY_MISMATCH,
        ReasonCode.SIGNATURE_INVALID,
        ReasonCode.MANDATE_SCOPE_VIOLATION,
        ReasonCode.AUTHORITY_IDENTITY_MISMATCH,
        ReasonCode.AMOUNT_LIMIT_EXCEEDED,
        ReasonCode.CURRENCY_MISMATCH,
        ReasonCode.MERCHANT_BINDING_MISMATCH,
        ReasonCode.PAYEE_BINDING_MISMATCH,
        ReasonCode.CATEGORY_SCOPE_VIOLATION,
        ReasonCode.PRODUCT_SCOPE_VIOLATION,
        ReasonCode.CART_HASH_MISMATCH,
        ReasonCode.PAYMENT_INTENT_HASH_MISMATCH,
        ReasonCode.CREDENTIAL_BINDING_MISMATCH,
        ReasonCode.TOKEN_SCOPE_VIOLATION,
        ReasonCode.CONSENT_BINDING_MISMATCH,
        ReasonCode.MANDATE_TIME_SCOPE_VIOLATION,
        ReasonCode.MANDATE_EXPIRED,
        ReasonCode.NONCE_REPLAY,
        ReasonCode.RECEIPT_CHAIN_BROKEN,
        ReasonCode.AUTHENTICATION_EVIDENCE_MISSING,
        ReasonCode.AUTHENTICATION_EVIDENCE_MISMATCH,
        ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED,
        ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY,
    }
    observed = {reason for reason in audit.observed_reasons if reason is not None}

    assert len(commands) == params.payment_count
    assert expected <= observed
    assert audit.valid_control_count >= 2
    assert all(timestamp <= population.horizon_end for timestamp in audit.schedule)
