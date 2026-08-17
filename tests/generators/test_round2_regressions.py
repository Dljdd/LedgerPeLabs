"""Round-two adversarial regressions for evaluator and parameter boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from typing import cast

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.decisions import ReasonCode
from apar.contracts.events import Rail
from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParameterError,
    CampaignParams,
    GenerationConstraintError,
    _CampaignEvaluator,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import PopulationGenerator
from apar.simulator.clock import Command
from apar.simulator.rails.card import AuthorizeCard
from tests.factories import make_scenario_config, make_threat_card

_MOTIFS = {
    "app_scam_mule": APP_SCAM_MULE_MOTIF,
    "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
    "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
}


def _bundle(seed: int, *, illicit: int = 32):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.A2A,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=illicit,
        duration_hours=24,
    )
    return compile_scenario(make_threat_card(rails=[Rail.A2A], default_config=config), config)


def _params(family: str, **updates: object) -> CampaignParams:
    values: dict[str, object] = {
        "campaign_id": "00000000-0000-4000-8000-000000000902",
        "seed": 2026,
        "payment_count": 10,
        "target_illicit_rate": Decimal("0.70"),
        "class_rate_tolerance": Decimal("0.01"),
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
    if family == "agentic_intent_abuse":
        values.update(
            payment_count=25,
            target_illicit_rate=Decimal("0.92"),
            agentic_attack_mix=Decimal("0.92"),
        )
    values.update(updates)
    return CampaignParams(**values)  # type: ignore[arg-type]


def _changed_authorization(
    command: Command,
    **updates: object,
) -> AuthorizeCard:
    payload = dict(command.payload)
    payload.update(updates)
    return AuthorizeCard(
        cast(str, payload["payment_id"]),
        amount=cast(Decimal, payload["amount"]),
        currency=cast(str, payload["currency"]),
        payer_account=cast(str, payload["payer_account"]),
        payee_account=cast(str, payload["payee_account"]),
        actor_id=cast(str, payload["actor_id"]),
        counterparty_id=cast(str, payload["counterparty_id"]),
        campaign_id=cast(str, payload["campaign_id"]),
        trace_id=cast(str, payload["trace_id"]),
        fee=cast(Decimal, payload["fee"]),
        hold_account=cast(str, payload["hold_account"]),
        fee_account=cast(str, payload["fee_account"]),
        chargeback_account=cast(str, payload["chargeback_account"]),
        idempotency_key=cast(str, payload["idempotency_key"]),
    )


def test_external_evaluator_recomputes_every_constraint_from_candidate() -> None:
    population = PopulationGenerator(seed=11).generate(_bundle(11))
    params = _params("synthetic_merchant_refund", seed=11)
    evaluator = _CampaignEvaluator(seed=11)
    commands, audit = evaluator.generate("synthetic_merchant_refund", population, params)
    opening_index = next(
        index for index, command in enumerate(commands) if type(command) is AuthorizeCard
    )
    opening = commands[opening_index]

    evaluator.validate(
        "synthetic_merchant_refund",
        commands,
        population,
        params,
        schedule=audit.schedule,
    )

    unrelated = next(
        entity
        for entity in population.entities
        if entity.entity_id != opening.payload["actor_id"] and entity.account_id
    )
    mutations = (
        _changed_authorization(
            opening,
            amount=cast(Decimal, opening.payload["amount"]) + Decimal("1.00"),
        ),
        _changed_authorization(opening, actor_id=unrelated.entity_id),
        _changed_authorization(
            opening,
            campaign_id="00000000-0000-4000-8000-000000000999",
        ),
    )
    for mutation in mutations:
        changed = (*commands[:opening_index], mutation, *commands[opening_index + 1 :])
        with pytest.raises(GenerationConstraintError):
            evaluator.validate(
                "synthetic_merchant_refund",
                changed,
                population,
                params,
                schedule=audit.schedule,
            )

    malformed = (
        (commands[:-1], audit.schedule[:-1]),
        (commands + (commands[-1],), audit.schedule + (audit.schedule[-1] + timedelta(seconds=1),)),
        (
            (commands[1], commands[0], *commands[2:]),
            audit.schedule,
        ),
    )
    for changed, schedule in malformed:
        with pytest.raises(GenerationConstraintError):
            evaluator.validate(
                "synthetic_merchant_refund",
                tuple(changed),
                population,
                params,
                schedule=tuple(schedule),
            )


def test_external_app_evaluator_rejects_terminal_and_dependency_drift() -> None:
    population = PopulationGenerator(seed=111).generate(_bundle(111))
    params = _params("app_scam_mule", seed=111)
    evaluator = _CampaignEvaluator(seed=111)
    commands, audit = evaluator.generate("app_scam_mule", population, params)
    evaluator.validate(
        "app_scam_mule",
        commands,
        population,
        params,
        schedule=audit.schedule,
    )

    for terminal in ("a2a.return", "a2a.freeze", "a2a.recover", "a2a.report"):
        kept = tuple(command.name != terminal for command in commands)
        changed = tuple(command for command, include in zip(commands, kept, strict=True) if include)
        schedule = tuple(
            timestamp
            for timestamp, include in zip(audit.schedule, kept, strict=True)
            if include
        )
        with pytest.raises(GenerationConstraintError):
            evaluator.validate(
                "app_scam_mule",
                changed,
                population,
                params,
                schedule=schedule,
            )

    groups: list[tuple[Command, ...]] = []
    for command in commands:
        if not groups or groups[-1][0].payment_id != command.payment_id:  # type: ignore[attr-defined]
            groups.append((command,))
        else:
            groups[-1] = (*groups[-1], command)
    mule_ids = {entity.entity_id for entity in population.by_role("mule")}
    attacker_ids = {entity.entity_id for entity in population.by_role("attacker")}
    cash_index = next(
        index
        for index, group in enumerate(groups)
        if group[0].payload["actor_id"] in mule_ids
        and group[0].payload["counterparty_id"] in attacker_ids
    )
    remaining = tuple(
        command
        for index, group in enumerate(groups)
        if index != cash_index
        for command in group
    )
    reordered = (*groups[cash_index], *remaining)
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "app_scam_mule",
            tuple(reordered),
            population,
            params,
            schedule=audit.schedule,
        )


def test_agentic_baseline_matrix_cannot_be_removed_by_adaptive_selection() -> None:
    population = PopulationGenerator(seed=12).generate(_bundle(12))
    params = _params(
        "agentic_intent_abuse",
        seed=12,
        agentic_mutations=("amount",),
    )

    evaluator = _CampaignEvaluator(seed=12)
    commands, audit = evaluator.generate(
        "agentic_intent_abuse", population, params
    )
    evaluator.validate(
        "agentic_intent_abuse",
        commands,
        population,
        params,
        schedule=audit.schedule,
        fixture=audit.agentic_fixture,
    )

    observed = {reason for reason in audit.observed_reasons if reason is not None}
    assert observed == {
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
        ReasonCode.AUTHENTICATION_EVIDENCE_MISSING,
        ReasonCode.AUTHENTICATION_EVIDENCE_MISMATCH,
        ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED,
        ReasonCode.NONCE_REPLAY,
        ReasonCode.RECEIPT_CHAIN_BROKEN,
        ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY,
    }
    assert audit.valid_control_count >= 2

    for impossible in (
        _params("agentic_intent_abuse", payment_count=24),
        _params(
            "agentic_intent_abuse",
            target_illicit_rate=Decimal("0.50"),
            agentic_attack_mix=Decimal("0.50"),
        ),
    ):
        with pytest.raises(GenerationConstraintError) as caught:
            _CampaignEvaluator(seed=12).generate(
                "agentic_intent_abuse", population, impossible
            )
        assert caught.value.attempts == 100


def test_deep_motifs_require_terminals_and_zero_rate_testing_rejects() -> None:
    population = PopulationGenerator(seed=13).generate(_bundle(13))
    app, _ = _CampaignEvaluator(seed=13).generate(
        "app_scam_mule", population, _params("app_scam_mule", seed=13)
    )
    for terminal in ("a2a.return", "a2a.freeze", "a2a.recover", "a2a.report"):
        with pytest.raises(ValueError):
            motif_signature(tuple(command for command in app if command.name != terminal))

    zero = _params(
        "card_testing_cnp",
        seed=13,
        target_illicit_rate=Decimal("0.00"),
        class_rate_tolerance=Decimal("0.00"),
    )
    with pytest.raises(GenerationConstraintError) as caught:
        _CampaignEvaluator(seed=13).generate("card_testing_cnp", population, zero)
    assert caught.value.attempts == 100


@pytest.mark.parametrize(
    "family",
    [
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
        "agentic_intent_abuse",
    ],
)
def test_minimum_equals_maximum_delay_is_exact_for_every_family(family: str) -> None:
    population = PopulationGenerator(seed=14).generate(_bundle(14))
    updates: dict[str, object] = {
        "seed": 14,
        "min_delay_seconds": 1,
        "max_delay_seconds": 1,
    }
    if family == "app_scam_mule":
        updates.update(cash_out_delay_seconds=1, cash_out_strategy="delayed")
    _, audit = _CampaignEvaluator(seed=14).generate(
        family, population, _params(family, **updates)
    )

    assert all(
        right - left == timedelta(seconds=1)
        for left, right in zip(audit.schedule, audit.schedule[1:], strict=False)
    )


def test_app_cash_out_delay_outside_command_bounds_rejects() -> None:
    population = PopulationGenerator(seed=140).generate(_bundle(140))
    params = _params(
        "app_scam_mule",
        seed=140,
        min_delay_seconds=1,
        max_delay_seconds=2,
        cash_out_delay_seconds=3,
        cash_out_strategy="delayed",
    )
    with pytest.raises(GenerationConstraintError) as caught:
        _CampaignEvaluator(seed=140).generate("app_scam_mule", population, params)
    assert caught.value.attempts == 100


@pytest.mark.parametrize(
    "updates",
    [
        {"seed": -1},
        {"seed": 2**63},
        {"target_value_total": Decimal("1e999999")},
        {"min_amount": Decimal("1e999999")},
        {"value_tolerance": Decimal("1000000.00")},
    ],
)
def test_parameter_numeric_extremes_use_stable_error(updates: dict[str, object]) -> None:
    with pytest.raises(CampaignParameterError) as caught:
        _params("app_scam_mule", **updates)
    assert caught.value.code == "CAMPAIGN_PARAMETER_INVALID"


def test_parameter_numeric_subclasses_and_generator_seed_bounds_reject() -> None:
    class IntegerSubclass(int):
        pass

    class DecimalSubclass(Decimal):
        pass

    for updates in (
        {"seed": IntegerSubclass(1)},
        {"target_value_total": DecimalSubclass("500.00")},
    ):
        with pytest.raises(CampaignParameterError):
            _params("app_scam_mule", **updates)
    for seed in (-1, 2**63, IntegerSubclass(1)):
        with pytest.raises(TypeError):
            _CampaignEvaluator(seed=seed)


def test_no_op_adaptive_regions_reject_or_change_concrete_output() -> None:
    population = PopulationGenerator(seed=15).generate(_bundle(15))
    app_base = _params(
        "app_scam_mule",
        seed=15,
        payment_count=15,
        target_illicit_rate=Decimal("0.80"),
        target_value_total=Decimal("750.00"),
        cash_out_fraction=Decimal("0.20"),
        mule_count=2,
        mule_layers=1,
    )
    two, _ = _CampaignEvaluator(seed=15).generate("app_scam_mule", population, app_base)
    three, _ = _CampaignEvaluator(seed=15).generate(
        "app_scam_mule",
        population,
        replace(app_base, mule_count=3, mule_layers=2),
    )
    assert campaign_bytes(two) != campaign_bytes(three)
    fanout_three, _ = _CampaignEvaluator(seed=15).generate(
        "app_scam_mule",
        population,
        replace(app_base, mule_fanout=3),
    )
    assert campaign_bytes(two) != campaign_bytes(fanout_three)

    burst, burst_audit = _CampaignEvaluator(seed=15).generate(
        "app_scam_mule",
        population,
        replace(
            app_base,
            cash_out_strategy="burst",
            cash_out_delay_seconds=50,
        ),
    )
    delayed, delayed_audit = _CampaignEvaluator(seed=15).generate(
        "app_scam_mule",
        population,
        replace(
            app_base,
            cash_out_strategy="delayed",
            cash_out_delay_seconds=50,
        ),
    )
    assert burst
    assert delayed
    assert burst_audit.schedule != delayed_audit.schedule
    _, later_audit = _CampaignEvaluator(seed=15).generate(
        "app_scam_mule",
        population,
        replace(
            app_base,
            cash_out_strategy="delayed",
            cash_out_delay_seconds=100,
        ),
    )
    assert delayed_audit.schedule != later_audit.schedule

    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=15).generate(
            "app_scam_mule",
            population,
            replace(app_base, mule_count=16, mule_layers=1),
        )
    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=15).generate(
            "card_testing_cnp",
            population,
            _params("card_testing_cnp", seed=15, retry_intensity=0),
        )

    card = _params("card_testing_cnp", seed=15)
    reused, _ = _CampaignEvaluator(seed=15).generate(
        "card_testing_cnp",
        population,
        replace(card, device_reuse_rate=Decimal("1.00")),
    )
    distributed, _ = _CampaignEvaluator(seed=15).generate(
        "card_testing_cnp",
        population,
        replace(card, device_reuse_rate=Decimal("0.00")),
    )
    assert campaign_bytes(reused) != campaign_bytes(distributed)
    concentrated, _ = _CampaignEvaluator(seed=15).generate(
        "card_testing_cnp",
        population,
        replace(card, merchant_concentration=Decimal("1.00")),
    )
    dispersed, _ = _CampaignEvaluator(seed=15).generate(
        "card_testing_cnp",
        population,
        replace(card, merchant_concentration=Decimal("0.00")),
    )
    assert campaign_bytes(concentrated) != campaign_bytes(dispersed)

    agentic = _params(
        "agentic_intent_abuse",
        seed=15,
        payment_count=26,
        target_illicit_rate=Decimal("0.9230769230769230769230769231"),
        class_rate_tolerance=Decimal("0.0000000000000000000000000001"),
        agentic_attack_mix=Decimal("0.9230769230769230769230769231"),
    )
    amount, _ = _CampaignEvaluator(seed=15).generate(
        "agentic_intent_abuse",
        population,
        replace(agentic, agentic_mutations=("amount",)),
    )
    currency, _ = _CampaignEvaluator(seed=15).generate(
        "agentic_intent_abuse",
        population,
        replace(agentic, agentic_mutations=("currency",)),
    )
    assert campaign_bytes(amount) != campaign_bytes(currency)

    low_mix = _params(
        "agentic_intent_abuse",
        seed=15,
        payment_count=30,
        target_illicit_rate=Decimal("0.80"),
        class_rate_tolerance=Decimal("0.00"),
        agentic_attack_mix=Decimal("0.80"),
        agentic_mutations=("currency",),
    )
    high_mix = replace(
        low_mix,
        target_illicit_rate=Decimal("0.90"),
        agentic_attack_mix=Decimal("0.90"),
    )
    low_commands, low_audit = _CampaignEvaluator(seed=15).generate(
        "agentic_intent_abuse", population, low_mix
    )
    high_commands, high_audit = _CampaignEvaluator(seed=15).generate(
        "agentic_intent_abuse", population, high_mix
    )
    assert low_audit.illicit_rate == Decimal("0.8")
    assert high_audit.illicit_rate == Decimal("0.9")
    assert campaign_bytes(low_commands) != campaign_bytes(high_commands)
