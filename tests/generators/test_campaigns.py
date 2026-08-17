"""Campaigns are causal rail schedules, not independently sampled rows."""

from __future__ import annotations

from datetime import UTC, timedelta
from decimal import Decimal
from typing import cast

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignGenerator,
    CampaignParams,
    GenerationConstraintError,
    _CampaignEvaluator,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import Population, PopulationGenerator
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference
from apar.simulator.rails import (
    A2ACommand,
    A2ARailAdapter,
    AgenticPaymentCommand,
    AgenticRailAdapter,
    CardCommand,
    CardRailAdapter,
)
from apar.simulator.rails.base import AdapterFactory
from apar.simulator.rails.card import AuthorizeCard, DeclineCardAuthorization
from apar.trust.verifier import TrustVerifier
from tests.factories import make_scenario_config, make_threat_card

FAMILIES = {
    "app_scam_mule": APP_SCAM_MULE_MOTIF,
    "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
    "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
}


def _bundle(seed: int = 260_816, rail: Rail = Rail.A2A):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=rail,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
    )
    return compile_scenario(
        make_threat_card(rails=[rail], default_config=config),
        config,
    )


@pytest.fixture
def population() -> Population:
    return PopulationGenerator(seed=260_816).generate(_bundle())


def _params(family: str, **updates: object) -> CampaignParams:
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
        "expected_motif": FAMILIES[family],
    }
    if family == "agentic_intent_abuse":
        values.update(
            {
                "payment_count": 25,
                "target_illicit_rate": Decimal("0.92"),
                "class_rate_tolerance": Decimal("0.01"),
            }
        )
    values.update(updates)
    return CampaignParams(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("family", FAMILIES)
def test_campaign_has_declared_entities_and_motif(
    family: str,
    population: Population,
) -> None:
    params = _params(family)
    commands, evidence = _CampaignEvaluator(seed=260_816).generate(
        family, population, params
    )

    assert commands
    assert all(isinstance(command, Command) for command in commands)
    assert all(command.campaign_id == params.campaign_id for command in commands)  # type: ignore[attr-defined]
    assert motif_signature(commands) == params.expected_motif
    assert evidence.motif_signature == params.expected_motif
    assert set(evidence.declared_entity_ids) <= {entity.entity_id for entity in population.entities}
    assert set(evidence.account_ids) <= set(population.opening_balances)
    assert len(evidence.schedule) == len(commands)
    assert evidence.schedule == tuple(sorted(evidence.schedule))
    assert evidence.schedule[0].tzinfo is UTC
    assert evidence.schedule[-1] - evidence.schedule[0] <= timedelta(hours=params.duration_hours)


@pytest.mark.parametrize("seed", [7, 19, 101, 260_816, 900_001])
@pytest.mark.parametrize("family", FAMILIES)
def test_campaign_is_byte_reproducible_across_five_seeds(
    seed: int,
    family: str,
) -> None:
    population = PopulationGenerator(seed=seed).generate(_bundle(seed))
    params = _params(family, seed=seed)
    first, first_evidence = _CampaignEvaluator(seed=seed).generate(family, population, params)
    second, second_evidence = _CampaignEvaluator(seed=seed).generate(family, population, params)

    assert campaign_bytes(first) == campaign_bytes(second)
    assert (
        first_evidence.canonical_bytes()
        == second_evidence.canonical_bytes()
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_class_rate_value_and_schedule_constraints_are_visible_and_satisfied(
    family: str,
    population: Population,
) -> None:
    params = _params(family)
    _, evidence = _CampaignEvaluator(seed=901).generate(family, population, params)

    assert abs(evidence.illicit_rate - params.target_illicit_rate) <= params.class_rate_tolerance
    assert abs(evidence.value_total - params.target_value_total) <= params.value_tolerance
    assert evidence.payment_count == params.payment_count
    assert evidence.attempts <= 100
    assert evidence.graph_digest != evidence.schedule_digest


def test_family_outputs_use_only_their_public_rail_commands(population: Population) -> None:
    generator = CampaignGenerator(seed=73)

    app = generator.generate("app_scam_mule", population, _params("app_scam_mule"))
    testing = generator.generate("card_testing_cnp", population, _params("card_testing_cnp"))
    merchant = generator.generate(
        "synthetic_merchant_refund", population, _params("synthetic_merchant_refund")
    )
    agentic = generator.generate(
        "agentic_intent_abuse", population, _params("agentic_intent_abuse")
    )

    assert all(isinstance(command, A2ACommand) for command in app)
    assert all(isinstance(command, CardCommand) for command in testing + merchant)
    assert all(isinstance(command, AgenticPaymentCommand) for command in agentic)


def test_deep_family_signatures_are_distinct(population: Population) -> None:
    signatures = {
        motif_signature(CampaignGenerator(seed=79).generate(family, population, _params(family)))
        for family in FAMILIES
    }

    assert signatures == set(FAMILIES.values())


def test_card_testing_probes_are_low_value_before_a_tighter_success_burst(
    population: Population,
) -> None:
    commands, evidence = _CampaignEvaluator(seed=81).generate(
        "card_testing_cnp",
        population,
        _params("card_testing_cnp"),
    )
    probe_positions = [
        index for index, command in enumerate(commands) if type(command) is DeclineCardAuthorization
    ]
    success_positions = [
        index for index, command in enumerate(commands) if type(command) is AuthorizeCard
    ]
    probe_amounts = [cast(Decimal, commands[index].payload["amount"]) for index in probe_positions]
    success_amounts = [
        cast(Decimal, commands[index].payload["amount"]) for index in success_positions
    ]
    delays = [
        int((right - left).total_seconds())
        for left, right in zip(evidence.schedule, evidence.schedule[1:], strict=False)
    ]

    assert max(probe_amounts) < min(success_amounts)
    assert max(delays[index - 1] for index in success_positions if index > 0) < min(
        delays[index - 1] for index in probe_positions if index > 0
    )


def test_a2a_cash_out_explicitly_names_upstream_settled_dependencies(
    population: Population,
) -> None:
    commands, evidence = _CampaignEvaluator(seed=82).generate(
        "app_scam_mule",
        population,
        _params("app_scam_mule"),
    )
    openings = [cast(A2ACommand, command) for command in commands if command.name == "a2a.initiate"]
    mule_accounts = {
        entity.account_id for entity in population.by_role("mule") if entity.account_id
    }
    cash_out_ids = {
        command.payment_id
        for command in openings
        if command.payload["payer_account"] in mule_accounts
    }
    dependency_ids = {dependency.payment_id for dependency in evidence.dependencies}

    assert cash_out_ids == dependency_ids
    assert all(dependency.upstream_payment_ids for dependency in evidence.dependencies)


def test_agentic_family_covers_valid_and_bound_integrity_mutations(
    population: Population,
) -> None:
    commands, evidence = _CampaignEvaluator(seed=83).generate(
        "agentic_intent_abuse", population, _params("agentic_intent_abuse")
    )
    requests = [
        command.request for command in commands if isinstance(command, AgenticPaymentCommand)
    ]
    mutation_kinds = set(evidence.mutation_kinds)

    assert evidence.agentic_fixture is not None
    assert {
        "AGENT_IDENTITY_MISMATCH",
        "SIGNATURE_INVALID",
        "MANDATE_SCOPE_VIOLATION",
        "NONCE_REPLAY",
        "AUTHENTICATION_EVIDENCE_REPLAY",
    } <= mutation_kinds
    assert len({request.nonce for request in requests}) < len(requests)
    assert all(len(request.signature) == 64 for request in requests)
    assert b"private" not in evidence.canonical_bytes().lower()


@pytest.mark.parametrize(
    ("family", "rail"),
    [
        ("app_scam_mule", Rail.A2A),
        ("card_testing_cnp", Rail.CARD),
        ("synthetic_merchant_refund", Rail.CARD),
        ("agentic_intent_abuse", Rail.AGENTIC),
    ],
)
@pytest.mark.parametrize("seed", [7, 19, 101, 260_816, 900_001])
def test_generated_schedule_executes_through_real_rail_and_conserves_value(
    family: str,
    rail: Rail,
    seed: int,
) -> None:
    bundle = _bundle(seed, rail)
    population = PopulationGenerator(seed=seed).generate(bundle)
    commands, evidence = _CampaignEvaluator(seed=seed).generate(
        family, population, _params(family, seed=seed)
    )
    factory: AdapterFactory
    if rail is Rail.A2A:

        def a2a_factory() -> A2ARailAdapter:
            return A2ARailAdapter()

        factory = a2a_factory
    elif rail is Rail.CARD:

        def card_factory() -> CardRailAdapter:
            return CardRailAdapter()

        factory = card_factory
    else:
        fixture = evidence.agentic_fixture
        assert fixture is not None
        verifier = TrustVerifier(
            registered_agents={(fixture.agent_id, fixture.key_id): fixture.public_key},
            mandates={fixture.mandate.mandate_id: fixture.mandate},
            authentication_evidence={
                item.evidence_id: item for item in fixture.authentication_evidence
            },
        )

        def agentic_factory() -> AgenticRailAdapter:
            return AgenticRailAdapter(
                verifier,
                lambda _request, _receipt: Action.APPROVE,
            )

        factory = agentic_factory
    opening_balances: dict[AccountReference, Decimal] = {
        cast(AccountReference, account): amount
        for account, amount in population.opening_balances.items()
    }
    engine = SimulationEngine(
        bundle,
        {rail: factory},
        opening_balances=opening_balances,
    )
    for priority, (timestamp, command) in enumerate(zip(evidence.schedule, commands, strict=True)):
        engine.schedule(timestamp, priority, command)

    events = engine.run()

    assert events
    assert all(event.campaign_id == evidence.campaign_id for event in events)
    engine.ledger.assert_conserved()


def test_impossible_constraints_fail_after_exactly_100_attempts_without_partial_state(
    population: Population,
) -> None:
    generator = CampaignGenerator(seed=89)
    impossible = _params(
        "card_testing_cnp",
        target_value_total=Decimal("2000.00"),
        min_amount=Decimal("10.00"),
        max_amount=Decimal("20.00"),
    )

    with pytest.raises(GenerationConstraintError) as caught:
        generator.generate("card_testing_cnp", population, impossible)

    assert caught.value.code == "GENERATION_CONSTRAINT_UNSATISFIED"
    assert caught.value.attempts == 100
    assert not hasattr(generator, "last_evidence")

    valid = _params("card_testing_cnp")
    after_failure = generator.generate("card_testing_cnp", population, valid)
    fresh = CampaignGenerator(seed=89).generate("card_testing_cnp", population, valid)
    assert campaign_bytes(after_failure) == campaign_bytes(fresh)


@pytest.mark.parametrize(
    "updates",
    [
        {"payment_count": 0},
        {"target_illicit_rate": Decimal("1.01")},
        {"class_rate_tolerance": Decimal("-0.01")},
        {"min_amount": Decimal("0.00")},
        {"max_amount": Decimal("9.99"), "min_amount": Decimal("10.00")},
        {"duration_hours": 0},
        {"query_budget": 0},
        {"min_delay_seconds": 0},
        {"max_delay_seconds": 0},
    ],
)
def test_campaign_params_reject_values_outside_declared_bounds(
    updates: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _params("app_scam_mule", **updates)
