"""G1: production rail lifecycle, accounting, and integrity invariants."""

from decimal import Decimal

import numpy as np

from apar.compiler import compile_scenario
from apar.contracts.events import EventKind, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.generators import PopulationGenerator
from apar.redteam import FixedPolicy
from apar.simulator.engine import SimulationEngine
from apar.simulator.rails.a2a import (
    A2ARailAdapter,
    AcceptA2A,
    FreezeA2AFunds,
    InitiateA2A,
    PostA2A,
    RecoverA2A,
    ReportA2AFraud,
)
from apar.simulator.rails.card import (
    AuthorizeCard,
    CardRailAdapter,
    ChargebackCard,
    ClearCard,
    OpenCardDispute,
    RecoverCard,
    ReportCardFraud,
    SettleCard,
)
from tests.factories import NOW, make_scenario_config, make_threat_card
from tests.redteam.conftest import campaign_benchmark

CAMPAIGN_ID = "00000000-0000-4000-8000-000000000701"
TRACE_ID = "00000000-0000-4000-8000-000000000702"
ACTOR_ID = "00000000-0000-4000-8000-000000000703"
COUNTERPARTY_ID = "00000000-0000-4000-8000-000000000704"


def _bundle(rail: Rail, *, seed: int = 701) -> ScenarioBundle:
    viewpoint = "agentic_commerce_gateway" if rail is Rail.AGENTIC else (
        "network_native" if rail is Rail.CARD else "network_with_bank_enrichment"
    )
    config = make_scenario_config(
        rail=rail,
        viewpoint=viewpoint,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    return compile_scenario(
        make_threat_card(rails=[rail], viewpoint=viewpoint, default_config=config),
        config,
    )


def test_g1_card_report_chargeback_recovery_is_linked_and_conserved() -> None:
    """Prove the production card adapter recovers principal and fees without value drift."""
    engine = SimulationEngine(
        _bundle(Rail.CARD),
        {Rail.CARD: CardRailAdapter},
        opening_balances={("payer", "USD"): Decimal("100.00")},
    )
    payment_id = "g1-card-payment"
    commands = (
        AuthorizeCard(
            payment_id,
            amount=Decimal("10.00"),
            currency="USD",
            payer_account="payer",
            payee_account="merchant",
            actor_id=ACTOR_ID,
            counterparty_id=COUNTERPARTY_ID,
            campaign_id=CAMPAIGN_ID,
            trace_id=TRACE_ID,
            fee=Decimal("1.00"),
        ),
        ClearCard(payment_id),
        SettleCard(payment_id),
        ReportCardFraud(payment_id),
        OpenCardDispute(payment_id),
        ChargebackCard(payment_id),
        RecoverCard(payment_id),
    )
    for priority, command in enumerate(commands):
        engine.schedule(NOW, priority, command)

    events = engine.run()

    assert tuple(event.event_type for event in events) == (
        EventKind.AUTHORIZATION,
        EventKind.CLEARING,
        EventKind.SETTLEMENT,
        EventKind.FRAUD_REPORTED,
        EventKind.DISPUTE_OPENED,
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
    )
    assert tuple(event.lineage.get("previous_event_id") for event in events) == (
        None,
        *(event.event_id for event in events[:-1]),
    )
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("merchant") == Decimal("0.00")
    assert engine.ledger.balance("card:fees") == Decimal("0.00")
    engine.ledger.assert_conserved()


def test_g1_a2a_report_freeze_recovery_is_linked_and_conserved() -> None:
    """Prove the production A2A adapter freezes and recovers only concrete posted value."""
    engine = SimulationEngine(
        _bundle(Rail.A2A),
        {Rail.A2A: A2ARailAdapter},
        opening_balances={("payer", "USD"): Decimal("100.00")},
    )
    payment_id = "g1-a2a-payment"
    commands = (
        InitiateA2A(
            payment_id,
            amount=Decimal("10.00"),
            currency="USD",
            payer_account="payer",
            payee_account="payee",
            actor_id=ACTOR_ID,
            counterparty_id=COUNTERPARTY_ID,
            campaign_id=CAMPAIGN_ID,
            trace_id=TRACE_ID,
            fee=Decimal("1.00"),
        ),
        AcceptA2A(payment_id),
        PostA2A(payment_id),
        ReportA2AFraud(payment_id),
        FreezeA2AFunds(payment_id),
        RecoverA2A(payment_id),
    )
    for priority, command in enumerate(commands):
        engine.schedule(NOW, priority, command)

    events = engine.run()

    assert tuple(event.event_type for event in events) == (
        EventKind.TRANSFER_INITIATED,
        EventKind.TRANSFER_ACCEPTED,
        EventKind.TRANSFER_POSTED,
        EventKind.FRAUD_REPORTED,
        EventKind.FUNDS_FROZEN,
        EventKind.RECOVERY,
    )
    assert tuple(event.lineage.get("previous_event_id") for event in events) == (
        None,
        *(event.event_id for event in events[:-1]),
    )
    assert engine.ledger.balance("payer") == Decimal("99.00")
    assert engine.ledger.balance("payee") == Decimal("0.00")
    assert engine.ledger.balance("a2a:fees") == Decimal("1.00")
    engine.ledger.assert_conserved()


def test_g1_agentic_mandatory_mutation_matrix_fails_closed_with_controls() -> None:
    """Prove all 23 mandatory integrity attacks decline while two controls post."""
    population = PopulationGenerator(seed=701).generate(_bundle(Rail.AGENTIC))
    benchmark = campaign_benchmark("agentic_intent_abuse", population)
    candidate = FixedPolicy().propose(
        (), benchmark.public_bounds, np.random.default_rng(701)
    )

    _, observation = benchmark.evaluate_with_observation(candidate)
    event_counts = dict(observation.event_type_counts)

    assert observation.fresh_replay_succeeded
    assert observation.ledger_conserved
    assert observation.command_count == 25
    assert observation.event_count == 25
    assert event_counts == {
        EventKind.AUTHORIZATION.value: 2,
        EventKind.AUTHORIZATION_DECLINED.value: 23,
    }
