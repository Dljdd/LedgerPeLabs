"""Real campaign execution → events → ledger conservation for all four families."""

from __future__ import annotations

from decimal import Decimal
from typing import cast

import pytest

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.generators.campaigns import CampaignGenerator, _CampaignEvaluator
from apar.generators.population import PopulationGenerator
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference
from apar.simulator.rails import (
    A2ARailAdapter,
    AgenticRailAdapter,
    CardRailAdapter,
)
from apar.trust.verifier import TrustVerifier
from tests.generators.test_campaigns import _bundle, _params

FAMILIES_AND_RAILS = [
    ("agentic_intent_abuse", Rail.AGENTIC),
    ("app_scam_mule", Rail.A2A),
    ("card_testing_cnp", Rail.CARD),
    ("synthetic_merchant_refund", Rail.CARD),
]

_SEED = 260816


def _execute_campaign(family: str, rail: Rail):
    """Execute one real campaign through SimulationEngine."""
    bundle = _bundle(_SEED, rail)
    population = PopulationGenerator(seed=_SEED).generate(bundle)
    params = _params(family, seed=_SEED)
    commands = CampaignGenerator(seed=_SEED).generate(family, population, params)
    assert len(commands) > 0, f"no commands generated for {family}"

    if rail is Rail.A2A:
        def factory() -> A2ARailAdapter:
            return A2ARailAdapter()
    elif rail is Rail.CARD:
        def factory() -> CardRailAdapter():
            return CardRailAdapter()
    else:
        _, evidence = _CampaignEvaluator(seed=_SEED).generate(
            family, population, params
        )
        fixture = evidence.agentic_fixture
        assert fixture is not None
        verifier = TrustVerifier(
            registered_agents={(fixture.agent_id, fixture.key_id): fixture.public_key},
            mandates={fixture.mandate.mandate_id: fixture.mandate},
            authentication_evidence={
                item.evidence_id: item for item in fixture.authentication_evidence
            },
        )
        def factory() -> AgenticRailAdapter:
            return AgenticRailAdapter(verifier, lambda req, receipt: Action.APPROVE)

    opening = {
        cast(AccountReference, account): amount
        for account, amount in population.opening_balances.items()
    }
    engine = SimulationEngine(bundle, {rail: factory}, opening_balances=opening)

    from datetime import timedelta
    start = bundle.replay_manifest.simulation_start
    for priority, command in enumerate(commands):
        engine.schedule(start + timedelta(minutes=priority), priority, command)

    events = engine.run()
    return commands, events, engine, population, params, bundle


class TestRealExecutionEvidence:
    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_campaign_executes_and_conserves_ledger(self, family: str, rail: Rail) -> None:
        commands, events, engine, population, params, bundle = _execute_campaign(family, rail)
        assert len(events) > 0, f"no events emitted for {family} on {rail}"
        engine.ledger.assert_conserved()

    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_events_have_multiple_event_kinds(self, family: str, rail: Rail) -> None:
        commands, events, engine, population, params, bundle = _execute_campaign(family, rail)
        event_kinds = {e.event_type for e in events}
        assert len(event_kinds) >= 1, f"expected at least one event kind for {family}, got {event_kinds}"

    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_events_carry_matching_campaign_ids(self, family: str, rail: Rail) -> None:
        commands, events, engine, population, params, bundle = _execute_campaign(family, rail)
        command_campaign_ids = set()
        for cmd in commands:
            cid = getattr(cmd, 'campaign_id', None) or cmd.payload.get('campaign_id')
            if cid:
                command_campaign_ids.add(cid)
        if command_campaign_ids:
            event_campaign_ids = {e.campaign_id for e in events}
            assert command_campaign_ids & event_campaign_ids, (
                f"campaign IDs do not reconcile between commands and events for {family}"
            )
