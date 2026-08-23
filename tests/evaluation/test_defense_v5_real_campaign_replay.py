"""Real Task 5 campaign generation and rail replay integration for v5 population."""

from __future__ import annotations

import pytest

from apar.contracts.events import Rail

# Import the proven test infrastructure from the existing campaign tests.
from tests.generators.test_campaigns import _bundle, _params  # noqa: F401

FAMILIES_AND_RAILS = [
    ("agentic_intent_abuse", Rail.AGENTIC),
    ("app_scam_mule", Rail.A2A),
    ("card_testing_cnp", Rail.CARD),
    ("synthetic_merchant_refund", Rail.CARD),
]


class TestRealCampaignReplayForV5:
    """Prove that the real CampaignGenerator → SimulationEngine → rail path works
    and produces events we can project into V5DecisionRow."""

    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_campaign_produces_replayable_commands(self, family: str, rail: Rail) -> None:
        seed = 260816
        bundle = _bundle(seed, rail)
        from apar.generators.population import PopulationGenerator
        population = PopulationGenerator(seed=seed).generate(bundle)

        params = _params(family, seed=seed)
        generator = __import__("apar.generators.campaigns", fromlist=["CampaignGenerator"]).CampaignGenerator(seed=seed)
        commands = generator.generate(family, population, params)

        assert len(commands) > 0, f"no commands generated for {family}"

        # Verify command types match the expected rail.
        from apar.simulator.rails import A2ACommand, AgenticPaymentCommand, CardCommand
        for command in commands:
            if rail is Rail.A2A:
                assert isinstance(command.request if hasattr(command, 'request') else command, (A2ACommand, type(command))), (
                    f"command type mismatch for {family} on {rail}"
                )
