"""Shared public-only fixtures for bounded attacker tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import Rail
from apar.generators import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParams,
    Population,
    PopulationGenerator,
)
from apar.redteam import DisclosureProfile, ParameterBounds
from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules
from tests.factories import make_scenario_config, make_threat_card


def campaign_params(family: str = "card_testing_cnp") -> CampaignParams:
    motifs = {
        "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
        "app_scam_mule": APP_SCAM_MULE_MOTIF,
        "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
        "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    }
    values: dict[str, object] = dict(
        campaign_id="00000000-0000-4000-8000-000000000960",
        seed=960,
        payment_count=10,
        target_illicit_rate=Decimal("0.70"),
        class_rate_tolerance=Decimal("0.01"),
        target_value_total=Decimal("500.00"),
        value_tolerance=Decimal("0.01"),
        min_amount=Decimal("10.00"),
        max_amount=Decimal("90.00"),
        currency="USD",
        duration_hours=12,
        query_budget=40,
        min_delay_seconds=1,
        max_delay_seconds=300,
        expected_motif=motifs[family],
    )
    if family == "agentic_intent_abuse":
        values.update(
            payment_count=25,
            target_illicit_rate=Decimal("0.92"),
            agentic_attack_mix=Decimal("0.92"),
        )
    return CampaignParams(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def benchmark_population() -> Population:
    config = make_scenario_config(
        rail=Rail.A2A,
        seed=960,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 960}),
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
    )
    bundle = compile_scenario(
        make_threat_card(rails=[Rail.A2A], default_config=config),
        config,
    )
    return PopulationGenerator(seed=960).generate(bundle)


def campaign_benchmark(
    family: str,
    population: Population,
    *,
    expose_realized_value: bool = False,
) -> CampaignBenchmark:
    return CampaignBenchmark(
        family=family,
        population=population,
        hidden_template=campaign_params(family),
        defender=default_defender_rules(),
        disclosure_profile=DisclosureProfile(
            profile_id=(
                "artifact-decision-and-value-v1"
                if expose_realized_value
                else "artifact-decision-only-v1"
            ),
            expose_realized_value=expose_realized_value,
        ),
        generator_seed=960,
    )


@pytest.fixture(scope="session")
def card_benchmark(benchmark_population: Population) -> CampaignBenchmark:
    return campaign_benchmark("card_testing_cnp", benchmark_population)


@pytest.fixture(scope="session")
def card_bounds(card_benchmark: CampaignBenchmark) -> ParameterBounds:
    return card_benchmark.public_bounds
