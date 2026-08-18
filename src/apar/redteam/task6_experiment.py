"""Evaluator-owned production construction for frozen Task 6 experiments.

This module is intentionally excluded from the policy-facing ``apar.redteam`` import
surface.  It owns scenario compilation, population generation, hidden campaign templates,
and benchmark construction for reproducible development and confirmatory runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import Rail
from apar.contracts.scenarios import (
    AttackerMode,
    CampaignStage,
    FeedbackField,
    ReplayConfig,
    ReplayOrdering,
    ScenarioConfig,
    StageTransition,
)
from apar.generators import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParams,
    Population,
    PopulationGenerator,
)
from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules
from apar.redteam.search import DisclosureProfile
from apar.registry.models import ThreatCard

_GENERATOR_SEED = 960
_CAMPAIGN_ID = "00000000-0000-4000-8000-000000000960"


@dataclass(frozen=True, slots=True)
class Task6Experiment:
    population: Population
    benchmarks: Mapping[str, CampaignBenchmark]
    negative_control: CampaignBenchmark

    @property
    def population_digest(self) -> str:
        return hashlib.sha256(self.population.canonical_bytes()).hexdigest()


def _scenario_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="app-mule-personalized-v1",
        version="1.0.0",
        rail=Rail.A2A,
        viewpoint="network_with_bank_enrichment",
        attacker_mode=AttackerMode.DECISION_ONLY,
        attacker_objective="expected_net_settled_value",
        query_budget=40,
        feedback=[
            FeedbackField.APPROVE,
            FeedbackField.CHALLENGE,
            FeedbackField.DECLINE,
            FeedbackField.REALIZED_VALUE,
        ],
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
        seed=_GENERATOR_SEED,
        campaign_stages=[
            CampaignStage(
                stage_id="persuasion",
                description="Synthetic persuasion stage",
            ),
            CampaignStage(
                stage_id="transfer",
                description="Synthetic transfer stage",
            ),
            CampaignStage(
                stage_id="mule_dispersion",
                description="Synthetic mule dispersion stage",
            ),
        ],
        transition_rules=[
            StageTransition(
                from_stage="persuasion",
                to_stage="transfer",
                condition="stage_completed",
            ),
            StageTransition(
                from_stage="transfer",
                to_stage="mule_dispersion",
                condition="stage_completed",
            ),
        ],
        replay=ReplayConfig(
            random_seed=_GENERATOR_SEED,
            simulation_start=datetime(2026, 8, 16, 12, tzinfo=UTC),
            generator_version="0.1.0",
            event_ordering=ReplayOrdering.EVENT_TIME_THEN_EVENT_ID,
        ),
        export_level="sanitized",
        economics={"acquisition_cost": "configured", "mule_commission": "configured"},
        lifecycle={"label_delay_days": "configured"},
        hidden_validity={"profile": "hidden-oracle-a"},
    )


def _population(root: Path) -> Population:
    fixture = root / "fixtures/golden/threat-card.json"
    card = ThreatCard.model_validate_json(fixture.read_text(encoding="utf-8"))
    config = _scenario_config()
    reviewed = card.model_copy(
        update={
            "rails": [Rail.A2A],
            "default_config": config,
        }
    )
    bundle = compile_scenario(reviewed, config)
    return PopulationGenerator(seed=_GENERATOR_SEED).generate(bundle)


def task6_campaign_params(family: str) -> CampaignParams:
    motifs = {
        "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
        "app_scam_mule": APP_SCAM_MULE_MOTIF,
        "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
        "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    }
    try:
        motif = motifs[family]
    except KeyError as error:
        raise ValueError("unsupported Task 6 campaign family") from error
    values: dict[str, object] = {
        "campaign_id": _CAMPAIGN_ID,
        "seed": _GENERATOR_SEED,
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
        "expected_motif": motif,
    }
    if family == "agentic_intent_abuse":
        values.update(
            payment_count=25,
            target_illicit_rate=Decimal("0.92"),
            agentic_attack_mix=Decimal("0.92"),
        )
    return CampaignParams(**values)  # type: ignore[arg-type]


def task6_campaign_benchmark(
    family: str,
    population: Population,
    *,
    expose_realized_value: bool,
) -> CampaignBenchmark:
    return CampaignBenchmark(
        family=family,
        population=population,
        hidden_template=task6_campaign_params(family),
        defender=default_defender_rules(),
        disclosure_profile=DisclosureProfile(
            profile_id=(
                "artifact-decision-and-value-v1"
                if expose_realized_value
                else "artifact-decision-only-v1"
            ),
            expose_realized_value=expose_realized_value,
        ),
        generator_seed=_GENERATOR_SEED,
    )


def build_task6_experiment(root: Path) -> Task6Experiment:
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    population = _population(root.resolve())
    benchmarks = {
        family: task6_campaign_benchmark(
            family,
            population,
            expose_realized_value=True,
        )
        for family in ("app_scam_mule", "card_testing_cnp")
    }
    return Task6Experiment(
        population=population,
        benchmarks=MappingProxyType(benchmarks),
        negative_control=task6_campaign_benchmark(
            "agentic_intent_abuse",
            population,
            expose_realized_value=False,
        ),
    )


__all__ = [
    "Task6Experiment",
    "build_task6_experiment",
    "task6_campaign_benchmark",
    "task6_campaign_params",
]
