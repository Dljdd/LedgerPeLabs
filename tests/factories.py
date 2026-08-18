"""Factories for valid public-contract fixtures."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from apar.contracts.decisions import Action, Decision, ReasonCode
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.contracts.reports import EvaluationReport, PromotionDecision
from apar.contracts.scenarios import (
    AttackerMode,
    CampaignStage,
    FeedbackField,
    ReplayConfig,
    ReplayOrdering,
    ScenarioConfig,
    StageTransition,
)
from apar.registry.models import EvidenceRecord, ThreatCard

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)


def make_threat_card(**overrides: Any) -> ThreatCard:
    """Return a valid evidence-backed threat card with optional direct overrides."""
    card = ThreatCard(
        threat_id="app-personalized-mule",
        title="AI-personalized APP scam and mule campaign",
        version=2,
        status="approved",
        family="app_scam_mule",
        confidence=0.9,
        implementation_status="deep_scenario",
        rails=[Rail.A2A, Rail.AGENTIC],
        viewpoint="network_with_bank_enrichment",
        genai_capability={"personalization": True, "iteration_speed": True},
        attacker_objective="expected_net_settled_value",
        observables=["beneficiary_age", "device_fan_out", "transfer_velocity"],
        defender_knowledge_boundary="decision-available payment signals only",
        safety_class="synthetic_only",
        evidence=[
            EvidenceRecord(
                evidence_id="fca-app-fraud-controls-2023",
                direct_source_url=(
                    "https://www.fca.org.uk/publications/multi-firm-reviews/"
                    "anti-fraud-controls-complaint-handling-firms-focus-app-fraud"
                ),
                source_type="regulator_guidance",
                publisher="Financial Conduct Authority",
                published_on=date(2023, 11, 7),
                accessed_on=date(2026, 8, 16),
                claim=(
                    "The FCA states that APP fraud happens when someone is tricked into "
                    "sending money to a fraudster posing as a genuine payee."
                ),
                is_project_inference=False,
                quality_grade="A",
                reviewer_notes="Direct public FCA review; scenario remains synthetic.",
            )
        ],
        default_config=ScenarioConfig(
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
            benign_entity_count=5_000,
            illicit_entity_count=60,
            duration_hours=24,
            seed=260_816,
            campaign_stages=[
                CampaignStage(
                    stage_id="persuasion", description="Synthetic persuasion stage"
                ),
                CampaignStage(stage_id="transfer", description="Synthetic transfer stage"),
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
                random_seed=260_816,
                simulation_start=NOW,
                generator_version="0.1.0",
                event_ordering=ReplayOrdering.EVENT_TIME_THEN_EVENT_ID,
            ),
            export_level="sanitized",
            economics={"acquisition_cost": "configured", "mule_commission": "configured"},
            lifecycle={"label_delay_days": "configured"},
            hidden_validity={"profile": "hidden-oracle-a"},
        ),
    )
    return card.model_copy(update=overrides)


def make_payment_event(**overrides: Any) -> PaymentEvent:
    """Return a valid payment event with optional direct model-field overrides."""
    event = PaymentEvent(
        schema_version="1.0.0",
        event_id="00000000-0000-4000-8000-000000000001",
        campaign_id="00000000-0000-4000-8000-000000000002",
        trace_id="00000000-0000-4000-8000-000000000003",
        rail=Rail.CARD,
        viewpoint="network_native",
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("10.00"),
        currency="USD",
        event_time=NOW,
        ingested_at=NOW + timedelta(milliseconds=25),
        available_at=NOW + timedelta(milliseconds=25),
        decision_at=NOW + timedelta(milliseconds=40),
        actor_id="00000000-0000-4000-8000-000000000004",
        counterparty_id="00000000-0000-4000-8000-000000000005",
    )
    return event.model_copy(update=overrides)


def make_scenario_config(**overrides: Any) -> ScenarioConfig:
    """Return a valid scenario configuration with optional direct model-field overrides."""
    scenario = ScenarioConfig(
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
        benign_entity_count=5_000,
        illicit_entity_count=60,
        duration_hours=24,
        seed=260_816,
        campaign_stages=[
            CampaignStage(stage_id="persuasion", description="Synthetic persuasion stage"),
            CampaignStage(stage_id="transfer", description="Synthetic transfer stage"),
            CampaignStage(
                stage_id="mule_dispersion", description="Synthetic mule dispersion stage"
            ),
        ],
        transition_rules=[
            StageTransition(
                from_stage="persuasion", to_stage="transfer", condition="stage_completed"
            ),
            StageTransition(
                from_stage="transfer",
                to_stage="mule_dispersion",
                condition="stage_completed",
            ),
        ],
        replay=ReplayConfig(
            random_seed=260_816,
            simulation_start=NOW,
            generator_version="0.1.0",
            event_ordering=ReplayOrdering.EVENT_TIME_THEN_EVENT_ID,
        ),
        export_level="sanitized",
        economics={"acquisition_cost": "configured", "mule_commission": "configured"},
        lifecycle={"label_delay_days": "configured"},
        hidden_validity={"profile": "hidden-oracle-a"},
    )
    return scenario.model_copy(update=overrides)


def make_decision(**overrides: Any) -> Decision:
    """Return a valid event-time decision with optional direct model-field overrides."""
    decision = Decision(
        decision_id="00000000-0000-4000-8000-000000000006",
        event_id="00000000-0000-4000-8000-000000000001",
        decision_time=NOW + timedelta(milliseconds=40),
        max_source_timestamp=NOW + timedelta(milliseconds=25),
        score=0.7,
        action=Action.CHALLENGE,
        reason_codes=[ReasonCode.VELOCITY_1M],
        model_version="rules-v1",
    )
    return decision.model_copy(update=overrides)


def make_evaluation_report(**overrides: Any) -> EvaluationReport:
    """Return a valid evaluation report with optional direct model-field overrides."""
    report = EvaluationReport(
        run_id="run-2026-08-16-a",
        scenario_id="app-mule-personalized-v1",
        generator_hash="sha256:generator",
        model_hash="sha256:model",
        policy_hash="sha256:policy",
        evaluator_hash="sha256:evaluator",
        dataset_partitions={"development": 1000, "hidden": 400},
        sample_counts={"events": 1400},
        fraud_prevalence={"development": 0.10, "hidden": 0.08},
        fraud_value_distribution={"total": Decimal("1234.56")},
        family_metrics={"mule": {"average_precision": 0.62}},
        segment_metrics={"retail": {"false_positive_rate": 0.01}},
        operational_action_rates={"challenge": 0.02},
        operational_budgets={"review_value": Decimal("500.00")},
        calibration={"expected_calibration_error": 0.03},
        latency={"p95_ms": 24.0},
        leakage_tests={"future_source_timestamp_rejected": True},
        metamorphic_tests={"id_permutation_invariance": True},
        adaptive_search_ablation={"evasion_rate": 0.2},
        hidden_evaluation_results={"status": "passed"},
        failed_gates=["calibration_drift"],
        limitations=["synthetic_only"],
        reviewer_id="00000000-0000-4000-8000-000000000007",
        promotion_decision=PromotionDecision.HOLD,
        reviewed_at=NOW,
    )
    return report.model_copy(update=overrides)
