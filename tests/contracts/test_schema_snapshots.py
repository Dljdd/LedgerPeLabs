from typing import Any, cast

from apar.contracts.decisions import Decision
from apar.contracts.events import PaymentEvent
from apar.contracts.reports import EvaluationReport
from apar.contracts.scenarios import ScenarioBundle, ScenarioConfig


def test_external_model_schema_required_fields_are_stable() -> None:
    """Catches accidental removal of fields needed at each external contract boundary."""
    required_by_model = {
        PaymentEvent: {"event_id", "campaign_id", "trace_id", "amount", "event_time"},
        ScenarioBundle: {"scenario_id", "threat_card_ref", "rail", "attacker_mode"},
        ScenarioConfig: {
            "scenario_id",
            "rail",
            "benign_entity_count",
            "illicit_entity_count",
            "duration_hours",
            "seed",
        },
        Decision: {"decision_id", "event_id", "decision_time", "action", "score"},
        EvaluationReport: {
            "run_id",
            "scenario_id",
            "generator_hash",
            "model_hash",
            "promotion_decision",
        },
    }

    for model, expected_required in required_by_model.items():
        schema = cast(Any, model).model_json_schema()
        assert expected_required <= set(schema["properties"])
        assert expected_required <= set(schema["required"])
