from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action, Decision
from apar.contracts.events import PaymentEvent
from apar.contracts.reports import EvaluationReport, PromotionDecision
from apar.contracts.scenarios import AttackerMode, FeedbackField, ScenarioBundle
from tests.factories import (
    make_decision,
    make_evaluation_report,
    make_payment_event,
    make_scenario_config,
)


def test_event_rejects_ingestion_before_event_time() -> None:
    """Catches a timeline regression that makes an event appear before ingestion."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with pytest.raises(ValidationError, match="ingested_at"):
        PaymentEvent.model_validate(
            make_payment_event(event_time=now, ingested_at=now - timedelta(seconds=1)).model_dump()
        )


def test_event_rejects_naive_timestamp() -> None:
    """Catches acceptance of timestamps that cannot be placed on a UTC timeline."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        candidate = make_payment_event(event_time=datetime(2026, 8, 16))
        PaymentEvent.model_validate(candidate.model_dump())


def test_event_rejects_named_non_utc_zero_offset_timestamp() -> None:
    """Catches treating a non-UTC timezone as UTC merely because its offset is zero."""
    london_winter = datetime(2026, 1, 16, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(ValidationError, match="timezone-aware and UTC"):
        PaymentEvent.model_validate(make_payment_event(event_time=london_winter).model_dump())


def test_event_accepts_canonical_utc_z_timestamp() -> None:
    """Preserves support for canonical ISO-8601 UTC timestamps at the event boundary."""
    event = PaymentEvent.model_validate(make_payment_event(event_time="2026-08-16T12:00:00Z"))
    assert event.event_time.tzname() == "UTC"


@pytest.mark.parametrize(
    "field_name",
    ["event_id", "campaign_id", "trace_id", "actor_id", "counterparty_id"],
)
def test_event_rejects_non_uuid_external_identifiers(field_name: str) -> None:
    """Catches a boundary regression that permits non-pseudonymous event identifiers."""
    with pytest.raises(ValidationError, match="UUID"):
        PaymentEvent.model_validate(make_payment_event(**{field_name: "not-a-uuid"}).model_dump())


def test_event_rejects_unknown_major_schema_version() -> None:
    """Catches accidental acceptance of an incompatible event payload major version."""
    with pytest.raises(ValidationError, match="unsupported schema major"):
        PaymentEvent.model_validate(make_payment_event(schema_version="2.0.0").model_dump())


def test_event_forbids_unknown_data_but_retains_declared_extensions() -> None:
    """Catches payload drift escaping the explicit extensions boundary."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PaymentEvent.model_validate(
            {**make_payment_event().model_dump(), "partner_signal": "unexpected"}
        )

    event = make_payment_event(extensions={"partner_signal": "retained"})
    assert event.extensions == {"partner_signal": "retained"}


def test_event_uses_decimal_money_and_is_frozen() -> None:
    """Catches money coercion or mutation after a validated event is created."""
    event = make_payment_event(amount=Decimal("10.25"))
    assert event.amount == Decimal("10.25")
    with pytest.raises(ValidationError, match="frozen_instance"):
        event.amount = Decimal("9.99")


def test_scenario_validates_feedback_and_query_budget() -> None:
    """Catches scenarios that cannot support bounded decision-only attacker feedback."""
    scenario = make_scenario_config(
        attacker_mode=AttackerMode.DECISION_ONLY,
        feedback=[FeedbackField.APPROVE, FeedbackField.REALIZED_VALUE],
        query_budget=40,
    )
    assert scenario.feedback == [FeedbackField.APPROVE, FeedbackField.REALIZED_VALUE]

    with pytest.raises(ValidationError, match="query_budget"):
        ScenarioBundle.model_validate(make_scenario_config(query_budget=0).model_dump())


def test_scenario_accepts_independent_semantic_scenario_version() -> None:
    """Catches applying envelope schema-major compatibility to scenario revisions."""
    scenario = ScenarioBundle.model_validate(make_scenario_config(version="2.0.0").model_dump())
    assert scenario.schema_version == "1.0.0"
    assert scenario.version == "2.0.0"


def test_decision_rejects_future_or_equal_source() -> None:
    """Catches data leakage from source events at or after the decision moment."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with pytest.raises(ValidationError, match="strictly before"):
        Decision.model_validate(
            make_decision(decision_time=now, max_source_timestamp=now).model_dump()
        )


def test_non_approve_decision_requires_reason_codes() -> None:
    """Catches an unexplained customer-impacting challenge or decline."""
    with pytest.raises(ValidationError, match="reason_codes"):
        candidate = make_decision(action=Action.CHALLENGE, reason_codes=[])
        Decision.model_validate(candidate.model_dump())


def test_decision_validates_uuid_score_and_utc_time() -> None:
    """Catches malformed decision identity, score range, and timeline data."""
    with pytest.raises(ValidationError, match="UUID"):
        Decision.model_validate(make_decision(decision_id="decision-1").model_dump())
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Decision.model_validate(make_decision(score=1.01).model_dump())
    with pytest.raises(ValidationError, match="timezone-aware"):
        Decision.model_validate(make_decision(decision_time=datetime(2026, 8, 16)).model_dump())


def test_model_validate_revalidates_existing_contract_instances() -> None:
    """Catches model_copy overrides bypassing contract validation at the external boundary."""
    with pytest.raises(ValidationError, match="UUID"):
        Decision.model_validate(make_decision(decision_id="not-a-uuid"))
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Decision.model_validate(make_decision(score=1.01))
    with pytest.raises(ValidationError, match="unsupported schema major"):
        PaymentEvent.model_validate(make_payment_event(schema_version="9.0.0"))


def test_evaluation_report_captures_required_audit_metadata() -> None:
    """Catches reports that lose reproducibility hashes or human promotion review."""
    report = make_evaluation_report()
    assert report.promotion_decision is PromotionDecision.HOLD
    assert report.fraud_value_distribution["total"] == Decimal("1234.56")
    assert UUID(report.reviewer_id).version == 4
    assert report.failed_gates == ["calibration_drift"]


def test_evaluation_report_rejects_unknown_major_schema_version() -> None:
    """Catches an incompatible report schema from entering the evaluator boundary."""
    with pytest.raises(ValidationError, match="unsupported schema major"):
        EvaluationReport.model_validate(make_evaluation_report(schema_version="9.0.0").model_dump())
