"""Closed contracts for the defender-visible event boundary."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import PolicyThresholds, scrub_event
from tests.factories import make_payment_event


def test_scrub_event_removes_every_evaluator_semantic() -> None:
    event = make_payment_event(
        rail_data={"payment_id": "pay-1", "hidden_family": "app_scam_mule"},
        lineage={"synthetic": True, "campaign_role": "attack", "generator": "dev"},
        party_refs={"actor_role": "mule", "merchant_id": "merchant-1"},
    )

    observed = scrub_event(event)

    dumped = observed.model_dump(mode="json")
    encoded = json.dumps(dumped, sort_keys=True).lower()
    assert observed.optional_refs == {"merchant_id": "merchant-1"}
    assert all(
        token not in encoded
        for token in ("hidden_family", "campaign_role", "actor_role", "generator")
    )


@pytest.mark.parametrize(
    ("rail", "event_type", "expected"),
    [
        (Rail.CARD, EventKind.AUTHORIZATION, True),
        (Rail.CARD, EventKind.AUTHORIZATION_DECLINED, True),
        (Rail.CARD, EventKind.CLEARING, False),
        (Rail.A2A, EventKind.TRANSFER_INITIATED, True),
        (Rail.A2A, EventKind.TRANSFER_POSTED, False),
        (Rail.AGENTIC, EventKind.AUTHORIZATION, True),
        (Rail.AGENTIC, EventKind.AUTHENTICATION_CHALLENGE, True),
        (Rail.AGENTIC, EventKind.AUTHORIZATION_DECLINED, True),
        (Rail.AGENTIC, EventKind.CLEARING, False),
    ],
)
def test_scrub_event_marks_only_declared_payment_openings_as_decision_points(
    rail: Rail, event_type: EventKind, expected: bool
) -> None:
    observed = scrub_event(
        make_payment_event(
            rail=rail,
            event_type=event_type,
            rail_data={"payment_id": "pay-1", "integrity": "pass"},
        )
    )

    assert observed.is_decision_point is expected


def test_scrub_event_projects_agentic_integrity_receipt_status() -> None:
    observed = scrub_event(
        make_payment_event(
            rail=Rail.AGENTIC,
            rail_data={
                "payment_id": "pay-1",
                "integrity": "fail",
                "reason_code": "payee_substitution",
                "receipt_hash": "0" * 64,
                "receipt_outcome": "preview",
            },
        )
    )

    assert observed.integrity_status == "fail"
    assert observed.integrity_reason == "payee_substitution"


def test_policy_thresholds_require_challenge_not_to_exceed_decline() -> None:
    with pytest.raises(
        ValidationError, match="challenge threshold must not exceed decline threshold"
    ):
        PolicyThresholds(challenge=0.8, decline=0.7)
