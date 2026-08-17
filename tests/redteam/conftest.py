"""Shared public-only fixtures for bounded attacker tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from apar.generators import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParams,
)
from apar.redteam import ParameterBounds


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


@pytest.fixture
def card_bounds() -> ParameterBounds:
    return ParameterBounds.for_campaign(
        "card_testing_cnp", campaign_params("card_testing_cnp")
    )
