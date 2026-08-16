"""Validation and curation invariants for the shipped threat portfolio."""

from pathlib import Path
from urllib.parse import urlparse

from apar.contracts.events import Rail
from apar.contracts.scenarios import FeedbackField, ScenarioConfig
from apar.registry.models import ThreatCard

PORTFOLIO_ROOT = Path("fixtures/threats")
EXPECTED_IDS = {
    "adaptive-card-testing",
    "agentic-cart-tampering",
    "agentic-mandate-escalation",
    "agentic-payee-substitution",
    "app-personalized-mule",
    "cnp-checkout-automation",
    "credential-stuffing-ato",
    "first-party-dispute-automation",
    "instant-payment-velocity",
    "investment-persuasion",
    "invoice-payee-substitution",
    "merchant-laundering",
    "mule-fanout-layering",
    "mule-recruitment",
    "promotion-abuse",
    "qr-social-engineering",
    "remote-access-guidance",
    "synthetic-identity-bustout",
    "synthetic-merchant-refund",
    "voice-clone-app",
}
EXPECTED_DEEP_SCENARIOS = {
    "adaptive-card-testing": "card_testing_cnp",
    "agentic-payee-substitution": "agentic_intent_abuse",
    "app-personalized-mule": "app_scam_mule",
    "synthetic-merchant-refund": "synthetic_merchant_refund",
}
AUTHORITATIVE_SOURCE_TYPES = {
    "government_advisory",
    "industry_body_report",
    "original_research",
    "payment_network_guidance",
    "regulator_guidance",
    "standards_guidance",
}


def _load_portfolio() -> list[ThreatCard]:
    return [
        ThreatCard.model_validate_json(path.read_text())
        for path in sorted(PORTFOLIO_ROOT.glob("*.json"))
    ]


def test_portfolio_has_exactly_the_approved_twenty_ids() -> None:
    """Catch missing, extra, renamed, or invalid threat-card files."""
    paths = sorted(PORTFOLIO_ROOT.glob("*.json"))
    cards = _load_portfolio()

    assert len(paths) == 20
    assert {path.stem for path in paths} == EXPECTED_IDS
    assert {card.threat_id for card in cards} == EXPECTED_IDS


def test_portfolio_has_exactly_four_family_aligned_deep_scenarios() -> None:
    """Catch deep-scenario scope drift or a family misalignment."""
    deep_scenarios = {
        card.threat_id: card.family
        for card in _load_portfolio()
        if card.implementation_status == "deep_scenario"
    }

    assert deep_scenarios == EXPECTED_DEEP_SCENARIOS


def test_every_card_has_direct_authoritative_fact_and_separate_inference() -> None:
    """Catch unsourced hypotheses or evidence that conflates fact with inference."""
    cards = _load_portfolio()
    assert len(cards) == 20

    for card in cards:
        facts = [record for record in card.evidence if not record.is_project_inference]
        inferences = [record for record in card.evidence if record.is_project_inference]
        authoritative = [
            record
            for record in facts
            if record.source_type in AUTHORITATIVE_SOURCE_TYPES
            and record.quality_grade in {"A", "B"}
            and urlparse(record.direct_source_url).scheme in {"http", "https"}
            and bool(urlparse(record.direct_source_url).netloc)
        ]

        assert authoritative, card.threat_id
        assert inferences, card.threat_id
        assert all(record.accessed_on.isoformat() == "2026-08-16" for record in card.evidence)
        assert card.observables
        assert card.rails
        assert any(card.genai_capability.values())
        assert card.simulation_status == "simulatable"
        assert card.safety_class == "synthetic_only"
        assert card.default_config.export_level == "sanitized"


def test_golden_app_contract_has_the_fixed_controller_values() -> None:
    """Catch changes to the deterministic APP challenge contract."""
    card = ThreatCard.model_validate_json(Path("fixtures/golden/threat-card.json").read_text())
    config = ScenarioConfig.model_validate_json(
        Path("fixtures/golden/scenario-config.json").read_text()
    )

    assert card.threat_id == "app-personalized-mule"
    assert card.family == "app_scam_mule"
    assert card.rails == [Rail.A2A, Rail.AGENTIC]
    assert card.genai_capability == {
        "personalization": True,
        "iteration_speed": True,
    }
    assert config.seed == 260816
    assert config.duration_hours == 24
    assert config.benign_entity_count == 5_000
    assert config.illicit_entity_count == 60
    assert config.feedback == [FeedbackField.ACTION, FeedbackField.REASON_FAMILY]
