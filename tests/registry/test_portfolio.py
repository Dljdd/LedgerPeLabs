"""Validation and curation invariants for the shipped threat portfolio."""

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest

from apar.contracts.events import Rail
from apar.contracts.scenarios import FeedbackField, ScenarioConfig
from apar.registry.models import ThreatCard
from scripts import verify_g0 as g0_verifier

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


def _load_portfolio(root: Path = PORTFOLIO_ROOT) -> list[ThreatCard]:
    cards = []
    for path in sorted(root.glob("*.json")):
        raw = json.loads(path.read_text())
        assert raw.get("simulation_status") == "simulatable", path
        card = ThreatCard.model_validate(raw)
        assert path.stem == card.threat_id, f"{path}: filename does not match threat_id"
        cards.append(card)
    return cards


def _portfolio_with_swapped_bodies(tmp_path: Path) -> Path:
    swapped_root = tmp_path / "threats"
    shutil.copytree(PORTFOLIO_ROOT, swapped_root)
    first = swapped_root / "adaptive-card-testing.json"
    second = swapped_root / "agentic-cart-tampering.json"
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    first.write_bytes(second_payload)
    second.write_bytes(first_payload)
    return swapped_root


def test_test_loader_rejects_filename_body_swap(tmp_path: Path) -> None:
    """Catch a same-ID-set body swap that masks filename-to-ID corruption in tests."""
    swapped_root = _portfolio_with_swapped_bodies(tmp_path)

    with pytest.raises(AssertionError, match="filename does not match threat_id"):
        _load_portfolio(swapped_root)


def test_g0_loader_rejects_filename_body_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch the G0 command accepting correct ID sets assigned to the wrong files."""
    swapped_root = _portfolio_with_swapped_bodies(tmp_path)
    monkeypatch.setattr(g0_verifier, "PORTFOLIO_ROOT", swapped_root)

    with pytest.raises(g0_verifier.G0VerificationError, match="filename does not match threat_id"):
        g0_verifier._load_portfolio()


def test_simulation_status_is_explicit_in_every_raw_threat_fixture() -> None:
    """Catch fixture omissions that Pydantic would silently fill from a model default."""
    paths = [*sorted(PORTFOLIO_ROOT.glob("*.json")), Path("fixtures/golden/threat-card.json")]

    for path in paths:
        raw = json.loads(path.read_text())
        assert raw.get("simulation_status") == "simulatable", path


def test_instant_payment_card_has_direct_faster_payments_abuse_evidence() -> None:
    """Catch regression to risk tooling that does not evidence instant-payment abuse."""
    card = ThreatCard.model_validate_json(
        (PORTFOLIO_ROOT / "instant-payment-velocity.json").read_text()
    )
    fact_urls = {
        record.direct_source_url
        for record in card.evidence
        if not record.is_project_inference
    }

    assert (
        "https://www.psr.org.uk/news-and-updates/speeches/speeches/"
        "chris-hemsley-speech-at-the-fraud-leaders-summit-on-28-february-2024/"
        in fact_urls
    )


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
