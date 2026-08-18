"""Verify the complete G0 foundation contract from a clean temporary state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from urllib.parse import urlparse

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from apar import __version__
from apar.api import create_app
from apar.compiler import compile_scenario
from apar.config import Settings
from apar.contracts.events import Rail
from apar.contracts.scenarios import FeedbackField, ScenarioConfig
from apar.registry.models import ThreatCard
from apar.registry.repository import ThreatRepository
from apar.storage.artifacts import ArtifactStore

PASS_LINE = (
    "G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_ROOT = REPOSITORY_ROOT / "fixtures" / "threats"
GOLDEN_ROOT = REPOSITORY_ROOT / "fixtures" / "golden"
GOLDEN_HASH_MANIFEST = GOLDEN_ROOT / "canonical-sha256.json"
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


class G0VerificationError(RuntimeError):
    """A failed G0 invariant with a concise command-line message."""


class GoldenHashManifest(BaseModel):
    """Strict review pins for the complete canonical golden artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    threat_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G0VerificationError(message)


def _require_not_none[ValueT](value: ValueT | None, message: str) -> ValueT:
    if value is None:
        raise G0VerificationError(message)
    return value


def _load_raw_object(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), f"{path}: fixture must be a JSON object")
    return cast(dict[str, object], loaded)


def _load_golden_hash_manifest() -> GoldenHashManifest:
    return GoldenHashManifest.model_validate_json(
        GOLDEN_HASH_MANIFEST.read_text(encoding="utf-8")
    )


def _load_portfolio() -> list[ThreatCard]:
    paths = sorted(PORTFOLIO_ROOT.glob("*.json"))
    _require(len(paths) == 20, "portfolio must contain exactly 20 JSON files")
    _require({path.stem for path in paths} == EXPECTED_IDS, "portfolio threat IDs changed")
    cards = []
    for path in paths:
        raw = _load_raw_object(path)
        _require(
            raw.get("simulation_status") == "simulatable",
            f"{path}: simulation_status must be explicit",
        )
        card = ThreatCard.model_validate(raw)
        _require(
            path.stem == card.threat_id,
            f"{path}: filename does not match threat_id",
        )
        cards.append(card)
    return cards


def _verify_portfolio(cards: list[ThreatCard]) -> None:
    _require({card.threat_id for card in cards} == EXPECTED_IDS, "validated IDs changed")
    deep_scenarios = {
        card.threat_id: card.family
        for card in cards
        if card.implementation_status == "deep_scenario"
    }
    _require(deep_scenarios == EXPECTED_DEEP_SCENARIOS, "deep-scenario mapping changed")

    for card in cards:
        facts = [record for record in card.evidence if not record.is_project_inference]
        inferences = [record for record in card.evidence if record.is_project_inference]
        authoritative_direct_facts = [
            record
            for record in facts
            if record.source_type in AUTHORITATIVE_SOURCE_TYPES
            and record.quality_grade in {"A", "B"}
            and urlparse(record.direct_source_url).scheme in {"http", "https"}
            and bool(urlparse(record.direct_source_url).netloc)
        ]
        _require(bool(authoritative_direct_facts), f"{card.threat_id}: direct evidence missing")
        _require(bool(inferences), f"{card.threat_id}: project inference missing")
        _require(
            all(record.accessed_on.isoformat() == "2026-08-16" for record in card.evidence),
            f"{card.threat_id}: evidence access date changed",
        )
        _require(bool(card.observables), f"{card.threat_id}: observables missing")
        _require(bool(card.rails), f"{card.threat_id}: affected rails missing")
        _require(any(card.genai_capability.values()), f"{card.threat_id}: capability delta missing")
        _require(card.safety_class == "synthetic_only", f"{card.threat_id}: unsafe class")
        _require(
            card.default_config.export_level == "sanitized",
            f"{card.threat_id}: unsafe export",
        )


def _load_and_verify_golden(cards: list[ThreatCard]) -> tuple[ThreatCard, ScenarioConfig]:
    golden_card_path = GOLDEN_ROOT / "threat-card.json"
    golden_card_raw = _load_raw_object(golden_card_path)
    _require(
        golden_card_raw.get("simulation_status") == "simulatable",
        f"{golden_card_path}: simulation_status must be explicit",
    )
    golden_card = ThreatCard.model_validate(golden_card_raw)
    golden_config = ScenarioConfig.model_validate_json(
        (GOLDEN_ROOT / "scenario-config.json").read_text(encoding="utf-8")
    )
    portfolio_card = next(card for card in cards if card.threat_id == golden_card.threat_id)
    _require(golden_card == portfolio_card, "golden APP card differs from portfolio card")
    _require(golden_card.family == "app_scam_mule", "golden APP family changed")
    _require(golden_card.rails == [Rail.A2A, Rail.AGENTIC], "golden APP rails changed")
    _require(
        golden_card.genai_capability == {"personalization": True, "iteration_speed": True},
        "golden APP capability delta changed",
    )
    _require(golden_config.seed == 260816, "golden seed changed")
    _require(golden_config.duration_hours == 24, "golden duration changed")
    _require(golden_config.benign_entity_count == 5_000, "golden benign count changed")
    _require(golden_config.illicit_entity_count == 60, "golden illicit count changed")
    _require(
        golden_config.feedback == [FeedbackField.ACTION, FeedbackField.REASON_FAMILY],
        "golden feedback contract changed",
    )
    return golden_card, golden_config


def _verify_real_integrations(
    cards: list[ThreatCard],
    golden_card: ThreatCard,
    golden_config: ScenarioConfig,
    pinned_hashes: GoldenHashManifest,
) -> None:
    with TemporaryDirectory(prefix="apar-g0-") as temporary_root:
        settings = Settings.from_root(Path(temporary_root))
        repository = ThreatRepository(settings.database_path)
        for card in cards:
            repository.upsert(card)
        _require(
            repository.list() == sorted(cards, key=lambda card: card.threat_id),
            "registry drift",
        )

        registered = _require_not_none(
            repository.get(golden_card.threat_id), "golden card missing from registry"
        )
        first_bundle = compile_scenario(registered, golden_config)
        second_bundle = compile_scenario(registered, golden_config)
        _require(first_bundle == second_bundle, "compiler output is not deterministic")
        _require(first_bundle.seed == 260816, "compiled seed changed")

        store = ArtifactStore(settings.artifact_root)
        card_ref = store.put_json(golden_card)
        first_ref = store.put_json(first_bundle)
        second_ref = store.put_json(second_bundle)
        _require(first_ref == second_ref, "artifact digest was not reused")
        _require(store.read(first_ref) == store.read(second_ref), "artifact payload changed")
        _require(
            card_ref.sha256 == pinned_hashes.threat_card_sha256,
            "golden threat-card canonical hash changed",
        )
        _require(
            first_ref.sha256 == pinned_hashes.scenario_bundle_sha256,
            "golden scenario-bundle canonical hash changed",
        )

        with TestClient(create_app(settings)) as client:
            health_response = client.get("/api/v1/health")
            registry_response = client.get("/api/v1/threats")
            golden_response = client.get(f"/api/v1/threats/{golden_card.threat_id}")
            missing_response = client.get("/api/v1/threats/not-present")

        _require(health_response.status_code == 200, "health API failed")
        health = TypeAdapter(dict[str, str]).validate_python(health_response.json())
        _require(health == {"status": "ok", "version": __version__}, "health API drift")
        _require(registry_response.status_code == 200, "registry list API failed")
        api_cards = TypeAdapter(list[ThreatCard]).validate_python(registry_response.json())
        _require(api_cards == sorted(cards, key=lambda card: card.threat_id), "registry API drift")
        _require(golden_response.status_code == 200, "registry get API failed")
        _require(
            ThreatCard.model_validate(golden_response.json()) == golden_card,
            "registry get API returned the wrong card",
        )
        _require(missing_response.status_code == 404, "registry missing-card behavior changed")


def verify_g0() -> None:
    """Raise when any portfolio or end-to-end G0 invariant fails."""
    cards = _load_portfolio()
    _verify_portfolio(cards)
    golden_card, golden_config = _load_and_verify_golden(cards)
    pinned_hashes = _load_golden_hash_manifest()
    _verify_real_integrations(cards, golden_card, golden_config, pinned_hashes)


def main() -> int:
    """Return a shell-friendly status and print the stable G0 success line."""
    try:
        verify_g0()
    except Exception as error:
        print(f"G0 FAIL: {error}", file=sys.stderr)
        return 1
    print(PASS_LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
