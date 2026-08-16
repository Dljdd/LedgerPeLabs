"""Golden G0 flow across the real registry, compiler, and artifact store."""

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from apar.api import create_app
from apar.compiler.compiler import compile_scenario
from apar.config import Settings
from apar.contracts.scenarios import FeedbackField, ScenarioConfig
from apar.registry.models import ThreatCard
from apar.registry.repository import ThreatRepository
from apar.storage.artifacts import ArtifactStore

GOLDEN_THREAT = Path("fixtures/golden/threat-card.json")
GOLDEN_CONFIG = Path("fixtures/golden/scenario-config.json")


def test_threat_compiles_and_freezes_deterministically(tmp_path: Path) -> None:
    """Catch broken golden validation, compilation, or immutable digest reuse."""
    golden_threat = ThreatCard.model_validate_json(GOLDEN_THREAT.read_text())
    golden_config = ScenarioConfig.model_validate_json(GOLDEN_CONFIG.read_text())
    repository = ThreatRepository(tmp_path / "state.db")
    store = ArtifactStore(tmp_path / "artifacts")

    repository.upsert(golden_threat)
    registered = repository.get("app-personalized-mule")
    assert registered is not None

    first_bundle = compile_scenario(registered, golden_config)
    second_bundle = compile_scenario(registered, golden_config)
    first_ref = store.put_json(first_bundle)
    second_ref = store.put_json(second_bundle)

    assert first_bundle == second_bundle
    assert first_ref == second_ref
    assert first_ref.sha256 == second_ref.sha256
    assert first_bundle.seed == 260816
    assert first_bundle.feedback == [FeedbackField.ACTION, FeedbackField.REASON_FAMILY]


def test_golden_threat_is_available_through_real_api(tmp_path: Path) -> None:
    """Catch API wiring that cannot expose a card persisted by the real repository."""
    golden_threat = ThreatCard.model_validate_json(GOLDEN_THREAT.read_text())
    settings = Settings.from_root(tmp_path)
    repository = ThreatRepository(settings.database_path)
    repository.upsert(golden_threat)

    with TestClient(create_app(settings)) as client:
        health_response = client.get("/api/v1/health")
        registry_response = client.get("/api/v1/threats")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert registry_response.status_code == 200
    assert [card["threat_id"] for card in registry_response.json()] == [
        "app-personalized-mule"
    ]


def test_one_command_g0_verification() -> None:
    """Catch a missing or incomplete clean-state G0 verification entry point."""
    result = subprocess.run(
        [sys.executable, "scripts/verify_g0.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith(
        "G0 PASS: 20 threat cards, contracts, registry, compiler, API, "
        "and artifact store"
    )
