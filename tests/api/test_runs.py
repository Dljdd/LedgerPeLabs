"""Compiled-scenario and authenticated-run API contracts."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from apar.contracts.events import Rail
from apar.contracts.scenarios import AttackerMode, ScenarioBundle
from apar.storage.artifacts import ArtifactStore
from tests.factories import make_scenario_config, make_threat_card


@contextmanager
def _registered_client(root: Path) -> Iterator[tuple[TestClient, dict[str, object]]]:
    config = make_scenario_config(
        rail=Rail.A2A,
        query_budget=1,
        seed=960,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 960}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    card = make_threat_card(rails=[Rail.A2A], default_config=config)
    client = TestClient(create_app(Settings.from_root(root)))
    with client:
        response = client.put(
            f"/api/v1/threats/{card.threat_id}",
            json=card.model_dump(mode="json"),
        )
        assert response.status_code == 200
        yield client, {
            "threat_id": card.threat_id,
            "config": config.model_dump(mode="json"),
        }


def test_compile_returns_a_verified_scenario_artifact_id(tmp_path: Path) -> None:
    """Catches compilation accepting paths or returning mutable scenario payloads."""
    with _registered_client(tmp_path) as (client, request):
        response = client.post("/api/v1/scenarios/compile", json=request)
        assert response.status_code == 201
        body = response.json()
        assert set(body) == {"scenario_artifact_id", "scenario_id"}
        store = ArtifactStore(Settings.from_root(tmp_path).artifact_root)
        ref = store.resolve(body["scenario_artifact_id"])
        bundle = ScenarioBundle.model_validate_json(store.read(ref))

    assert body["scenario_id"] == "app-mule-personalized-v1"
    assert bundle.seed == 960
    assert bundle.extensions["apar_run_binding_v1"] == {
        "attacker_mode": AttackerMode.DECISION_ONLY.value,
        "threat_card_ref": "app-personalized-mule@2",
        "threat_family": "app_scam_mule",
    }


def test_run_rejects_same_rail_policy_from_a_different_reviewed_family(
    tmp_path: Path,
) -> None:
    """Catch rail-only pairing of card-testing and synthetic-refund policies."""
    config = make_scenario_config(
        rail=Rail.CARD,
        query_budget=1,
        seed=962,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 962}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    card = make_threat_card(
        threat_id="reviewed-card-testing",
        family="card_testing_cnp",
        rails=[Rail.CARD],
        default_config=config,
    )
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        assert client.put(
            f"/api/v1/threats/{card.threat_id}", json=card.model_dump(mode="json")
        ).status_code == 200
        compiled = client.post(
            "/api/v1/scenarios/compile",
            json={"threat_id": card.threat_id, "config": config.model_dump(mode="json")},
        ).json()
        response = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": compiled["scenario_artifact_id"],
                "policy": {
                    "family": "synthetic_merchant_refund",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                },
            },
        )

    assert response.status_code == 409


def test_run_accepts_only_a_compiled_id_and_typed_policy_then_is_gettable(
    tmp_path: Path,
) -> None:
    """Catches routes accepting executable policy authority or losing signed run lineage."""
    with _registered_client(tmp_path) as (client, compile_request):
        compiled = client.post(
            "/api/v1/scenarios/compile", json=compile_request
        ).json()
        response = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": compiled["scenario_artifact_id"],
                "policy": {
                    "family": "app_scam_mule",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                },
            },
        )
        assert response.status_code == 201
        manifest = response.json()
        fetched = client.get(f"/api/v1/runs/{manifest['run_id']}")
        internal = cast(FastAPI, client.app).state.run_runner.get(manifest["run_id"])
        restricted_ref = internal.artifacts["restricted_validity"]

        assert fetched.status_code == 200
        assert fetched.json() == manifest
        assert set(manifest["artifacts"]) == {
            "events",
            "feedback",
            "policy",
            "population",
            "scenario",
        }
        assert restricted_ref.sha256 not in json.dumps(manifest, sort_keys=True)
        assert restricted_ref.relative_path not in json.dumps(manifest, sort_keys=True)
        assert "restricted_validity" not in str(manifest)
        assert "restricted_evaluation" not in str(manifest)
        assert "reasons" not in str(manifest)

        rejected = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": compiled["scenario_artifact_id"],
                "policy": {
                    "family": "app_scam_mule",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                    "path": "/tmp/untrusted.py",
                },
            },
        )
        disclosure_escalation = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": compiled["scenario_artifact_id"],
                "policy": {
                    "family": "app_scam_mule",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                    "expose_realized_value": True,
                },
            },
        )
        mismatched = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": compiled["scenario_artifact_id"],
                "policy": {
                    "family": "card_testing_cnp",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                },
            },
        )
    assert rejected.status_code == 422
    assert disclosure_escalation.status_code == 422
    assert mismatched.status_code == 409
    assert mismatched.json() == {
        "detail": {
            "code": "RUN_REJECTED",
            "message": "run rejected by execution boundary",
        }
    }


def test_unknown_scenario_artifact_and_run_use_structured_errors(tmp_path: Path) -> None:
    """Catches framework errors leaking storage paths or internal runner exceptions."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        run_response = client.post(
            "/api/v1/runs",
            json={
                "scenario_artifact_id": "0" * 64,
                "policy": {
                    "family": "app_scam_mule",
                    "attacker_mode": "decision_only",
                    "kind": "fixed",
                    "query_budget": 1,
                    "worker_timeout_ms": 2_000,
                },
            },
        )
        get_response = client.get("/api/v1/runs/run-00000000000000000000000000000000")

    assert run_response.status_code == 404
    assert run_response.json() == {
        "detail": {
            "code": "SCENARIO_ARTIFACT_NOT_FOUND",
            "message": "compiled scenario artifact not found",
        }
    }
    assert get_response.status_code == 404
    assert get_response.json() == {
        "detail": {"code": "RUN_NOT_FOUND", "message": "run not found"}
    }
