from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from tests.factories import make_threat_card


def test_registry_lists_cards_and_reads_a_card_written_through_the_api(tmp_path) -> None:
    """Catch routes that do not use the lifespan-owned registry repository."""
    threat_payload = make_threat_card().model_dump(mode="json")

    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        assert client.get("/api/v1/threats").json() == []
        write_response = client.put("/api/v1/threats/app-personalized-mule", json=threat_payload)
        read_response = client.get("/api/v1/threats/app-personalized-mule")
        list_response = client.get("/api/v1/threats")

    assert write_response.status_code == 200
    assert write_response.json() == threat_payload
    assert read_response.status_code == 200
    assert read_response.json() == threat_payload
    assert list_response.status_code == 200
    assert list_response.json() == [threat_payload]


def test_missing_threat_uses_the_structured_not_found_error(tmp_path) -> None:
    """Catch missing registry records being returned as a framework-default 404."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/api/v1/threats/no-such-threat")

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"code": "THREAT_NOT_FOUND", "message": "threat card not found"}
    }


def test_path_and_payload_threat_ids_must_match(tmp_path) -> None:
    """Catch an overwrite of a card whose payload ID does not match the target path."""
    threat_payload = make_threat_card().model_dump(mode="json")

    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.put("/api/v1/threats/other-id", json=threat_payload)

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "THREAT_ID_MISMATCH",
            "message": "path threat ID must match payload threat_id",
        }
    }


def test_invalid_threat_payload_uses_the_structured_validation_error(tmp_path) -> None:
    """Catch request validation errors that leak FastAPI's list-shaped error response."""
    threat_payload = make_threat_card().model_dump(mode="json")
    threat_payload["unexpected"] = "not allowed"

    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.put("/api/v1/threats/app-personalized-mule", json=threat_payload)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"code": "VALIDATION_FAILED", "message": "request validation failed"}
    }


def test_openapi_has_only_the_declared_registry_operations(tmp_path) -> None:
    """Catch missing or accidental registry operations in the published API contract."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert set(paths["/api/v1/threats"]) == {"get"}
    assert set(paths["/api/v1/threats/{threat_id}"]) == {"get", "put"}
