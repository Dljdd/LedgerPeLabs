from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings


APPROVED_OPENAPI_PATHS = frozenset(
    {
        "/api/v1/health",
        "/api/v1/defense/evaluations",
        "/api/v1/defense/evaluations/{evaluation_id}",
        "/api/v1/defense/evaluations/{evaluation_id}/artifacts/{name}",
        "/api/v1/defense/v2/scorecard",
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/scenarios/compile",
        "/api/v1/threats",
        "/api/v1/threats/{threat_id}",
        "/defense/v2/scorecard",
    }
)


def test_health_is_versioned(tmp_path) -> None:
    """Catch an unversioned or unavailable health endpoint."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_openapi_exposes_only_the_approved_api_paths(tmp_path) -> None:
    """Catch accidental publication of routes outside the approved API surface."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert set(response.json()["paths"]) == APPROVED_OPENAPI_PATHS
