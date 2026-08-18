"""Health endpoint for local orchestration."""

from fastapi import APIRouter

from apar import __version__

router = APIRouter()


@router.get("/api/v1/health")
def get_health() -> dict[str, str]:
    """Report that the local API process is available and identify its version."""
    return {"status": "ok", "version": __version__}
