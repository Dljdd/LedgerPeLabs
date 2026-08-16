"""Dependencies backed by one application instance's lifespan state."""

from typing import cast

from fastapi import Request

from apar.registry.repository import ThreatRepository


def get_repository(request: Request) -> ThreatRepository:
    """Return the repository initialized by the current application's lifespan."""
    return cast(ThreatRepository, request.app.state.repository)
