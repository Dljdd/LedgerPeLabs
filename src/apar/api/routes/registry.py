"""Threat registry endpoints."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends

from apar.api.app import ApiError
from apar.api.dependencies import get_repository
from apar.registry.models import ThreatCard
from apar.registry.repository import ThreatRepository

THREAT_NOT_FOUND: Final = "THREAT_NOT_FOUND"
THREAT_ID_MISMATCH: Final = "THREAT_ID_MISMATCH"

router = APIRouter(prefix="/api/v1")
Repository = Annotated[ThreatRepository, Depends(get_repository)]


@router.get("/threats", response_model=list[ThreatCard])
def list_threats(repository: Repository) -> list[ThreatCard]:
    """Return the current registry in stable threat-ID order."""
    return repository.list()


@router.get("/threats/{threat_id}", response_model=ThreatCard)
def get_threat(threat_id: str, repository: Repository) -> ThreatCard:
    """Return one registered threat card or a structured missing-resource error."""
    card = repository.get(threat_id)
    if card is None:
        raise ApiError(404, THREAT_NOT_FOUND, "threat card not found")
    return card


@router.put("/threats/{threat_id}", response_model=ThreatCard)
def put_threat(threat_id: str, card: ThreatCard, repository: Repository) -> ThreatCard:
    """Replace the registered card when its stable ID matches the target path."""
    if threat_id != card.threat_id:
        raise ApiError(409, THREAT_ID_MISMATCH, "path threat ID must match payload threat_id")
    repository.upsert(card)
    return card
