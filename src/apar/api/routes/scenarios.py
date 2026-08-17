"""Evidence-backed scenario compilation into immutable artifacts."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends

from apar.api.app import ApiError, ErrorEnvelope
from apar.api.dependencies import get_artifact_store, get_repository
from apar.compiler import CompilerError, compile_scenario
from apar.contracts._validation import ExternalContract
from apar.contracts.scenarios import ScenarioConfig
from apar.registry.repository import ThreatRepository
from apar.runs import bind_scenario_for_run
from apar.storage.artifacts import ArtifactStore

THREAT_NOT_FOUND: Final = "THREAT_NOT_FOUND"

router = APIRouter(prefix="/api/v1")
Repository = Annotated[ThreatRepository, Depends(get_repository)]
Store = Annotated[ArtifactStore, Depends(get_artifact_store)]


class CompileScenarioRequest(ExternalContract):
    """Select reviewed evidence and supply a closed bounded configuration."""

    threat_id: str
    config: ScenarioConfig


class CompiledScenarioArtifact(ExternalContract):
    """Public handle for one immutable compiled bundle."""

    scenario_artifact_id: str
    scenario_id: str


@router.post(
    "/scenarios/compile",
    response_model=CompiledScenarioArtifact,
    status_code=201,
    responses={
        404: {"model": ErrorEnvelope, "description": "Threat card not found"},
        422: {"model": ErrorEnvelope, "description": "Compilation rejected"},
    },
)
def compile_scenario_artifact(
    request: CompileScenarioRequest,
    repository: Repository,
    store: Store,
) -> CompiledScenarioArtifact:
    """Compile one registered card and return only its content address."""
    card = repository.get(request.threat_id)
    if card is None:
        raise ApiError(404, THREAT_NOT_FOUND, "threat card not found")
    try:
        bundle = bind_scenario_for_run(
            compile_scenario(card, request.config), threat_family=card.family
        )
    except CompilerError as error:
        raise ApiError(422, error.code, str(error)) from None
    reference = store.put_json(bundle)
    return CompiledScenarioArtifact(
        scenario_artifact_id=reference.sha256,
        scenario_id=bundle.scenario_id,
    )


__all__ = ["CompileScenarioRequest", "CompiledScenarioArtifact", "router"]
