"""Typed execution and authenticated manifest retrieval endpoints."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends
from pydantic import Field, ValidationError

from apar.api.app import ApiError, ErrorEnvelope
from apar.api.dependencies import get_artifact_store, get_run_runner
from apar.contracts._validation import ExternalContract
from apar.contracts.scenarios import ScenarioBundle
from apar.runs import AttackerPolicy, PublicRunManifest, RunExecutionError, RunRunner
from apar.storage.artifacts import ArtifactStore

SCENARIO_ARTIFACT_NOT_FOUND: Final = "SCENARIO_ARTIFACT_NOT_FOUND"
INVALID_SCENARIO_ARTIFACT: Final = "INVALID_SCENARIO_ARTIFACT"
RUN_NOT_FOUND: Final = "RUN_NOT_FOUND"
RUN_REJECTED: Final = "RUN_REJECTED"
RUN_VERIFICATION_FAILED: Final = "RUN_VERIFICATION_FAILED"

router = APIRouter(prefix="/api/v1")
Store = Annotated[ArtifactStore, Depends(get_artifact_store)]
Runner = Annotated[RunRunner, Depends(get_run_runner)]


class CreateRunRequest(ExternalContract):
    """Select immutable scenario bytes and a closed built-in worker policy."""

    scenario_artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: AttackerPolicy


@router.post(
    "/runs",
    response_model=PublicRunManifest,
    status_code=201,
    responses={
        404: {"model": ErrorEnvelope, "description": "Scenario artifact not found"},
        409: {"model": ErrorEnvelope, "description": "Run rejected"},
        422: {"model": ErrorEnvelope, "description": "Invalid request or artifact"},
    },
)
def create_run(
    request: CreateRunRequest, store: Store, runner: Runner
) -> PublicRunManifest:
    """Execute only verified compiled bytes through a typed disposable policy."""
    try:
        reference = store.resolve(request.scenario_artifact_id)
    except ValueError:
        raise ApiError(
            404,
            SCENARIO_ARTIFACT_NOT_FOUND,
            "compiled scenario artifact not found",
        ) from None
    try:
        bundle = ScenarioBundle.model_validate_json(store.read(reference))
    except (ValidationError, ValueError):
        raise ApiError(
            422,
            INVALID_SCENARIO_ARTIFACT,
            "artifact is not a compiled scenario",
        ) from None
    try:
        return runner.public_view(runner.execute(bundle, request.policy))
    except RunExecutionError:
        raise ApiError(
            409,
            RUN_REJECTED,
            "run rejected by execution boundary",
        ) from None


@router.get(
    "/runs/{run_id}",
    response_model=PublicRunManifest,
    responses={
        404: {"model": ErrorEnvelope, "description": "Run not found"},
        409: {"model": ErrorEnvelope, "description": "Run verification failed"},
        422: {"model": ErrorEnvelope, "description": "Request validation failed"},
    },
)
def get_run(run_id: str, runner: Runner) -> PublicRunManifest:
    """Return one fully reverified signed manifest without restricted payloads."""
    try:
        return runner.public_view(runner.get(run_id))
    except KeyError:
        raise ApiError(404, RUN_NOT_FOUND, "run not found") from None
    except RunExecutionError:
        raise ApiError(
            409,
            RUN_VERIFICATION_FAILED,
            "stored run failed authenticated verification",
        ) from None


__all__ = ["CreateRunRequest", "router"]
