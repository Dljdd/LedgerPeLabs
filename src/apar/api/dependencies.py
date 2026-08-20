"""Dependencies backed by one application instance's lifespan state."""

from typing import cast

from fastapi import Request

from apar.evaluation.service import DefenseEvaluationService
from apar.registry.repository import ThreatRepository
from apar.runs import RunRunner
from apar.storage.artifacts import ArtifactStore


def get_repository(request: Request) -> ThreatRepository:
    """Return the repository initialized by the current application's lifespan."""
    return cast(ThreatRepository, request.app.state.repository)


def get_artifact_store(request: Request) -> ArtifactStore:
    """Return the immutable store initialized by the current application's lifespan."""
    return cast(ArtifactStore, request.app.state.artifact_store)


def get_run_runner(request: Request) -> RunRunner:
    """Return the signer-backed runner initialized by the application lifespan."""
    return cast(RunRunner, request.app.state.run_runner)


def get_defense_evaluation_service(request: Request) -> DefenseEvaluationService:
    """Return the signer- and verifier-pinned Defend service for this lifespan."""
    return cast(DefenseEvaluationService, request.app.state.defense_service)
