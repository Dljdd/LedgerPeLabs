"""Versioned aggregate-only Defend scorecard and artifact endpoints."""

from __future__ import annotations

import base64
from typing import Annotated, Final, Never

from fastapi import APIRouter, Depends, Request, Response
from pydantic import Field, field_validator

from apar.api.app import ApiError, ErrorEnvelope
from apar.api.dependencies import get_defense_evaluation_service
from apar.contracts._validation import ExternalContract
from apar.evaluation.reporting import DefenseScorecard
from apar.evaluation.service import (
    DefenseArtifactInvalid,
    DefenseEvaluationService,
    DefenseExecutionConflict,
    DefenseResourceNotFound,
)
from apar.evaluation.v2_reporting import DefenseV2Scorecard

DEFENSE_EVALUATION_NOT_FOUND: Final = "DEFENSE_EVALUATION_NOT_FOUND"
DEFENSE_ARTIFACT_NOT_FOUND: Final = "DEFENSE_ARTIFACT_NOT_FOUND"
DEFENSE_ARTIFACT_INVALID: Final = "DEFENSE_ARTIFACT_INVALID"
DEFENSE_EVALUATION_CONFLICT: Final = "DEFENSE_EVALUATION_CONFLICT"

router = APIRouter(prefix="/api/v1/defense")
public_router = APIRouter(prefix="/defense")
Service = Annotated[DefenseEvaluationService, Depends(get_defense_evaluation_service)]

_PUBLIC_V2_SCORECARD = DefenseV2Scorecard.from_json(
    base64.b64decode(
        b"eyJhcm1zIjpbeyJhcm0iOiJydWxlc19vbmx5IiwiZ2F0ZSI6eyJhcm0iOiJydWxlc19vbmx5Iiwib3V0"
        b"Y29tZSI6eyJjb2RlcyI6WyJOT1RfRVhFQ1VURUQiXSwicGFzc2VkIjpmYWxzZX19LCJzdGF0dXMiOiJu"
        b"b3RfZXhlY3V0ZWQifSx7ImFybSI6ImdiZHRfb25seSIsImdhdGUiOnsiYXJtIjoiZ2JkdF9vbmx5Iiwi"
        b"b3V0Y29tZSI6eyJjb2RlcyI6WyJOT1RfRVhFQ1VURUQiXSwicGFzc2VkIjpmYWxzZX19LCJzdGF0dXMi"
        b"OiJub3RfZXhlY3V0ZWQifSx7ImFybSI6ImxheWVyZWRfaHlicmlkIiwiZ2F0ZSI6eyJhcm0iOiJsYXll"
        b"cmVkX2h5YnJpZCIsIm91dGNvbWUiOnsiY29kZXMiOlsiTk9UX0VYRUNVVEVEIl0sInBhc3NlZCI6ZmFs"
        b"c2V9fSwic3RhdHVzIjoibm90X2V4ZWN1dGVkIn1dLCJwcm90b2NvbF9kaWdlc3QiOiI3MzkyZWM0ZmFj"
        b"MTMzNzU0NzNlNmMzMzU1MjllZjc1NGJjNTlhNGNhYTc0Yzk1ZmYxOWVlZTJkNWZkOWYyNjZkIiwicHVi"
        b"bGljX2tleV9iYXNlNjQiOiJHU0JNakR1b1c2Qlp6Umlza0JlOUFxR05wZEVNYVB0M21qUytRKzhHZDVF"
        b"PSIsInNjaGVtYV92ZXJzaW9uIjoiMi4wLjAiLCJzaWduYXR1cmVfYmFzZTY0Ijoielk5bytOMDc3VDUr"
        b"MzYrbFNENnhjZmpLSERnbDI1eEhZeUdkRkdXc2JyZFdNUXVENkZkdWwyOVk3V1Q0UzlLS2htNFFNZmZv"
        b"V0RTclhDWXJZUkkxQ1E9PSIsInNpZ25lcl9rZXlfaWQiOiI4MWViYzIzYzVhOWRkNjEzNGIxNWE3ODll"
        b"Zjg2ZWZhMGE0OWJhMDU1YTQ2MTQ0ODJhNjQxNzQ5ZmFmOTVjZDFiIiwic3RhdHVzIjoibm90X2V4ZWN1"
        b"dGVkIiwic3ludGhldGljX3Njb3BlIjoiU3ludGhldGljLW9ubHkgZXZhbHVhdGlvbjsgbm90IGEgcmVh"
        b"bC13b3JsZCBwcmV2YWxlbmNlIG9yIGV4dGVybmFsLXZhbGlkaXR5IGNsYWltLiJ9"
    )
)


class CreateDefenseEvaluationRequest(ExternalContract):
    """Select exactly one immutable synthetic corpus and frozen defender."""

    corpus_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    defender_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("corpus_artifact_digest", "defender_artifact_digest", mode="before")
    @classmethod
    def digests_are_exact_strings(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("artifact digests must be exact strings")
        return value


@router.get("/v2/scorecard", response_model=DefenseV2Scorecard)
@public_router.get("/v2/scorecard", response_model=DefenseV2Scorecard)
def get_v2_scorecard() -> DefenseV2Scorecard:
    """Read the signed public v2 status; this route cannot start an evaluation."""
    return _PUBLIC_V2_SCORECARD


@router.post(
    "/evaluations",
    response_model=DefenseScorecard,
    status_code=201,
    responses={
        404: {"model": ErrorEnvelope, "description": "Input artifact not found"},
        409: {"model": ErrorEnvelope, "description": "Evaluation conflict"},
        422: {"model": ErrorEnvelope, "description": "Invalid artifact or request"},
        413: {"model": ErrorEnvelope, "description": "Request body too large"},
        503: {"model": ErrorEnvelope, "description": "Defense service unavailable"},
    },
)
def create_evaluation(
    request: CreateDefenseEvaluationRequest,
    service: Service,
) -> DefenseScorecard:
    """Execute an injected frozen evaluation and publish it atomically."""
    try:
        return service.create(
            corpus_artifact_digest=request.corpus_artifact_digest,
            defender_artifact_digest=request.defender_artifact_digest,
        )
    except DefenseResourceNotFound:
        raise ApiError(
            404,
            DEFENSE_EVALUATION_NOT_FOUND,
            "evaluation input artifact not found",
        ) from None
    except DefenseArtifactInvalid:
        raise ApiError(
            422,
            DEFENSE_ARTIFACT_INVALID,
            "evaluation artifact failed validation",
        ) from None
    except DefenseExecutionConflict:
        raise ApiError(
            409,
            DEFENSE_EVALUATION_CONFLICT,
            "defense evaluation could not be completed",
        ) from None


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=DefenseScorecard,
    responses={
        404: {"model": ErrorEnvelope, "description": "Evaluation not found"},
        422: {"model": ErrorEnvelope, "description": "Stored artifact invalid"},
    },
)
def get_evaluation(evaluation_id: str, service: Service) -> DefenseScorecard:
    """Return a scorecard only after fresh signed-artifact validation."""
    try:
        return service.get(evaluation_id)
    except DefenseResourceNotFound:
        raise ApiError(404, DEFENSE_EVALUATION_NOT_FOUND, "defense evaluation not found") from None
    except DefenseArtifactInvalid:
        raise ApiError(
            422,
            DEFENSE_ARTIFACT_INVALID,
            "published defense artifact failed validation",
        ) from None


@router.get(
    "/evaluations/{evaluation_id}/artifacts/{name}",
    responses={
        200: {"content": {"text/csv": {}, "text/markdown": {}, "application/json": {}}},
        404: {"model": ErrorEnvelope, "description": "Public artifact not found"},
        422: {"model": ErrorEnvelope, "description": "Stored artifact invalid"},
    },
)
def get_public_artifact(
    evaluation_id: str,
    name: str,
    request: Request,
    service: Service,
) -> Response:
    """Return only an exact allowlisted artifact name and its immutable bytes."""
    if not _raw_name_is_exact(request, name):
        _artifact_not_found()
    try:
        artifact = service.get_artifact(evaluation_id, name)
    except DefenseResourceNotFound:
        _artifact_not_found()
    except DefenseArtifactInvalid:
        raise ApiError(
            422,
            DEFENSE_ARTIFACT_INVALID,
            "published defense artifact failed validation",
        ) from None
    return Response(
        content=artifact.payload,
        media_type=artifact.reference.media_type,
        headers={
            "ETag": f'"{artifact.reference.sha256}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/evaluations/{evaluation_id}/artifacts/{name:path}", include_in_schema=False)
def reject_public_artifact_alias(
    evaluation_id: str,
    name: str,
) -> None:
    """Normalize all traversal and encoded-slash guesses to the same 404."""
    del evaluation_id, name
    _artifact_not_found()


def _raw_name_is_exact(request: Request, name: str) -> bool:
    raw_path = request.scope.get("raw_path")
    if type(raw_path) is not bytes or b"%" in raw_path or b"\\" in raw_path:
        return False
    try:
        encoded = name.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        return False
    return raw_path.rsplit(b"/", 1)[-1] == encoded


def _artifact_not_found() -> Never:
    raise ApiError(404, DEFENSE_ARTIFACT_NOT_FOUND, "public artifact not found")


__all__ = ["CreateDefenseEvaluationRequest", "public_router", "router"]
