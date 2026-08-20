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
from apar.evaluation.v2_preregistration import V2Preregistration
from apar.evaluation.v2_reporting import (
    DefenseV2Scorecard,
    V2ReportingContractError,
    load_current_v2_scorecard,
)

DEFENSE_EVALUATION_NOT_FOUND: Final = "DEFENSE_EVALUATION_NOT_FOUND"
DEFENSE_ARTIFACT_NOT_FOUND: Final = "DEFENSE_ARTIFACT_NOT_FOUND"
DEFENSE_ARTIFACT_INVALID: Final = "DEFENSE_ARTIFACT_INVALID"
DEFENSE_EVALUATION_CONFLICT: Final = "DEFENSE_EVALUATION_CONFLICT"

router = APIRouter(prefix="/api/v1/defense")
public_router = APIRouter(prefix="/defense")
Service = Annotated[DefenseEvaluationService, Depends(get_defense_evaluation_service)]

_PUBLIC_V2_PREREGISTRATION = V2Preregistration.from_json(
    base64.b64decode(
        b"eyJib290c3RyYXBfbWFuaWZlc3Rfc2hhMjU2IjoiZDI1YjU2ZmZjZjc0YWVhYzgyZjhkZWZlMjBhNTI3OWMyZWVjYWM2ODllNGJiYmEwNTA2MWU0YmRhZmRhYWYzMiIsImJ1ZGdldF9tYW5pZmVzdF9zaGEyNTYiOiI4ZGUwYWI3MDJmNjQxMDQyNzUwMDBhODI3MWQzZjA0ZWMyNjBmMWFjZmMyYTBhNGM1YmVhYzAwMGQ3MzQyMDYzIiwiY2FuZGlkYXRlX2dyaWRfc2hhMjU2IjoiMWFkNmIwM2EyY2E2ZjBhN2ZmNjFhMWNiODIyNTQ2Mzc3NTY0MGJjMWJiOTY2ODFjNTM3NGFhZjRhM2M1NDg2NiIsImNvbnRyb2xzX21hbmlmZXN0X3NoYTI1NiI6IjkzZjc0NDFkM2RhZDY5YzNjYzg1ZTVlZjE0YWQ5OGRjMjc1ZTE0MTBjYzVhNzRjMGVkMWY0N2EwM2E1MGUxNjkiLCJldmFsdWF0b3JfY2FwYWJpbGl0eV9zaGEyNTYiOiJjMGJlYjliZjU3MWYwNjg4NDhhNWUwZDAyZWM3ZDY5YTdhM2QzN2M0NjllYTBkNDA0YjU3NDc3YzIyODQ0ODEyIiwiZXZhbHVhdG9yX2tleV9pZCI6ImNkOWI4NzVkNGViOGNlNDc0NWEwNDk1YmNlZDFkYTk3NWEwZmVjODE3NTQwMjQyZmY5M2IwNGNiYmY4MDVjYTAiLCJldmFsdWF0b3JfcHVibGljX2tleV9iYXNlNjQiOiI3cEl5aDlSVlgwRzhHWXVXNWVTYmdYc21mREVsN29rVXJ5OGpyUTlycE9zPSIsImV4ZWN1dGlvbl9ub25jZSI6IjRiNGE5MjQwNWE4Yzg0YWU1MDM1YmNiYzUxMGUwNmUxNzI5MjM4ZmZlMGU3OGYxMDY1MTVkNzhkM2M2M2M5OGEiLCJmZWF0dXJlX21hbmlmZXN0X3NoYTI1NiI6IjAyNGY5ZjFiZjdjMjkxM2NiZmE4NjY5MTMyYjYwNTRjODZmOTQ0NTkwOTg1NGVlM2RjYjhiYTM0MzAxZGY4MDIiLCJmaWRlbGl0eV92YWxpZGF0aW9uX2J1bmRsZV9zaGEyNTYiOiI1N2VlNWYxZmUxNzNmYWUxOGFhNzI5ZGFhMjk1NmQyMDcxMzAzNzA3MmI3NWM2M2Y5MzAxOWI1N2MxMGI4YWI2IiwibWFuaWZlc3RfcmVnaXN0cnlfc2hhMjU2IjoiZDUxNDQ0ZmJkOWNiNmMyNDliMjJkYzBhN2I4YjZiMTM4MDYyYTMzMWE1M2NjYzMzMzRkYTlhZTUyY2NlOGVjYSIsIm1heGltdW1fY29uZmlybWF0b3J5X2F0dGVtcHRzIjoxLCJtZXRyaWNzX21hbmlmZXN0X3NoYTI1NiI6IjdiMzM3YWFmNTUwYzc0NGUzYjFlYjQwMmM1MTE5N2Q4MmMxNWE2MTY0ZjBmNjU3MTFlMzEwMDUwZjViZmFmYTYiLCJwb3B1bGF0aW9uX21hbmlmZXN0X3NoYTI1NiI6ImYyOWM4NmY5MzIzMGEyZDUwZTgwMGQxNTUxNzQ4MjM4MjhjNTE3ZTBkMDBhZWU2MTJlM2FhNmY3M2ZlMzM0ZjciLCJwcmVyZWdpc3RyYXRpb25faWQiOiJhcGFyLWRlZmVuZC12MiIsInByb3RvY29sX3Byb2ZpbGVfc2hhMjU2IjoiZGU5MWJiYmUzZjJhODM3ZGE1MTQ1ZmYyYTdmYTc2N2ZkMDIxZjJhZGU2ZWYzNjU1ZWMxYWQ0ZTUwM2M2ZTQ2YyIsInJlcG9ydGluZ19zY2hlbWFfc2hhMjU2IjoiZTEyYWZmNmY2ZGU5MWE2NDU2ZTk1MTU3YjczOWVhYTNjM2NkMWQ4YzM4ZTdlMWJiZjk4MzY3YWYzYjNhMjllZiIsInNjaGVtYV92ZXJzaW9uIjoiMS4wLjAiLCJzZWVkX2NvbW1pdG1lbnRzIjpbeyJjb21taXRtZW50X3NoYTI1NiI6IjExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTEiLCJuYW1lIjoib3BlcmF0aW5nX3BvcHVsYXRpb24ifSx7ImNvbW1pdG1lbnRfc2hhMjU2IjoiMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMiIsIm5hbWUiOiJjYW1wYWlnbl9pbmplY3Rpb24ifV0sInNpZ25hdHVyZV9iYXNlNjQiOiJxQitqeFBHSFZFaHZSU09XbTNLWFlvQ1p1RlJranYzYmFYcjVNUVR1d2l1SVdHTm5CWG5KOHBhRlZObFAvWnZ1eUxlc0tha0swYVJwZVNJOFZPMW1EZz09Iiwic291cmNlX21hbmlmZXN0X3NoYTI1NiI6IjUwYmVkODA4MjMwMTFkNGU1NjNhYzNjMDBjNDRmMzJkYWJmZTFiYmQwYTllZjdhYzNiYWJjZDA5N2Y5OGQyMDIiLCJzeW50aGV0aWNfc2NvcGUiOiJTeW50aGV0aWMtb25seSBldmFsdWF0aW9uOyBub3QgYSByZWFsLXdvcmxkIHByZXZhbGVuY2Ugb3IgZXh0ZXJuYWwtdmFsaWRpdHkgY2xhaW0uIiwic3ludGhldGljX3Njb3BlX3NoYTI1NiI6IjlkMTViZWIxMDk5OGRjYWU3YjViYTc3NjVjMmM3YTkxMWRlNTdjYTA3NjBjYmE5OGQxNTNiYTVhMmIwODM0NDcifQ=="
    )
)
_PUBLIC_V2_SCORECARD = DefenseV2Scorecard.from_json(
    base64.b64decode(
        b"eyJhcm1zIjpbeyJhcm0iOiJydWxlc19vbmx5IiwiZ2F0ZSI6eyJhcm0iOiJydWxlc19vbmx5Iiwib3V0Y29tZSI6eyJjb2RlcyI6WyJOT1RfRVhFQ1VURUQiXSwicGFzc2VkIjpmYWxzZX19LCJzdGF0dXMiOiJub3RfZXhlY3V0ZWQifSx7ImFybSI6ImdiZHRfb25seSIsImdhdGUiOnsiYXJtIjoiZ2JkdF9vbmx5Iiwib3V0Y29tZSI6eyJjb2RlcyI6WyJOT1RfRVhFQ1VURUQiXSwicGFzc2VkIjpmYWxzZX19LCJzdGF0dXMiOiJub3RfZXhlY3V0ZWQifSx7ImFybSI6ImxheWVyZWRfaHlicmlkIiwiZ2F0ZSI6eyJhcm0iOiJsYXllcmVkX2h5YnJpZCIsIm91dGNvbWUiOnsiY29kZXMiOlsiTk9UX0VYRUNVVEVEIl0sInBhc3NlZCI6ZmFsc2V9fSwic3RhdHVzIjoibm90X2V4ZWN1dGVkIn1dLCJleGVjdXRpb25fbm9uY2UiOiI0YjRhOTI0MDVhOGM4NGFlNTAzNWJjYmM1MTBlMDZlMTcyOTIzOGZmZTBlNzhmMTA2NTE1ZDc4ZDNjNjNjOThhIiwicHJlcmVnaXN0cmF0aW9uX2lkIjoiYXBhci1kZWZlbmQtdjIiLCJwcm90b2NvbF9kaWdlc3QiOiJkZTkxYmJiZTNmMmE4MzdkYTUxNDVmZjJhN2ZhNzY3ZmQwMjFmMmFkZTZlZjM2NTVlYzFhZDRlNTAzYzZlNDZjIiwicHVibGljX2tleV9iYXNlNjQiOiI3cEl5aDlSVlgwRzhHWXVXNWVTYmdYc21mREVsN29rVXJ5OGpyUTlycE9zPSIsInNjaGVtYV92ZXJzaW9uIjoiMi4wLjAiLCJzaWduYXR1cmVfYmFzZTY0IjoiZ2ZnVkVGN05jaXB3ZWo0c1YwZldzSWo2bWVhcThMdFZOQ2tnc1pvL09DaXY5MUxVOGpzdmVVNUhjcGRpNVRKdHpVRHlaODZqRHo4bE1YSXdKdXBhQmc9PSIsInNpZ25lcl9rZXlfaWQiOiJjZDliODc1ZDRlYjhjZTQ3NDVhMDQ5NWJjZWQxZGE5NzVhMGZlYzgxNzU0MDI0MmZmOTNiMDRjYmJmODA1Y2EwIiwic3RhdHVzIjoibm90X2V4ZWN1dGVkIiwic3ludGhldGljX3Njb3BlIjoiU3ludGhldGljLW9ubHkgZXZhbHVhdGlvbjsgbm90IGEgcmVhbC13b3JsZCBwcmV2YWxlbmNlIG9yIGV4dGVybmFsLXZhbGlkaXR5IGNsYWltLiJ9"
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
def get_v2_scorecard(request: Request) -> DefenseV2Scorecard:
    """Read the signed public v2 status; this route cannot start an evaluation."""
    try:
        preregistration = request.app.state.v2_preregistration or _PUBLIC_V2_PREREGISTRATION
        fallback = request.app.state.v2_scorecard or _PUBLIC_V2_SCORECARD
        return load_current_v2_scorecard(
            request.app.state.settings.root,
            fallback=fallback,
            preregistration=preregistration,
        )
    except V2ReportingContractError:
        raise ApiError(
            422,
            DEFENSE_ARTIFACT_INVALID,
            "published defense artifact failed validation",
        ) from None


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
