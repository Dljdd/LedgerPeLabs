"""Application factory for APAR's local API boundary."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from apar import __version__
from apar.config import Settings
from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.gates import EvaluatorReplayVerifier
from apar.evaluation.reporting import PublicArtifactVerifier
from apar.evaluation.service import (
    DefenseArtifactInvalid,
    DefenseEvaluationService,
    DefenseServiceUnavailable,
    EvaluationExecutor,
)
from apar.registry.repository import ThreatRepository
from apar.runs import RunRunner, RunSigningIdentity
from apar.storage.artifacts import ArtifactStore

RESOURCE_NOT_FOUND: Final = "RESOURCE_NOT_FOUND"
VALIDATION_FAILED: Final = "VALIDATION_FAILED"
DEFENSE_BODY_LIMIT: Final = 4096


class ErrorDetail(BaseModel):
    """Machine-readable information for every client-facing API error."""

    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """The stable shape used for every client-facing API error response."""

    detail: ErrorDetail


class ApiError(Exception):
    """A client-facing error with a stable machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorEnvelope(detail=ErrorDetail(code=code, message=message)).model_dump(),
    )


async def _api_error_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, ApiError):
        raise error
    return _error_response(error.status_code, error.code, error.message)


async def _validation_error_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, RequestValidationError):
        raise error
    return _error_response(422, VALIDATION_FAILED, "request validation failed")


async def _http_error_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, StarletteHTTPException):
        raise error
    if error.status_code == 404:
        return _error_response(404, RESOURCE_NOT_FOUND, "resource not found")
    message = error.detail if isinstance(error.detail, str) else "request failed"
    return _error_response(error.status_code, f"HTTP_{error.status_code}", message)


async def _defense_unavailable_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, DefenseServiceUnavailable):
        raise error
    return _error_response(
        503,
        "DEFENSE_SERVICE_UNAVAILABLE",
        "defense evaluation service unavailable",
    )


async def _defense_invalid_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, DefenseArtifactInvalid):
        raise error
    return _error_response(
        422,
        "DEFENSE_ARTIFACT_INVALID",
        "published defense artifact failed validation",
    )


class _DefenseBodyLimitMiddleware:
    """Reject oversized Defend POST bodies before JSON/Pydantic parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/v1/defense/evaluations"
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                length = int(raw_length.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                await self._reject(scope, receive, send)
                return
            if length < 0 or length > DEFENSE_BODY_LIMIT:
                await self._reject(scope, receive, send)
                return
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if type(chunk) is not bytes:
                await self._reject(scope, receive, send)
                return
            body.extend(chunk)
            if len(body) > DEFENSE_BODY_LIMIT:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = _error_response(
            413, "DEFENSE_REQUEST_TOO_LARGE", "defense request body too large"
        )
        await response(scope, receive, send)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    run_state_root = app.state.settings.root / ".apar" / "private-run-state"
    signer = RunSigningIdentity.load_or_create(run_state_root / "run-signing-key.ed25519")
    app.state.repository = ThreatRepository(app.state.settings.database_path)
    app.state.artifact_store = ArtifactStore(app.state.settings.artifact_root)
    app.state.run_runner = RunRunner(
        artifact_store=app.state.artifact_store,
        signer=signer,
        run_index_root=run_state_root / "runs",
    )
    if app.state.defense_configured:
        defender_verifier = DefenderBundleVerifier(
            app.state.artifact_store,
            signer_key_id=app.state.defender_signer_key_id,
            public_key_base64=app.state.defender_public_key_base64,
        )
        try:
            app.state.defense_service = DefenseEvaluationService(
                artifact_store=app.state.artifact_store,
                publication_signer=app.state.publication_signer,
                publication_verifier=app.state.publication_verifier,
                evaluator_verifier=app.state.evaluator_verifier,
                hidden_proof_verifier=app.state.hidden_proof_verifier,
                defender_verifier=defender_verifier,
                executor=app.state.defense_executor,
            )
        except DefenseArtifactInvalid:
            app.state.defense_service = None
            app.state.defense_startup_invalid = True
    else:
        app.state.defense_service = None
    yield


def create_app(
    settings: Settings,
    *,
    defense_executor: EvaluationExecutor | None = None,
    evaluator_verifier: EvaluatorReplayVerifier | None = None,
    hidden_proof_verifier: EvaluatorReplayVerifier | None = None,
    publication_signer: RunSigningIdentity | None = None,
    publication_verifier: PublicArtifactVerifier | None = None,
    defender_signer_key_id: str | None = None,
    defender_public_key_base64: str | None = None,
) -> FastAPI:
    """Build an unbound local API application for the supplied settings."""
    from apar.api.routes.defense import router as defense_router
    from apar.api.routes.health import router as health_router
    from apar.api.routes.registry import router as registry_router
    from apar.api.routes.runs import router as runs_router
    from apar.api.routes.scenarios import router as scenarios_router

    defense_dependencies = (
        defense_executor,
        evaluator_verifier,
        hidden_proof_verifier,
        publication_signer,
        publication_verifier,
        defender_signer_key_id,
        defender_public_key_base64,
    )
    configured = all(item is not None for item in defense_dependencies)
    if any(item is not None for item in defense_dependencies) and not configured:
        raise TypeError("all defense authorities and capabilities must be supplied together")
    if configured and (
        type(defense_executor) is not EvaluationExecutor
        or type(evaluator_verifier) is not EvaluatorReplayVerifier
        or type(hidden_proof_verifier) is not EvaluatorReplayVerifier
        or type(publication_signer) is not RunSigningIdentity
        or type(publication_verifier) is not PublicArtifactVerifier
        or type(defender_signer_key_id) is not str
        or type(defender_public_key_base64) is not str
    ):
        raise TypeError("defense authorities and capabilities must be exact")
    if configured:
        assert evaluator_verifier is not None
        assert hidden_proof_verifier is not None
        assert publication_verifier is not None
        if (
            len(
                {
                    evaluator_verifier.key_id,
                    hidden_proof_verifier.key_id,
                    publication_verifier.key_id,
                }
            )
            != 3
        ):
            raise TypeError("defense trust roots must be independent")
    app = FastAPI(
        title="APAR API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.defense_configured = configured
    app.state.defense_startup_invalid = False
    app.state.defense_executor = defense_executor
    app.state.evaluator_verifier = evaluator_verifier
    app.state.hidden_proof_verifier = hidden_proof_verifier
    app.state.publication_signer = publication_signer
    app.state.publication_verifier = publication_verifier
    app.state.defender_signer_key_id = defender_signer_key_id
    app.state.defender_public_key_base64 = defender_public_key_base64
    app.add_middleware(_DefenseBodyLimitMiddleware)
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(DefenseServiceUnavailable, _defense_unavailable_handler)
    app.add_exception_handler(DefenseArtifactInvalid, _defense_invalid_handler)
    app.include_router(health_router)
    app.include_router(defense_router)
    app.include_router(registry_router)
    app.include_router(scenarios_router)
    app.include_router(runs_router)
    return app
