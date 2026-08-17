"""Application factory for APAR's local API boundary."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from apar import __version__
from apar.config import Settings
from apar.registry.repository import ThreatRepository
from apar.runs import RunRunner, RunSigningIdentity
from apar.storage.artifacts import ArtifactStore

RESOURCE_NOT_FOUND: Final = "RESOURCE_NOT_FOUND"
VALIDATION_FAILED: Final = "VALIDATION_FAILED"


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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.repository = ThreatRepository(app.state.settings.database_path)
    app.state.artifact_store = ArtifactStore(app.state.settings.artifact_root)
    signer = RunSigningIdentity.load_or_create(
        app.state.settings.root / ".apar" / "run-signing-key.ed25519"
    )
    app.state.run_runner = RunRunner(
        artifact_store=app.state.artifact_store,
        signer=signer,
        run_index_root=app.state.settings.root / ".apar" / "runs",
    )
    yield


def create_app(settings: Settings) -> FastAPI:
    """Build an unbound local API application for the supplied settings."""
    from apar.api.routes.health import router as health_router
    from apar.api.routes.registry import router as registry_router
    from apar.api.routes.runs import router as runs_router
    from apar.api.routes.scenarios import router as scenarios_router

    app = FastAPI(
        title="APAR API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.include_router(health_router)
    app.include_router(registry_router)
    app.include_router(scenarios_router)
    app.include_router(runs_router)
    return app
