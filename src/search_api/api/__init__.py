"""HTTP layer: request parsing and validation, response mapping, authentication. Business logic
lives in the services; routes call one service method each and map domain objects and errors to
the API's schemas and status codes."""

from fastapi import APIRouter, Depends, FastAPI

from search_api.api import clients, documents, health, search
from search_api.api.auth import require_api_key
from search_api.api.errors import UNAUTHORIZED, register_error_handlers


def include_api(app: FastAPI) -> None:
    protected = APIRouter(dependencies=[Depends(require_api_key)], responses=UNAUTHORIZED)
    protected.include_router(clients.router)
    protected.include_router(documents.router)
    protected.include_router(search.router)
    app.include_router(protected)
    app.include_router(health.router)  # open for probes
    register_error_handlers(app)
