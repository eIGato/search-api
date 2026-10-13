"""FastAPI dependencies: the only place where the HTTP layer reaches the application's state.

The container is created by the lifespan in main.create_app and handed to requests through the
lifespan state, so routes depend on typed services instead of reading attributes off the app.
"""

import uuid
from typing import Annotated, cast

from fastapi import Depends, Request

from search_api.container import Container
from search_api.errors import NotFoundError
from search_api.services import ClientService, DocumentService, SearchService, SummaryService


def get_container(request: Request) -> Container:
    return cast(Container, request.state.container)


AppContainer = Annotated[Container, Depends(get_container)]


def get_client_service(container: AppContainer) -> ClientService:
    return container.clients


def get_document_service(container: AppContainer) -> DocumentService:
    return container.documents


def get_summary_service(container: AppContainer) -> SummaryService:
    return container.summaries


def get_search_service(container: AppContainer) -> SearchService:
    return container.search


Clients = Annotated[ClientService, Depends(get_client_service)]
Documents = Annotated[DocumentService, Depends(get_document_service)]
Summaries = Annotated[SummaryService, Depends(get_summary_service)]
Search = Annotated[SearchService, Depends(get_search_service)]


def parse_id(value: str, entity: str) -> uuid.UUID:
    # IDs are opaque strings in the API contract: a malformed ID is just an ID that does not
    # exist, so it is a 404 rather than a validation error.
    try:
        return uuid.UUID(value)
    except ValueError:
        raise NotFoundError(entity) from None
