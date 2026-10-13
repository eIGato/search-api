from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from search_api import schemas
from search_api.errors import ConflictError, DomainError, InvalidRequestError, NotFoundError

ResponsesDoc = dict[int | str, dict[str, Any]]
NOT_FOUND: ResponsesDoc = {status.HTTP_404_NOT_FOUND: {"model": schemas.ErrorResponse}}
CONFLICT: ResponsesDoc = {status.HTTP_409_CONFLICT: {"model": schemas.ErrorResponse}}
UNAUTHORIZED: ResponsesDoc = {status.HTTP_401_UNAUTHORIZED: {"model": schemas.ErrorResponse}}

_STATUS_CODES: dict[type[DomainError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    InvalidRequestError: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DomainError)
    return JSONResponse(
        status_code=_STATUS_CODES[type(exc)],
        content=schemas.ErrorResponse(detail=exc.message).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    for error in _STATUS_CODES:
        app.add_exception_handler(error, _domain_error_handler)
