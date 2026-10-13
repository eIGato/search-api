from fastapi import APIRouter, Request, Response, status

from search_api import schemas
from search_api.api.dependencies import Clients, parse_id
from search_api.api.errors import CONFLICT, NOT_FOUND
from search_api.domain import NewClient

router = APIRouter(tags=["clients"])


@router.post(
    "/clients",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Client,
    responses=CONFLICT,
    summary="Create a client",
)
async def create_client(
    payload: schemas.ClientCreate, clients: Clients, request: Request, response: Response
) -> schemas.Client:
    client = await clients.create(NewClient(**payload.model_dump()))
    response.headers["Location"] = str(request.url_for("get_client", client_id=str(client.id)))
    return schemas.Client.model_validate(client)


@router.get(
    "/clients/{client_id}",
    response_model=schemas.Client,
    responses=NOT_FOUND,
    summary="Get a client",
)
async def get_client(client_id: str, clients: Clients) -> schemas.Client:
    client = await clients.get(parse_id(client_id, "Client"))
    return schemas.Client.model_validate(client)
