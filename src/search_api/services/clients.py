import uuid

from search_api.db import Database
from search_api.domain import Client, NewClient
from search_api.errors import ConflictError, NotFoundError


class ClientService:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, new: NewClient) -> Client:
        async with self._db.transaction() as uow:
            client = await uow.clients.add(new)
        if client is None:
            raise ConflictError("A client with this email already exists")
        return client

    async def get(self, client_id: uuid.UUID) -> Client:
        async with self._db.transaction(read_only=True) as uow:
            client = await uow.clients.get(client_id)
        if client is None:
            raise NotFoundError("Client")
        return client
