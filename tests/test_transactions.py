"""Transaction boundaries and consistency across phases separated by slow external work."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta

import asyncpg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from search_api.config import Settings
from search_api.db import Database
from search_api.domain import NewClient
from search_api.main import create_app
from tests.conftest import HashingEmbedder, create_client, create_document


def run_with_database[T](settings: Settings, body: Callable[[Database], Awaitable[T]]) -> T:
    async def run() -> T:
        database = Database(settings)
        try:
            return await body(database)
        finally:
            await database.dispose()

    return asyncio.run(run())


def new_client(email: str) -> NewClient:
    return NewClient(first_name="Ann", last_name="Lee", email=email)


def test_read_only_transaction_reads_one_snapshot(client: TestClient, settings: Settings) -> None:
    async def body(database: Database) -> None:
        async with database.transaction(read_only=True) as reader:
            assert await reader.clients.get_many([uuid.uuid4()]) == {}  # takes the snapshot
            async with database.transaction() as writer:
                added = await writer.clients.add(new_client("ann@example.org"))
            assert added is not None
            # Committed meanwhile, but not part of the reader's snapshot.
            assert not await reader.clients.exists(added.id)
        async with database.transaction(read_only=True) as reader:
            assert await reader.clients.exists(added.id)

    run_with_database(settings, body)


def test_read_only_transaction_rejects_writes(client: TestClient, settings: Settings) -> None:
    async def body(database: Database) -> None:
        with pytest.raises(DBAPIError, match="read-only transaction"):
            async with database.transaction(read_only=True) as uow:
                await uow.clients.add(new_client("ann@example.org"))

    run_with_database(settings, body)


def test_transaction_rolls_back_on_error(client: TestClient, settings: Settings) -> None:
    async def body(database: Database) -> None:
        added = None
        with pytest.raises(RuntimeError):
            async with database.transaction() as uow:
                added = await uow.clients.add(new_client("ann@example.org"))
                raise RuntimeError("fails after the insert")
        assert added is not None
        async with database.transaction(read_only=True) as uow:
            assert not await uow.clients.exists(added.id)

    run_with_database(settings, body)


class DeletingEmbedder(HashingEmbedder):
    """Deletes a client while a document of theirs is being embedded."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url.replace("postgresql+asyncpg", "postgresql")
        self.client_id: str | None = None

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if self.client_id is not None:
            asyncio.run(self._delete(self.client_id))  # runs in a worker thread
        return super().embed_passages(texts)

    async def _delete(self, client_id: str) -> None:
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute("DELETE FROM clients WHERE id = $1", uuid.UUID(client_id))
        finally:
            await conn.close()


def test_client_deleted_during_embedding_is_a_404_without_orphans(
    client: TestClient, settings: Settings, database_url: str
) -> None:
    embedder = DeletingEmbedder(database_url)
    with TestClient(create_app(settings, embedder=embedder)) as api:
        owner = create_client(api)
        embedder.client_id = owner["id"]
        response = api.post(
            f"/clients/{owner['id']}/documents", json={"title": "Memo", "content": "Notes."}
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Client not found"}

    async def count_documents() -> int:
        conn = await asyncpg.connect(embedder.database_url)
        try:
            count: int = await conn.fetchval(
                "SELECT (SELECT count(*) FROM documents) + (SELECT count(*) FROM document_chunks)"
            )
            return count
        finally:
            await conn.close()

    assert asyncio.run(count_documents()) == 0


def test_expired_summary_lease_cannot_release_a_newer_one(
    client: TestClient, settings: Settings
) -> None:
    owner = create_client(client)
    document_id = uuid.UUID(create_document(client, owner["id"], "Memo", "Notes.")["id"])
    # The duration passed to a claim decides whether an existing lease has expired.
    instant, minute = timedelta(0), timedelta(minutes=1)

    async def body(database: Database) -> None:
        async with database.transaction() as uow:
            stale = await uow.documents.claim_summary_lease(document_id, minute)
        async with database.transaction() as uow:
            assert await uow.documents.claim_summary_lease(document_id, minute) is None  # live
            current = await uow.documents.claim_summary_lease(document_id, instant)  # expired
        assert stale is not None and current is not None

        async with database.transaction() as uow:
            assert not await uow.documents.release_summary_lease(document_id, stale, "Stale.")
            # The current lease is still held: nobody else can claim it.
            assert await uow.documents.claim_summary_lease(document_id, minute) is None
            assert await uow.documents.get_summary(document_id) is None

        async with database.transaction() as uow:
            assert await uow.documents.release_summary_lease(document_id, current, "Fresh.")
            summary = await uow.documents.get_summary(document_id)
        assert summary is not None and (summary.text, summary.method) == ("Fresh.", "llm")

    run_with_database(settings, body)


def test_extractive_summary_does_not_overwrite_a_stored_one(
    client: TestClient, settings: Settings
) -> None:
    owner = create_client(client)
    document_id = uuid.UUID(create_document(client, owner["id"], "Memo", "Notes.")["id"])

    async def body(database: Database) -> None:
        async with database.transaction() as uow:
            token = await uow.documents.claim_summary_lease(document_id, timedelta(minutes=1))
            assert token is not None
            await uow.documents.release_summary_lease(document_id, token, "From the LLM.")
        async with database.transaction() as uow:
            stored = await uow.documents.store_summary_if_absent(
                document_id, "Extracted.", "extractive"
            )
        assert stored is not None and (stored.text, stored.method) == ("From the LLM.", "llm")

    run_with_database(settings, body)
