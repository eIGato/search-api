import uuid
from collections.abc import Collection
from dataclasses import asdict

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import domain, models


def to_domain(row: models.Client) -> domain.Client:
    return domain.Client(
        id=row.id,
        first_name=row.first_name,
        last_name=row.last_name,
        email=row.email,
        description=row.description,
        social_links=list(row.social_links),
        created_at=row.created_at,
    )


class ClientRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, new: domain.NewClient) -> domain.Client | None:
        """Insert a client; None if one with the same email (in any case) already exists.

        ON CONFLICT instead of catching the unique violation: concurrent inserts of one email
        resolve to exactly one row, and the transaction is not aborted by an error.
        """
        stmt = (
            insert(models.Client)
            .values(**asdict(new))
            .on_conflict_do_nothing(index_elements=[func.lower(models.Client.email)])
            .returning(models.Client)
        )
        row = await self._session.scalar(stmt)
        return None if row is None else to_domain(row)

    async def get(self, client_id: uuid.UUID) -> domain.Client | None:
        row = await self._session.get(models.Client, client_id)
        return None if row is None else to_domain(row)

    async def exists(self, client_id: uuid.UUID) -> bool:
        stmt = select(models.Client.id).where(models.Client.id == client_id)
        return await self._session.scalar(stmt) is not None

    async def lock_for_reference(self, client_id: uuid.UUID) -> bool:
        """Keep the client from being deleted until the transaction ends (FOR KEY SHARE).

        Taken before inserting rows that reference the client. Returns False if the client does
        not exist.
        """
        stmt = (
            select(models.Client.id)
            .where(models.Client.id == client_id)
            .with_for_update(read=True, key_share=True)
        )
        return await self._session.scalar(stmt) is not None

    async def get_many(self, client_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, domain.Client]:
        if not client_ids:
            return {}
        stmt = select(models.Client).where(models.Client.id.in_(client_ids))
        return {row.id: to_domain(row) for row in await self._session.scalars(stmt)}
