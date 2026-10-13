from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from search_api.config import Settings
from search_api.repositories import UnitOfWork


class Database:
    """Owns the connection pool and defines transaction boundaries.

    All database access goes through `transaction()`: one transaction per block, committed when
    the block exits normally, rolled back when it raises, and its connection returned to the
    pool right after. Services open one block per phase of their work and never keep one open
    across slow work that does not need the database (embedding, LLM calls), so a small pool
    serves many concurrent requests.
    """

    def __init__(self, settings: Settings) -> None:
        self._engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
        )
        self._sessionmaker = async_sessionmaker(self._engine)

    @asynccontextmanager
    async def transaction(self, *, read_only: bool = False) -> AsyncIterator[UnitOfWork]:
        """One transaction and the repositories that work in it.

        Read-only transactions run at REPEATABLE READ, so all their statements see the same
        snapshot: a search ranks and loads its results from one consistent state of the data.
        Read-only transactions at this level never fail with serialization errors. Write
        transactions run at READ COMMITTED and lock what they depend on explicitly.
        """
        async with self._sessionmaker.begin() as session:
            if read_only:
                await session.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                )
            yield UnitOfWork(session)

    async def ping(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def dispose(self) -> None:
        await self._engine.dispose()
