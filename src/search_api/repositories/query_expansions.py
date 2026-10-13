from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import models


class QueryExpansionRepository:
    """LLM query expansions, cached per normalized query and model (shared by all workers)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, query_key: str, model: str) -> list[str] | None:
        qe = models.QueryExpansion
        phrases = await self._session.scalar(
            select(qe.phrases).where(qe.query_key == query_key, qe.model == model)
        )
        return None if phrases is None else list(phrases)

    async def add(self, query_key: str, model: str, phrases: Sequence[str]) -> None:
        # Concurrent requests may compute the same entry; the first one stored wins.
        await self._session.execute(
            insert(models.QueryExpansion)
            .values(query_key=query_key, model=model, phrases=list(phrases))
            .on_conflict_do_nothing()
        )
