"""Data access. Repositories run SQL inside a transaction they are given and never commit:
transaction boundaries belong to the services (see db.Database.transaction)."""

from sqlalchemy.ext.asyncio import AsyncSession

from search_api.repositories.clients import ClientRepository
from search_api.repositories.documents import DocumentRepository
from search_api.repositories.query_expansions import QueryExpansionRepository
from search_api.repositories.search import SearchRepository


class UnitOfWork:
    """The repositories of one database transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.clients = ClientRepository(session)
        self.documents = DocumentRepository(session)
        self.query_expansions = QueryExpansionRepository(session)
        self.search = SearchRepository(session)


__all__ = [
    "ClientRepository",
    "DocumentRepository",
    "QueryExpansionRepository",
    "SearchRepository",
    "UnitOfWork",
]
