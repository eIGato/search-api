"""Business logic. Services own transaction boundaries: each phase of their work runs in its own
db.Database.transaction(), and slow work that does not need the database (embedding, LLM calls)
runs between transactions, never inside one."""

from search_api.services.clients import ClientService
from search_api.services.documents import DocumentService
from search_api.services.search import SearchService
from search_api.services.summaries import SummaryService

__all__ = ["ClientService", "DocumentService", "SearchService", "SummaryService"]
