"""Composition root: builds the application's resources and services once, at startup."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from search_api.config import Settings
from search_api.db import Database
from search_api.embeddings import Embedder, FastEmbedEmbedder
from search_api.llm import LanguageModel, LLMClient
from search_api.models import EMBEDDING_DIM
from search_api.services import ClientService, DocumentService, SearchService, SummaryService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Container:
    """Everything a running application uses, wired explicitly. Immutable once built."""

    settings: Settings
    database: Database
    clients: ClientService
    documents: DocumentService
    summaries: SummaryService
    search: SearchService


@asynccontextmanager
async def create_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LanguageModel | None = None,
) -> AsyncIterator[Container]:
    """Build the container and, on exit, close the resources it created.

    `embedder` and `llm` replace the ones built from the settings (tests use this). Resources
    passed in stay owned by the caller and are not closed here.
    """
    async with AsyncExitStack() as owned:
        database = Database(settings)
        owned.push_async_callback(database.dispose)

        if embedder is None:
            embedder = FastEmbedEmbedder(
                settings.embedding_model, settings.embedding_cache_dir, settings.embedding_threads
            )
        if embedder.dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding model {settings.embedding_model} produces {embedder.dim}-d vectors "
                f"but the database stores {EMBEDDING_DIM}-d vectors; add a migration."
            )

        if llm is None and settings.anthropic_api_key is not None:
            client = LLMClient(
                settings.anthropic_api_key.get_secret_value(),
                settings.llm_model,
                settings.llm_timeout_seconds,
            )
            owned.push_async_callback(client.close)
            llm = client
        if llm is not None:
            logger.info("LLM features enabled with model %s", llm.model)
        else:
            logger.info("ANTHROPIC_API_KEY not set: extractive summaries, thesaurus-only expansion")

        yield Container(
            settings=settings,
            database=database,
            clients=ClientService(database),
            documents=DocumentService(database, embedder, settings),
            summaries=SummaryService(database, embedder, llm),
            search=SearchService(database, embedder, llm, settings),
        )
