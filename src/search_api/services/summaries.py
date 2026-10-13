import asyncio
import time
import uuid
from datetime import timedelta

import numpy as np

from search_api.chunking import split_sentences
from search_api.db import Database
from search_api.domain import Document, Summary
from search_api.embeddings import Embedder, embed_passages
from search_api.errors import NotFoundError
from search_api.llm import LanguageModel

# Bound the work for very long documents; the opening of a document is usually the most
# informative part anyway.
MAX_SENTENCES_CONSIDERED = 200
POLL_INTERVAL_SECONDS = 0.25
LEASE_MARGIN_SECONDS = 5.0


async def extractive_summary(embedder: Embedder, content: str, max_sentences: int = 3) -> str:
    """Pick the sentences closest to the document's centroid embedding, in original order.

    A cheap, deterministic fallback for when no LLM is configured: it never invents facts,
    but it can only quote, not paraphrase.
    """
    sentences = split_sentences(content)[:MAX_SENTENCES_CONSIDERED]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectors = np.asarray(await embed_passages(embedder, sentences), dtype=np.float32)
    centroid = vectors.mean(axis=0)
    scores = vectors @ centroid
    chosen = sorted(np.argsort(-scores)[:max_sentences].tolist())
    return " ".join(sentences[i] for i in chosen)


class SummaryService:
    """Document summaries, generated on first request and cached in the documents table.

    - No LLM configured: an extractive summary, cached.
    - LLM configured: exactly one request (across all workers) calls the LLM; concurrent
      requests for the same document wait for its result. If the LLM fails, the caller gets an
      extractive summary that is not cached, so a later request retries the LLM.

    Documents are immutable, so a summary generated from content read in an earlier
    transaction is still the summary of the stored document.
    """

    def __init__(self, database: Database, embedder: Embedder, llm: LanguageModel | None) -> None:
        self._db = database
        self._embedder = embedder
        self._llm = llm

    async def get(self, client_id: uuid.UUID, document_id: uuid.UUID) -> Summary:
        async with self._db.transaction(read_only=True) as uow:
            document = await uow.documents.get(client_id, document_id)
            stored = None if document is None else await uow.documents.get_summary(document_id)
        if document is None:
            raise NotFoundError("Document")

        if stored is not None and (stored.method == "llm" or self._llm is None):
            return stored
        if self._llm is None:
            return await self._extractive(document)
        return await self._llm_summary(document, self._llm)

    async def _extractive(self, document: Document) -> Summary:
        text = await extractive_summary(self._embedder, document.content)
        async with self._db.transaction() as uow:
            stored = await uow.documents.store_summary_if_absent(document.id, text, "extractive")
        if stored is None:
            raise NotFoundError("Document")
        return stored

    async def _llm_summary(self, document: Document, llm: LanguageModel) -> Summary:
        lease = timedelta(seconds=llm.max_call_seconds + LEASE_MARGIN_SECONDS)
        deadline = time.monotonic() + lease.total_seconds() + LEASE_MARGIN_SECONDS
        while time.monotonic() < deadline:
            async with self._db.transaction() as uow:
                token = await uow.documents.claim_summary_lease(document.id, lease)
            if token is not None:
                return await self._generate(document, llm, token)

            # Someone else is generating it: wait for their result.
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            async with self._db.transaction(read_only=True) as uow:
                stored = await uow.documents.get_summary(document.id)
            if stored is not None and stored.method == "llm":
                return stored

        return await self._unstored_extractive(document)

    async def _generate(self, document: Document, llm: LanguageModel, token: uuid.UUID) -> Summary:
        text = await llm.summarize(document.title, document.content)
        async with self._db.transaction() as uow:
            # Releases the lease either way; stores the summary only if the LLM produced one.
            # If the lease expired meanwhile and another request took it, that request's result
            # is stored instead and this one is just returned.
            await uow.documents.release_summary_lease(document.id, token, text)
        if text is not None:
            return Summary(document.id, text, "llm")
        return await self._unstored_extractive(document)

    async def _unstored_extractive(self, document: Document) -> Summary:
        """Fallback when the LLM is configured but failed: not cached, so it is retried."""
        text = await extractive_summary(self._embedder, document.content)
        return Summary(document.id, text, "extractive")
