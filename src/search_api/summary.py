import asyncio
import time
import uuid
from datetime import timedelta
from typing import Literal, cast

import numpy as np
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import models
from search_api.chunking import split_sentences
from search_api.db import release_connection
from search_api.embeddings import Embedder, embed_passages
from search_api.llm import LLMClient

# Bound the work for very long documents; the opening of a document is usually the most
# informative part anyway.
MAX_SENTENCES_CONSIDERED = 200


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


SummaryMethod = Literal["llm", "extractive"]
POLL_INTERVAL_SECONDS = 0.25
LEASE_MARGIN_SECONDS = 5.0


async def get_summary(
    session: AsyncSession,
    document: models.Document,
    embedder: Embedder,
    llm: LLMClient | None,
) -> tuple[str, SummaryMethod]:
    """Return the document's summary, generating and caching it on first use.

    - No LLM configured: an extractive summary, cached.
    - LLM configured: exactly one request (across all workers) calls the LLM; concurrent
      requests for the same document wait for its result. If the LLM fails, the caller gets an
      extractive summary that is not cached, so a later request retries the LLM.
    No database connection is held while the summary is being generated.
    """
    if document.summary is not None and (document.summary_method == "llm" or llm is None):
        return document.summary, cast(SummaryMethod, document.summary_method)
    await release_connection(session)

    if llm is None:
        summary = await extractive_summary(embedder, document.content)
        await session.execute(
            update(models.Document)
            .where(models.Document.id == document.id)
            .values(summary=summary, summary_method="extractive")
        )
        await session.commit()
        return summary, "extractive"

    lease = timedelta(seconds=llm.max_call_seconds + LEASE_MARGIN_SECONDS)
    deadline = time.monotonic() + lease.total_seconds() + LEASE_MARGIN_SECONDS
    while time.monotonic() < deadline:
        if await _claim(session, document.id, lease):
            return await _generate(session, document, embedder, llm)
        # Someone else is generating it: wait for their result.
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        row = (
            await session.execute(
                select(models.Document.summary, models.Document.summary_method).where(
                    models.Document.id == document.id
                )
            )
        ).one()
        await session.commit()
        if row.summary_method == "llm":
            return row.summary, "llm"

    return await extractive_summary(embedder, document.content), "extractive"


async def _claim(session: AsyncSession, document_id: uuid.UUID, lease: timedelta) -> bool:
    """Atomically take the generation lease unless an LLM summary exists or a lease is live."""
    doc = models.Document
    claimed = await session.scalar(
        update(doc)
        .where(
            doc.id == document_id,
            or_(doc.summary_method.is_(None), doc.summary_method != "llm"),
            or_(doc.summary_claimed_at.is_(None), doc.summary_claimed_at < func.now() - lease),
        )
        .values(summary_claimed_at=func.now())
        .returning(doc.id)
    )
    await session.commit()
    return claimed is not None


async def _generate(
    session: AsyncSession, document: models.Document, embedder: Embedder, llm: LLMClient
) -> tuple[str, SummaryMethod]:
    summary = await llm.summarize(document.title, document.content)
    values: dict[str, object] = {"summary_claimed_at": None}  # release the lease either way
    if summary is not None:
        values |= {"summary": summary, "summary_method": "llm"}
    await session.execute(
        update(models.Document).where(models.Document.id == document.id).values(**values)
    )
    await session.commit()
    if summary is not None:
        return summary, "llm"
    return await extractive_summary(embedder, document.content), "extractive"
