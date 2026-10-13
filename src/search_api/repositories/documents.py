import uuid
from collections.abc import Collection, Sequence
from datetime import timedelta
from typing import cast

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import domain, models


def to_domain(row: models.Document) -> domain.Document:
    return domain.Document(
        id=row.id,
        client_id=row.client_id,
        title=row.title,
        content=row.content,
        created_at=row.created_at,
    )


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        client_id: uuid.UUID,
        new: domain.NewDocument,
        chunks: Sequence[tuple[str, list[float]]],
    ) -> domain.Document:
        """Insert a document with its (text, embedding) chunks."""
        row = models.Document(client_id=client_id, title=new.title, content=new.content)
        row.chunks = [
            models.DocumentChunk(chunk_index=i, content=text, embedding=embedding)
            for i, (text, embedding) in enumerate(chunks)
        ]
        self._session.add(row)
        await self._session.flush()  # fetches server defaults (created_at) through RETURNING
        return to_domain(row)

    async def get(self, client_id: uuid.UUID, document_id: uuid.UUID) -> domain.Document | None:
        """The document, if it exists and belongs to the client."""
        stmt = select(models.Document).where(
            models.Document.id == document_id, models.Document.client_id == client_id
        )
        row = await self._session.scalar(stmt)
        return None if row is None else to_domain(row)

    async def list_for_client(self, client_id: uuid.UUID) -> list[domain.Document]:
        stmt = (
            select(models.Document)
            .where(models.Document.client_id == client_id)
            .order_by(models.Document.created_at.desc(), models.Document.id)
        )
        return [to_domain(row) for row in await self._session.scalars(stmt)]

    async def get_many(
        self, document_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, domain.Document]:
        if not document_ids:
            return {}
        stmt = select(models.Document).where(models.Document.id.in_(document_ids))
        return {row.id: to_domain(row) for row in await self._session.scalars(stmt)}

    # --- summaries ---------------------------------------------------------------------------

    async def get_summary(self, document_id: uuid.UUID) -> domain.Summary | None:
        doc = models.Document
        row = (
            await self._session.execute(
                select(doc.summary, doc.summary_method).where(doc.id == document_id)
            )
        ).one_or_none()
        if row is None or row.summary is None:
            return None
        return domain.Summary(
            document_id, row.summary, cast(domain.SummaryMethod, row.summary_method)
        )

    async def store_summary_if_absent(
        self, document_id: uuid.UUID, text: str, method: domain.SummaryMethod
    ) -> domain.Summary | None:
        """Store a summary unless one exists already; return the stored one (None: no document).

        Keeps a summary stored concurrently by another request, which may be a better (LLM)
        one, instead of overwriting it.
        """
        doc = models.Document
        stmt = (
            update(doc)
            .where(doc.id == document_id)
            .values(
                summary=func.coalesce(doc.summary, text),
                summary_method=func.coalesce(doc.summary_method, method),
            )
            .returning(doc.summary, doc.summary_method)
        )
        row = (await self._session.execute(stmt)).one_or_none()
        if row is None:
            return None
        return domain.Summary(
            document_id, row.summary, cast(domain.SummaryMethod, row.summary_method)
        )

    async def claim_summary_lease(
        self, document_id: uuid.UUID, duration: timedelta
    ) -> uuid.UUID | None:
        """Take the LLM summary generation lease; return its token, or None if not taken.

        Not taken when an LLM summary exists already or another request holds a live lease.
        """
        doc = models.Document
        token = uuid.uuid4()
        claimed = await self._session.scalar(
            update(doc)
            .where(
                doc.id == document_id,
                or_(doc.summary_method.is_(None), doc.summary_method != "llm"),
                or_(
                    doc.summary_claimed_at.is_(None),
                    doc.summary_claimed_at < func.now() - duration,
                ),
            )
            .values(summary_claimed_at=func.now(), summary_lease_token=token)
            .returning(doc.id)
        )
        return None if claimed is None else token

    async def release_summary_lease(
        self, document_id: uuid.UUID, token: uuid.UUID, llm_summary: str | None
    ) -> bool:
        """Release the lease, storing the LLM summary if there is one.

        Only while the lease is still the one identified by `token`: a request whose lease
        expired and was taken over must not release (or overwrite) the new holder's. Returns
        whether it was.
        """
        doc = models.Document
        values: dict[str, object] = {"summary_claimed_at": None, "summary_lease_token": None}
        if llm_summary is not None:
            values |= {"summary": llm_summary, "summary_method": "llm"}
        released = await self._session.scalar(
            update(doc)
            .where(doc.id == document_id, doc.summary_lease_token == token)
            .values(**values)
            .returning(doc.id)
        )
        return released is not None
