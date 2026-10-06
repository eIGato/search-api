from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Computed, DateTime, ForeignKey, Index, Integer, Text, func, text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Must match the migration; see Settings.embedding_dim.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name: Mapped[str] = mapped_column(Text)
    last_name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    social_links: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Search helpers maintained by Postgres (see the initial migration for the expressions).
    # "John", "Doe", "john.doe@neviswealth.com" -> "john doe john.doe@neviswealth.com"
    name_email_text: Mapped[str] = mapped_column(
        Text, Computed("lower(first_name || ' ' || last_name || ' ' || email)", persisted=True)
    )
    # Alphanumerics only, so "NevisWealth", "nevis wealth" and "nevis-wealth" all match
    # "johndoe|johndoeneviswealthcom". The "|" keeps matches from spanning name and email.
    compact_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            "regexp_replace(lower(first_name || last_name), '[^a-z0-9]', '', 'g') || '|' || "
            "regexp_replace(lower(email), '[^a-z0-9]', '', 'g')",
            persisted=True,
        ),
    )
    description_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(description, ''))", persisted=True),
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="client", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("uq_clients_email_lower", func.lower(email), unique=True),
        Index(
            "ix_clients_name_email_trgm",
            "name_email_text",
            postgresql_using="gin",
            postgresql_ops={"name_email_text": "gin_trgm_ops"},
        ),
        Index(
            "ix_clients_compact_trgm",
            "compact_text",
            postgresql_using="gin",
            postgresql_ops={"compact_text": "gin_trgm_ops"},
        ),
        Index("ix_clients_description_tsv", "description_tsv", postgresql_using="gin"),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Lazily generated and cached on first request.
    summary: Mapped[str | None] = mapped_column(Text)
    summary_method: Mapped[str | None] = mapped_column(Text)
    # Lease taken by the request generating the LLM summary, so that concurrent requests (in any
    # worker) wait for its result instead of calling the LLM again. Expires if that request dies.
    summary_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', title), 'A') || "
            "setweight(to_tsvector('english', content), 'B')",
            persisted=True,
        ),
    )

    client: Mapped[Client] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_documents_search_tsv", "search_tsv", postgresql_using="gin"),)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class QueryExpansion(Base):
    """LLM query expansions, cached per normalized query and model (shared by all workers)."""

    __tablename__ = "query_expansions"

    query_key: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    phrases: Mapped[list[str]] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
