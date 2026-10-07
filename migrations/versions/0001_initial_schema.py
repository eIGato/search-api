"""Initial schema: clients, documents, document chunk embeddings, LLM expansion cache.

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "clients",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("first_name", sa.Text(), nullable=False),
        sa.Column("last_name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "social_links", sa.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "name_email_text",
            sa.Text(),
            sa.Computed("lower(first_name || ' ' || last_name || ' ' || email)", persisted=True),
            nullable=False,
        ),
        sa.Column(
            "compact_text",
            sa.Text(),
            sa.Computed(
                "regexp_replace(lower(first_name || last_name), '[^a-z0-9]', '', 'g') || '|' || "
                "regexp_replace(lower(email), '[^a-z0-9]', '', 'g')",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "description_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', coalesce(description, ''))", persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "uq_clients_email_lower", "clients", [sa.literal_column("lower(email)")], unique=True
    )
    op.create_index(
        "ix_clients_name_email_trgm",
        "clients",
        ["name_email_text"],
        postgresql_using="gin",
        postgresql_ops={"name_email_text": "gin_trgm_ops"},
    )
    op.create_index(
        "ix_clients_compact_trgm",
        "clients",
        ["compact_text"],
        postgresql_using="gin",
        postgresql_ops={"compact_text": "gin_trgm_ops"},
    )
    op.create_index(
        "ix_clients_description_tsv", "clients", ["description_tsv"], postgresql_using="gin"
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "client_id",
            sa.UUID(),
            sa.ForeignKey("clients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_method", sa.Text(), nullable=True),
        sa.Column("summary_claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "search_tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', title), 'A') || "
                "setweight(to_tsvector('english', content), 'B')",
                persisted=True,
            ),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_client_id", "documents", ["client_id"])
    op.create_index("ix_documents_search_tsv", "documents", ["search_tsv"], postgresql_using="gin")

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "query_expansions",
        sa.Column("query_key", sa.Text(), primary_key=True),
        sa.Column("model", sa.Text(), primary_key=True),
        sa.Column("phrases", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("query_expansions")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("clients")
