"""Summary lease token: only the lease holder may store its summary and release the lease.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("summary_lease_token", sa.UUID(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "summary_lease_token")
