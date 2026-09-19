"""Add expiring ingestion worker ownership.

Revision ID: 20260918_07
Revises: 20260918_06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260918_07"
down_revision: str | None = "20260918_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    # A pre-migration running row has no verifiable live owner. Make it
    # recoverable rather than inventing a lease for an unknown worker.
    op.execute("UPDATE ingestion_jobs SET status = 'queued' WHERE status = 'running'")
    op.create_check_constraint(
        "ck_ingestion_jobs_lease_matches_running",
        "ingestion_jobs",
        "(status = 'running') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
    )
    op.create_index(
        "ix_ingestion_jobs_status_lease",
        "ingestion_jobs",
        ["status", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_status_lease", table_name="ingestion_jobs")
    op.drop_constraint(
        "ck_ingestion_jobs_lease_matches_running",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_column("ingestion_jobs", "lease_expires_at")
    op.drop_column("ingestion_jobs", "lease_token")
