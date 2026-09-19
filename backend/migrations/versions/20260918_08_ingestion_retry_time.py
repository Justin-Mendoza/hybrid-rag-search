"""Add durable ingestion retry eligibility.

Revision ID: 20260918_08
Revises: 20260918_07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260918_08"
down_revision: str | None = "20260918_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_ingestion_jobs_retry_time_requires_queued",
        "ingestion_jobs",
        "next_attempt_at IS NULL OR status = 'queued'",
    )
    op.create_index(
        "ix_ingestion_jobs_status_retry",
        "ingestion_jobs",
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_status_retry", table_name="ingestion_jobs")
    op.drop_constraint(
        "ck_ingestion_jobs_retry_time_requires_queued",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_column("ingestion_jobs", "next_attempt_at")
