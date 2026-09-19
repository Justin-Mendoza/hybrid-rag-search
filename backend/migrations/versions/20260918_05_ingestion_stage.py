"""Add durable ingestion stage progress.

Revision ID: 20260918_05
Revises: 20260915_04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260918_05"
down_revision: str | None = "20260915_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("stage", sa.Text(), server_default="parse", nullable=False),
    )
    op.execute("UPDATE ingestion_jobs SET stage = 'complete' WHERE status = 'succeeded'")
    op.create_check_constraint(
        "ck_ingestion_jobs_stage_allowed",
        "ingestion_jobs",
        "stage IN ('parse', 'chunk', 'embed', 'index', 'complete')",
    )
    op.create_check_constraint(
        "ck_ingestion_jobs_completion_matches_status",
        "ingestion_jobs",
        "(status = 'succeeded') = (stage = 'complete')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ingestion_jobs_completion_matches_status",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_constraint(
        "ck_ingestion_jobs_stage_allowed",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_column("ingestion_jobs", "stage")
