"""Snapshot ingestion content and pipeline identities.

Revision ID: 20260918_06
Revises: 20260918_05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260918_06"
down_revision: str | None = "20260918_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ingestion_jobs", sa.Column("content_hash", sa.Text(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("pipeline_version", sa.Text(), nullable=True))
    op.execute(
        "UPDATE ingestion_jobs AS job SET content_hash = document.content_hash "
        "FROM documents AS document WHERE document.id = job.document_id"
    )
    # Old jobs predate versioned pipelines. Unique legacy identities preserve
    # their history without treating that work as reusable by a current recipe.
    op.execute(
        "UPDATE ingestion_jobs SET pipeline_version = 'pipe_' || "
        "replace(id::text, '-', '') || replace(id::text, '-', '')"
    )
    op.alter_column("ingestion_jobs", "content_hash", nullable=False)
    op.alter_column("ingestion_jobs", "pipeline_version", nullable=False)
    op.create_check_constraint(
        "ck_ingestion_jobs_content_hash_sha256",
        "ingestion_jobs",
        "content_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_ingestion_jobs_pipeline_version_format",
        "ingestion_jobs",
        "pipeline_version ~ '^pipe_[0-9a-f]{64}$'",
    )
    op.create_unique_constraint(
        "uq_ingestion_jobs_document_recipe",
        "ingestion_jobs",
        ["tenant_id", "document_id", "content_hash", "pipeline_version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_ingestion_jobs_document_recipe",
        "ingestion_jobs",
        type_="unique",
    )
    op.drop_constraint(
        "ck_ingestion_jobs_pipeline_version_format",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_constraint(
        "ck_ingestion_jobs_content_hash_sha256",
        "ingestion_jobs",
        type_="check",
    )
    op.drop_column("ingestion_jobs", "pipeline_version")
    op.drop_column("ingestion_jobs", "content_hash")
