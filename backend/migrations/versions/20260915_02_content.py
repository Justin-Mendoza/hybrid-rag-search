"""Create collections, grants, documents, and ingestion jobs.

Revision ID: 20260915_02
Revises: 20260915_01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260915_02"
down_revision: str | None = "20260915_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_collections_name_not_blank"),
        sa.CheckConstraint(
            "slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'", name="ck_collections_slug_format"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_collections_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_collections"),
        sa.UniqueConstraint("tenant_id", "id", name="uq_collections_tenant_id_id"),
        sa.UniqueConstraint("tenant_id", "slug", name="uq_collections_tenant_id_slug"),
    )
    op.create_table(
        "collection_grants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("membership_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("permission", sa.Text(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "permission IN ('read', 'write', 'manage')",
            name="ck_collection_grants_permission_allowed",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "collection_id"],
            ["collections.tenant_id", "collections.id"],
            name="fk_collection_grants_tenant_id_collections",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "membership_id"],
            ["memberships.tenant_id", "memberships.id"],
            name="fk_collection_grants_tenant_id_memberships",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_collection_grants"),
        sa.UniqueConstraint(
            "tenant_id",
            "collection_id",
            "membership_id",
            name="uq_collection_grants_tenant_id_collection_id_membership_id",
        ),
    )
    op.create_index(
        "ix_collection_grants_tenant_membership",
        "collection_grants",
        ["tenant_id", "membership_id"],
    )
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "source_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("index_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'", name="ck_documents_content_hash_sha256"
        ),
        sa.CheckConstraint(
            "(status = 'deleted') = (deleted_at IS NOT NULL)",
            name="ck_documents_deleted_at_matches_status",
        ),
        sa.CheckConstraint("index_version >= 0", name="ck_documents_index_version_nonnegative"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_documents_size_bytes_nonnegative"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed', 'deleting', 'deleted')",
            name="ck_documents_status_allowed",
        ),
        sa.CheckConstraint(
            "length(trim(storage_key)) > 0", name="ck_documents_storage_key_not_blank"
        ),
        sa.CheckConstraint(
            "length(trim(source_key)) > 0", name="ck_documents_source_key_not_blank"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "collection_id"],
            ["collections.tenant_id", "collections.id"],
            name="fk_documents_tenant_id_collections",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("tenant_id", "id", name="uq_documents_tenant_id_id"),
        sa.UniqueConstraint(
            "tenant_id",
            "collection_id",
            "source_key",
            name="uq_documents_tenant_id_collection_id_source_key",
        ),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_tenant_status", "documents", ["tenant_id", "status"])
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column(
            "checkpoint",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("attempts >= 0", name="ck_ingestion_jobs_attempts_nonnegative"),
        sa.CheckConstraint(
            "attempts <= max_attempts", name="ck_ingestion_jobs_attempts_within_limit"
        ),
        sa.CheckConstraint("max_attempts > 0", name="ck_ingestion_jobs_max_attempts_positive"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_ingestion_jobs_status_allowed",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.id"],
            name="fk_ingestion_jobs_tenant_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_jobs"),
    )
    op.create_index("ix_ingestion_jobs_tenant_status", "ingestion_jobs", ["tenant_id", "status"])

    for table in ("collections", "collection_grants", "documents", "ingestion_jobs"):
        op.execute(
            f"CREATE TRIGGER trg_{table}_updated_at BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
        )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_table("documents")
    op.drop_table("collection_grants")
    op.drop_table("collections")
