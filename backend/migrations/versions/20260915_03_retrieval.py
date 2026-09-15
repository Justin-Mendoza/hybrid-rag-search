"""Create retrieval configurations and query traces.

Revision ID: 20260915_03
Revises: 20260915_02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260915_03"
down_revision: str | None = "20260915_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION prevent_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION '% is append-only', TG_TABLE_NAME
                USING ERRCODE = 'integrity_constraint_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.create_table(
        "retrieval_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("rerank_model", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "length(trim(embedding_model)) > 0",
            name="ck_retrieval_configs_embedding_model_not_blank",
        ),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_retrieval_configs_name_not_blank"),
        sa.CheckConstraint(
            "jsonb_typeof(parameters) = 'object'",
            name="ck_retrieval_configs_parameters_object",
        ),
        sa.CheckConstraint("version > 0", name="ck_retrieval_configs_version_positive"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_retrieval_configs_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_configs"),
        sa.UniqueConstraint("tenant_id", "id", name="uq_retrieval_configs_tenant_id_id"),
        sa.UniqueConstraint(
            "tenant_id",
            "name",
            "version",
            name="uq_retrieval_configs_tenant_id_name_version",
        ),
    )
    op.execute(
        "CREATE TRIGGER trg_retrieval_configs_immutable "
        "BEFORE UPDATE OR DELETE ON retrieval_configs "
        "FOR EACH ROW EXECUTE FUNCTION prevent_mutation()"
    )
    op.create_table(
        "query_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("retrieval_config_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("normalized_query_hash", sa.Text(), nullable=False),
        sa.Column("content_captured", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("selected_path", sa.Text(), nullable=False),
        sa.Column("total_latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "timings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "rankings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "usage",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "fallbacks",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "content_captured OR query_text IS NULL",
            name="ck_query_traces_query_text_capture_policy",
        ),
        sa.CheckConstraint(
            "normalized_query_hash ~ '^[0-9a-f]{64}$'",
            name="ck_query_traces_normalized_query_hash_sha256",
        ),
        sa.CheckConstraint(
            "length(trim(selected_path)) > 0", name="ck_query_traces_selected_path_not_blank"
        ),
        sa.CheckConstraint(
            "total_latency_ms IS NULL OR total_latency_ms >= 0",
            name="ck_query_traces_total_latency_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "collection_id"],
            ["collections.tenant_id", "collections.id"],
            name="fk_query_traces_tenant_id_collections",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "retrieval_config_id"],
            ["retrieval_configs.tenant_id", "retrieval_configs.id"],
            name="fk_query_traces_tenant_id_retrieval_configs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_query_traces"),
    )


def downgrade() -> None:
    op.drop_table("query_traces")
    op.execute("DROP TRIGGER trg_retrieval_configs_immutable ON retrieval_configs")
    op.drop_table("retrieval_configs")
    op.execute("DROP FUNCTION prevent_mutation()")
