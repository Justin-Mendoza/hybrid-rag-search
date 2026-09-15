"""Create evaluation and audit tables.

Revision ID: 20260915_04
Revises: 20260915_03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260915_04"
down_revision: str | None = "20260915_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column[object]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("corpus_hash", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "dataset_metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        _created_at(),
        sa.CheckConstraint(
            "corpus_hash ~ '^[0-9a-f]{64}$'", name="ck_evaluation_datasets_corpus_hash_sha256"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(dataset_metadata) = 'object'",
            name="ck_evaluation_datasets_metadata_object",
        ),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_evaluation_datasets_name_not_blank"),
        sa.CheckConstraint("version > 0", name="ck_evaluation_datasets_version_positive"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_evaluation_datasets_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_datasets"),
        sa.UniqueConstraint("tenant_id", "id", name="uq_evaluation_datasets_tenant_id_id"),
        sa.UniqueConstraint(
            "tenant_id", "name", "version", name="uq_evaluation_datasets_tenant_id_name_version"
        ),
    )
    op.create_table(
        "judgments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_key", sa.Text(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("target_type", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Text(), nullable=False),
        sa.Column("relevance_grade", sa.Integer(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column(
            "key_facts", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column(
            "metadata_filters",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        _created_at(),
        sa.CheckConstraint(
            "jsonb_typeof(key_facts) = 'array'", name="ck_judgments_key_facts_array"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metadata_filters) = 'object'", name="ck_judgments_metadata_filters_object"
        ),
        sa.CheckConstraint("length(trim(query_key)) > 0", name="ck_judgments_query_key_not_blank"),
        sa.CheckConstraint(
            "length(trim(query_text)) > 0", name="ck_judgments_query_text_not_blank"
        ),
        sa.CheckConstraint(
            "relevance_grade BETWEEN 0 AND 3", name="ck_judgments_relevance_grade_range"
        ),
        sa.CheckConstraint("length(trim(target_id)) > 0", name="ck_judgments_target_id_not_blank"),
        sa.CheckConstraint(
            "target_type IN ('document', 'chunk')", name="ck_judgments_target_type_allowed"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            ["evaluation_datasets.tenant_id", "evaluation_datasets.id"],
            name="fk_judgments_tenant_id_evaluation_datasets",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_judgments"),
        sa.UniqueConstraint(
            "tenant_id",
            "dataset_id",
            "query_key",
            "target_type",
            "target_id",
            name="uq_judgments_dataset_query_target",
        ),
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("retrieval_config_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column(
            "metrics", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metrics) = 'object'", name="ck_evaluation_runs_metrics_object"
        ),
        sa.CheckConstraint("seed >= 0", name="ck_evaluation_runs_seed_nonnegative"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_evaluation_runs_status_allowed",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            ["evaluation_datasets.tenant_id", "evaluation_datasets.id"],
            name="fk_evaluation_runs_tenant_id_evaluation_datasets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "retrieval_config_id"],
            ["retrieval_configs.tenant_id", "retrieval_configs.id"],
            name="fk_evaluation_runs_tenant_id_retrieval_configs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_runs"),
    )
    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_membership_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "action ~ '^[a-z][a-z0-9_]*\\.[a-z][a-z0-9_]*$'",
            name="ck_audit_events_action_namespaced",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(details) = 'object'", name="ck_audit_events_details_object"
        ),
        sa.CheckConstraint(
            "length(trim(entity_id)) > 0", name="ck_audit_events_entity_id_not_blank"
        ),
        sa.CheckConstraint(
            "entity_type ~ '^[a-z][a-z0-9_]*$'", name="ck_audit_events_entity_type_format"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "actor_membership_id"],
            ["memberships.tenant_id", "memberships.id"],
            name="fk_audit_events_tenant_id_memberships",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_audit_events_tenant_id_tenants",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_events"),
    )

    for table in ("evaluation_datasets", "judgments", "audit_events"):
        op.execute(
            f"CREATE TRIGGER trg_{table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION prevent_mutation()"
        )
    op.execute(
        "CREATE TRIGGER trg_evaluation_runs_updated_at BEFORE UPDATE ON evaluation_runs "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
    )


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("evaluation_runs")
    op.drop_table("judgments")
    op.drop_table("evaluation_datasets")
