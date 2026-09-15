import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from hybrid_rag_search.database import Base


class JudgmentTarget(enum.StrEnum):
    DOCUMENT = "document"
    CHUNK = "chunk"


class EvaluationRunStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    corpus_hash: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    dataset_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "name", "version"),
        CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint("corpus_hash ~ '^[0-9a-f]{64}$'", name="corpus_hash_sha256"),
        CheckConstraint("jsonb_typeof(dataset_metadata) = 'object'", name="metadata_object"),
    )


class Judgment(Base):
    __tablename__ = "judgments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    query_key: Mapped[str] = mapped_column(Text)
    query_text: Mapped[str] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(Text)
    target_id: Mapped[str] = mapped_column(Text)
    relevance_grade: Mapped[int] = mapped_column(Integer)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    key_facts: Mapped[list[object]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    metadata_filters: Mapped[dict[str, object]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            ["evaluation_datasets.tenant_id", "evaluation_datasets.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "tenant_id",
            "dataset_id",
            "query_key",
            "target_type",
            "target_id",
            name="uq_judgments_dataset_query_target",
        ),
        CheckConstraint("length(trim(query_key)) > 0", name="query_key_not_blank"),
        CheckConstraint("length(trim(query_text)) > 0", name="query_text_not_blank"),
        CheckConstraint("target_type IN ('document', 'chunk')", name="target_type_allowed"),
        CheckConstraint("length(trim(target_id)) > 0", name="target_id_not_blank"),
        CheckConstraint("relevance_grade BETWEEN 0 AND 3", name="relevance_grade_range"),
        CheckConstraint("jsonb_typeof(key_facts) = 'array'", name="key_facts_array"),
        CheckConstraint(
            "jsonb_typeof(metadata_filters) = 'object'", name="metadata_filters_object"
        ),
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    retrieval_config_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(Text, server_default=EvaluationRunStatus.QUEUED.value)
    seed: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            ["evaluation_datasets.tenant_id", "evaluation_datasets.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "retrieval_config_id"],
            ["retrieval_configs.tenant_id", "retrieval_configs.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="status_allowed",
        ),
        CheckConstraint("seed >= 0", name="seed_nonnegative"),
        CheckConstraint("jsonb_typeof(metrics) = 'object'", name="metrics_object"),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT")
    )
    actor_membership_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(Text)
    entity_id: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "actor_membership_id"],
            ["memberships.tenant_id", "memberships.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "action ~ '^[a-z][a-z0-9_]*\\.[a-z][a-z0-9_]*$'",
            name="action_namespaced",
        ),
        CheckConstraint("entity_type ~ '^[a-z][a-z0-9_]*$'", name="entity_type_format"),
        CheckConstraint("length(trim(entity_id)) > 0", name="entity_id_not_blank"),
        CheckConstraint("jsonb_typeof(details) = 'object'", name="details_object"),
    )
