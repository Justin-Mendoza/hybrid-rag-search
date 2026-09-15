import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
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


class RetrievalConfig(Base):
    __tablename__ = "retrieval_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    parameters: Mapped[dict[str, object]] = mapped_column(JSONB)
    embedding_model: Mapped[str] = mapped_column(Text)
    rerank_model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "name", "version"),
        CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint("jsonb_typeof(parameters) = 'object'", name="parameters_object"),
        CheckConstraint("length(trim(embedding_model)) > 0", name="embedding_model_not_blank"),
    )


class QueryTrace(Base):
    __tablename__ = "query_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    collection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    retrieval_config_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    query_text: Mapped[str | None] = mapped_column(Text)
    normalized_query_hash: Mapped[str] = mapped_column(Text)
    content_captured: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    selected_path: Mapped[str] = mapped_column(Text)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    timings: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    rankings: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    usage: Mapped[dict[str, object]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    fallbacks: Mapped[list[object]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    error: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "collection_id"],
            ["collections.tenant_id", "collections.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "retrieval_config_id"],
            ["retrieval_configs.tenant_id", "retrieval_configs.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "normalized_query_hash ~ '^[0-9a-f]{64}$'", name="normalized_query_hash_sha256"
        ),
        CheckConstraint("content_captured OR query_text IS NULL", name="query_text_capture_policy"),
        CheckConstraint(
            "total_latency_ms IS NULL OR total_latency_ms >= 0",
            name="total_latency_nonnegative",
        ),
        CheckConstraint("length(trim(selected_path)) > 0", name="selected_path_not_blank"),
    )
