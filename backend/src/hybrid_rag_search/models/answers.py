"""Author-owned independent question/answer history."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from hybrid_rag_search.database import Base


class SavedAnswer(Base):
    __tablename__ = "saved_answers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    author_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    question: Mapped[str] = mapped_column(Text)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "author_user_id"],
            ["memberships.tenant_id", "memberships.user_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("length(trim(question)) BETWEEN 1 AND 4096", name="question_length"),
        CheckConstraint("jsonb_typeof(result) = 'object'", name="result_object"),
        Index("ix_saved_answers_author_history", "tenant_id", "author_user_id", "created_at", "id"),
    )
