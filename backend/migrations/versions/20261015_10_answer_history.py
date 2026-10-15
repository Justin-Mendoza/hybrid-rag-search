"""Add author-owned grounded answer history.

Revision ID: 20261015_10
Revises: 20261013_09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261015_10"
down_revision: str | None = "20261013_09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "saved_answers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("author_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "author_user_id"],
            ["memberships.tenant_id", "memberships.user_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("length(trim(question)) BETWEEN 1 AND 4096", name="question_length"),
        sa.CheckConstraint("jsonb_typeof(result) = 'object'", name="result_object"),
    )
    op.create_index(
        "ix_saved_answers_author_history",
        "saved_answers",
        ["tenant_id", "author_user_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_table("saved_answers")
