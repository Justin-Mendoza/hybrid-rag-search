"""Atomic PostgreSQL ownership claims for at-least-once job delivery."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import func, or_, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.models.content import (
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)

LEASE_DURATION_SECONDS = 300


@dataclass(frozen=True)
class ClaimedJob:
    id: UUID
    tenant_id: UUID
    document_id: UUID
    content_hash: str
    pipeline_version: str
    stage: IngestionStage
    attempts: int
    max_attempts: int
    checkpoint: dict[str, object]
    lease_token: UUID
    lease_expires_at: datetime


class JobClaimer:
    """Let exactly one worker transition a queued job to running."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def claim(self, job_id: UUID) -> ClaimedJob | None:
        lease_token = uuid4()
        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == IngestionJobStatus.QUEUED,
                IngestionJob.attempts < IngestionJob.max_attempts,
                or_(
                    IngestionJob.next_attempt_at.is_(None),
                    IngestionJob.next_attempt_at <= func.now(),
                ),
            )
            .values(
                status=IngestionJobStatus.RUNNING,
                attempts=IngestionJob.attempts + 1,
                started_at=func.coalesce(IngestionJob.started_at, func.now()),
                finished_at=None,
                lease_token=lease_token,
                lease_expires_at=func.now() + text(f"INTERVAL '{LEASE_DURATION_SECONDS} seconds'"),
                next_attempt_at=None,
            )
            .returning(
                IngestionJob.id,
                IngestionJob.tenant_id,
                IngestionJob.document_id,
                IngestionJob.content_hash,
                IngestionJob.pipeline_version,
                IngestionJob.stage,
                IngestionJob.attempts,
                IngestionJob.max_attempts,
                IngestionJob.checkpoint,
                IngestionJob.lease_token,
                IngestionJob.lease_expires_at,
            )
        )
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            row = result.mappings().one_or_none()
        if row is None:
            return None
        return ClaimedJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            document_id=row["document_id"],
            content_hash=row["content_hash"],
            pipeline_version=row["pipeline_version"],
            stage=IngestionStage(row["stage"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            checkpoint=dict(row["checkpoint"]),
            lease_token=row["lease_token"],
            lease_expires_at=row["lease_expires_at"],
        )

    async def renew(self, job_id: UUID, lease_token: UUID) -> datetime | None:
        """Extend a live lease only for the worker that currently owns it."""

        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.lease_token == lease_token,
                IngestionJob.lease_expires_at > func.now(),
            )
            .values(
                lease_expires_at=func.now() + text(f"INTERVAL '{LEASE_DURATION_SECONDS} seconds'")
            )
            .returning(IngestionJob.lease_expires_at)
        )
        async with self.sessions.begin() as session:
            result = await session.scalar(statement)
        return result
