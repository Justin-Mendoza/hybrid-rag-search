"""Publish durable queued job identities to the transient Redis broker."""

import asyncio
from typing import Protocol
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.models.content import IngestionJob, IngestionJobStatus


class JobPublisher(Protocol):
    async def publish(self, job_id: UUID) -> None:
        """Publish only the durable job identity."""
        ...


class ActorSender(Protocol):
    def send(self, job_id: str) -> object:
        """Send a serialized actor argument through its configured broker."""
        ...

    def send_with_options(self, *, args: tuple[str], delay: int) -> object:
        """Send a serialized actor argument after a delay in milliseconds."""
        ...


class DramatiqJobPublisher:
    """Adapt Dramatiq's synchronous actor send method to async services."""

    def __init__(self, actor: ActorSender) -> None:
        self.actor = actor

    async def publish(self, job_id: UUID) -> None:
        await asyncio.to_thread(self.actor.send, str(job_id))

    async def schedule(self, job_id: UUID, *, delay_ms: int) -> None:
        if not isinstance(delay_ms, int) or isinstance(delay_ms, bool) or delay_ms < 0:
            raise ValueError("Job publication delay must be a nonnegative integer")
        await asyncio.to_thread(
            self.actor.send_with_options,
            args=(str(job_id),),
            delay=delay_ms,
        )


class RecoveryDispatcher:
    """Republish queued PostgreSQL jobs without claiming or changing them."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        publisher: JobPublisher,
        *,
        batch_size: int = 100,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("Recovery batch size must be a positive integer")
        self.sessions = sessions
        self.publisher = publisher
        self.batch_size = batch_size

    async def dispatch_queued(self) -> int:
        async with self.sessions.begin() as session:
            result = await session.scalars(
                select(IngestionJob.id)
                .where(
                    IngestionJob.status == IngestionJobStatus.QUEUED,
                    or_(
                        IngestionJob.next_attempt_at.is_(None),
                        IngestionJob.next_attempt_at <= func.now(),
                    ),
                )
                .order_by(IngestionJob.created_at, IngestionJob.id)
                .limit(self.batch_size)
            )
            job_ids = tuple(result.all())

        published = 0
        for job_id in job_ids:
            await self.publisher.publish(job_id)
            published += 1
        return published
