"""Recover jobs whose worker ownership lease expired."""

from dataclasses import dataclass

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.ingestion.dispatch import RecoveryDispatcher
from hybrid_rag_search.models.content import (
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
)

LEASE_EXPIRED_CODE = "worker_lease_expired"
LEASE_EXPIRED_MESSAGE = "Worker ownership expired before the ingestion attempt completed"


@dataclass(frozen=True)
class LeaseRecoveryResult:
    requeued: int
    failed: int


class ExpiredLeaseRecovery:
    """Atomically release expired jobs according to their remaining attempts."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def recover(self) -> LeaseRecoveryResult:
        retry_statement = (
            update(IngestionJob)
            .where(
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.lease_expires_at <= func.now(),
                IngestionJob.attempts < IngestionJob.max_attempts,
            )
            .values(
                status=IngestionJobStatus.QUEUED,
                error_code=LEASE_EXPIRED_CODE,
                error_message=LEASE_EXPIRED_MESSAGE,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=func.now(),
            )
            .returning(IngestionJob.id)
        )
        fail_statement = (
            update(IngestionJob)
            .where(
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.lease_expires_at <= func.now(),
                IngestionJob.attempts >= IngestionJob.max_attempts,
            )
            .values(
                status=IngestionJobStatus.FAILED,
                error_code=LEASE_EXPIRED_CODE,
                error_message=LEASE_EXPIRED_MESSAGE,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                finished_at=func.now(),
            )
            .returning(
                IngestionJob.tenant_id,
                IngestionJob.document_id,
                IngestionJob.content_hash,
            )
        )
        async with self.sessions.begin() as session:
            retried = await session.execute(retry_statement)
            failed = await session.execute(fail_statement)
            retried_ids = tuple(retried.scalars().all())
            failed_rows = tuple(failed.mappings().all())
            for row in failed_rows:
                await session.execute(
                    update(Document)
                    .where(
                        Document.tenant_id == row["tenant_id"],
                        Document.id == row["document_id"],
                        Document.content_hash == row["content_hash"],
                    )
                    .values(
                        status=DocumentStatus.FAILED,
                        error_message=LEASE_EXPIRED_MESSAGE,
                    )
                )
        return LeaseRecoveryResult(requeued=len(retried_ids), failed=len(failed_rows))


@dataclass(frozen=True)
class RecoveryRunResult:
    requeued_expired: int
    failed_expired: int
    published: int


class IngestionRecovery:
    """Run the bounded startup/manual repair pass, then publish eligible jobs."""

    def __init__(
        self,
        expired_leases: ExpiredLeaseRecovery,
        queued_jobs: RecoveryDispatcher,
    ) -> None:
        self.expired_leases = expired_leases
        self.queued_jobs = queued_jobs

    async def run(self) -> RecoveryRunResult:
        leases = await self.expired_leases.recover()
        published = await self.queued_jobs.dispatch_queued()
        return RecoveryRunResult(leases.requeued, leases.failed, published)
