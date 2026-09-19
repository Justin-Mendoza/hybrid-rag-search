"""Durable retry scheduling and terminal ingestion failure handling."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.models.content import (
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
)


@dataclass(frozen=True)
class RetryPolicy:
    base_delay_seconds: float = 5.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_delay_seconds) or self.base_delay_seconds <= 0:
            raise ValueError("Retry base delay must be positive and finite")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("Retry multiplier must be at least one and finite")
        if not math.isfinite(self.jitter_ratio) or not 0 <= self.jitter_ratio < 1:
            raise ValueError("Retry jitter ratio must be between zero and one")

    def delay(self, attempt: int, random_sample: float) -> timedelta:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
            raise ValueError("Retry attempt must be a positive integer")
        if not math.isfinite(random_sample) or not 0 <= random_sample <= 1:
            raise ValueError("Retry random sample must be between zero and one")
        base = self.base_delay_seconds * self.multiplier ** (attempt - 1)
        jitter = 1 - self.jitter_ratio + (2 * self.jitter_ratio * random_sample)
        return timedelta(seconds=base * jitter)


DEFAULT_RETRY_POLICY = RetryPolicy()


class FailureDisposition(StrEnum):
    RETRY_SCHEDULED = "retry_scheduled"
    TERMINAL = "terminal"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True)
class FailureOutcome:
    disposition: FailureDisposition
    next_attempt_at: datetime | None = None


class JobFailureHandler:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        random_sample: Callable[[], float],
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.sessions = sessions
        self.retry_policy = retry_policy
        self.clock = clock
        self.random_sample = random_sample

    async def record(
        self,
        claimed: ClaimedJob,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> FailureOutcome:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", error_code):
            raise ValueError("Job error code must use lowercase snake_case")
        if not isinstance(error_message, str) or not error_message.strip():
            raise ValueError("Job error message must be nonblank")
        if not isinstance(retryable, bool):
            raise ValueError("Job retryable flag must be boolean")

        now = self.clock()
        will_retry = retryable and claimed.attempts < claimed.max_attempts
        next_attempt_at = (
            now + self.retry_policy.delay(claimed.attempts, self.random_sample())
            if will_retry
            else None
        )
        status = IngestionJobStatus.QUEUED if will_retry else IngestionJobStatus.FAILED
        message = error_message.strip()[:1000]
        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.id == claimed.id,
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.lease_token == claimed.lease_token,
                IngestionJob.lease_expires_at > func.now(),
            )
            .values(
                status=status,
                error_code=error_code,
                error_message=message,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=next_attempt_at,
                finished_at=None if will_retry else now,
            )
            .returning(IngestionJob.document_id)
        )
        async with self.sessions.begin() as session:
            document_id = await session.scalar(statement)
            if document_id is None:
                return FailureOutcome(FailureDisposition.LEASE_LOST)
            if not will_retry:
                await session.execute(
                    update(Document)
                    .where(
                        Document.id == claimed.document_id,
                        Document.tenant_id == claimed.tenant_id,
                        Document.content_hash == claimed.content_hash,
                    )
                    .values(status=DocumentStatus.FAILED, error_message=message)
                )
        return FailureOutcome(
            FailureDisposition.RETRY_SCHEDULED if will_retry else FailureDisposition.TERMINAL,
            next_attempt_at,
        )
