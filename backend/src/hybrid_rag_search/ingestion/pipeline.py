"""One-document worker orchestration across resumable ingestion stages."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import ArtifactContractError
from hybrid_rag_search.ingestion.artifacts import ArtifactConflictError, ArtifactIntegrityError
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.failures import (
    FailureDisposition,
    JobFailureHandler,
)
from hybrid_rag_search.ingestion.indexing import IndexWriteError
from hybrid_rag_search.ingestion.parse_stage import SourceReadError
from hybrid_rag_search.models.content import IngestionStage
from hybrid_rag_search.parsers.errors import ParseError
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.tokenization import TokenizerLoadError


class JobClaimService(Protocol):
    async def claim(self, job_id: UUID) -> ClaimedJob | None: ...


class ArtifactStage(Protocol):
    async def run(self, claimed: ClaimedJob) -> ClaimedJob | None: ...


class FinalIndexStage(Protocol):
    async def run(self, claimed: ClaimedJob) -> bool: ...


class RetryScheduler(Protocol):
    async def schedule(self, job_id: UUID, *, delay_ms: int) -> None: ...


class JobRunResult(StrEnum):
    NOT_CLAIMED = "not_claimed"
    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True)
class ClassifiedFailure:
    code: str
    message: str
    retryable: bool


def classify_failure(error: Exception) -> ClassifiedFailure:
    """Translate internal exceptions into safe durable failure details."""

    if isinstance(error, ProviderError):
        return ClassifiedFailure(f"provider_{error.code}", str(error), error.retryable)
    if isinstance(error, IndexWriteError):
        return ClassifiedFailure(f"index_{error.code}", str(error), error.retryable)
    if isinstance(error, SourceReadError):
        return ClassifiedFailure(error.code.value, str(error), error.retryable)
    if isinstance(error, ParseError):
        return ClassifiedFailure(f"parse_{error.code.value}", str(error), False)
    if isinstance(error, (ArtifactContractError, ArtifactConflictError, ArtifactIntegrityError)):
        return ClassifiedFailure(
            "artifact_integrity", "Ingestion artifact validation failed", False
        )
    if isinstance(error, TokenizerLoadError):
        return ClassifiedFailure("tokenizer_unavailable", str(error), False)
    if isinstance(error, DocumentNotFound):
        return ClassifiedFailure("document_not_found", str(error), False)
    if isinstance(error, DocumentUnavailable):
        return ClassifiedFailure("document_unavailable", str(error), False)
    if isinstance(error, OSError):
        return ClassifiedFailure("storage_unavailable", "Ingestion storage is unavailable", True)
    return ClassifiedFailure("internal_error", "Unexpected ingestion worker failure", True)


class IngestionPipeline:
    """Claim once, resume from the durable stage, and own all retry decisions."""

    def __init__(
        self,
        claimer: JobClaimService,
        parse_stage: ArtifactStage,
        chunk_stage: ArtifactStage,
        embed_stage: ArtifactStage,
        index_stage: FinalIndexStage,
        failures: JobFailureHandler,
        retries: RetryScheduler,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.claimer = claimer
        self.stages = {
            IngestionStage.PARSE: parse_stage,
            IngestionStage.CHUNK: chunk_stage,
            IngestionStage.EMBED: embed_stage,
        }
        self.index_stage = index_stage
        self.failures = failures
        self.retries = retries
        self.clock = clock

    async def run(self, job_id: UUID) -> JobRunResult:
        if not isinstance(job_id, UUID):
            raise ValueError("Ingestion job ID must be a UUID")
        claimed = await self.claimer.claim(job_id)
        if claimed is None:
            return JobRunResult.NOT_CLAIMED

        try:
            current = claimed
            while current.stage in self.stages:
                advanced = await self.stages[current.stage].run(current)
                if advanced is None:
                    return JobRunResult.LEASE_LOST
                current = advanced
            if current.stage is not IngestionStage.INDEX:
                raise ValueError("Claimed job has an invalid resumable stage")
            if not await self.index_stage.run(current):
                return JobRunResult.LEASE_LOST
            return JobRunResult.SUCCEEDED
        except Exception as error:
            failure = classify_failure(error)
            outcome = await self.failures.record(
                current,
                error_code=failure.code,
                error_message=failure.message,
                retryable=failure.retryable,
            )
            if outcome.disposition is FailureDisposition.LEASE_LOST:
                return JobRunResult.LEASE_LOST
            if outcome.disposition is FailureDisposition.TERMINAL:
                return JobRunResult.FAILED
            assert outcome.next_attempt_at is not None
            delay_seconds = max(0.0, (outcome.next_attempt_at - self.clock()).total_seconds())
            await self.retries.schedule(job_id, delay_ms=math.ceil(delay_seconds * 1000))
            return JobRunResult.RETRY_SCHEDULED
