from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import ArtifactContractError
from hybrid_rag_search.ingestion.artifacts import ArtifactConflictError, ArtifactIntegrityError
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.failures import FailureDisposition, FailureOutcome
from hybrid_rag_search.ingestion.indexing import IndexWriteError
from hybrid_rag_search.ingestion.parse_stage import SourceReadError, SourceReadErrorCode
from hybrid_rag_search.ingestion.pipeline import (
    ClassifiedFailure,
    IngestionPipeline,
    JobRunResult,
    classify_failure,
)
from hybrid_rag_search.models.content import IngestionStage
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.tokenization import TokenizerLoadError

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def claimed(stage: IngestionStage = IngestionStage.PARSE) -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash="a" * 64,
        pipeline_version="pipe_" + "b" * 64,
        stage=stage,
        attempts=1,
        max_attempts=3,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=NOW + timedelta(minutes=5),
    )


def pipeline(job: ClaimedJob | None):
    claimer = AsyncMock()
    claimer.claim.return_value = job
    parse_stage, chunk_stage, embed_stage, index_stage = (AsyncMock() for _ in range(4))
    if job is not None:
        parse_stage.run.return_value = replace(job, stage=IngestionStage.CHUNK)
        chunk_stage.run.return_value = replace(job, stage=IngestionStage.EMBED)
        embed_stage.run.return_value = replace(job, stage=IngestionStage.INDEX)
    index_stage.run.return_value = True
    failures = AsyncMock()
    retries = AsyncMock()
    value = IngestionPipeline(
        claimer,
        parse_stage,
        chunk_stage,
        embed_stage,
        index_stage,
        failures,
        retries,
        clock=lambda: NOW,
    )
    return value, claimer, parse_stage, chunk_stage, embed_stage, index_stage, failures, retries


@pytest.mark.anyio
async def test_pipeline_runs_all_stages_and_completes() -> None:
    job = claimed()
    value, _, parse_stage, chunk_stage, embed_stage, index_stage, failures, _ = pipeline(job)
    assert await value.run(job.id) is JobRunResult.SUCCEEDED
    parse_stage.run.assert_awaited_once_with(job)
    chunk_stage.run.assert_awaited_once()
    embed_stage.run.assert_awaited_once()
    index_stage.run.assert_awaited_once()
    failures.record.assert_not_awaited()


@pytest.mark.anyio
async def test_pipeline_resumes_and_duplicate_delivery_does_nothing() -> None:
    job = claimed(IngestionStage.EMBED)
    value, _, parse_stage, chunk_stage, embed_stage, index_stage, *_ = pipeline(job)
    assert await value.run(job.id) is JobRunResult.SUCCEEDED
    parse_stage.run.assert_not_awaited()
    chunk_stage.run.assert_not_awaited()
    embed_stage.run.assert_awaited_once_with(job)
    index_stage.run.assert_awaited_once()

    value, claimer, *_ = pipeline(None)
    assert await value.run(job.id) is JobRunResult.NOT_CLAIMED
    claimer.claim.assert_awaited_once_with(job.id)


@pytest.mark.anyio
async def test_pipeline_stops_when_stage_or_completion_loses_lease() -> None:
    job = claimed()
    value, _, parse_stage, *rest = pipeline(job)
    parse_stage.run.return_value = None
    assert await value.run(job.id) is JobRunResult.LEASE_LOST
    rest[2].run.assert_not_awaited()

    index_job = claimed(IngestionStage.INDEX)
    value, *items = pipeline(index_job)
    index_stage = items[4]
    index_stage.run.return_value = False
    assert await value.run(index_job.id) is JobRunResult.LEASE_LOST


@pytest.mark.anyio
async def test_pipeline_schedules_durable_retry() -> None:
    job = claimed(IngestionStage.EMBED)
    value, _, _, _, embed_stage, _, failures, retries = pipeline(job)
    embed_stage.run.side_effect = ProviderError("rate_limited", retryable=True)
    next_attempt = NOW + timedelta(seconds=6.001)
    failures.record.return_value = FailureOutcome(
        FailureDisposition.RETRY_SCHEDULED,
        next_attempt,
    )
    assert await value.run(job.id) is JobRunResult.RETRY_SCHEDULED
    failures.record.assert_awaited_once_with(
        job,
        error_code="provider_rate_limited",
        error_message="Model provider error: rate_limited",
        retryable=True,
    )
    retries.schedule.assert_awaited_once_with(job.id, delay_ms=6001)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        (FailureDisposition.TERMINAL, JobRunResult.FAILED),
        (FailureDisposition.LEASE_LOST, JobRunResult.LEASE_LOST),
    ],
)
async def test_pipeline_reports_terminal_or_stale_failure(
    disposition: FailureDisposition, expected: JobRunResult
) -> None:
    job = claimed(IngestionStage.INDEX)
    value, *items = pipeline(job)
    index_stage, failures, retries = items[4:7]
    index_stage.run.side_effect = ValueError("bad stage data")
    failures.record.return_value = FailureOutcome(disposition)
    assert await value.run(job.id) is expected
    retries.schedule.assert_not_awaited()


@pytest.mark.anyio
async def test_pipeline_rejects_non_uuid_and_invalid_resumable_stage() -> None:
    job = claimed(IngestionStage.COMPLETE)
    value, *items = pipeline(job)
    failures, retries = items[5:7]
    failures.record.return_value = FailureOutcome(
        FailureDisposition.RETRY_SCHEDULED,
        NOW - timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="UUID"):
        await value.run("job")  # type: ignore[arg-type]
    assert await value.run(job.id) is JobRunResult.RETRY_SCHEDULED
    retries.schedule.assert_awaited_once_with(job.id, delay_ms=0)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ProviderError("timeout", retryable=True),
            ClassifiedFailure("provider_timeout", "Model provider error: timeout", True),
        ),
        (
            IndexWriteError("partial_write", retryable=True),
            ClassifiedFailure("index_partial_write", "partial_write", True),
        ),
        (
            SourceReadError(
                SourceReadErrorCode.ORIGINAL_MISSING,
                "Original missing",
                retryable=True,
            ),
            ClassifiedFailure("original_missing", "Original missing", True),
        ),
        (
            ParseError(ParseErrorCode.CORRUPT_DOCUMENT, "Corrupt"),
            ClassifiedFailure("parse_corrupt_document", "Corrupt", False),
        ),
        (
            ArtifactContractError("details"),
            ClassifiedFailure("artifact_integrity", "Ingestion artifact validation failed", False),
        ),
        (
            ArtifactConflictError("details"),
            ClassifiedFailure("artifact_integrity", "Ingestion artifact validation failed", False),
        ),
        (
            ArtifactIntegrityError("details"),
            ClassifiedFailure("artifact_integrity", "Ingestion artifact validation failed", False),
        ),
        (
            TokenizerLoadError("Tokenizer missing"),
            ClassifiedFailure("tokenizer_unavailable", "Tokenizer missing", False),
        ),
        (
            DocumentNotFound("Missing"),
            ClassifiedFailure("document_not_found", "Missing", False),
        ),
        (
            DocumentUnavailable("Deleted"),
            ClassifiedFailure("document_unavailable", "Deleted", False),
        ),
        (
            OSError("secret path"),
            ClassifiedFailure("storage_unavailable", "Ingestion storage is unavailable", True),
        ),
        (
            RuntimeError("secret details"),
            ClassifiedFailure("internal_error", "Unexpected ingestion worker failure", True),
        ),
    ],
)
def test_failure_classification_is_safe_and_stable(
    error: Exception, expected: ClassifiedFailure
) -> None:
    assert classify_failure(error) == expected
