from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.failures import (
    FailureDisposition,
    JobFailureHandler,
    RetryPolicy,
)
from hybrid_rag_search.models.content import IngestionStage

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def claimed(*, attempts: int = 1, max_attempts: int = 3) -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash="a" * 64,
        pipeline_version="pipe_" + "b" * 64,
        stage=IngestionStage.EMBED,
        attempts=attempts,
        max_attempts=max_attempts,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=NOW + timedelta(minutes=5),
    )


def handler(returned_document_id, *, random_sample: Mock | None = None):
    session = AsyncMock()
    session.scalar.return_value = returned_document_id
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    sample = random_sample or Mock(return_value=0.5)
    service = JobFailureHandler(sessions, random_sample=sample, clock=lambda: NOW)
    return service, session, sample


def test_retry_policy_uses_exponential_backoff_and_bounded_jitter() -> None:
    policy = RetryPolicy()
    assert policy.delay(1, 0.0) == timedelta(seconds=4)
    assert policy.delay(1, 0.5) == timedelta(seconds=5)
    assert policy.delay(1, 1.0) == timedelta(seconds=6)
    assert policy.delay(2, 0.5) == timedelta(seconds=10)


@pytest.mark.parametrize(
    "changes",
    [
        {"base_delay_seconds": 0},
        {"base_delay_seconds": float("inf")},
        {"multiplier": 0.5},
        {"multiplier": float("nan")},
        {"jitter_ratio": -0.1},
        {"jitter_ratio": 1.0},
    ],
)
def test_retry_policy_rejects_invalid_configuration(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(RetryPolicy(), **changes)


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5])
def test_retry_policy_requires_positive_attempt(attempt: object) -> None:
    with pytest.raises(ValueError, match="attempt"):
        RetryPolicy().delay(attempt, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample", [-0.1, 1.1, float("nan")])
def test_retry_policy_requires_unit_random_sample(sample: float) -> None:
    with pytest.raises(ValueError, match="random sample"):
        RetryPolicy().delay(1, sample)


@pytest.mark.anyio
async def test_retryable_failure_schedules_next_attempt_and_releases_lease() -> None:
    job = claimed(attempts=1)
    service, session, sample = handler(job.document_id)

    outcome = await service.record(
        job,
        error_code="provider_unavailable",
        error_message="  Cohere temporarily unavailable  ",
        retryable=True,
    )

    assert outcome.disposition is FailureDisposition.RETRY_SCHEDULED
    assert outcome.next_attempt_at == NOW + timedelta(seconds=5)
    sample.assert_called_once_with()
    session.execute.assert_not_awaited()
    compiled = str(
        session.scalar.await_args.args[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "status='queued'" in compiled
    assert "lease_token=NULL" in compiled
    assert "error_code='provider_unavailable'" in compiled


@pytest.mark.anyio
@pytest.mark.parametrize("retryable", [False, True])
async def test_permanent_or_final_attempt_fails_job_and_document(retryable: bool) -> None:
    job = claimed(attempts=3 if retryable else 1)
    sample = Mock(return_value=0.5)
    service, session, _ = handler(job.document_id, random_sample=sample)

    outcome = await service.record(
        job,
        error_code="invalid_document",
        error_message="cannot parse",
        retryable=retryable,
    )

    assert outcome == (outcome.__class__(FailureDisposition.TERMINAL))
    sample.assert_not_called()
    session.execute.assert_awaited_once()
    job_update = str(
        session.scalar.await_args.args[0].compile(compile_kwargs={"literal_binds": True})
    )
    document_update = str(
        session.execute.await_args.args[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "status='failed'" in job_update
    assert "documents.content_hash" in document_update
    assert "status='failed'" in document_update


@pytest.mark.anyio
async def test_lost_lease_cannot_change_job_or_document() -> None:
    job = claimed()
    service, session, _ = handler(None)
    outcome = await service.record(
        job,
        error_code="provider_unavailable",
        error_message="unavailable",
        retryable=True,
    )
    assert outcome.disposition is FailureDisposition.LEASE_LOST
    session.execute.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "message", "retryable", "expected"),
    [
        ("Bad-Code", "message", True, "snake_case"),
        ("bad_code", " ", True, "nonblank"),
        ("bad_code", "message", 1, "boolean"),
    ],
)
async def test_failure_details_are_validated(
    code: str, message: str, retryable: object, expected: str
) -> None:
    job = claimed()
    service, _, _ = handler(job.document_id)
    with pytest.raises(ValueError, match=expected):
        await service.record(
            job,
            error_code=code,
            error_message=message,
            retryable=retryable,  # type: ignore[arg-type]
        )
