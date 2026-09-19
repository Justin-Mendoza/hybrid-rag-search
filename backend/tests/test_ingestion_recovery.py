from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.recovery import (
    ExpiredLeaseRecovery,
    IngestionRecovery,
    LeaseRecoveryResult,
    RecoveryRunResult,
)


def recovery(*, retried: int, failed_rows: list[dict[str, object]]):
    retried_result = MagicMock()
    retried_result.scalars.return_value.all.return_value = [uuid4() for _ in range(retried)]
    failed_result = MagicMock()
    failed_result.mappings.return_value.all.return_value = failed_rows
    session = AsyncMock()
    session.execute.side_effect = [
        retried_result,
        failed_result,
        *(MagicMock() for _ in failed_rows),
    ]
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    return ExpiredLeaseRecovery(sessions), session


@pytest.mark.anyio
async def test_expired_jobs_with_attempts_remaining_are_requeued() -> None:
    service, session = recovery(retried=2, failed_rows=[])
    assert await service.recover() == LeaseRecoveryResult(requeued=2, failed=0)
    assert session.execute.await_count == 2
    retry_statement = str(
        session.execute.await_args_list[0].args[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "ingestion_jobs.lease_expires_at <= now()" in retry_statement
    assert "ingestion_jobs.attempts < ingestion_jobs.max_attempts" in retry_statement
    assert "status='queued'" in retry_statement
    assert "lease_token=NULL" in retry_statement


@pytest.mark.anyio
async def test_expired_final_attempt_fails_matching_document() -> None:
    tenant_id, document_id = uuid4(), uuid4()
    service, session = recovery(
        retried=0,
        failed_rows=[
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "content_hash": "a" * 64,
            }
        ],
    )
    assert await service.recover() == LeaseRecoveryResult(requeued=0, failed=1)
    assert session.execute.await_count == 3
    fail_statement = str(
        session.execute.await_args_list[1].args[0].compile(compile_kwargs={"literal_binds": True})
    )
    document_statement = str(
        session.execute.await_args_list[2].args[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "ingestion_jobs.attempts >= ingestion_jobs.max_attempts" in fail_statement
    assert "status='failed'" in fail_statement
    assert "documents.content_hash = '" + "a" * 64 + "'" in document_statement


@pytest.mark.anyio
async def test_recovery_repairs_expired_leases_before_dispatching() -> None:
    expired = AsyncMock()
    expired.recover.return_value = LeaseRecoveryResult(requeued=2, failed=1)
    queued = AsyncMock()
    queued.dispatch_queued.return_value = 4
    service = IngestionRecovery(expired, queued)
    assert await service.run() == RecoveryRunResult(2, 1, 4)
    expired.recover.assert_awaited_once_with()
    queued.dispatch_queued.assert_awaited_once_with()
