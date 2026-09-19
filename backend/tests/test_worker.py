from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search import worker
from hybrid_rag_search.ingestion.pipeline import JobRunResult
from hybrid_rag_search.ingestion.recovery import RecoveryRunResult


def test_ingestion_actor_runs_configured_job(monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = uuid4()
    run = AsyncMock(return_value=JobRunResult.SUCCEEDED)
    monkeypatch.setattr(worker, "run_configured_job", run)
    assert worker.ingest_document.fn(str(job_id)) == "succeeded"
    run.assert_awaited_once_with(
        worker.settings,
        worker.ingest_document,
        worker.document_index,
        job_id,
    )


def test_ingestion_actor_rejects_invalid_job_id() -> None:
    with pytest.raises(ValueError):
        worker.ingest_document.fn("not-a-uuid")


def test_heartbeat_actor() -> None:
    assert worker.heartbeat.fn() == "ok"


def test_startup_middleware_runs_one_recovery_pass() -> None:
    recover = AsyncMock(return_value=RecoveryRunResult(0, 0, 0))
    middleware = worker.StartupRecoveryMiddleware(recover)
    middleware.after_worker_boot(MagicMock(), MagicMock())
    recover.assert_awaited_once_with()


@pytest.mark.anyio
async def test_configured_startup_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    result = RecoveryRunResult(1, 2, 3)
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(worker, "run_configured_recovery", run)
    assert await worker.recover_at_startup() == result
    run.assert_awaited_once_with(worker.settings, worker.ingest_document)
