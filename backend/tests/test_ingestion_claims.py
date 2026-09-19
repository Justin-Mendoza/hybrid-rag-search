from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.claims import JobClaimer
from hybrid_rag_search.models.content import IngestionStage


def claimer(returned: dict[str, object] | None):
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = returned
    session = AsyncMock()
    session.execute.return_value = result
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    return JobClaimer(sessions), session


@pytest.mark.anyio
async def test_atomic_claim_returns_durable_job_snapshot() -> None:
    job_id, tenant_id, document_id = uuid4(), uuid4(), uuid4()
    lease_token = uuid4()
    lease_expires_at = datetime(2026, 9, 18, 12, 5, tzinfo=UTC)
    service, session = claimer(
        {
            "id": job_id,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "content_hash": "a" * 64,
            "pipeline_version": "pipe_" + "b" * 64,
            "stage": "chunk",
            "attempts": 2,
            "max_attempts": 3,
            "checkpoint": {"parsed": "artifact-key"},
            "lease_token": lease_token,
            "lease_expires_at": lease_expires_at,
        }
    )

    claimed = await service.claim(job_id)

    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.tenant_id == tenant_id
    assert claimed.document_id == document_id
    assert claimed.stage is IngestionStage.CHUNK
    assert claimed.attempts == 2
    assert claimed.checkpoint == {"parsed": "artifact-key"}
    assert claimed.lease_token == lease_token
    assert claimed.lease_expires_at == lease_expires_at

    statement = session.execute.await_args.args[0]
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "ingestion_jobs.status = 'queued'" in compiled
    assert "ingestion_jobs.attempts < ingestion_jobs.max_attempts" in compiled
    assert "ingestion_jobs.next_attempt_at IS NULL" in compiled
    assert "ingestion_jobs.next_attempt_at <= now()" in compiled
    assert "attempts=(ingestion_jobs.attempts + 1)" in compiled
    assert "RETURNING ingestion_jobs.id" in compiled
    assert "lease_token=" in compiled
    assert "INTERVAL '300 seconds'" in compiled


@pytest.mark.anyio
async def test_unclaimable_job_returns_none() -> None:
    service, session = claimer(None)
    assert await service.claim(uuid4()) is None
    session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_current_worker_can_renew_lease() -> None:
    service, session = claimer(None)
    expires_at = datetime(2026, 9, 18, 12, 10, tzinfo=UTC)
    session.scalar.return_value = expires_at
    job_id, lease_token = uuid4(), uuid4()

    assert await service.renew(job_id, lease_token) == expires_at

    statement = session.scalar.await_args.args[0]
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "ingestion_jobs.status = 'running'" in compiled
    assert "ingestion_jobs.lease_expires_at > now()" in compiled
    assert "INTERVAL '300 seconds'" in compiled


@pytest.mark.anyio
async def test_stale_or_wrong_lease_cannot_be_renewed() -> None:
    service, session = claimer(None)
    session.scalar.return_value = None
    assert await service.renew(uuid4(), uuid4()) is None
