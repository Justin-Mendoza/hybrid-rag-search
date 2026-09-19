from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.jobs import EnqueueResult, IngestionJobService
from hybrid_rag_search.models.content import Document, DocumentStatus, IngestionJobStatus

PIPELINE_VERSION = "pipe_" + "b" * 64


def service(document: Document | None, inserted_job_id=None, existing=None):
    session = AsyncMock()
    session.scalar.side_effect = [document, inserted_job_id]
    if existing is not None:
        result = MagicMock()
        result.one.return_value = existing
        session.execute.return_value = result
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    publisher = AsyncMock()
    return IngestionJobService(sessions, publisher), session, publisher


def document(status: DocumentStatus = DocumentStatus.PENDING) -> Document:
    return Document(
        id=uuid4(),
        tenant_id=uuid4(),
        collection_id=uuid4(),
        content_hash="a" * 64,
        status=status,
    )


@pytest.mark.anyio
async def test_new_job_is_persisted_before_publication() -> None:
    value = document()
    job_id = uuid4()
    jobs, session, publisher = service(value, job_id)
    result = await jobs.enqueue_document(value.tenant_id, value.id, PIPELINE_VERSION)
    assert result == EnqueueResult(job_id, True, IngestionJobStatus.QUEUED)
    publisher.publish.assert_awaited_once_with(job_id)
    statement = session.scalar.await_args_list[1].args[0]
    compiled = statement.compile()
    assert "ON CONFLICT ON CONSTRAINT uq_ingestion_jobs_document_recipe DO NOTHING" in str(compiled)
    assert value.content_hash in compiled.params.values()
    update_sql = str(session.execute.await_args.args[0].compile())
    assert "status=:status" in update_sql


@pytest.mark.anyio
async def test_existing_queued_job_is_republished_without_new_work() -> None:
    value = document(DocumentStatus.PROCESSING)
    job_id = uuid4()
    existing = MagicMock(id=job_id, status=IngestionJobStatus.QUEUED.value)
    jobs, session, publisher = service(value, None, existing)
    result = await jobs.enqueue_document(value.tenant_id, value.id, PIPELINE_VERSION)
    assert result == EnqueueResult(job_id, False, IngestionJobStatus.QUEUED)
    publisher.publish.assert_awaited_once_with(job_id)
    sql = str(session.execute.await_args.args[0].compile())
    assert "ingestion_jobs.content_hash" in sql


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status",
    [
        IngestionJobStatus.RUNNING,
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED,
        IngestionJobStatus.CANCELLED,
    ],
)
async def test_existing_nonqueued_job_is_reused_without_publication(
    status: IngestionJobStatus,
) -> None:
    value = document(DocumentStatus.PROCESSING)
    job_id = uuid4()
    jobs, _, publisher = service(value, None, MagicMock(id=job_id, status=status.value))
    assert await jobs.enqueue_document(
        value.tenant_id, value.id, PIPELINE_VERSION
    ) == EnqueueResult(job_id, False, status)
    publisher.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_enqueue_validates_identity_and_document_lifecycle() -> None:
    value = document()
    jobs, _, publisher = service(value)
    with pytest.raises(ValueError, match="UUIDs"):
        await jobs.enqueue_document("tenant", value.id, PIPELINE_VERSION)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pipeline version"):
        await jobs.enqueue_document(value.tenant_id, value.id, "bad")

    missing, _, _ = service(None)
    with pytest.raises(DocumentNotFound):
        await missing.enqueue_document(value.tenant_id, value.id, PIPELINE_VERSION)
    for status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
        unavailable = document(status)
        jobs, _, _ = service(unavailable)
        with pytest.raises(DocumentUnavailable):
            await jobs.enqueue_document(
                unavailable.tenant_id,
                unavailable.id,
                PIPELINE_VERSION,
            )
    publisher.publish.assert_not_awaited()
