import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.ingestion.claims import JobClaimer
from hybrid_rag_search.models.content import (
    Collection,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)
from hybrid_rag_search.models.identity import Tenant


@pytest.mark.integration
@pytest.mark.anyio
async def test_only_one_concurrent_worker_claims_a_queued_job() -> None:
    engine = create_async_engine(async_database_url(get_settings().database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    service = JobClaimer(sessions)
    tenant_id, collection_id, document_id, job_id = [uuid4() for _ in range(4)]
    try:
        async with sessions.begin() as session:
            session.add(Tenant(id=tenant_id, name="Claim Test", slug=f"claim-{tenant_id}"))
            await session.flush()
            session.add(
                Collection(id=collection_id, tenant_id=tenant_id, name="Claims", slug="claims")
            )
            await session.flush()
            session.add(
                Document(
                    id=document_id,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    source_key=str(document_id),
                    storage_key=f"{tenant_id}/{document_id}/original",
                    original_filename="claim.txt",
                    media_type="text/plain",
                    content_hash="a" * 64,
                    size_bytes=1,
                    status=DocumentStatus.PENDING,
                )
            )
            await session.flush()
            session.add(
                IngestionJob(
                    id=job_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    content_hash="a" * 64,
                    pipeline_version="pipe_" + "b" * 64,
                )
            )

        claims = await asyncio.gather(service.claim(job_id), service.claim(job_id))
        assert sum(claim is not None for claim in claims) == 1
        claimed = next(claim for claim in claims if claim is not None)
        assert claimed.attempts == 1
        assert claimed.stage is IngestionStage.PARSE

        async with sessions() as session:
            job = await session.get(IngestionJob, job_id)
            assert job is not None
            assert job.status == IngestionJobStatus.RUNNING
            assert job.attempts == 1
            assert job.started_at is not None
            assert job.lease_token == claimed.lease_token
            assert job.lease_expires_at == claimed.lease_expires_at

        renewed = await service.renew(job_id, claimed.lease_token)
        assert renewed is not None and renewed >= claimed.lease_expires_at
        assert await service.renew(job_id, uuid4()) is None
    finally:
        async with sessions.begin() as session:
            await session.execute(delete(IngestionJob).where(IngestionJob.id == job_id))
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.execute(delete(Collection).where(Collection.id == collection_id))
            await session.execute(delete(Tenant).where(Tenant.id == tenant_id))
        await engine.dispose()
