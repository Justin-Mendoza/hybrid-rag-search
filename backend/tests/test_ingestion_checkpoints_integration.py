from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.ingestion.artifacts import ArtifactKind, ArtifactRef, artifact_key
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
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
async def test_only_current_lease_advances_a_durable_checkpoint() -> None:
    engine = create_async_engine(async_database_url(get_settings().database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    claimer = JobClaimer(sessions)
    checkpointer = JobCheckpointer(sessions)
    tenant_id, collection_id, document_id, job_id = [uuid4() for _ in range(4)]
    try:
        async with sessions.begin() as session:
            session.add(Tenant(id=tenant_id, name="Checkpoint Test", slug=f"point-{tenant_id}"))
            await session.flush()
            session.add(
                Collection(id=collection_id, tenant_id=tenant_id, name="Checkpoints", slug="points")
            )
            await session.flush()
            session.add(
                Document(
                    id=document_id,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    source_key=str(document_id),
                    storage_key=f"{tenant_id}/{document_id}/original",
                    original_filename="checkpoint.txt",
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

        claimed = await claimer.claim(job_id)
        assert claimed is not None
        artifact = ArtifactRef(
            artifact_key(tenant_id, job_id, ArtifactKind.PARSED, "c" * 64),
            "d" * 64,
            123,
        )
        advanced = await checkpointer.advance_artifact(claimed, artifact)
        assert advanced is not None
        assert advanced.stage is IngestionStage.CHUNK
        assert advanced.checkpoint["parsed"] == artifact.checkpoint_value()
        assert advanced.lease_expires_at >= claimed.lease_expires_at

        assert await checkpointer.advance_artifact(claimed, artifact) is None
        async with sessions() as session:
            job = await session.get(IngestionJob, job_id)
            assert job is not None
            assert job.status == IngestionJobStatus.RUNNING
            assert job.stage == IngestionStage.CHUNK
            assert job.checkpoint == advanced.checkpoint
    finally:
        async with sessions.begin() as session:
            await session.execute(delete(IngestionJob).where(IngestionJob.id == job_id))
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.execute(delete(Collection).where(Collection.id == collection_id))
            await session.execute(delete(Tenant).where(Tenant.id == tenant_id))
        await engine.dispose()
