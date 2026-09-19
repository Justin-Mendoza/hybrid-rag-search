"""Create or reuse durable ingestion work before publishing its Redis hint."""

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.dispatch import JobPublisher
from hybrid_rag_search.models.content import (
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
)


@dataclass(frozen=True)
class EnqueueResult:
    job_id: UUID
    created: bool
    status: IngestionJobStatus


class IngestionJobService:
    """Persist idempotent work and publish only after its transaction commits."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        publisher: JobPublisher,
    ) -> None:
        self.sessions = sessions
        self.publisher = publisher

    async def enqueue_document(
        self,
        tenant_id: UUID,
        document_id: UUID,
        pipeline_version: str,
    ) -> EnqueueResult:
        if not isinstance(tenant_id, UUID) or not isinstance(document_id, UUID):
            raise ValueError("Ingestion tenant and document IDs must be UUIDs")
        if not isinstance(pipeline_version, str) or not re.fullmatch(
            r"pipe_[0-9a-f]{64}", pipeline_version
        ):
            raise ValueError("Ingestion pipeline version must be a stable identity")

        async with self.sessions.begin() as session:
            document = await session.scalar(
                select(Document)
                .where(Document.tenant_id == tenant_id, Document.id == document_id)
                .with_for_update()
            )
            if document is None:
                raise DocumentNotFound("Document not found")
            if document.status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
                raise DocumentUnavailable("Document is deleted or deleting")

            statement = (
                insert(IngestionJob)
                .values(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    content_hash=document.content_hash,
                    pipeline_version=pipeline_version,
                )
                .on_conflict_do_nothing(constraint="uq_ingestion_jobs_document_recipe")
                .returning(IngestionJob.id)
            )
            job_id = await session.scalar(statement)
            created = job_id is not None
            if created:
                status = IngestionJobStatus.QUEUED
                await session.execute(
                    update(Document)
                    .where(Document.tenant_id == tenant_id, Document.id == document_id)
                    .values(status=DocumentStatus.PROCESSING, error_message=None)
                )
            else:
                existing = (
                    await session.execute(
                        select(IngestionJob.id, IngestionJob.status).where(
                            IngestionJob.tenant_id == tenant_id,
                            IngestionJob.document_id == document_id,
                            IngestionJob.content_hash == document.content_hash,
                            IngestionJob.pipeline_version == pipeline_version,
                        )
                    )
                ).one()
                job_id = existing.id
                status = IngestionJobStatus(existing.status)

        assert job_id is not None
        if status is IngestionJobStatus.QUEUED:
            await self.publisher.publish(job_id)
        return EnqueueResult(job_id, created, status)
