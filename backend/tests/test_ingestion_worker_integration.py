import re
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.chunking.contracts import DEFAULT_CHUNKING_CONFIG
from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.documents import DocumentService
from hybrid_rag_search.ingestion.artifacts import LocalArtifactStorage
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.chunk_stage import (
    ChunkStageRunner,
    PostgresDocumentIdentityResolver,
)
from hybrid_rag_search.ingestion.claims import JobClaimer
from hybrid_rag_search.ingestion.embed_stage import EmbeddingStageRunner
from hybrid_rag_search.ingestion.failures import JobFailureHandler
from hybrid_rag_search.ingestion.index_stage import (
    IndexStageRunner,
    PostgresIndexDocumentResolver,
)
from hybrid_rag_search.ingestion.indexing import FakeDocumentIndex
from hybrid_rag_search.ingestion.jobs import IngestionJobService
from hybrid_rag_search.ingestion.parse_stage import ParseStageRunner, PostgresParseSourceLoader
from hybrid_rag_search.ingestion.pipeline import IngestionPipeline, JobRunResult
from hybrid_rag_search.models.content import (
    Collection,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
)
from hybrid_rag_search.models.identity import Tenant
from hybrid_rag_search.parsers.dispatcher import default_parser_dispatcher
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider
from hybrid_rag_search.storage import LocalFileStorage
from hybrid_rag_search.tokenization import TokenSpan

PIPELINE_VERSION = "pipe_" + "b" * 64


class WordTokenizer:
    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(
            TokenSpan(index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\S+", text), start=1)
        )


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def embed(self, request):
        self.calls += 1
        return await super().embed(request)


def failing_pipeline(claimer, failures, retries, error: Exception) -> IngestionPipeline:
    parse_stage = AsyncMock()
    parse_stage.run.side_effect = error
    return IngestionPipeline(
        claimer,
        parse_stage,
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        failures,
        retries,
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_worker_success_retry_and_permanent_failure_are_durable(tmp_path: Path) -> None:
    engine = create_async_engine(async_database_url(get_settings().database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    originals = LocalFileStorage(tmp_path / "originals")
    artifacts = LocalArtifactStorage(tmp_path / "artifacts")
    documents = DocumentService(sessions, originals)
    publisher = AsyncMock()
    jobs = IngestionJobService(sessions, publisher)
    tenant_id, collection_id = uuid4(), uuid4()
    document_ids = []
    job_ids = []
    try:
        async with sessions.begin() as session:
            session.add(Tenant(id=tenant_id, name="Worker Test", slug=f"worker-{tenant_id}"))
            await session.flush()
            session.add(
                Collection(
                    id=collection_id,
                    tenant_id=tenant_id,
                    name="Worker",
                    slug="worker",
                )
            )

        upload = await documents.upload(
            tenant_id,
            collection_id,
            "success.txt",
            "text/plain",
            b"A durable ingestion document with enough words to embed.",
        )
        document_ids.append(upload.document_id)
        enqueued = await jobs.enqueue_document(tenant_id, upload.document_id, PIPELINE_VERSION)
        job_ids.append(enqueued.job_id)
        duplicate = await jobs.enqueue_document(tenant_id, upload.document_id, PIPELINE_VERSION)
        assert not duplicate.created and duplicate.job_id == enqueued.job_id

        claimer = JobClaimer(sessions)
        checkpointer = JobCheckpointer(sessions)
        provider = CountingEmbeddingProvider()
        pipeline = IngestionPipeline(
            claimer,
            ParseStageRunner(
                PostgresParseSourceLoader(sessions, originals),
                default_parser_dispatcher(),
                artifacts,
                checkpointer,
            ),
            ChunkStageRunner(
                PostgresDocumentIdentityResolver(sessions),
                WordTokenizer(),
                artifacts,
                checkpointer,
            ),
            EmbeddingStageRunner(
                provider,
                artifacts,
                claimer,
                checkpointer,
                model=provider.MODEL,
                dimensions=provider.dimensions,
                chunk_config_version=DEFAULT_CHUNKING_CONFIG.version,
            ),
            IndexStageRunner(
                PostgresIndexDocumentResolver(sessions),
                artifacts,
                FakeDocumentIndex(),
                checkpointer,
            ),
            JobFailureHandler(sessions, random_sample=lambda: 0.5),
            AsyncMock(),
        )
        assert await pipeline.run(enqueued.job_id) is JobRunResult.SUCCEEDED
        assert await pipeline.run(enqueued.job_id) is JobRunResult.NOT_CLAIMED
        assert provider.calls == 1
        reused = await jobs.enqueue_document(tenant_id, upload.document_id, PIPELINE_VERSION)
        assert reused.status is IngestionJobStatus.SUCCEEDED

        retry_upload = await documents.upload(
            tenant_id, collection_id, "retry.txt", "text/plain", b"retry bytes"
        )
        document_ids.append(retry_upload.document_id)
        retry_job = await jobs.enqueue_document(
            tenant_id, retry_upload.document_id, PIPELINE_VERSION
        )
        job_ids.append(retry_job.job_id)
        retries = AsyncMock()
        retry_pipeline = failing_pipeline(
            claimer,
            JobFailureHandler(sessions, random_sample=lambda: 0.5),
            retries,
            OSError("temporary storage failure"),
        )
        assert await retry_pipeline.run(retry_job.job_id) is JobRunResult.RETRY_SCHEDULED
        retries.schedule.assert_awaited_once()

        failed_upload = await documents.upload(
            tenant_id, collection_id, "failed.txt", "text/plain", b"failed bytes"
        )
        document_ids.append(failed_upload.document_id)
        failed_job = await jobs.enqueue_document(
            tenant_id, failed_upload.document_id, PIPELINE_VERSION
        )
        job_ids.append(failed_job.job_id)
        failed_pipeline = failing_pipeline(
            claimer,
            JobFailureHandler(sessions, random_sample=lambda: 0.5),
            AsyncMock(),
            ParseError(ParseErrorCode.CORRUPT_DOCUMENT, "Document is corrupt"),
        )
        assert await failed_pipeline.run(failed_job.job_id) is JobRunResult.FAILED

        async with sessions() as session:
            success = await session.get(IngestionJob, enqueued.job_id)
            success_document = await session.get(Document, upload.document_id)
            retry = await session.get(IngestionJob, retry_job.job_id)
            retry_document = await session.get(Document, retry_upload.document_id)
            failed = await session.get(IngestionJob, failed_job.job_id)
            failed_document = await session.get(Document, failed_upload.document_id)
        assert success is not None and success.status == IngestionJobStatus.SUCCEEDED
        assert success_document is not None and success_document.status == DocumentStatus.READY
        assert success_document.index_version == 1
        assert retry is not None and retry.status == IngestionJobStatus.QUEUED
        assert retry.next_attempt_at is not None and retry.attempts == 1
        assert retry_document is not None and retry_document.status == DocumentStatus.PROCESSING
        assert failed is not None and failed.status == IngestionJobStatus.FAILED
        assert failed.error_code == "parse_corrupt_document"
        assert failed_document is not None and failed_document.status == DocumentStatus.FAILED
    finally:
        async with sessions.begin() as session:
            await session.execute(delete(IngestionJob).where(IngestionJob.id.in_(job_ids)))
            await session.execute(delete(Document).where(Document.id.in_(document_ids)))
            await session.execute(delete(Collection).where(Collection.id == collection_id))
            await session.execute(delete(Tenant).where(Tenant.id == tenant_id))
        await engine.dispose()
