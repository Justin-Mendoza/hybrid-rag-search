"""Composition root for the configured ingestion worker and recovery pass."""

import random
from uuid import UUID

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.chunking.contracts import DEFAULT_CHUNKING_CONFIG
from hybrid_rag_search.config import Settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.ingestion.artifacts import LocalArtifactStorage
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.chunk_stage import (
    ChunkStageRunner,
    PostgresDocumentIdentityResolver,
)
from hybrid_rag_search.ingestion.claims import JobClaimer
from hybrid_rag_search.ingestion.dispatch import (
    ActorSender,
    DramatiqJobPublisher,
    RecoveryDispatcher,
)
from hybrid_rag_search.ingestion.embed_stage import EmbeddingStageRunner
from hybrid_rag_search.ingestion.failures import JobFailureHandler
from hybrid_rag_search.ingestion.identity import PipelineConfig
from hybrid_rag_search.ingestion.index_stage import (
    IndexStageRunner,
    PostgresIndexDocumentResolver,
)
from hybrid_rag_search.ingestion.indexing import DocumentIndex
from hybrid_rag_search.ingestion.parse_stage import ParseStageRunner, PostgresParseSourceLoader
from hybrid_rag_search.ingestion.pipeline import IngestionPipeline, JobRunResult
from hybrid_rag_search.ingestion.recovery import (
    ExpiredLeaseRecovery,
    IngestionRecovery,
    RecoveryRunResult,
)
from hybrid_rag_search.parsers.dispatcher import default_parser_dispatcher
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.storage import LocalFileStorage
from hybrid_rag_search.tokenization import load_configured_tokenizer


def configured_pipeline_version(settings: Settings) -> str:
    """Identify every configured input that changes searchable output."""

    return PipelineConfig(
        parser_name="default-parser-set",
        parser_version="plain-text:1,markdown-it-py:1,stdlib-html:1,pypdf:1",
        chunk_config_version=DEFAULT_CHUNKING_CONFIG.version,
        embedding_model=settings.cohere_embed_model,
        embedding_dimensions=settings.cohere_embed_dimensions,
        embedding_input_type="search_document",
        embedding_type="float",
        index_schema_version=settings.opensearch_index_schema_version,
    ).version


async def run_configured_job(
    settings: Settings,
    actor: ActorSender,
    index: DocumentIndex,
    job_id: UUID,
) -> JobRunResult:
    """Build short-lived provider resources around one durable job attempt."""

    engine = create_async_engine(async_database_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    publisher = DramatiqJobPublisher(actor)
    claimer = JobClaimer(sessions)
    checkpointer = JobCheckpointer(sessions)
    artifacts = LocalArtifactStorage(settings.ingestion_artifact_root)
    try:
        async with configured_cohere_embeddings(settings) as provider:
            pipeline = IngestionPipeline(
                claimer,
                ParseStageRunner(
                    PostgresParseSourceLoader(
                        sessions,
                        LocalFileStorage(settings.storage_root),
                    ),
                    default_parser_dispatcher(),
                    artifacts,
                    checkpointer,
                ),
                ChunkStageRunner(
                    PostgresDocumentIdentityResolver(sessions),
                    load_configured_tokenizer(settings),
                    artifacts,
                    checkpointer,
                ),
                EmbeddingStageRunner(
                    provider,
                    artifacts,
                    claimer,
                    checkpointer,
                    model=settings.cohere_embed_model,
                    dimensions=settings.cohere_embed_dimensions,
                    chunk_config_version=DEFAULT_CHUNKING_CONFIG.version,
                ),
                IndexStageRunner(
                    PostgresIndexDocumentResolver(sessions),
                    artifacts,
                    index,
                    checkpointer,
                ),
                JobFailureHandler(sessions, random_sample=random.random),
                publisher,
            )
            return await pipeline.run(job_id)
    finally:
        await engine.dispose()


async def run_configured_recovery(
    settings: Settings,
    actor: ActorSender,
) -> RecoveryRunResult:
    """Repair expired ownership and republish eligible durable work once."""

    engine = create_async_engine(async_database_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        publisher = DramatiqJobPublisher(actor)
        return await IngestionRecovery(
            ExpiredLeaseRecovery(sessions),
            RecoveryDispatcher(sessions, publisher),
        ).run()
    finally:
        await engine.dispose()
