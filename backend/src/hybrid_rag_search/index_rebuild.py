"""Explicit two-phase rebuild of the derived OpenSearch chunks index."""

import argparse
import asyncio
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.ingestion.dispatch import DramatiqJobPublisher
from hybrid_rag_search.ingestion.jobs import IngestionJobService
from hybrid_rag_search.ingestion.runtime import configured_pipeline_version
from hybrid_rag_search.models.content import Document, DocumentStatus
from hybrid_rag_search.opensearch_index import OpenSearchIndexManager, physical_index_name
from hybrid_rag_search.worker import ingest_document


@dataclass(frozen=True)
class RebuildStartResult:
    index_name: str
    enqueued: int


async def start_rebuild(settings: Settings, build_id: str) -> RebuildStartResult:
    index_name = physical_index_name(settings.opensearch_index_schema_version, build_id)
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        settings.opensearch_read_alias,
        settings.opensearch_write_alias,
        settings.cohere_embed_dimensions,
    )
    await manager.create(index_name)
    await manager.point_write_alias(index_name)

    engine = create_async_engine(async_database_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            rows = await session.execute(
                select(Document.tenant_id, Document.id)
                .where(Document.status.not_in((DocumentStatus.DELETING, DocumentStatus.DELETED)))
                .order_by(Document.tenant_id, Document.id)
            )
            documents = tuple(rows.all())
        jobs = IngestionJobService(sessions, DramatiqJobPublisher(ingest_document))
        pipeline_version = configured_pipeline_version(settings)
        for tenant_id, document_id in documents:
            await jobs.enqueue_document(tenant_id, document_id, pipeline_version)
    finally:
        await engine.dispose()
    return RebuildStartResult(index_name, len(documents))


async def promote_rebuild(settings: Settings, index_name: str) -> None:
    engine = create_async_engine(async_database_url(settings.database_url))
    sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(engine)
    try:
        async with sessions() as session:
            unfinished = await session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.status.not_in((DocumentStatus.READY, DocumentStatus.DELETED)))
            )
        if unfinished:
            raise RuntimeError(f"Cannot promote rebuild while {unfinished} documents are not ready")
    finally:
        await engine.dispose()
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        settings.opensearch_read_alias,
        settings.opensearch_write_alias,
        settings.cohere_embed_dimensions,
    )
    await manager.promote_read_alias(index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "promote"))
    parser.add_argument("value", help="build ID for start; physical index name for promote")
    args = parser.parse_args()
    settings = get_settings()
    if args.action == "start":
        result = asyncio.run(start_rebuild(settings, args.value))
        print(f"rebuild started: index={result.index_name} enqueued={result.enqueued}")
    else:
        asyncio.run(promote_rebuild(settings, args.value))
        print(f"rebuild promoted: index={args.value}")


if __name__ == "__main__":  # pragma: no cover
    main()
