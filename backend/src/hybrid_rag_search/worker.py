"""Dramatiq entry point for durable document ingestion."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID

import dramatiq
from dramatiq import Broker, Middleware, Worker
from dramatiq.brokers.redis import RedisBroker

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.ingestion.recovery import RecoveryRunResult
from hybrid_rag_search.ingestion.runtime import run_configured_job, run_configured_recovery
from hybrid_rag_search.opensearch_index import OpenSearchDocumentIndex

settings = get_settings()
broker = RedisBroker(url=settings.redis_url)
dramatiq.set_broker(broker)
document_index = OpenSearchDocumentIndex(
    settings.opensearch_url,
    settings.opensearch_write_alias,
    settings.cohere_embed_dimensions,
)


@dramatiq.actor(max_retries=0)
def ingest_document(job_id: str) -> str:
    """Run one claimed attempt; durable PostgreSQL state owns further retries."""

    parsed_job_id = UUID(job_id)
    result = asyncio.run(
        run_configured_job(settings, ingest_document, document_index, parsed_job_id)
    )
    return result.value


@dramatiq.actor(max_retries=0)
def heartbeat() -> str:
    """Minimal actor proving that the worker can import application code."""

    return "ok"


class StartupRecoveryMiddleware(Middleware):
    """Run one bounded recovery pass when the worker becomes available."""

    def __init__(
        self,
        recover: Callable[[], Coroutine[Any, Any, RecoveryRunResult]],
    ) -> None:
        self.recover = recover

    def after_worker_boot(self, broker: Broker, worker: Worker) -> None:
        del broker, worker
        asyncio.run(self.recover())


async def recover_at_startup() -> RecoveryRunResult:
    return await run_configured_recovery(settings, ingest_document)


broker.add_middleware(StartupRecoveryMiddleware(recover_at_startup))
