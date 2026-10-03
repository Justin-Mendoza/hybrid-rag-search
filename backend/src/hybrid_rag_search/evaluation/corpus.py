"""Prepare reusable evaluation scopes using the existing durable ingestion pipeline."""

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.config import Settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.evaluation.datasets import EvaluationDataset, load_dataset
from hybrid_rag_search.evaluation.scifact import DEFAULT_OUTPUT, SUBSET_OUTPUT
from hybrid_rag_search.ingestion.jobs import IngestionJobService
from hybrid_rag_search.ingestion.pipeline import JobRunResult
from hybrid_rag_search.ingestion.runtime import configured_pipeline_version, run_configured_job
from hybrid_rag_search.models.content import Collection, Document, IngestionJob
from hybrid_rag_search.models.identity import Tenant
from hybrid_rag_search.opensearch_index import OpenSearchDocumentIndex, OpenSearchIndexManager
from hybrid_rag_search.storage import LocalFileStorage, storage_key

EVALUATION_TENANT = UUID("00000000-0000-4000-8000-000000000201")
DATASETS = {
    "synthetic-workspace": Path("datasets/synthetic-workspace/v1"),
    "scifact": DEFAULT_OUTPUT,
    "scifact-subset": SUBSET_OUTPUT,
}


@dataclass(frozen=True)
class EvaluationScope:
    tenant_id: UUID
    collection_id: UUID
    pipeline_version: str
    document_ids: dict[str, str]


def evaluation_scope(dataset: EvaluationDataset, settings: Settings) -> EvaluationScope:
    if any(record.type != "document" for record in dataset.corpus):
        raise ValueError("Day 15 evaluates document corpora only")
    pipeline = configured_pipeline_version(settings)
    collection = uuid5(EVALUATION_TENANT, f"{dataset.corpus_hash}:{pipeline}")
    return EvaluationScope(
        EVALUATION_TENANT,
        collection,
        pipeline,
        {str(uuid5(collection, record.id)): record.id for record in dataset.corpus},
    )


def selected_dataset(name: str) -> EvaluationDataset:
    if name not in DATASETS:
        raise ValueError("Unknown evaluation dataset")
    return load_dataset(DATASETS[name])


def evaluation_index_settings(dataset: EvaluationDataset, settings: Settings) -> Settings:
    """Keep evaluation index creation separate from demo aliases and manual rebuilds."""
    scope = evaluation_scope(dataset, settings)
    prefix = f"hybrid-rag-eval-{scope.collection_id}"
    return settings.model_copy(
        update={
            "opensearch_read_alias": f"{prefix}-read",
            "opensearch_write_alias": f"{prefix}-write",
        }
    )


async def ensure_evaluation_index(dataset: EvaluationDataset, settings: Settings) -> None:
    scope = evaluation_scope(dataset, settings)
    physical = f"hybrid-rag-eval-{scope.collection_id}"
    async with httpx.AsyncClient(base_url=settings.opensearch_url) as client:
        response = await client.head(f"/{physical}")
        if response.status_code != 404:
            response.raise_for_status()
            return
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        settings.opensearch_read_alias,
        settings.opensearch_write_alias,
        settings.cohere_embed_dimensions,
    )
    await manager.create(physical)
    await manager.point_write_alias(physical)
    await manager.promote_read_alias(physical)


class ManualActor:
    """The preparation CLI executes durable jobs inline; retries remain explicit."""

    def send(self, job_id: str) -> object:
        return job_id

    def send_with_options(self, *, args: tuple[str], delay: int) -> object:
        return args


class ManualPublisher:
    async def publish(self, job_id: UUID) -> None:
        pass


async def prepare_corpus(
    dataset: EvaluationDataset, settings: Settings, *, document_delay: float = 0
) -> EvaluationScope:
    if document_delay < 0:
        raise ValueError("Document delay must be nonnegative")
    settings = evaluation_index_settings(dataset, settings)
    await ensure_evaluation_index(dataset, settings)
    scope = evaluation_scope(dataset, settings)
    engine = create_async_engine(async_database_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    storage = LocalFileStorage(settings.storage_root)
    index = OpenSearchDocumentIndex(
        settings.opensearch_url, settings.opensearch_write_alias, settings.cohere_embed_dimensions
    )
    try:
        async with sessions.begin() as session:
            await session.execute(
                insert(Tenant)
                .values(id=scope.tenant_id, name="Evaluation", slug="evaluation")
                .on_conflict_do_nothing(index_elements=[Tenant.id])
            )
            await session.execute(
                insert(Collection)
                .values(
                    id=scope.collection_id,
                    tenant_id=scope.tenant_id,
                    name=f"Evaluation {dataset.manifest.dataset_id}",
                    slug=f"eval-{scope.collection_id}",
                    description=dataset.corpus_hash,
                )
                .on_conflict_do_nothing(index_elements=[Collection.id])
            )
        jobs = IngestionJobService(sessions, ManualPublisher())
        for number, record in enumerate(dataset.corpus, 1):
            document_id = uuid5(scope.collection_id, record.id)
            content = ((record.title + "\n\n" if record.title else "") + record.text).encode()
            digest = hashlib.sha256(content).hexdigest()
            key = storage_key(scope.tenant_id, document_id)
            async with sessions.begin() as session:
                existing = await session.get(Document, document_id)
                if existing is None:
                    storage.save(key, content)
                    session.add(
                        Document(
                            id=document_id,
                            tenant_id=scope.tenant_id,
                            collection_id=scope.collection_id,
                            source_key=record.id,
                            storage_key=key,
                            original_filename=f"{record.id}.txt",
                            media_type="text/plain",
                            content_hash=digest,
                            size_bytes=len(content),
                            source_metadata={**record.metadata, "evaluation_id": record.id},
                        )
                    )
                elif existing.content_hash != digest or existing.source_metadata != {
                    **record.metadata,
                    "evaluation_id": record.id,
                }:
                    raise ValueError("Prepared evaluation document differs from dataset")
            job = await jobs.enqueue_document(scope.tenant_id, document_id, scope.pipeline_version)
            if job.status != "succeeded":
                if document_delay:
                    await asyncio.sleep(document_delay)
                result = await run_configured_job(settings, ManualActor(), index, job.job_id)
                if result != JobRunResult.SUCCEEDED:
                    raise ValueError(
                        f"Ingestion {job.job_id}: {result}; inspect/recover durable job"
                    )
            if number % 100 == 0 or number == len(dataset.corpus):
                print(f"Prepared {number}/{len(dataset.corpus)} documents", flush=True)
    finally:
        await engine.dispose()
    await compact_corpus(dataset, settings)
    await verify_prepared(dataset, settings)
    return scope


async def compact_corpus(dataset: EvaluationDataset, settings: Settings) -> None:
    """Settle deleted promotion records before benchmarking a completed static corpus."""
    scope = evaluation_scope(dataset, settings)
    async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=300) as client:
        response = await client.post(
            f"/hybrid-rag-eval-{scope.collection_id}/_forcemerge",
            params={"max_num_segments": 1, "flush": "true"},
        )
        response.raise_for_status()


async def verify_prepared(dataset: EvaluationDataset, settings: Settings) -> int:
    """Fail closed on missing/stale jobs or index generations, including after a reset."""
    settings = evaluation_index_settings(dataset, settings)
    scope = evaluation_scope(dataset, settings)
    engine = create_async_engine(async_database_url(settings.database_url))
    try:
        async with async_sessionmaker(engine)() as session:
            jobs = (
                (
                    await session.execute(
                        select(IngestionJob).where(
                            IngestionJob.tenant_id == scope.tenant_id,
                            IngestionJob.document_id.in_([UUID(key) for key in scope.document_ids]),
                            IngestionJob.pipeline_version == scope.pipeline_version,
                            IngestionJob.status == "succeeded",
                        )
                    )
                )
                .scalars()
                .all()
            )
            if len(jobs) != len(dataset.corpus):
                raise ValueError("Evaluation corpus is not prepared; run evaluation prepare")
            expected: dict[str, tuple[str, int]] = {}
            for job in jobs:
                receipt = job.checkpoint.get("index")
                if (
                    not isinstance(receipt, dict)
                    or not isinstance(receipt.get("generation_id"), str)
                    or type(receipt.get("chunk_count")) is not int
                ):
                    raise ValueError("Completed ingestion has an invalid index receipt")
                expected[str(job.document_id)] = (receipt["generation_id"], receipt["chunk_count"])
    finally:
        await engine.dispose()
    async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=30) as client:
        response = await client.post(
            f"/{settings.opensearch_read_alias}/_search",
            json={
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": str(scope.tenant_id)}},
                            {"term": {"collection_id": str(scope.collection_id)}},
                            {"term": {"visibility": "ready"}},
                        ]
                    }
                },
                "aggs": {
                    "documents": {
                        "terms": {"field": "document_id", "size": len(expected)},
                        "aggs": {"generations": {"terms": {"field": "generation_id", "size": 2}}},
                    }
                },
            },
        )
        response.raise_for_status()
        buckets = response.json()["aggregations"]["documents"]["buckets"]
    actual = {
        item["key"]: ([g["key"] for g in item["generations"]["buckets"]], item["doc_count"])
        for item in buckets
    }
    if actual != {key: ([value[0]], value[1]) for key, value in expected.items()}:
        raise ValueError(
            "Evaluation index differs from completed ingestion; rebuild before running"
        )
    return sum(int(item["doc_count"]) for item in buckets)
