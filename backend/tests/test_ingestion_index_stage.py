from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    ChunkEmbedding,
    EmbeddingsArtifact,
    encode_chunks_artifact,
    encode_embeddings_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactRef,
    LocalArtifactStorage,
    artifact_key,
)
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.index_stage import (
    IndexDocumentMetadata,
    IndexStageRunner,
    PostgresIndexDocumentResolver,
)
from hybrid_rag_search.ingestion.indexing import FakeDocumentIndex
from hybrid_rag_search.models.content import DocumentStatus, IngestionStage
from hybrid_rag_search.parsers.contracts import SourceLocator

NOW = datetime(2026, 9, 19, tzinfo=UTC)


def claimed() -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash="a" * 64,
        pipeline_version="pipe_" + "b" * 64,
        stage=IngestionStage.INDEX,
        attempts=1,
        max_attempts=3,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=NOW + timedelta(minutes=5),
    )


def runner(tmp_path: Path):
    job = claimed()
    collection_id = uuid4()
    content_id = DocumentContentIdentity(job.tenant_id, collection_id, job.content_hash).document_id
    chunks = tuple(
        Chunk(
            content_id,
            "chunkcfg_" + "c" * 64,
            order,
            f"chunk {order}",
            f"chunk {order}",
            2,
            (SourceSpan(SourceLocator(order), 0, 7),),
        )
        for order in (1, 2)
    )
    embeddings = EmbeddingsArtifact(
        content_id,
        "embed-test",
        2,
        "fake",
        tuple(ChunkEmbedding(chunk.chunk_id, (float(chunk.order), 0.0)) for chunk in chunks),
    )
    artifacts = LocalArtifactStorage(tmp_path)
    chunks_identity = "d" * 64
    embeddings_identity = "e" * 64
    chunks_ref = artifacts.save(
        artifact_key(job.tenant_id, job.id, ArtifactKind.CHUNKS, chunks_identity),
        encode_chunks_artifact(chunks, chunks_identity),
    )
    embeddings_ref = artifacts.save(
        artifact_key(job.tenant_id, job.id, ArtifactKind.EMBEDDINGS, embeddings_identity),
        encode_embeddings_artifact(embeddings, embeddings_identity),
    )
    job = replace(
        job,
        checkpoint={
            "chunks": chunks_ref.checkpoint_value(),
            "embeddings": embeddings_ref.checkpoint_value(),
        },
    )
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=IndexDocumentMetadata(collection_id))
    checkpointer = AsyncMock()
    checkpointer.complete_index.return_value = True
    index = FakeDocumentIndex()
    return (
        IndexStageRunner(resolver, artifacts, index, checkpointer),
        job,
        resolver,
        checkpointer,
        index,
    )


@pytest.mark.anyio
async def test_index_stage_replaces_complete_document_and_finishes_job(tmp_path: Path) -> None:
    stage, job, resolver, checkpointer, index = runner(tmp_path)
    assert await stage.run(job)
    request = index.active_document(job.tenant_id, job.document_id)
    assert request is not None
    assert tuple(record.chunk.order for record in request.records) == (1, 2)
    assert tuple(record.embedding.chunk_id for record in request.records) == tuple(
        record.chunk.chunk_id for record in request.records
    )
    resolver.resolve.assert_awaited_once_with(job)
    receipt = checkpointer.complete_index.await_args.args[1]
    assert receipt.generation_id == request.generation_id


@pytest.mark.anyio
async def test_index_stage_returns_false_when_completion_loses_lease(tmp_path: Path) -> None:
    stage, job, _, checkpointer, index = runner(tmp_path)
    checkpointer.complete_index.return_value = False
    assert not await stage.run(job)
    assert index.active_document(job.tenant_id, job.document_id) is not None


@pytest.mark.anyio
async def test_index_stage_validates_job_and_stage(tmp_path: Path) -> None:
    stage, job, *_ = runner(tmp_path)
    with pytest.raises(ValueError, match="ClaimedJob"):
        await stage.run("job")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index stage"):
        await stage.run(replace(job, stage=IngestionStage.EMBED))


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["chunks", "embeddings"])
async def test_index_stage_rejects_invalid_or_cross_job_reference(
    tmp_path: Path, name: str
) -> None:
    stage, job, *_ = runner(tmp_path)
    invalid = replace(job, checkpoint={**job.checkpoint, name: {"key": "bad"}})
    with pytest.raises(ValueError, match="checkpoint reference is invalid"):
        await stage.run(invalid)

    reference = ArtifactRef.from_checkpoint(job.checkpoint[name])
    other = replace(reference, key=reference.key.replace(str(job.id), str(uuid4())))
    cross_job = replace(job, checkpoint={**job.checkpoint, name: other.checkpoint_value()})
    with pytest.raises(ValueError, match="does not match this job"):
        await stage.run(cross_job)


@pytest.mark.anyio
async def test_index_stage_rejects_mismatched_artifact_document(tmp_path: Path) -> None:
    stage, job, _, _, _ = runner(tmp_path)
    # A separately valid artifact is easier to construct than mutating immutable storage.
    chunks_reference = ArtifactRef.from_checkpoint(job.checkpoint["chunks"])
    chunks_content = stage.artifacts.fetch(chunks_reference)
    from hybrid_rag_search.ingestion.artifact_payloads import decode_chunks_artifact

    chunks = decode_chunks_artifact(
        chunks_content,
        chunks_reference.key.rsplit("/", 1)[1].removesuffix(".json"),
    )
    mismatched = EmbeddingsArtifact(
        "doc_" + "f" * 64,
        "embed-test",
        2,
        "fake",
        tuple(ChunkEmbedding(chunk.chunk_id, (1.0, 0.0)) for chunk in chunks),
    )
    different_identity = "f" * 64
    different_ref = stage.artifacts.save(
        artifact_key(job.tenant_id, job.id, ArtifactKind.EMBEDDINGS, different_identity),
        encode_embeddings_artifact(mismatched, different_identity),
    )
    with pytest.raises(ValueError, match="different document identities"):
        await stage.run(
            replace(
                job,
                checkpoint={**job.checkpoint, "embeddings": different_ref.checkpoint_value()},
            )
        )


@pytest.mark.anyio
async def test_index_stage_rejects_mismatched_chunk_order(tmp_path: Path) -> None:
    stage, job, *_ = runner(tmp_path)
    chunks_reference = ArtifactRef.from_checkpoint(job.checkpoint["chunks"])
    from hybrid_rag_search.ingestion.artifact_payloads import decode_chunks_artifact

    chunks = decode_chunks_artifact(
        stage.artifacts.fetch(chunks_reference),
        chunks_reference.key.rsplit("/", 1)[1].removesuffix(".json"),
    )
    mismatched = EmbeddingsArtifact(
        chunks[0].document_id,
        "embed-test",
        2,
        "fake",
        tuple(ChunkEmbedding(chunk.chunk_id, (1.0, 0.0)) for chunk in reversed(chunks)),
    )
    identity = "f" * 64
    reference = stage.artifacts.save(
        artifact_key(job.tenant_id, job.id, ArtifactKind.EMBEDDINGS, identity),
        encode_embeddings_artifact(mismatched, identity),
    )
    with pytest.raises(ValueError, match="different chunk order"):
        await stage.run(
            replace(job, checkpoint={**job.checkpoint, "embeddings": reference.checkpoint_value()})
        )


def resolver(row: dict[str, object] | None):
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = row
    session = AsyncMock()
    session.execute.return_value = result
    sessions = MagicMock()
    sessions.return_value.__aenter__.return_value = session
    return PostgresIndexDocumentResolver(sessions), session


@pytest.mark.anyio
async def test_postgres_index_document_resolver_validates_current_document() -> None:
    job = claimed()
    collection_id = uuid4()
    service, session = resolver(
        {
            "collection_id": collection_id,
            "content_hash": job.content_hash,
            "status": DocumentStatus.PROCESSING,
        }
    )
    assert await service.resolve(job) == IndexDocumentMetadata(collection_id)
    sql = str(session.execute.await_args.args[0].compile())
    assert "documents.tenant_id" in sql
    with pytest.raises(ValueError, match="ClaimedJob"):
        await service.resolve("job")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_postgres_index_document_resolver_rejects_missing_deleted_or_changed() -> None:
    job = claimed()
    service, _ = resolver(None)
    with pytest.raises(DocumentNotFound):
        await service.resolve(job)
    for status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
        service, _ = resolver(
            {"collection_id": uuid4(), "content_hash": job.content_hash, "status": status}
        )
        with pytest.raises(DocumentUnavailable):
            await service.resolve(job)
    service, _ = resolver(
        {
            "collection_id": uuid4(),
            "content_hash": "f" * 64,
            "status": DocumentStatus.PROCESSING,
        }
    )
    with pytest.raises(ValueError, match="no longer matches"):
        await service.resolve(job)
