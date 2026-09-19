"""Build and atomically commit one complete document index generation."""

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    decode_chunks_artifact,
    decode_embeddings_artifact,
)
from hybrid_rag_search.ingestion.artifacts import ArtifactKind, ArtifactRef, ArtifactStorage
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.indexing import (
    DocumentIndex,
    IndexRecord,
    ReplaceDocumentRequest,
)
from hybrid_rag_search.models.content import Document, DocumentStatus, IngestionStage


@dataclass(frozen=True)
class IndexDocumentMetadata:
    collection_id: UUID


class IndexDocumentResolver(Protocol):
    async def resolve(self, claimed: ClaimedJob) -> IndexDocumentMetadata: ...


class PostgresIndexDocumentResolver:
    """Resolve tenant-scoped metadata while verifying the claimed content is current."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def resolve(self, claimed: ClaimedJob) -> IndexDocumentMetadata:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Index document resolution requires a ClaimedJob")
        statement = select(
            Document.collection_id,
            Document.content_hash,
            Document.status,
        ).where(Document.id == claimed.document_id, Document.tenant_id == claimed.tenant_id)
        async with self.sessions() as session:
            result = await session.execute(statement)
            row = result.mappings().one_or_none()
        if row is None:
            raise DocumentNotFound("Document not found")
        if row["status"] in (DocumentStatus.DELETING, DocumentStatus.DELETED):
            raise DocumentUnavailable("Document is deleted or deleting")
        if row["content_hash"] != claimed.content_hash:
            raise ValueError("Document content no longer matches the ingestion job")
        return IndexDocumentMetadata(row["collection_id"])


def _checkpoint_reference(
    claimed: ClaimedJob,
    checkpoint_name: str,
    kind: ArtifactKind,
) -> ArtifactRef:
    try:
        reference = ArtifactRef.from_checkpoint(claimed.checkpoint.get(checkpoint_name))
    except ValueError:
        raise ArtifactContractError(
            f"{checkpoint_name.capitalize()} checkpoint reference is invalid"
        ) from None
    expected_prefix = f"{claimed.tenant_id}/{claimed.id}/{kind.value}/"
    if not reference.key.startswith(expected_prefix):
        raise ArtifactContractError(
            f"{checkpoint_name.capitalize()} checkpoint reference does not match this job"
        )
    return reference


def _identity(reference: ArtifactRef) -> str:
    return reference.key.rsplit("/", maxsplit=1)[1].removesuffix(".json")


class IndexStageRunner:
    """Validate durable artifacts, replace one index generation, and finish the job."""

    def __init__(
        self,
        resolver: IndexDocumentResolver,
        artifacts: ArtifactStorage,
        index: DocumentIndex,
        checkpointer: JobCheckpointer,
    ) -> None:
        self.resolver = resolver
        self.artifacts = artifacts
        self.index = index
        self.checkpointer = checkpointer

    async def run(self, claimed: ClaimedJob) -> bool:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Index stage requires a ClaimedJob")
        if claimed.stage is not IngestionStage.INDEX:
            raise ValueError("Index stage requires a job at the index stage")

        chunks_reference = _checkpoint_reference(claimed, "chunks", ArtifactKind.CHUNKS)
        embeddings_reference = _checkpoint_reference(
            claimed,
            "embeddings",
            ArtifactKind.EMBEDDINGS,
        )
        chunks_content, embeddings_content = await asyncio.gather(
            asyncio.to_thread(self.artifacts.fetch, chunks_reference),
            asyncio.to_thread(self.artifacts.fetch, embeddings_reference),
        )
        chunks = decode_chunks_artifact(chunks_content, _identity(chunks_reference))
        embeddings = decode_embeddings_artifact(
            embeddings_content,
            _identity(embeddings_reference),
        )
        if embeddings.document_id != chunks[0].document_id:
            raise ArtifactContractError("Index artifacts use different document identities")
        if tuple(item.chunk_id for item in embeddings.embeddings) != tuple(
            chunk.chunk_id for chunk in chunks
        ):
            raise ArtifactContractError("Index artifacts use different chunk order")

        metadata = await self.resolver.resolve(claimed)
        request = ReplaceDocumentRequest(
            tenant_id=claimed.tenant_id,
            collection_id=metadata.collection_id,
            document_id=claimed.document_id,
            content_hash=claimed.content_hash,
            pipeline_version=claimed.pipeline_version,
            embedding_model=embeddings.model,
            embedding_dimensions=embeddings.dimensions,
            embedding_adapter=embeddings.adapter,
            records=tuple(
                IndexRecord(chunk, embedding)
                for chunk, embedding in zip(chunks, embeddings.embeddings, strict=True)
            ),
        )
        receipt = await self.index.replace_document(request)
        return await self.checkpointer.complete_index(claimed, receipt)
