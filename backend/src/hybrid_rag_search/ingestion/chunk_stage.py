"""Recoverable chunk-stage execution for a claimed ingestion job."""

import asyncio
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.chunking.chunker import chunk_document
from hybrid_rag_search.chunking.contracts import DEFAULT_CHUNKING_CONFIG, ChunkingConfig
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    decode_chunks_artifact,
    decode_parsed_artifact,
    encode_chunks_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.models.content import Document, DocumentStatus, IngestionStage
from hybrid_rag_search.tokenization import TextTokenizer


class DocumentIdentityResolver(Protocol):
    async def resolve(self, claimed: ClaimedJob) -> str: ...


class PostgresDocumentIdentityResolver:
    """Resolve search identity from current tenant, collection, and claimed content."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def resolve(self, claimed: ClaimedJob) -> str:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Document identity resolution requires a ClaimedJob")
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
        return DocumentContentIdentity(
            claimed.tenant_id,
            row["collection_id"],
            claimed.content_hash,
        ).document_id


class ChunkStageRunner:
    """Recover or produce chunks from the durable parsed checkpoint."""

    def __init__(
        self,
        identity_resolver: DocumentIdentityResolver,
        tokenizer: TextTokenizer,
        artifacts: ArtifactStorage,
        checkpointer: JobCheckpointer,
        *,
        config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
    ) -> None:
        self.identity_resolver = identity_resolver
        self.tokenizer = tokenizer
        self.artifacts = artifacts
        self.checkpointer = checkpointer
        self.config = config

    async def run(self, claimed: ClaimedJob) -> ClaimedJob | None:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Chunk stage requires a ClaimedJob")
        if claimed.stage is not IngestionStage.CHUNK:
            raise ValueError("Chunk stage requires a job at the chunk stage")

        parsed_identity = artifact_identity(
            ArtifactKind.PARSED,
            claimed.content_hash,
            claimed.pipeline_version,
        )
        expected_parsed_key = artifact_key(
            claimed.tenant_id,
            claimed.id,
            ArtifactKind.PARSED,
            parsed_identity,
        )
        try:
            parsed_reference = ArtifactRef.from_checkpoint(claimed.checkpoint.get("parsed"))
        except ValueError:
            raise ArtifactContractError("Parsed checkpoint reference is invalid") from None
        if parsed_reference.key != expected_parsed_key:
            raise ArtifactContractError("Parsed checkpoint reference does not match this job")

        document_id = await self.identity_resolver.resolve(claimed)
        chunks_identity = artifact_identity(
            ArtifactKind.CHUNKS,
            parsed_reference.sha256,
            document_id,
            self.config.version,
        )
        chunks_key = artifact_key(
            claimed.tenant_id,
            claimed.id,
            ArtifactKind.CHUNKS,
            chunks_identity,
        )
        chunks_reference = await asyncio.to_thread(self.artifacts.lookup, chunks_key)
        if chunks_reference is not None:
            chunks_content = await asyncio.to_thread(self.artifacts.fetch, chunks_reference)
            chunks = decode_chunks_artifact(chunks_content, chunks_identity)
            if (
                chunks[0].document_id != document_id
                or chunks[0].config_version != self.config.version
            ):
                raise ArtifactContractError("Chunks artifact does not match this pipeline")
        else:
            parsed_content = await asyncio.to_thread(
                self.artifacts.fetch,
                parsed_reference,
            )
            parsed = decode_parsed_artifact(parsed_content, parsed_identity)
            chunks = await asyncio.to_thread(
                chunk_document,
                document_id=document_id,
                document=parsed,
                tokenizer=self.tokenizer,
                config=self.config,
            )
            chunks_content = encode_chunks_artifact(chunks, chunks_identity)
            chunks_reference = await asyncio.to_thread(
                self.artifacts.save,
                chunks_key,
                chunks_content,
            )
        return await self.checkpointer.advance_artifact(claimed, chunks_reference)
