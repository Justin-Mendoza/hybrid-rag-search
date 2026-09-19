"""Recoverable parse-stage execution for a claimed ingestion job."""

import asyncio
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    decode_parsed_artifact,
    encode_parsed_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.models.content import Document, DocumentStatus, IngestionStage
from hybrid_rag_search.parsers.contracts import ParseRequest
from hybrid_rag_search.parsers.dispatcher import ParserDispatcher
from hybrid_rag_search.storage import FileStorage, storage_key


class SourceReadErrorCode(StrEnum):
    INVALID_STORAGE_KEY = "invalid_storage_key"
    ORIGINAL_MISSING = "original_missing"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"


class SourceReadError(RuntimeError):
    """Safe, classifiable failure while loading immutable source bytes."""

    def __init__(self, code: SourceReadErrorCode, message: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class ParseSource:
    collection_id: UUID
    content: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.collection_id, UUID):
            raise ValueError("Parse source collection ID must be a UUID")
        if not isinstance(self.content, bytes):
            raise ValueError("Parse source content must be bytes")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("Parse source media type must be nonblank")


class ParseSourceLoader(Protocol):
    async def load(self, claimed: ClaimedJob) -> ParseSource: ...


class PostgresParseSourceLoader:
    """Load and verify the exact immutable original named by a claimed job."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        originals: FileStorage,
    ) -> None:
        self.sessions = sessions
        self.originals = originals

    async def load(self, claimed: ClaimedJob) -> ParseSource:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Parse source loading requires a ClaimedJob")
        statement = select(
            Document.collection_id,
            Document.storage_key,
            Document.media_type,
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
        expected_key = storage_key(claimed.tenant_id, claimed.document_id)
        if row["storage_key"] != expected_key:
            raise SourceReadError(
                SourceReadErrorCode.INVALID_STORAGE_KEY,
                "Document storage key does not match its identity",
                retryable=False,
            )
        if row["content_hash"] != claimed.content_hash:
            raise SourceReadError(
                SourceReadErrorCode.CONTENT_HASH_MISMATCH,
                "Document content no longer matches the ingestion job",
                retryable=False,
            )
        try:
            content = await asyncio.to_thread(self.originals.fetch, expected_key)
        except FileNotFoundError:
            raise SourceReadError(
                SourceReadErrorCode.ORIGINAL_MISSING,
                "Original document is not available in storage",
                retryable=True,
            ) from None
        except OSError:
            raise SourceReadError(
                SourceReadErrorCode.STORAGE_UNAVAILABLE,
                "Original document storage is unavailable",
                retryable=True,
            ) from None
        if hashlib.sha256(content).hexdigest() != claimed.content_hash:
            raise SourceReadError(
                SourceReadErrorCode.CONTENT_HASH_MISMATCH,
                "Original document bytes do not match their content hash",
                retryable=False,
            )
        return ParseSource(row["collection_id"], content, row["media_type"])


class ParseStageRunner:
    """Recover or produce one parsed artifact, then checkpoint it."""

    def __init__(
        self,
        source_loader: ParseSourceLoader,
        parser: ParserDispatcher,
        artifacts: ArtifactStorage,
        checkpointer: JobCheckpointer,
    ) -> None:
        self.source_loader = source_loader
        self.parser = parser
        self.artifacts = artifacts
        self.checkpointer = checkpointer

    async def run(self, claimed: ClaimedJob) -> ClaimedJob | None:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Parse stage requires a ClaimedJob")
        if claimed.stage is not IngestionStage.PARSE:
            raise ValueError("Parse stage requires a job at the parse stage")
        identity = artifact_identity(
            ArtifactKind.PARSED,
            claimed.content_hash,
            claimed.pipeline_version,
        )
        key = artifact_key(claimed.tenant_id, claimed.id, ArtifactKind.PARSED, identity)
        reference = await asyncio.to_thread(self.artifacts.lookup, key)
        if reference is not None:
            content = await asyncio.to_thread(self.artifacts.fetch, reference)
            decode_parsed_artifact(content, identity)
        else:
            source = await self.source_loader.load(claimed)
            parsed = await asyncio.to_thread(
                self.parser.parse,
                ParseRequest(source.content, source.media_type),
            )
            content = encode_parsed_artifact(parsed, identity)
            reference = await asyncio.to_thread(self.artifacts.save, key, content)
        return await self.checkpointer.advance_artifact(claimed, reference)
