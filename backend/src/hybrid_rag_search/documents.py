"""Original-file lifecycle, independent of HTTP and future ingestion workers."""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.models.content import Collection, Document, DocumentStatus
from hybrid_rag_search.storage import FileStorage, storage_key


class DocumentNotFound(LookupError):
    """No document/collection exists within the supplied tenant scope."""


class DocumentUnavailable(ValueError):
    """The document is being deleted or has already been deleted."""


@dataclass(frozen=True)
class UploadResult:
    document_id: UUID
    duplicate: bool


class DocumentService:
    """Own transactions; callers must supply an already authorized tenant/scope.

    A collection row lock serializes duplicate checks, including concurrent
    uploads. Filesystem work runs outside the event loop. PostgreSQL and disk
    cannot commit atomically: a crash/commit failure may leave an orphan file,
    but must never cause deletion of a possibly committed original.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession], storage: FileStorage) -> None:
        self.sessions = sessions
        self.storage = storage

    async def upload(
        self,
        tenant_id: UUID,
        collection_id: UUID,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> UploadResult:
        if not filename.strip() or not media_type.strip():
            raise ValueError("Filename and media type must be nonblank")
        digest = hashlib.sha256(content).hexdigest()
        async with self.sessions.begin() as session:
            collection = await session.scalar(
                select(Collection)
                .where(Collection.tenant_id == tenant_id, Collection.id == collection_id)
                .with_for_update()
            )
            if collection is None:
                raise DocumentNotFound("Collection not found")
            duplicate = await session.scalar(
                select(Document).where(
                    Document.tenant_id == tenant_id,
                    Document.collection_id == collection_id,
                    Document.content_hash == digest,
                    Document.status.not_in([DocumentStatus.DELETING, DocumentStatus.DELETED]),
                )
            )
            if duplicate is not None:
                return UploadResult(duplicate.id, duplicate=True)
            document_id = uuid4()
            key = storage_key(tenant_id, document_id)
            session.add(
                Document(
                    id=document_id,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    source_key=str(document_id),
                    storage_key=key,
                    original_filename=filename,
                    media_type=media_type,
                    content_hash=digest,
                    size_bytes=len(content),
                    status=DocumentStatus.PENDING,
                )
            )
            await session.flush()
            await asyncio.to_thread(self.storage.save, key, content)
        return UploadResult(document_id, duplicate=False)

    async def _document(
        self, session: AsyncSession, tenant_id: UUID, document_id: UUID
    ) -> Document:
        document = await session.scalar(
            select(Document)
            .where(Document.tenant_id == tenant_id, Document.id == document_id)
            .with_for_update()
        )
        if document is None:
            raise DocumentNotFound("Document not found")
        # Defend against a corrupted/cross-tenant storage pointer as well as IDs.
        if document.storage_key != storage_key(tenant_id, document_id):
            raise ValueError("Document storage key does not match its identity")
        return document

    async def fetch(self, tenant_id: UUID, document_id: UUID) -> bytes:
        async with self.sessions.begin() as session:
            document = await self._document(session, tenant_id, document_id)
            if document.status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
                raise DocumentUnavailable("Document is deleted or deleting")
            return await asyncio.to_thread(self.storage.fetch, document.storage_key)

    async def delete(self, tenant_id: UUID, document_id: UUID) -> None:
        # Commit intent first. An I/O failure leaves a durable, retryable state.
        async with self.sessions.begin() as session:
            document = await self._document(session, tenant_id, document_id)
            if document.status == DocumentStatus.DELETED:
                return
            document.status = DocumentStatus.DELETING
        async with self.sessions.begin() as session:
            document = await self._document(session, tenant_id, document_id)
            await asyncio.to_thread(self.storage.delete, document.storage_key)
            document.status = DocumentStatus.DELETED
            document.deleted_at = datetime.now(UTC)
            document.error_message = None
