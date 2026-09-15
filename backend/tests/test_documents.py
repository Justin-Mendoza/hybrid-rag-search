from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.documents import DocumentNotFound, DocumentService, DocumentUnavailable
from hybrid_rag_search.models.content import Document, DocumentStatus
from hybrid_rag_search.storage import LocalFileStorage, storage_key


@pytest.fixture
def context(tmp_path: Path) -> tuple[DocumentService, AsyncMock, Document]:
    session = AsyncMock()
    session.add = MagicMock()
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    service = DocumentService(sessions, LocalFileStorage(tmp_path))
    tenant_id, document_id = uuid4(), uuid4()
    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        storage_key=storage_key(tenant_id, document_id),
        status=DocumentStatus.PENDING,
    )
    return service, session, document


@pytest.mark.anyio
async def test_upload_metadata_and_duplicate(
    context: tuple[DocumentService, AsyncMock, Document],
) -> None:
    service, session, document = context
    session.scalar.side_effect = [object(), None]
    result = await service.upload(
        document.tenant_id, uuid4(), "../name.txt", "text/plain", b"hello"
    )
    stored = session.add.call_args.args[0]
    assert result.document_id == stored.id
    assert not result.duplicate
    assert stored.original_filename == "../name.txt"
    assert stored.size_bytes == 5
    assert stored.media_type == "text/plain"
    assert stored.content_hash == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert stored.status == DocumentStatus.PENDING
    assert service.storage.fetch(stored.storage_key) == b"hello"
    session.scalar.side_effect = [object(), stored]
    duplicate = await service.upload(
        document.tenant_id, uuid4(), "other.txt", "text/plain", b"hello"
    )
    assert duplicate.document_id == result.document_id
    assert duplicate.duplicate
    assert session.add.call_count == 1


@pytest.mark.anyio
async def test_upload_validation(context: tuple[DocumentService, AsyncMock, Document]) -> None:
    service, session, document = context
    with pytest.raises(ValueError):
        await service.upload(document.tenant_id, uuid4(), " ", "text/plain", b"")
    session.scalar.return_value = None
    with pytest.raises(DocumentNotFound):
        await service.upload(document.tenant_id, uuid4(), "a.txt", "text/plain", b"")
    session.add.assert_not_called()


@pytest.mark.anyio
async def test_fetch_and_retryable_deletion(
    context: tuple[DocumentService, AsyncMock, Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    service, session, document = context
    session.scalar.return_value = document
    service.storage.save(document.storage_key, b"original")
    assert await service.fetch(document.tenant_id, document.id) == b"original"
    original_delete = service.storage.delete

    def fail(_key: str) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(service.storage, "delete", fail)
    with pytest.raises(OSError):
        await service.delete(document.tenant_id, document.id)
    assert document.status == DocumentStatus.DELETING
    with pytest.raises(DocumentUnavailable):
        await service.fetch(document.tenant_id, document.id)
    monkeypatch.setattr(service.storage, "delete", original_delete)
    await service.delete(document.tenant_id, document.id)
    assert document.status == DocumentStatus.DELETED
    assert document.deleted_at is not None
    with pytest.raises(FileNotFoundError):
        service.storage.fetch(document.storage_key)
    await service.delete(document.tenant_id, document.id)


@pytest.mark.anyio
async def test_missing_and_corrupt_pointer(
    context: tuple[DocumentService, AsyncMock, Document],
) -> None:
    service, session, document = context
    session.scalar.return_value = None
    with pytest.raises(DocumentNotFound):
        await service.fetch(document.tenant_id, document.id)
    document.storage_key = storage_key(uuid4(), document.id)
    session.scalar.return_value = document
    with pytest.raises(ValueError, match="identity"):
        await service.delete(document.tenant_id, document.id)
