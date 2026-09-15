import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.documents import DocumentNotFound, DocumentService, DocumentUnavailable
from hybrid_rag_search.models.content import Collection, Document, DocumentStatus
from hybrid_rag_search.models.identity import Tenant
from hybrid_rag_search.storage import LocalFileStorage


@pytest.mark.integration
@pytest.mark.anyio
async def test_original_lifecycle_in_postgres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(async_database_url(get_settings().database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    storage = LocalFileStorage(tmp_path)
    service = DocumentService(sessions, storage)
    tenant_a, tenant_b, collection_a, collection_b = [uuid4() for _ in range(4)]
    try:
        async with sessions.begin() as session:
            session.add_all(
                [
                    Tenant(id=tenant_a, name="Storage A", slug=f"storage-{tenant_a}"),
                    Tenant(id=tenant_b, name="Storage B", slug=f"storage-{tenant_b}"),
                ]
            )
            await session.flush()
            session.add_all(
                [
                    Collection(id=collection_a, tenant_id=tenant_a, name="A", slug="a"),
                    Collection(id=collection_b, tenant_id=tenant_b, name="B", slug="b"),
                ]
            )
        uploads = await asyncio.gather(
            *[
                service.upload(tenant_a, collection_a, "../../original.txt", "text/plain", b"hello")
                for _ in range(3)
            ]
        )
        document_id = uploads[0].document_id
        assert len({result.document_id for result in uploads}) == 1
        assert sum(not result.duplicate for result in uploads) == 1
        assert await service.fetch(tenant_a, document_id) == b"hello"
        async with sessions() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.content_hash == hashlib.sha256(b"hello").hexdigest()
            assert document.size_bytes == 5
            assert document.media_type == "text/plain"
            assert document.status == DocumentStatus.PENDING
            assert document.original_filename == "../../original.txt"
            key = document.storage_key
        assert (tmp_path / key).read_bytes() == b"hello"
        for operation in (service.fetch, service.delete):
            with pytest.raises(DocumentNotFound):
                await operation(tenant_b, document_id)
        with pytest.raises(DocumentNotFound):
            await service.upload(tenant_b, collection_a, "x", "text/plain", b"hello")
        other = await service.upload(tenant_b, collection_b, "x", "text/plain", b"hello")
        assert other.document_id != document_id
        assert not other.duplicate

        original_delete = storage.delete

        def fail(_key: str) -> None:
            raise OSError("storage unavailable")

        monkeypatch.setattr(storage, "delete", fail)
        with pytest.raises(OSError):
            await service.delete(tenant_a, document_id)
        async with sessions() as session:
            document = await session.get(Document, document_id)
            assert document is not None and document.status == DocumentStatus.DELETING
        with pytest.raises(DocumentUnavailable):
            await service.fetch(tenant_a, document_id)
        monkeypatch.setattr(storage, "delete", original_delete)
        await service.delete(tenant_a, document_id)
        await service.delete(tenant_a, document_id)
        assert not (tmp_path / key).exists()
        async with sessions() as session:
            document = await session.get(Document, document_id)
            assert document is not None and document.status == DocumentStatus.DELETED
            assert document.deleted_at is not None
        replacement = await service.upload(tenant_a, collection_a, "x", "text/plain", b"hello")
        assert replacement.document_id != document_id and not replacement.duplicate
        assert await service.fetch(tenant_b, other.document_id) == b"hello"

        def fail_save(_key: str, _content: bytes) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(storage, "save", fail_save)
        with pytest.raises(OSError):
            await service.upload(tenant_a, collection_a, "bad", "text/plain", b"new bytes")
        async with sessions() as session:
            assert (
                await session.scalar(
                    select(Document).where(
                        Document.original_filename == "bad", Document.tenant_id == tenant_a
                    )
                )
                is None
            )
    finally:
        async with sessions.begin() as session:
            await session.execute(
                delete(Document).where(Document.tenant_id.in_([tenant_a, tenant_b]))
            )
            await session.execute(
                delete(Collection).where(Collection.tenant_id.in_([tenant_a, tenant_b]))
            )
            await session.execute(delete(Tenant).where(Tenant.id.in_([tenant_a, tenant_b])))
        await engine.dispose()
