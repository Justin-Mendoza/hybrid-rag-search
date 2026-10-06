"""Real PostgreSQL/OpenSearch access checks; all provider calls are deterministic fakes."""

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from test_hybrid_retrieval_integration import indexed_document

from hybrid_rag_search import api
from hybrid_rag_search import search as search_module
from hybrid_rag_search.authorization import Authorization
from hybrid_rag_search.chunking.contracts import SourceSpan
from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.documents import DocumentService
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import IndexRecord
from hybrid_rag_search.main import create_app
from hybrid_rag_search.models import CollectionGrant, Membership, User
from hybrid_rag_search.opensearch_index import OpenSearchDocumentIndex, OpenSearchIndexManager
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider, FakeRerankingProvider
from hybrid_rag_search.seed import (
    DEMO_COLLECTION_ID,
    DEMO_TENANT_ID,
    DEMO_USER_ID,
    DEMO_USER_IDS,
    ENGINEERING_COLLECTION_ID,
    ENGINEERING_USER_ID,
    HR_COLLECTION_ID,
    HR_USER_ID,
    RESTRICTED_USER_ID,
    SECOND_COLLECTION_ID,
    SECOND_TENANT_ID,
    SECOND_USER_ID,
    seed_statements,
)
from hybrid_rag_search.storage import LocalFileStorage


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_demo_api_enforces_real_permissions_across_all_modes_and_direct_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    index_name = f"hybrid-rag-chunks-v3-auth-{uuid4().hex[:12]}"
    settings = get_settings().model_copy(
        update={
            "opensearch_read_alias": index_name,
            "cohere_embed_model": FakeEmbeddingProvider.MODEL,
            "cohere_embed_dimensions": 3,
            "storage_root": tmp_path / "originals",
        }
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    seen_candidates = []

    @asynccontextmanager
    async def embeddings(_settings):
        yield FakeEmbeddingProvider(3)

    class Reranking(FakeRerankingProvider):
        async def rerank(self, request, **kwargs):
            seen_candidates.append(tuple(item.text for item in request.candidates))
            return await super().rerank(request, **kwargs)

    @asynccontextmanager
    async def reranking(_settings):
        yield Reranking()

    monkeypatch.setattr(search_module, "configured_cohere_embeddings", embeddings)
    monkeypatch.setattr(search_module, "configured_cohere_reranking", reranking)
    engine = create_async_engine(async_database_url(settings.database_url))
    manager = OpenSearchIndexManager(settings.opensearch_url, "unused-read", "unused-write", 3)
    try:
        await manager.create(index_name)
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                for _ in range(2):
                    for statement in seed_statements():
                        await connection.execute(statement)
                sessions = async_sessionmaker(
                    connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
                )
                async with sessions() as session:
                    assert (
                        len(
                            list(
                                await session.scalars(
                                    select(User.id).where(User.id.in_(DEMO_USER_IDS))
                                )
                            )
                        )
                        == 5
                    )
                auth = Authorization(sessions)
                originals = DocumentService(sessions, LocalFileStorage(settings.storage_root))
                index = OpenSearchDocumentIndex(settings.opensearch_url, index_name, 3)
                docs = {}
                texts = {}
                for number, (tenant, collection, label) in enumerate(
                    (
                        (DEMO_TENANT_ID, DEMO_COLLECTION_ID, "handbook"),
                        (DEMO_TENANT_ID, ENGINEERING_COLLECTION_ID, "engineeringsecret"),
                        (DEMO_TENANT_ID, HR_COLLECTION_ID, "hrsecret"),
                        (SECOND_TENANT_ID, SECOND_COLLECTION_ID, "othersecret"),
                    )
                ):
                    text = f"Password recovery policy {label}."
                    uploaded = await originals.upload(
                        tenant, collection, f"{label}.txt", "text/plain", text.encode()
                    )
                    request = indexed_document(
                        tenant, collection, author=label, suffix=chr(ord("a") + number)
                    )
                    chunk = replace(
                        request.records[0].chunk,
                        content_text=text,
                        embedding_text=text,
                        source_spans=(SourceSpan(SourceLocator(1), 0, len(text)),),
                    )
                    await index.replace_document(
                        replace(
                            request,
                            document_id=uploaded.document_id,
                            embedding_model=FakeEmbeddingProvider.MODEL,
                            records=(
                                IndexRecord(chunk, ChunkEmbedding(chunk.chunk_id, (1.0, 0.0, 0.0))),
                            ),
                        )
                    )
                    docs[collection] = uploaded.document_id
                    texts[collection] = text
                app = create_app()
                app.dependency_overrides[api.authorization] = lambda: auth
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    assert (await client.get("/v1/demo-users")).status_code == 200
                    assert len((await client.get("/v1/demo-users")).json()) == 5
                    request = {
                        "workspace_id": str(DEMO_TENANT_ID),
                        "query": "password recovery",
                        "debug": True,
                    }
                    headers = {"X-Demo-User-ID": str(ENGINEERING_USER_ID)}
                    assert (await client.post("/v1/search", json=request)).status_code == 401
                    assert (
                        await client.post(
                            "/v1/search", headers={"X-Demo-User-ID": str(uuid4())}, json=request
                        )
                    ).status_code == 401
                    expected = {
                        DEMO_USER_ID: (
                            DEMO_COLLECTION_ID,
                            ENGINEERING_COLLECTION_ID,
                            HR_COLLECTION_ID,
                        ),
                        ENGINEERING_USER_ID: (DEMO_COLLECTION_ID, ENGINEERING_COLLECTION_ID),
                        HR_USER_ID: (DEMO_COLLECTION_ID, HR_COLLECTION_ID),
                        RESTRICTED_USER_ID: (DEMO_COLLECTION_ID,),
                        SECOND_USER_ID: (SECOND_COLLECTION_ID,),
                    }
                    for user, collections in expected.items():
                        tenant = SECOND_TENANT_ID if user == SECOND_USER_ID else DEMO_TENANT_ID
                        headers = {"X-Demo-User-ID": str(user)}
                        workspaces = (await client.get("/v1/workspaces", headers=headers)).json()
                        assert [row["id"] for row in workspaces] == [str(tenant)]
                        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                            seen_candidates.clear()
                            response = await client.post(
                                "/v1/search",
                                headers=headers,
                                json={**request, "workspace_id": str(tenant), "mode": mode},
                            )
                            assert response.status_code == 200, response.text
                            body = response.json()
                            assert {UUID(item["collection_id"]) for item in body["chunks"]} == set(
                                collections
                            )
                            for denied in set(texts) - set(collections):
                                assert texts[denied] not in response.text
                                assert all(
                                    texts[denied] not in candidates
                                    for candidates in seen_candidates
                                )
                            for item in body["chunks"]:
                                assert item["tenant_id"] == str(tenant)
                        for collection, document in docs.items():
                            path = f"/v1/workspaces/{tenant}/documents/{document}/content"
                            response = await client.get(path, headers=headers)
                            if collection in collections:
                                assert (
                                    response.status_code == 200
                                    and response.text == texts[collection]
                                )
                            else:
                                assert response.status_code == 404 and response.json() == {
                                    "detail": "Document not found"
                                }
                        foreign = DEMO_TENANT_ID if tenant == SECOND_TENANT_ID else SECOND_TENANT_ID
                        assert (
                            await client.post(
                                "/v1/search",
                                headers=headers,
                                json={**request, "workspace_id": str(foreign)},
                            )
                        ).status_code == 403
                    # Both privileged roles work without collection grants.
                    await connection.execute(
                        delete(CollectionGrant).where(
                            CollectionGrant.membership_id == UUID(int=DEMO_USER_ID.int + 1)
                        )
                    )
                    for role in ("admin", "owner"):
                        await connection.execute(
                            update(Membership)
                            .where(Membership.user_id == DEMO_USER_ID)
                            .values(role=role)
                        )
                        assert (
                            len((await auth.scope(DEMO_USER_ID, DEMO_TENANT_ID)).collection_ids)
                            == 3
                        )
                    headers = {"X-Demo-User-ID": str(ENGINEERING_USER_ID)}
                    response = await client.post(
                        "/v1/search",
                        headers=headers,
                        json={
                            **request,
                            "mode": "bm25",
                            "collection_id": str(ENGINEERING_COLLECTION_ID),
                        },
                    )
                    assert {item["collection_id"] for item in response.json()["chunks"]} == {
                        str(ENGINEERING_COLLECTION_ID)
                    }
                    assert (
                        await client.post(
                            "/v1/search",
                            headers=headers,
                            json={**request, "collection_id": str(HR_COLLECTION_ID)},
                        )
                    ).status_code == 404
                    path = (
                        f"/v1/workspaces/{DEMO_TENANT_ID}/documents/"
                        f"{docs[ENGINEERING_COLLECTION_ID]}/content"
                    )
                    assert (await client.get(path, headers=headers)).status_code == 200
                    # A completed grant change affects the very next request.
                    await connection.execute(
                        delete(CollectionGrant).where(
                            CollectionGrant.membership_id == UUID(int=ENGINEERING_USER_ID.int + 1)
                        )
                    )
                    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                        seen_candidates.clear()
                        response = await client.post(
                            "/v1/search", headers=headers, json={**request, "mode": mode}
                        )
                        assert response.status_code == 200 and response.json()["chunks"] == []
                        assert not seen_candidates
                    assert (await client.get(path, headers=headers)).status_code == 404
                    assert (
                        await client.get(
                            path.replace(str(docs[ENGINEERING_COLLECTION_ID]), str(uuid4())),
                            headers=headers,
                        )
                    ).json() == {"detail": "Document not found"}
                    assert (
                        await client.post("/debug/search", json={"dataset": "synthetic-workspace"})
                    ).status_code == 404
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()
        async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=10) as client:
            response = await client.delete(f"/{index_name}")
            if response.status_code != 404:
                response.raise_for_status()


@pytest.mark.integration
@pytest.mark.anyio
async def test_committed_grant_removal_is_visible_to_the_next_independent_http_request(
    tmp_path, monkeypatch
):
    from hybrid_rag_search.models import Collection, Document, Tenant

    settings = get_settings().model_copy(update={"storage_root": tmp_path / "originals"})
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    tenant, collection, membership, grant = uuid4(), uuid4(), uuid4(), uuid4()
    engine = create_async_engine(async_database_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    auth = Authorization(sessions)
    try:
        async with sessions.begin() as session:
            session.add(Tenant(id=tenant, name="Grant refresh fixture", slug=f"auth-{tenant.hex}"))
            await session.flush()
            session.add_all(
                [
                    Collection(id=collection, tenant_id=tenant, name="Private", slug="private"),
                    Membership(
                        id=membership, tenant_id=tenant, user_id=ENGINEERING_USER_ID, role="member"
                    ),
                ]
            )
            await session.flush()
            session.add(
                CollectionGrant(
                    id=grant,
                    tenant_id=tenant,
                    collection_id=collection,
                    membership_id=membership,
                    permission="read",
                )
            )
        original = await DocumentService(sessions, LocalFileStorage(settings.storage_root)).upload(
            tenant, collection, "private.txt", "text/plain", b"private original"
        )
        assert (await auth.scope(ENGINEERING_USER_ID, tenant)).collection_ids == (collection,)
        # Use the default dependency: each HTTP request opens its own database connection.
        app = create_app()
        headers = {"X-Demo-User-ID": str(ENGINEERING_USER_ID)}
        path = f"/v1/workspaces/{tenant}/documents/{original.document_id}/content"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get(path, headers=headers)).content == b"private original"
            async with sessions.begin() as session:
                await session.execute(delete(CollectionGrant).where(CollectionGrant.id == grant))
            assert (await client.get(path, headers=headers)).status_code == 404
            response = await client.post(
                "/v1/search",
                headers=headers,
                json={"workspace_id": str(tenant), "query": "private", "mode": "hybrid_rerank"},
            )
            assert response.status_code == 200 and response.json()["chunks"] == []
        assert (await auth.scope(ENGINEERING_USER_ID, tenant)).collection_ids == ()
    finally:
        async with sessions.begin() as session:
            await session.execute(delete(Document).where(Document.tenant_id == tenant))
            await session.execute(
                delete(CollectionGrant).where(CollectionGrant.tenant_id == tenant)
            )
            await session.execute(delete(Collection).where(Collection.tenant_id == tenant))
            await session.execute(delete(Membership).where(Membership.tenant_id == tenant))
            await session.execute(delete(Tenant).where(Tenant.id == tenant))
        await engine.dispose()
