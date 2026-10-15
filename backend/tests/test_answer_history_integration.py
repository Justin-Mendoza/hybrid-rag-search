"""Real PostgreSQL history ownership, permission revocation, and migration checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from test_answers import evidence
from test_management_integration import management_database as answer_database  # noqa: F401

from hybrid_rag_search import api
from hybrid_rag_search.answer_history import AnswerHistory
from hybrid_rag_search.answers import AnswerResult
from hybrid_rag_search.authorization import Authorization, AuthorizedScope
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.main import create_app
from hybrid_rag_search.models import (
    Collection,
    CollectionGrant,
    Document,
    Membership,
    SavedAnswer,
    Tenant,
    User,
)
from hybrid_rag_search.seed import DEMO_USER_ID, ENGINEERING_USER_ID, HR_USER_ID, SECOND_USER_ID


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_private_history_permissions_deletion_and_database_boundary(answer_database):  # noqa: F811
    engine = create_async_engine(async_database_url(answer_database))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    tenant, foreign, collection, document, member = (uuid4() for _ in range(5))
    author = ENGINEERING_USER_ID
    try:
        async with sessions.begin() as session:
            session.add_all(
                [Tenant(id=t, name="History", slug=f"history-{t}") for t in (tenant, foreign)]
            )
            session.add_all(
                [
                    User(id=u, email=f"{u}@history.test", display_name="History user")
                    for u in (author, DEMO_USER_ID, HR_USER_ID, SECOND_USER_ID)
                ]
            )
            await session.flush()
            session.add_all(
                [
                    Membership(id=member, tenant_id=tenant, user_id=author, role="member"),
                    Membership(tenant_id=tenant, user_id=DEMO_USER_ID, role="owner"),
                    Membership(tenant_id=tenant, user_id=HR_USER_ID, role="admin"),
                    Membership(tenant_id=foreign, user_id=SECOND_USER_ID, role="owner"),
                ]
            )
            session.add(
                Collection(id=collection, tenant_id=tenant, name="Evidence", slug="evidence")
            )
            await session.flush()
            session.add(
                CollectionGrant(
                    tenant_id=tenant,
                    collection_id=collection,
                    membership_id=member,
                    permission="read",
                )
            )
            session.add(
                Document(
                    id=document,
                    tenant_id=tenant,
                    collection_id=collection,
                    source_key="source",
                    storage_key="original",
                    original_filename="a.txt",
                    media_type="text/plain",
                    content_hash="a" * 64,
                    size_bytes=10,
                    status="ready",
                )
            )
        policy = Authorization(sessions)
        history = AnswerHistory(policy)
        scope = AuthorizedScope(author, tenant, (collection,))
        source = evidence(scope).model_copy(update={"document_id": document})
        result = AnswerResult(
            outcome="partial", text="Protected source answer.", evidence=(source,)
        )
        service = SimpleNamespace(answer=AsyncMock(return_value=result))
        first = await history.answer_and_save(service, author, tenant, "First question", collection)
        second = await history.answer_and_save(service, author, tenant, "Second question")
        assert first != second
        # The composite membership foreign key prevents a cross-workspace author row.
        with pytest.raises(IntegrityError):
            async with sessions.begin() as session:
                session.add(
                    SavedAnswer(
                        tenant_id=foreign,
                        author_user_id=author,
                        question="forged",
                        result=result.model_dump(mode="json"),
                    )
                )
        app = create_app()
        app.dependency_overrides[api.authorization] = lambda: policy
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            headers = {"X-Demo-User-ID": str(author)}
            base = f"/v1/workspaces/{tenant}/answers"
            listing = await client.get(base, headers=headers)
            assert [row["id"] for row in listing.json()] == [str(second), str(first)]
            assert "Protected" not in listing.text
            assert (await client.get(base + "?limit=1&offset=1", headers=headers)).json()[0][
                "id"
            ] == str(first)
            detail = await client.get(f"{base}/{first}", headers=headers)
            assert detail.status_code == 200 and detail.json()["result"]["text"] == result.text
            assert detail.json()["result"]["evidence"][0]["document_id"] == str(document)
            for user in (DEMO_USER_ID, HR_USER_ID):
                private_headers = {"X-Demo-User-ID": str(user)}
                assert (await client.get(base, headers=private_headers)).json() == []
                assert (
                    await client.get(f"{base}/{first}", headers=private_headers)
                ).status_code == 404
            foreign_headers = {"X-Demo-User-ID": str(SECOND_USER_ID)}
            assert (await client.get(f"{base}/{first}", headers=foreign_headers)).status_code == 403
            assert (
                await client.get(
                    f"/v1/workspaces/{foreign}/answers/{first}", headers=foreign_headers
                )
            ).status_code == 404
            assert (await client.get(base)).status_code == 401
            async with sessions.begin() as session:
                await session.execute(
                    delete(CollectionGrant).where(CollectionGrant.membership_id == member)
                )
            hidden = await client.get(f"{base}/{first}", headers=headers)
            assert hidden.json()["available"] is False and hidden.json()["result"] is None
            assert "Protected" not in hidden.text and "content_text" not in hidden.text
            assert (await client.get(base, headers=headers)).json()[0]["available"] is False
            async with sessions.begin() as session:
                session.add(
                    CollectionGrant(
                        tenant_id=tenant,
                        collection_id=collection,
                        membership_id=member,
                        permission="read",
                    )
                )
            assert (await client.get(f"{base}/{first}", headers=headers)).json()[
                "available"
            ] is True
            async with sessions.begin() as session:
                await session.execute(
                    update(Document).where(Document.id == document).values(status="deleting")
                )
            assert (await client.get(f"{base}/{first}", headers=headers)).json()["result"] is None
            async with sessions.begin() as session:
                await session.execute(delete(Membership).where(Membership.user_id == author))
            assert (await client.get(f"{base}/{first}", headers=headers)).status_code == 403
    finally:
        await engine.dispose()
