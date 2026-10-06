from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from hybrid_rag_search import api
from hybrid_rag_search.authorization import AccessDenied, AuthorizedScope
from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.lexical_retrieval import RetrievalError
from hybrid_rag_search.main import create_app
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.seed import DEMO_USER_ID


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Policy:
    sessions = object()
    error = None

    async def demo_users(self):
        return [{"id": DEMO_USER_ID, "display_name": "Admin"}]

    async def workspaces(self, user):
        if self.error:
            raise self.error
        if user is None:
            raise AccessDenied(401, "Select a valid demo identity")
        return [{"id": uuid4(), "name": "Acme"}]

    async def scope(self, user, tenant, collection=None):
        if self.error:
            raise self.error
        return AuthorizedScope(user, tenant, () if collection is None else (collection,))

    async def document_scope(self, user, tenant, document):
        return await self.scope(user, tenant)


def test_api_contract_and_safe_error_responses(monkeypatch):
    policy = Policy()
    app = create_app()
    app.dependency_overrides[api.authorization] = lambda: policy
    headers = {"X-Demo-User-ID": str(DEMO_USER_ID)}
    workspace, document = uuid4(), uuid4()
    request = {"workspace_id": str(workspace), "query": "password reset", "mode": "bm25"}
    path = f"/v1/workspaces/{workspace}/documents/{document}/content"

    class Search:
        error = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def search(self, *args, **kwargs):
            if self.error:
                raise self.error
            return {"chunks": [], "trace": None}

    retrieval = Search()
    monkeypatch.setattr(api, "AuthorizedSearch", lambda _: retrieval)

    class Documents:
        error = None

        async def fetch(self, tenant, document):
            if self.error:
                raise self.error
            return b"authorized original"

    documents = Documents()
    monkeypatch.setattr(api, "DocumentService", lambda *args: documents)
    with TestClient(app) as client:
        assert client.get("/v1/demo-users").json()[0]["id"] == str(DEMO_USER_ID)
        assert client.get("/v1/workspaces").status_code == 401
        assert client.get("/v1/workspaces", headers=headers).status_code == 200
        assert client.post("/v1/search", headers=headers, json=request).status_code == 200
        for body in (
            {**request, "collection_ids": []},
            {**request, "dataset": "scifact"},
            {**request, "mode": "invalid"},
            {**request, "query": ""},
        ):
            assert client.post("/v1/search", headers=headers, json=body).status_code == 422
        assert (
            client.post(
                "/v1/search", headers={"X-Demo-User-ID": "invalid"}, json=request
            ).status_code
            == 422
        )
        assert client.get(path, headers=headers).content == b"authorized original"
        for error in (DocumentNotFound(), DocumentUnavailable()):
            documents.error = error
            response = client.get(path, headers=headers)
            assert response.status_code == 404 and response.json() == {
                "detail": "Document not found"
            }
        policy.error = AccessDenied(404, "Document not found")
        assert client.get(path, headers=headers).status_code == 404
        policy.error = AccessDenied(403, "Workspace access denied")
        assert client.post("/v1/search", headers=headers, json=request).status_code == 403
        assert client.get("/v1/workspaces", headers=headers).status_code == 403
        policy.error = None
        for error in (
            RetrievalError("opensearch_unavailable", retryable=True),
            ProviderError("unavailable"),
        ):
            retrieval.error = error
            response = client.post("/v1/search", headers=headers, json=request)
            assert response.status_code == 503 and response.json()["detail"] == error.code
        assert client.post("/debug/search", json={"dataset": "scifact"}).status_code == 404


@pytest.mark.anyio
async def test_authorization_dependency_disposes_engine(monkeypatch):
    class Engine:
        disposed = False

        async def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(api, "create_async_engine", lambda _: engine)
    monkeypatch.setattr(api, "async_sessionmaker", lambda *args, **kwargs: "sessions")
    generator = api.authorization()
    policy = await anext(generator)
    assert policy.sessions == "sessions"
    await generator.aclose()
    assert engine.disposed
