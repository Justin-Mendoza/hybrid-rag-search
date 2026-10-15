"""History service control flow and HTTP contract, without model or database calls."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from test_answers import evidence

from hybrid_rag_search import api
from hybrid_rag_search.answer_history import AnswerHistory
from hybrid_rag_search.answers import AnswerResult
from hybrid_rag_search.authorization import AccessDenied, AuthorizedScope
from hybrid_rag_search.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


class MemorySession:
    def __init__(self):
        self.row = None

    def add(self, row):
        row.id, row.created_at = uuid4(), datetime.now(UTC)
        self.row = row

    async def flush(self):
        pass

    async def scalar(self, query):
        return self.row

    async def scalars(self, query):
        return [] if self.row is None else [self.row]


class Sessions:
    def __init__(self):
        self.session = MemorySession()

    @asynccontextmanager
    async def __call__(self):
        yield self.session

    begin = __call__


def policy_fixture():
    scope = AuthorizedScope(uuid4(), uuid4(), (uuid4(),))
    policy = SimpleNamespace(
        sessions=Sessions(), scope=AsyncMock(return_value=scope), document_scope=AsyncMock()
    )
    return policy, scope


@pytest.mark.anyio
async def test_history_save_read_and_withholding_after_source_denial():
    policy, scope = policy_fixture()
    service = AnswerHistory(policy)
    result = AnswerResult(outcome="partial", text="Supported portion.", evidence=(evidence(scope),))
    generator = SimpleNamespace(answer=AsyncMock(return_value=result))
    assert await service.list(scope.user_id, scope.tenant_id) == []
    with pytest.raises(AccessDenied, match="Answer not found"):
        await service.get(scope.user_id, scope.tenant_id, uuid4())
    answer_id = await service.answer_and_save(generator, scope.user_id, scope.tenant_id, "Question")
    saved = await service.get(scope.user_id, scope.tenant_id, answer_id)
    assert saved.result == result and saved.available
    assert (await service.list(scope.user_id, scope.tenant_id))[0].available
    generator.answer.assert_awaited_once_with(scope, "Question")
    policy.document_scope.side_effect = AccessDenied(404, "Document not found")
    hidden = await service.get(scope.user_id, scope.tenant_id, answer_id)
    assert not hidden.available and hidden.result is None
    assert not (await service.list(scope.user_id, scope.tenant_id))[0].available
    assert "Supported portion." not in hidden.model_dump_json()


def test_history_routes_pagination_validation_and_safe_errors():
    policy, scope = policy_fixture()
    app = create_app()
    app.dependency_overrides[api.authorization] = lambda: policy
    with TestClient(app) as client:
        base = f"/v1/workspaces/{scope.tenant_id}/answers"
        assert client.get(base).json() == []
        assert client.get(base + "?limit=0").status_code == 422
        assert client.get(base + "?limit=101").status_code == 422
        assert client.get(base + "?offset=-1").status_code == 422
        assert client.get(base + "/bad-id").status_code == 422
        assert client.post(base, json={"question": "invented answer"}).status_code == 405
        missing = client.get(base + "/" + str(uuid4()))
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
        policy.scope.side_effect = AccessDenied(401, "Select an identity")
        assert client.get(base).status_code == 401
        schema = client.get("/openapi.json").json()
        assert "get" in schema["paths"]["/v1/workspaces/{workspace_id}/answers/{answer_id}"]
