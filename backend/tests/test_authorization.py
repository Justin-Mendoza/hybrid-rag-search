from types import SimpleNamespace
from uuid import uuid4

import pytest

from hybrid_rag_search.authorization import AccessDenied, Authorization
from hybrid_rag_search.seed import DEMO_USER_ID


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Session:
    def __init__(
        self, *, user=True, membership=None, collections=(), document=None, users=(), rows=()
    ):
        self.record = SimpleNamespace(id=DEMO_USER_ID, display_name="Admin") if user else None
        self.membership = membership
        self.collections = collections
        self.document = document
        self.users = users
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, *args):
        return self.record

    async def scalar(self, query):
        return self.document if query.column_descriptions[0]["name"] == "id" else self.membership

    async def scalars(self, query):
        return self.users if query.column_descriptions[0]["name"] == "User" else self.collections

    async def execute(self, query):
        return self.rows


def policy(**kwargs):
    return Authorization(lambda: Session(**kwargs))


@pytest.mark.anyio
async def test_invalid_or_missing_demo_identity_fails_closed():
    for user in (None, uuid4()):
        with pytest.raises(AccessDenied) as error:
            await policy().user(user)
        assert error.value.status_code == 401
    with pytest.raises(AccessDenied):
        await policy(user=False).user(DEMO_USER_ID)


@pytest.mark.anyio
async def test_demo_directory_and_workspace_selection():
    user = SimpleNamespace(id=DEMO_USER_ID, display_name="Admin")
    tenant = SimpleNamespace(id=uuid4(), name="Workspace")
    auth = policy(users=[user], rows=[(tenant, "owner")])
    assert await auth.demo_users() == [{"id": DEMO_USER_ID, "display_name": "Admin"}]
    assert await auth.workspaces(DEMO_USER_ID) == [
        {"id": tenant.id, "name": "Workspace", "role": "owner"}
    ]


@pytest.mark.anyio
async def test_scope_roles_narrowing_and_empty_permissions():
    tenant, collection = uuid4(), uuid4()
    for role in ("owner", "admin", "member"):
        auth = policy(membership=SimpleNamespace(id=uuid4(), role=role), collections=(collection,))
        scope = await auth.scope(DEMO_USER_ID, tenant)
        assert scope.collection_ids == (collection,)
        assert (await auth.scope(DEMO_USER_ID, tenant, collection)).collection_ids == (collection,)
        with pytest.raises(AccessDenied) as error:
            await auth.scope(DEMO_USER_ID, tenant, uuid4())
        assert error.value.status_code == 404
    auth = policy(membership=SimpleNamespace(id=uuid4(), role="member"))
    assert (await auth.scope(DEMO_USER_ID, tenant)).collection_ids == ()
    for membership in (None, SimpleNamespace(role="unknown")):
        with pytest.raises(AccessDenied) as error:
            await policy(membership=membership).scope(DEMO_USER_ID, tenant)
        assert error.value.status_code == 403


@pytest.mark.anyio
async def test_document_ids_require_current_authorized_scope():
    tenant, document, collection = uuid4(), uuid4(), uuid4()
    auth = policy(
        membership=SimpleNamespace(role="owner"), collections=(collection,), document=document
    )
    assert (await auth.document_scope(DEMO_USER_ID, tenant, document)).collection_ids == (
        collection,
    )
    for auth in (policy(), policy(membership=SimpleNamespace(role="owner"))):
        with pytest.raises(AccessDenied) as error:
            await auth.document_scope(DEMO_USER_ID, tenant, document)
        assert error.value.status_code == 404
    with pytest.raises(AccessDenied) as error:
        await policy().document_scope(None, tenant, document)
    assert error.value.status_code == 401
