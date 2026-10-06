"""Database-backed permissions for explicitly simulated demo identities."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.models import Collection, CollectionGrant, Document, Membership, Tenant, User
from hybrid_rag_search.models.content import DocumentStatus
from hybrid_rag_search.seed import DEMO_USER_IDS


class AccessDenied(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class AuthorizedScope:
    user_id: UUID
    tenant_id: UUID
    collection_ids: tuple[UUID, ...]


class Authorization:
    """Read current database permissions; scopes last for one request only."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def user(self, user_id: UUID | None) -> User:
        if user_id is None or user_id not in DEMO_USER_IDS:
            raise AccessDenied(401, "Select a valid demo identity")
        async with self.sessions() as session:
            user = await session.get(User, user_id)
            if user is None:
                raise AccessDenied(401, "Select a valid demo identity")
            return user

    async def demo_users(self) -> list[dict[str, object]]:
        async with self.sessions() as session:
            users = await session.scalars(
                select(User).where(User.id.in_(DEMO_USER_IDS)).order_by(User.display_name)
            )
            return [{"id": user.id, "display_name": user.display_name} for user in users]

    async def workspaces(self, user_id: UUID | None) -> list[dict[str, object]]:
        user = await self.user(user_id)
        async with self.sessions() as session:
            rows = await session.execute(
                select(Tenant, Membership.role)
                .join(Membership, Membership.tenant_id == Tenant.id)
                .where(Membership.user_id == user.id)
                .order_by(Tenant.name)
            )
            return [{"id": tenant.id, "name": tenant.name, "role": role} for tenant, role in rows]

    async def scope(
        self, user_id: UUID | None, tenant_id: UUID, collection_id: UUID | None = None
    ) -> AuthorizedScope:
        user = await self.user(user_id)
        async with self.sessions() as session:
            membership = await session.scalar(
                select(Membership).where(
                    Membership.user_id == user.id, Membership.tenant_id == tenant_id
                )
            )
            if membership is None or membership.role not in ("owner", "admin", "member"):
                raise AccessDenied(403, "Workspace access denied")
            query = select(Collection.id).where(Collection.tenant_id == tenant_id)
            if membership.role == "member":
                query = query.join(
                    CollectionGrant,
                    (CollectionGrant.collection_id == Collection.id)
                    & (CollectionGrant.tenant_id == Collection.tenant_id),
                ).where(
                    CollectionGrant.membership_id == membership.id,
                    CollectionGrant.permission.in_(("read", "write", "manage")),
                )
            allowed = tuple(sorted(await session.scalars(query)))
            if collection_id is not None:
                if collection_id not in allowed:
                    raise AccessDenied(404, "Collection not found")
                allowed = (collection_id,)
            return AuthorizedScope(user.id, tenant_id, allowed)

    async def document_scope(
        self, user_id: UUID | None, tenant_id: UUID, document_id: UUID
    ) -> AuthorizedScope:
        try:
            scope = await self.scope(user_id, tenant_id)
        except AccessDenied as error:
            if error.status_code == 401:
                raise
            raise AccessDenied(404, "Document not found") from None
        async with self.sessions() as session:
            document = await session.scalar(
                select(Document.id).where(
                    Document.id == document_id,
                    Document.tenant_id == tenant_id,
                    Document.collection_id.in_(scope.collection_ids),
                    Document.status.not_in((DocumentStatus.DELETING, DocumentStatus.DELETED)),
                )
            )
            if document is None:
                raise AccessDenied(404, "Document not found")
        return scope
