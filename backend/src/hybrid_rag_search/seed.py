import asyncio
import uuid

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql.dml import Insert

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.models import Collection, CollectionGrant, Membership, Tenant, User

DEMO_TENANT_ID = uuid.UUID("00000000-0000-4000-8000-000000000101")
DEMO_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000102")
DEMO_MEMBERSHIP_ID = uuid.UUID("00000000-0000-4000-8000-000000000103")
DEMO_COLLECTION_ID = uuid.UUID("00000000-0000-4000-8000-000000000104")
DEMO_GRANT_ID = uuid.UUID("00000000-0000-4000-8000-000000000105")


def seed_statements() -> list[Insert]:
    return [
        insert(Tenant)
        .values(id=DEMO_TENANT_ID, name="Acme Demo", slug="acme-demo")
        .on_conflict_do_update(
            index_elements=[Tenant.id], set_={"name": "Acme Demo", "slug": "acme-demo"}
        ),
        insert(User)
        .values(id=DEMO_USER_ID, email="admin@acme.test", display_name="Acme Admin")
        .on_conflict_do_update(
            index_elements=[User.id],
            set_={"email": "admin@acme.test", "display_name": "Acme Admin"},
        ),
        insert(Membership)
        .values(
            id=DEMO_MEMBERSHIP_ID,
            tenant_id=DEMO_TENANT_ID,
            user_id=DEMO_USER_ID,
            role="owner",
        )
        .on_conflict_do_update(
            index_elements=[Membership.id],
            set_={"tenant_id": DEMO_TENANT_ID, "user_id": DEMO_USER_ID, "role": "owner"},
        ),
        insert(Collection)
        .values(
            id=DEMO_COLLECTION_ID,
            tenant_id=DEMO_TENANT_ID,
            name="Company Handbook",
            slug="company-handbook",
            description="Seeded collection for local development.",
        )
        .on_conflict_do_update(
            index_elements=[Collection.id],
            set_={
                "tenant_id": DEMO_TENANT_ID,
                "name": "Company Handbook",
                "slug": "company-handbook",
                "description": "Seeded collection for local development.",
            },
        ),
        insert(CollectionGrant)
        .values(
            id=DEMO_GRANT_ID,
            tenant_id=DEMO_TENANT_ID,
            collection_id=DEMO_COLLECTION_ID,
            membership_id=DEMO_MEMBERSHIP_ID,
            permission="manage",
        )
        .on_conflict_do_update(
            index_elements=[CollectionGrant.id],
            set_={
                "tenant_id": DEMO_TENANT_ID,
                "collection_id": DEMO_COLLECTION_ID,
                "membership_id": DEMO_MEMBERSHIP_ID,
                "permission": "manage",
            },
        ),
    ]


async def seed_database(database_url: str | None = None) -> None:
    url = async_database_url(database_url or get_settings().database_url)
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        for statement in seed_statements():
            await connection.execute(statement)
    await engine.dispose()


def main() -> None:
    asyncio.run(seed_database())
    print("Seeded Acme Demo tenant, owner, collection, and grant.")


if __name__ == "__main__":  # pragma: no cover
    main()
