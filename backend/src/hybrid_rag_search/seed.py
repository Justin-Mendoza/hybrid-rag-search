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


ENGINEERING_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000112")
HR_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000122")
RESTRICTED_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000132")
SECOND_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000142")
SECOND_TENANT_ID = uuid.UUID("00000000-0000-4000-8000-000000000141")
ENGINEERING_COLLECTION_ID = uuid.UUID("00000000-0000-4000-8000-000000000114")
HR_COLLECTION_ID = uuid.UUID("00000000-0000-4000-8000-000000000124")
SECOND_COLLECTION_ID = uuid.UUID("00000000-0000-4000-8000-000000000144")
DEMO_USER_IDS = (DEMO_USER_ID, ENGINEERING_USER_ID, HR_USER_ID, RESTRICTED_USER_ID, SECOND_USER_ID)


def seed_statements() -> list[Insert]:
    # Fixed IDs preserve the original Day 3 records and make reseeding repeatable.
    tenants = [
        {"id": DEMO_TENANT_ID, "name": "Acme Demo", "slug": "acme-demo"},
        {"id": SECOND_TENANT_ID, "name": "Other Demo", "slug": "other-demo"},
    ]
    users = [
        {"id": user_id, "email": email, "display_name": name}
        for user_id, email, name in (
            (DEMO_USER_ID, "admin@acme.test", "Acme Admin"),
            (ENGINEERING_USER_ID, "engineering@acme.test", "Acme Engineering"),
            (HR_USER_ID, "hr@acme.test", "Acme HR"),
            (RESTRICTED_USER_ID, "restricted@acme.test", "Acme Restricted"),
            (SECOND_USER_ID, "member@other.test", "Other Member"),
        )
    ]
    memberships = [
        {
            "id": uuid.UUID(int=user_id.int + 1),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "role": role,
        }
        for user_id, tenant_id, role in (
            (DEMO_USER_ID, DEMO_TENANT_ID, "owner"),
            (ENGINEERING_USER_ID, DEMO_TENANT_ID, "member"),
            (HR_USER_ID, DEMO_TENANT_ID, "member"),
            (RESTRICTED_USER_ID, DEMO_TENANT_ID, "member"),
            (SECOND_USER_ID, SECOND_TENANT_ID, "member"),
        )
    ]
    collections = [
        {
            "id": collection_id,
            "tenant_id": tenant_id,
            "name": name,
            "slug": slug,
            "description": "Seeded collection for local development.",
        }
        for collection_id, tenant_id, name, slug in (
            (DEMO_COLLECTION_ID, DEMO_TENANT_ID, "Company Handbook", "company-handbook"),
            (ENGINEERING_COLLECTION_ID, DEMO_TENANT_ID, "Engineering", "engineering"),
            (HR_COLLECTION_ID, DEMO_TENANT_ID, "HR", "hr"),
            (SECOND_COLLECTION_ID, SECOND_TENANT_ID, "Other Handbook", "handbook"),
        )
    ]
    grants = [
        {
            "id": grant_id,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "membership_id": uuid.UUID(int=user_id.int + 1),
            "permission": permission,
        }
        for grant_id, tenant_id, collection_id, user_id, permission in (
            (DEMO_GRANT_ID, DEMO_TENANT_ID, DEMO_COLLECTION_ID, DEMO_USER_ID, "manage"),
            (uuid.UUID(int=0x201), DEMO_TENANT_ID, DEMO_COLLECTION_ID, ENGINEERING_USER_ID, "read"),
            (
                uuid.UUID(int=0x202),
                DEMO_TENANT_ID,
                ENGINEERING_COLLECTION_ID,
                ENGINEERING_USER_ID,
                "write",
            ),
            (uuid.UUID(int=0x203), DEMO_TENANT_ID, DEMO_COLLECTION_ID, HR_USER_ID, "read"),
            (uuid.UUID(int=0x204), DEMO_TENANT_ID, HR_COLLECTION_ID, HR_USER_ID, "manage"),
            (uuid.UUID(int=0x205), DEMO_TENANT_ID, DEMO_COLLECTION_ID, RESTRICTED_USER_ID, "read"),
            (uuid.UUID(int=0x206), SECOND_TENANT_ID, SECOND_COLLECTION_ID, SECOND_USER_ID, "read"),
        )
    ]
    statements: list[Insert] = []
    for model, rows in (
        (Tenant, tenants),
        (User, users),
        (Membership, memberships),
        (Collection, collections),
        (CollectionGrant, grants),
    ):
        statement = insert(model).values(rows)
        statements.append(
            statement.on_conflict_do_update(
                index_elements=[model.id],
                set_={key: getattr(statement.excluded, key) for key in rows[0] if key != "id"},
            )
        )
    return statements


async def seed_database(database_url: str | None = None) -> None:
    url = async_database_url(database_url or get_settings().database_url)
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        for statement in seed_statements():
            await connection.execute(statement)
    await engine.dispose()


def main() -> None:
    asyncio.run(seed_database())
    print("Seeded Acme Demo and Other Demo: five identities, four collections, and grants.")


if __name__ == "__main__":  # pragma: no cover
    main()
