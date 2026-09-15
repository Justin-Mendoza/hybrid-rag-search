import uuid

import asyncpg  # type: ignore[import-untyped]
import pytest

from hybrid_rag_search.config import get_settings


@pytest.mark.integration
@pytest.mark.anyio
async def test_postgres_rejects_core_constraint_violations() -> None:
    connection = await asyncpg.connect(get_settings().database_url)
    transaction = connection.transaction()
    await transaction.start()

    tenant_a, tenant_b, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    membership_id, collection_a, collection_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    dataset_id = uuid.uuid4()

    try:
        await connection.executemany(
            "INSERT INTO tenants (id, name, slug) VALUES ($1, $2, $3)",
            [
                (tenant_a, "Schema Test A", f"schema-test-{tenant_a.hex}"),
                (tenant_b, "Schema Test B", f"schema-test-{tenant_b.hex}"),
            ],
        )
        await connection.execute(
            "INSERT INTO users (id, email, display_name) VALUES ($1, $2, $3)",
            user_id,
            f"{user_id.hex}@example.test",
            "Schema Test User",
        )
        await connection.execute(
            "INSERT INTO memberships (id, tenant_id, user_id, role) VALUES ($1, $2, $3, 'member')",
            membership_id,
            tenant_a,
            user_id,
        )
        await connection.executemany(
            "INSERT INTO collections (id, tenant_id, name, slug) VALUES ($1, $2, $3, $4)",
            [
                (collection_a, tenant_a, "Collection A", f"collection-{collection_a.hex}"),
                (collection_b, tenant_b, "Collection B", f"collection-{collection_b.hex}"),
            ],
        )
        await connection.execute(
            "INSERT INTO evaluation_datasets (id, tenant_id, name, version, corpus_hash) "
            "VALUES ($1, $2, 'schema-test', 1, $3)",
            dataset_id,
            tenant_a,
            "a" * 64,
        )

        with pytest.raises(asyncpg.CheckViolationError):
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO memberships (id, tenant_id, user_id, role) "
                    "VALUES ($1, $2, $3, 'superuser')",
                    uuid.uuid4(),
                    tenant_a,
                    user_id,
                )

        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO collection_grants "
                    "(id, tenant_id, collection_id, membership_id, permission) "
                    "VALUES ($1, $2, $3, $4, 'read')",
                    uuid.uuid4(),
                    tenant_b,
                    collection_b,
                    membership_id,
                )

        with pytest.raises(asyncpg.CheckViolationError):
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO documents "
                    "(id, tenant_id, collection_id, source_key, storage_key, original_filename, "
                    "media_type, content_hash, size_bytes) "
                    "VALUES ($1, $2, $3, 'source', 'storage', 'file.txt', 'text/plain', 'bad', 1)",
                    uuid.uuid4(),
                    tenant_a,
                    collection_a,
                )

        with pytest.raises(asyncpg.CheckViolationError):
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO judgments "
                    "(id, tenant_id, dataset_id, query_key, query_text, target_type, target_id, "
                    "relevance_grade) VALUES ($1, $2, $3, 'q1', 'Question?', 'chunk', 'c1', 4)",
                    uuid.uuid4(),
                    tenant_a,
                    dataset_id,
                )
    finally:
        await transaction.rollback()
        await connection.close()
