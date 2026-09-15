# Day 04 — Local file storage and document lifecycle

[Issue #4](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/4)

## Implemented

- `FileStorage` defines save, fetch, and idempotent delete; `LocalFileStorage`
  implements it without new dependencies.
- Immutable originals use `<tenant UUID>/<document UUID>` keys. Original filenames
  are metadata only. Invalid keys and symlink components are rejected.
- Files are written to a temporary file, flushed and synced, then atomically
  published using a create-only hard link. Failed writes clean up temporary files;
  existing originals cannot be overwritten.
- `DocumentService` owns PostgreSQL transactions. Upload records SHA-256, byte
  count, caller-supplied MIME type, filename, key, and `pending` status.
- Equal bytes reuse an active document within the same tenant and collection,
  regardless of filename. A collection row lock serializes concurrent uploads.
  Different collections/tenants own separate originals. Deleted/deleting documents
  do not suppress a new upload. Source keys are generated document UUIDs; upload
  does not replace an existing source in place.
- Fetch and delete use tenant-scoped lookups and validate the stored key against
  document identity. Fetch is blocked for deleting/deleted documents.
- Delete commits `deleting` before disk removal, then records `deleted` and a UTC
  deletion timestamp. Retrying after an I/O failure or already-missing file is safe.
  The metadata row remains for history.

## Use from the backend

After `make db-upgrade` and `make db-seed`, the following async code stores and
reads a small original using the seeded collection:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.database import async_database_url
from hybrid_rag_search.documents import DocumentService
from hybrid_rag_search.seed import DEMO_COLLECTION_ID, DEMO_TENANT_ID
from hybrid_rag_search.storage import LocalFileStorage

async def example():
    settings = get_settings()
    engine = create_async_engine(async_database_url(settings.database_url))
    try:
        service = DocumentService(
            async_sessionmaker(engine), LocalFileStorage(settings.storage_root)
        )
        result = await service.upload(
            DEMO_TENANT_ID, DEMO_COLLECTION_ID,
            "hello.txt", "text/plain", b"Hello, retrieval!",
        )
        assert await service.fetch(DEMO_TENANT_ID, result.document_id) == b"Hello, retrieval!"
        print(result.document_id, result.duplicate)
    finally:
        await engine.dispose()
```

Run inside an async caller or with `asyncio.run(example())`. Repeating it returns
the same ID with `duplicate=True`. The service accepts bytes in memory; upload
streaming, size limits, HTTP error mapping, and MIME inspection belong to the
later API/parser layers. Filesystem operations run in worker threads so they do
not block the async event loop.

## Boundaries and recovery

- Callers must authorize the requested tenant and collection. These services
  enforce tenant scope, but do not implement membership/grant policy (Day 16).
- Parsing and ingestion transitions (`processing`, `ready`, `failed`) belong to
  Days 6–8. HTTP APIs and coordinated search-index/audit deletion belong to Day 17.
  This delete method only handles original bytes and document metadata.
- The root and its ancestors must be owned by the application and protected from
  untrusted local writers. Symlink validation assumes that local processes cannot
  swap filesystem components during operations. This is the local deployment
  boundary, not a sandbox against a hostile user with filesystem access.
- Disk and PostgreSQL cannot share one atomic transaction. Upload saves bytes
  before committing metadata. A process crash or database commit failure can
  leave an unreferenced original; the service intentionally retains it because a
  lost commit acknowledgement may still mean the row committed. During maintenance
  with writers stopped, compare keys with PostgreSQL and quarantine unreferenced
  files. Automated reconciliation and power-loss durability are not provided.
- A failed delete remains `deleting`; repeat `service.delete(tenant_id, document_id)`
  after storage recovers. If bytes were removed before a crash, the retry completes
  the metadata transition. Missing originals during fetch raise `FileNotFoundError`.
- Collection-level serialization favors correctness and simplicity for this local
  project; high-volume upload concurrency may warrant finer-grained locks later.

## Verification

- `make check`: formatting, lint, strict types, 47 backend unit tests at 100%
  statement coverage, frontend test, and production frontend build pass.
- `make db-test`: schema constraints plus real PostgreSQL/filesystem integration
  pass. Storage integration covers metadata persistence, concurrent deduplication,
  tenant isolation, failed-save rollback, soft deletion, failure/retry, and reupload.
- Unit tests cover path traversal, malformed/noncanonical keys, symlinks, immutable
  writes, temporary-file cleanup, missing files, and lifecycle validation.
- CI runs both integration suites on Python 3.12 and 3.13 with PostgreSQL 17.

All four issue acceptance criteria are covered; no schema migration is required.
