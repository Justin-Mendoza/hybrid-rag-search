# Database migrations

Alembic reads `DATABASE_URL` through the application settings. Create revisions from the repository root with `make db-revision MESSAGE="describe change"`, apply them with `make db-upgrade`, and roll back one revision with `make db-downgrade`.

Run `make db-seed` after upgrading to load deterministic local development records. Run `make db-test` while the Compose PostgreSQL service is available to exercise database constraints against PostgreSQL itself. Downgrades may delete data introduced by the reverted revision.
