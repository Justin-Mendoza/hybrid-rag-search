# Day 03 - Define the PostgreSQL schema and migrations

GitHub: [Issue #1](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/1)

Depends on: Day 02

## Problem

The stack can reach PostgreSQL, but the application has no durable domain model. Upcoming document, authorization, ingestion, retrieval, and evaluation work needs one authoritative schema that prevents invalid lifecycle values and cross-tenant relationships rather than relying only on application code.

## Outcome

A clean PostgreSQL database can migrate to the latest schema, load deterministic development records without duplication, enforce core constraints, and roll back safely. The schema establishes durable identities and relationships without implementing the APIs and workflows owned by later tickets.

## User stories

- As a workspace administrator, I want users, memberships, collections, and grants represented consistently so that later authorization logic has a trustworthy source of truth.
- As an ingestion worker, I want document and job lifecycle state persisted so that retries and failures survive process restarts.
- As a retrieval engineer, I want versioned configurations, traces, datasets, judgments, and runs so that experiments remain reproducible.
- As a contributor, I want one migration and seed workflow so that a reset local database becomes usable predictably.

## Confirmed decisions

### Database tooling and identity

- Use SQLAlchemy 2.x typed models with async PostgreSQL support and Alembic migrations.
- Use application-generated UUIDv4 primary keys for major entities.
- Store timezone-aware timestamps in UTC with database-generated defaults.
- Represent lifecycle values as text columns constrained by PostgreSQL `CHECK` constraints and Python enums, rather than native PostgreSQL enum types.

### Tenant isolation and authorization

- Users are global identities; memberships connect users to one or more tenants.
- Tenant-owned rows carry `tenant_id`.
- Composite foreign keys include `tenant_id` wherever they can prevent a relationship from crossing tenant boundaries.
- Membership roles are `owner`, `admin`, and `member`.
- Collection grant permissions are `read`, `write`, and `manage` and target tenant memberships.

### Lifecycle and reproducibility

- Documents are soft-deleted using lifecycle state plus `deleted_at`; audit history remains durable.
- Retrieval configurations are immutable versions, unique by tenant, name, and version.
- Core relational and filterable data uses normal columns. Flexible source metadata, configuration payloads, timings, ranking details, and event context use JSONB.
- Audit events are append-only, enforced by a PostgreSQL trigger.
- Evaluation datasets are included as the versioned parent of judgments and evaluation runs.
- Development seed data is separate from migrations and uses deterministic UUIDs so repeated execution is idempotent.

## P0 schema

| Table | Responsibility | Core constraints |
|---|---|---|
| `tenants` | Workspace isolation boundary | Unique slug |
| `users` | Global demo identities | Unique normalized email |
| `memberships` | User role within a tenant | Unique tenant/user pair; constrained role |
| `collections` | Tenant-owned searchable scope | Unique tenant/slug pair |
| `collection_grants` | Membership access to a collection | Same-tenant membership and collection; unique pair; constrained permission |
| `documents` | Source identity, storage, metadata, and lifecycle | Same-tenant collection; unique collection/content identity; constrained status; deletion-state consistency |
| `ingestion_jobs` | Durable document-processing attempts and errors | Same-tenant document; constrained status and non-negative attempts |
| `retrieval_configs` | Reproducible retrieval parameters | Unique tenant/name/version; positive version; immutable rows |
| `query_traces` | Sanitized query path, timing, ranking, usage, and failures | Same-tenant collection/config references; content capture flag |
| `evaluation_datasets` | Versioned evaluation-corpus metadata | Unique tenant/name/version |
| `judgments` | Graded relevance labels within a dataset | Same-tenant dataset; unique query/target judgment |
| `evaluation_runs` | Execution of a dataset against a retrieval config | Same-tenant dataset/config; constrained status |
| `audit_events` | Administrative and content-lifecycle history | Same-tenant actor membership when present; append-only trigger |

## P0 requirements

- Alembic exposes a single current head and reads the application database URL without storing credentials in migration files.
- `upgrade head` succeeds against an empty PostgreSQL database.
- `downgrade base` removes the Day 3 schema safely, including triggers and helper functions.
- All foreign keys define deliberate delete behavior rather than inheriting implicit defaults.
- Database constraints reject duplicate memberships/grants, invalid lifecycle values, negative counters, inconsistent soft deletion, and enforceable cross-tenant references.
- Indexes support tenant-scoped lookups and the expected lifecycle/status polling paths without prematurely optimizing retrieval queries.
- A separate seed command creates one tenant, one admin user and membership, one collection, and one managing collection grant.
- Re-running the seed command changes no identities and creates no duplicates.
- Automated schema tests run against real PostgreSQL and cover upgrade, rollback, seed idempotency, and core constraint failures.

## Non-goals

- Document chunks and chunk identity belong to Day 7.
- File-storage behavior and document APIs belong to Day 4 and later API tickets.
- Real ingestion actors, retries, and checkpoints belong to Day 8.
- OpenSearch mappings and index writes belong to Day 9.
- Runtime authorization policy belongs to Day 16; Day 3 provides its constrained data foundation.
- Partitioning, row-level security, production database roles, and online zero-downtime migrations are premature for the local project.

## Acceptance criteria

- [x] A clean database upgrades from base to the single Alembic head.
- [x] The head schema downgrades fully to base and can upgrade again.
- [x] Enforceable cross-tenant membership, collection, document, evaluation, and trace relationships are rejected by PostgreSQL.
- [x] Invalid lifecycle values and inconsistent document deletion fields are rejected.
- [x] Duplicate membership, grant, version, and seed records are rejected or reused as designed.
- [x] Audit-event update and delete operations are rejected while migration rollback remains possible.
- [x] Running the seed command twice produces the same deterministic records and row counts.
- [x] Schema tests pass against the Compose PostgreSQL service.
- [x] Existing formatting, linting, typing, unit tests, and frontend build continue to pass.

## Implementation checkpoints

1. **Schema contract (complete):** Confirm identity, tenancy, lifecycle, permission, versioning, metadata, audit, and seed behavior.
2. **Migration foundation (complete):** Add SQLAlchemy/Alembic configuration, shared conventions, and database commands; prove connectivity before creating the domain revision.
3. **Core application schema (complete):** Implement tenants, users, memberships, collections, grants, documents, ingestion jobs, retrieval configs, and constraints.
4. **Evaluation and audit schema (complete):** Implement traces, datasets, judgments, runs, append-only audit behavior, and cross-tenant constraints.
5. **Seed and verification (complete):** Add deterministic seed logic, real-PostgreSQL schema tests, documentation, and the final migration/rollback/upgrade verification.

## Verification evidence

Capture migration history, table and constraint inspection, negative tenant-isolation tests, seed row counts, rollback/re-upgrade results, and `make check` output in the Day 3 PR description.

### Checkpoint 2 evidence

- Alembic 1.20 and SQLAlchemy 2.0.53 install through the backend package constraints.
- Alembic translates the shared `postgresql://` application URL to the explicit `postgresql+asyncpg://` SQLAlchemy dialect without adding a second database driver.
- `make db-current` connects successfully to the Compose PostgreSQL service and confirms transactional DDL support.
- Shared metadata defines deterministic names for indexes, unique constraints, checks, foreign keys, and primary keys so future migrations remain stable.
- Ruff, strict mypy, and 12 backend tests at 100% coverage pass.

### Identity and content slice evidence

- The identity revision upgrades, downgrades to base, and reapplies cleanly; the content revision independently downgrades to the identity revision and reapplies.
- Alembic reports a single head and `alembic check` reports no model/schema differences.
- Disposable PostgreSQL transactions confirmed rejection of duplicate memberships, invalid roles, cross-tenant collection grants, malformed SHA-256 hashes, and a deleted document without `deleted_at`.
- The content model keeps original bytes outside PostgreSQL: `storage_key` locates the file, `content_hash` fingerprints its bytes, and later OpenSearch records hold derived searchable chunks.
- Ruff, strict mypy, and 14 backend tests at 100% coverage pass after the content slice.

### Retrieval, evaluation, and audit evidence

- Alembic reports one current head at `20260915_04`, and each layered migration independently downgrades and reapplies.
- `alembic check` reports no differences between SQLAlchemy metadata and the live PostgreSQL schema.
- PostgreSQL rejected retrieval-configuration updates and deletes, preserving historical configuration versions.
- A disposable transaction confirmed the 0-3 relevance-grade range and rejected mutation of evaluation datasets, judgments, and audit events.
- The first failed evaluation migration was rolled back atomically when PostgreSQL rejected an overlong constraint identifier; the corrected, intentionally short name then applied successfully.
- Ruff, strict mypy, and 16 backend tests at 100% coverage pass after the final schema group.

### Seed and automated verification evidence

- Two consecutive `make db-seed` runs produced the same deterministic identities and exactly one matching tenant, user, membership, collection, and grant.
- `make db-test` passed against PostgreSQL 17 and exercised invalid roles, cross-tenant grants, malformed document hashes, and out-of-range relevance grades.
- Backend CI now starts PostgreSQL 17 for both Python versions, validates upgrade/check/downgrade/re-upgrade, runs the seed twice, and executes the PostgreSQL constraint test.
- Local unit coverage contains 20 passing tests at 100%; the PostgreSQL integration test is intentionally separated from the default unit suite.
- A final full `downgrade base` removed every Day 3 table, `upgrade head` recreated all four revisions, and the seed and PostgreSQL test passed afterward.
