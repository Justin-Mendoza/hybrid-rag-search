# Day 02 - Build the Docker Compose local environment

GitHub: [Issue #2](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/2)

Depends on: Day 01

## Problem

The application currently runs as two host processes, but every upcoming ingestion and retrieval ticket depends on PostgreSQL, Redis, OpenSearch, and a background worker. Contributors need one reproducible environment whose services start only when their dependencies are ready and whose state survives ordinary restarts.

## Outcome

On the reference Apple Silicon MacBook Pro, one command builds and starts the API, frontend, worker, PostgreSQL, Redis, and OpenSearch. Docker reports each service healthy, application endpoints respond, and named-volume data survives a stop/start cycle.

## Mental model

```text
Browser -> Next.js -> FastAPI
                       |
                 Dramatiq worker
                   /    |     \
          PostgreSQL  Redis  OpenSearch
```

- PostgreSQL will become the authoritative application database.
- Redis coordinates background jobs now and caching/rate limits later; it is not authoritative state.
- OpenSearch holds the rebuildable lexical and vector search index.
- The Dramatiq worker executes ingestion work outside API requests.
- Compose creates a private network where services resolve one another by service name.

## P0 scope

- Build development images for the FastAPI backend and Next.js frontend.
- Run the backend image as both the API process and a separate Dramatiq worker process.
- Add pinned, ARM64-compatible PostgreSQL, Redis, and OpenSearch images.
- Persist PostgreSQL, Redis, OpenSearch, and application file storage in named volumes.
- Add meaningful health checks for all six services.
- Use `depends_on.condition: service_healthy` rather than fixed startup sleeps.
- Make the OpenSearch JVM heap configurable, defaulting to 512 MB for local development.
- Add stack lifecycle and diagnostic commands to the Makefile.
- Document Docker Desktop resource requirements, ports, persistence, and reset behavior.

## Confirmed decisions

### Application containers

Use development-oriented containers with source bind mounts and reload/watch behavior. Day 2 optimizes the contributor loop; production images and deployment are project non-goals.

Build the shared API/worker image from Python 3.12.14 slim. Keep `/health` as a compatibility liveness endpoint, add `/health/live` for explicit liveness, and make `/health/ready` perform functional PostgreSQL `SELECT 1`, Redis `PING`, and OpenSearch cluster-health operations. The worker's container health check performs Redis `PING` because a worker that cannot reach its broker cannot accept jobs.

Build the frontend from the exact `node:22.23.2-bookworm-slim` image. Bind-mount its source for live reload while keeping Linux `node_modules` and Next.js's `.next` cache in named volumes. Next.js needs limited write access to maintain `next-env.d.ts`, while its generated cache and dependencies remain separate from the macOS host. Use separate internal and browser-visible API URLs because Docker service names resolve inside the Compose network but not in the host browser.

### Data services

- PostgreSQL 17 Alpine, pinned to an exact patch release available for ARM64.
- Redis 8 Alpine, pinned to an exact patch release available for ARM64, with AOF persistence and `appendfsync everysec`.
- Single-node OpenSearch 3.6 LTS, pinned to an exact ARM64-compatible patch release.

Exact patch versions will be verified from image manifests before implementation. Version changes will be deliberate rather than inherited through floating tags.

### Local OpenSearch security

Disable the OpenSearch security plugin for the local-only Compose network and publish port 9200 only on loopback. This avoids local certificate/password ceremony while application-level authorization remains a separately tested requirement. Documentation must state that this setting is unsafe outside local development. Default the configurable JVM heap to 512 MB and increase it later when benchmark measurements require more memory.

### Commands

- `make stack-up`: build, start, and wait for the full container stack.
- `make stack-down`: stop containers while preserving named volumes.
- `make stack-status`: show container health.
- `make stack-logs`: follow service logs.
- `make stack-reset CONFIRM=1`: explicitly remove local stack volumes; without the flag, refuse and print a warning.

Keep the existing host-based `make dev` workflow available for fast frontend/backend work.

## Non-goals

- Database tables and migrations belong to Day 3.
- Document storage behavior and lifecycle belong to Day 4.
- Real ingestion actors and retry policy belong to Day 8; Day 2 only proves that a worker can connect and remain ready.
- OpenSearch mappings and indexing logic belong to Day 9.
- Production TLS, secret management, orchestration, and deployment remain outside the local-only project scope.

## Acceptance criteria

- [x] Given Docker Desktop is running, `make stack-up` builds and starts all six services without manual sequencing.
- [x] `docker compose ps` reports API, frontend, worker, PostgreSQL, Redis, and OpenSearch as healthy.
- [x] The API health endpoint responds on `127.0.0.1:8000` and the frontend responds on `127.0.0.1:3000`.
- [x] PostgreSQL, Redis, and OpenSearch data written before `make stack-down` remains after the next `make stack-up`.
- [x] Backend and worker startup waits for healthy dependencies rather than a fixed delay.
- [x] Changing `OPENSEARCH_JAVA_OPTS` changes the container JVM heap configuration.
- [x] Every selected image resolves to a native `linux/arm64` manifest on Apple Silicon.
- [x] `make check` continues to pass outside containers.
- [x] Documentation distinguishes ordinary shutdown from destructive volume reset.

## Implementation checkpoints

1. **Data plane (complete):** Compose PostgreSQL, Redis, and OpenSearch with volumes and health checks; verify native ARM64 manifests and persistence.
2. **Backend runtime (complete):** Add the backend image, dependency settings, API health behavior, and a bootable Dramatiq worker; verify readiness ordering.
3. **Frontend runtime (complete):** Add the frontend image and health check; wire it to the API through explicit internal/public URLs.
4. **Developer workflow (complete):** Add Make targets and documentation; verify the clean one-command startup, stop/start persistence, and full quality suite.

## Verification evidence

Capture the final `docker compose ps`, endpoint responses, architecture inspection, persistence probe, and `make check` output in the Day 2 PR description.

### Checkpoint 1 evidence

- Pulled images report native `arm64`: PostgreSQL 17.11 Alpine, Redis 8.10.1 Alpine, and OpenSearch 3.6.0.
- PostgreSQL `pg_isready`, Redis `PING`, and OpenSearch cluster health all pass.
- OpenSearch reports a green single-node cluster and uses `-Xms512m -Xmx512m`.
- Redis reports `appendonly yes` and `appendfsync everysec`.
- Disposable records written to all three services survived a complete container stop/start cycle and were removed after verification.

### Checkpoint 2 evidence

- The shared API/worker image reports Python 3.12.14 and native `arm64` architecture.
- The API and Dramatiq worker start only after PostgreSQL, Redis, and OpenSearch are healthy.
- `/health` and `/health/live` return HTTP 200 without dependency checks; `/health/ready` returns HTTP 200 only when all three dependency operations succeed.
- Stopping Redis made `/health/ready` return HTTP 503 with Redis identified as unavailable; restarting Redis restored readiness.
- Backend formatting, strict type checking, and 11 tests with 100% coverage pass locally.

### Checkpoint 3 evidence

- The frontend image uses Node 22.23.2 and resolves to native `arm64`.
- The source tree is bind-mounted for live reload while `/app/node_modules` and `/app/.next` use Docker named volumes.
- The frontend waits for the API readiness check and its own health check performs an HTTP request to port 3000.
- The rendered page responds successfully at `http://127.0.0.1:3000`.

### Checkpoint 4 evidence

- `make stack-down` followed by `make stack-up` rebuilt the application images, started services in readiness order, and returned only after all six services were healthy.
- A disposable Redis value survived the complete down/up lifecycle and was removed after verification.
- `/health/ready` returned all three dependencies ready and the frontend returned HTTP 200 after restart.
- `make stack-reset` without `CONFIRM=1` exited with an error before invoking Docker.
- `make check` passed: formatting, linting, strict backend and frontend type checks, 11 backend tests at 100% coverage, one frontend test, and the production frontend build.
