# Hybrid RAG Search

A production-style enterprise search and retrieval-augmented generation engine built to explore retrieval quality, reranking, adaptive search, citations, and performance tradeoffs.

The project is intentionally focused on the search system around the language model—not on building another generic “chat with a PDF” interface.

> Status: Day 12 filtered hybrid retrieval implemented. The system is designed for reproducible local operation and will not be deployed as a public service.

## What the system will do

A user selects a seeded demo identity, searches an authorized collection, and receives a streamed answer backed by citations to exact document passages or messages.

The retrieval pipeline will compare and combine:

1. OpenSearch BM25 lexical retrieval
2. OpenSearch vector retrieval using Cohere Embed
3. Reciprocal Rank Fusion
4. Cohere Rerank
5. Cohere Chat generation with citations

An adaptive retrieval policy will eventually choose between fast, standard, and bounded deep-search paths based on query and retrieval confidence signals.

## Architecture

```text
Files / JSON message archives
             |
             v
       FastAPI ingestion API
             |
             v
   Dramatiq workers via Redis
    parse -> chunk -> embed
             |
             v
         OpenSearch
   text + metadata + vectors
        /            \
     BM25          dense search
        \            /
         RRF hybrid ranking
                |
         Cohere Rerank
                |
          Cohere Chat
                |
     streamed answer + citations
```

Supporting components:

- **PostgreSQL:** authoritative users, permissions, document metadata and lifecycle, job history, retrieval configurations, and evaluation results
- **OpenSearch:** rebuildable lexical and vector search index
- **Redis:** Dramatiq broker, cache, rate limits, and temporary coordination
- **Local filesystem:** original documents, parsed artifacts, and evaluation reports
- **Next.js/React/TypeScript:** thin user interface; all application and authorization logic remains in FastAPI
- **Docker Compose:** reproducible local environment

## Evaluation

Retrieval quality is a first-class feature. The project will compare BM25, dense, hybrid, hybrid plus reranking, and adaptive retrieval using:

- Recall@5/10/50
- MRR@10
- nDCG@10
- Citation precision and recall
- Answer faithfulness
- p50/p95/p99 latency
- Query throughput and indexing throughput
- Cohere usage and estimated cost per query

Two separate evaluation corpora are planned:

- A synthetic company workspace containing policies, design documents, incidents, conflicting versions, permissions, and Discord-style message archives
- BEIR SciFact as a small, recognized external retrieval benchmark with existing relevance judgments

Real Cohere calls will be used for relevance evaluation and low-concurrency latency measurements. Deterministic fake adapters and synthetic vectors will be used for automated tests, sustained load tests, and larger infrastructure experiments so results remain reproducible and trial usage stays bounded.

## Access-control model

The local demo will provide pre-created users with different workspace roles and collection permissions. Identity verification is deliberately simulated; authorization enforcement is real and tested across retrieval, caches, answers, citations, and direct document access.

Example identities will include an administrator, engineering user, HR user, restricted user, and a user from a second tenant.

## Delivery plan

Implementation is organized into 30 GitHub issues, where each numbered “Day” represents one focused 2–4 hour work session rather than a calendar date:

- **Days 1–15:** P0 measurable search foundation
- **Days 16–22:** P0.5 cited-answer product
- **Days 23–28:** P1 adaptive retrieval and production-style hardening
- **Days 29–30:** final validation, benchmarks, documentation, and demo preparation

See the [GitHub issue backlog](https://github.com/Justin-Mendoza/hybrid-rag-search/issues) for ticket scope, dependencies, acceptance criteria, and verification checkpoints.

## Documentation

- [Product and system specification](SPEC.md)

## Development

Supported toolchain:

- Python 3.12 or 3.13
- Node.js 22 LTS and npm 10+
- macOS on Apple Silicon with Docker Desktop is the reference environment. Allocate at least 4 GB of memory to Docker Desktop; OpenSearch uses a configurable 512 MB JVM heap by default.

For the complete local development environment, run `make dev`. It installs dependencies, starts the backend and frontend together, and opens `http://localhost:3000` in the default browser. Press Ctrl+C to stop both servers.

To run the applications separately, use `make setup` once, followed by `make backend-dev` (API at `http://localhost:8000`) and `make frontend-dev` (web at `http://localhost:3000`). Run every CI quality gate with `make check`.

Focused commands are available as `make format`, `make format-check`, `make lint`, `make typecheck`, `make test`, and `make build`. Copy `.env.example` to `.env` for local configuration; never commit credentials.

### Model providers

The backend calls Cohere through typed embedding, reranking, and generation
interfaces. Production adapters use the official async Cohere SDK, while normal
tests and load tests use deterministic fakes. This keeps application code
independent of the vendor SDK and makes routine verification reproducible and
free of provider usage.

The current baselines are `embed-english-light-v3.0`, `rerank-v4.0-fast`, and
`command-r7b-12-2024`. Live smoke tests are excluded from normal `pytest`,
`make test`, `make check`, and CI runs. After configuring `COHERE_API_KEY` or its
supported `COHERE_TRIAL_KEY` alias, explicitly authorize the three-call smoke
suite with `make cohere-smoke`. See the [Day 5 implementation notes](docs/tickets/day-05.md)
for model decisions, usage metadata, error behavior, and verification evidence.

### Hybrid retrieval

Day 12 runs Day 10's BM25 and Day 11's dense retrievers concurrently against
one tenant-scoped metadata filter contract, then combines their ranks with
versioned reciprocal-rank fusion (`rrf-v1`). The index stores normalized typed
fields for source type, author, and source date while preserving the original
document metadata. This requires a complete `chunks-v3` rebuild.

Inspect the library boundary with
`python -m hybrid_rag_search.hybrid_retrieval --tenant <uuid> 'query'`. After
`make stack-up`, run `make opensearch-hybrid-test` for the deterministic real
OpenSearch filter-and-fusion check. See the [Day 12 implementation notes](docs/tickets/day-12.md).

### Document parsing

The backend parses PDF, Markdown, HTML, and plain-text originals into one shared
sequence of normalized text blocks. Blocks retain one-based PDF pages or nested
heading paths when available, giving later chunking and citations a stable source
location without exposing format-specific parser objects.

Parsing is deterministic and local: no Cohere calls, database writes, worker
messages, or OpenSearch indexing occur in this stage. HTML scripts/styles and
hidden content are excluded; encrypted, corrupt, unsupported, or textless inputs
produce stable safe errors. PDF extraction handles embedded text only—OCR and
layout reconstruction remain out of scope. See the [Day 6 implementation notes](docs/tickets/day-06.md)
for the block contract, normalization rules, and format-specific behavior.

### Docker Compose environment

Run `make stack-up` to build and start the complete six-service environment. The command waits until the frontend, API, worker, PostgreSQL, Redis, and OpenSearch health checks pass.

- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- PostgreSQL: `127.0.0.1:5432`
- Redis: `127.0.0.1:6379`
- OpenSearch: `http://127.0.0.1:9200`
- OpenSearch metrics: `http://127.0.0.1:9600`

Use `make stack-status` to inspect health and `make stack-logs` to follow recent logs. `make stack-down` removes the containers and network but preserves all named-volume data, so the next `make stack-up` restores it.

To deliberately erase PostgreSQL, Redis, OpenSearch, application storage, frontend dependencies, and the Next.js cache, run `make stack-reset CONFIRM=1`. The command refuses to delete anything without that flag. OpenSearch security is disabled in this local-only environment and must not be exposed beyond loopback.

See the [Day 2 implementation ticket](docs/tickets/day-02.md) for its outcome, scope, decisions, and verification evidence.

### Database schema

With PostgreSQL running, apply all schema migrations and load deterministic development records:

```bash
make db-upgrade
make db-seed
```

Use `make db-current` to show the applied revision and `make db-test` to run the real-PostgreSQL constraint suite. `make db-downgrade` reverses one revision and may delete data owned by that revision, so use it only when intentionally testing or revising the schema.

The seed command is idempotent. Re-running it reuses the same Acme Demo tenant, owner, Company Handbook collection, and managing grant. Schema history and the complete data-model decisions are documented in the [Day 3 implementation ticket](docs/tickets/day-03.md).

## Scope boundaries

Original uploads use the replaceable `FileStorage` interface and `DocumentService`.
Bytes live under `STORAGE_ROOT` (`.data/originals` locally, `/app/data/originals` in
the shared Compose volume); PostgreSQL stores their hash, size, media type, and
lifecycle. See the [Day 4 usage and recovery notes](docs/tickets/day-04.md).
`make db-test` now verifies document storage and lifecycle against PostgreSQL as
well as schema constraints.

This project will not include cloud deployment, Kubernetes, live Slack or Discord connectors, real account authentication, source-native ACL synchronization, autonomous agents, or proprietary model training.

The complete scope, success targets, design decisions, risks, and release definition are maintained in [SPEC.md](SPEC.md).
