# Hybrid RAG Search

A production-style enterprise search system built to explore the engineering behind modern retrieval-augmented generation systems: hybrid retrieval, reranking, authorization, evaluation, citations, latency, caching, and failure handling.

The project intentionally focuses on the **search system around the language model**, rather than building another generic “chat with a PDF” application.

> **Current status:** Retrieval, hybrid ranking, neural reranking, ingestion, parsing, storage, infrastructure, and versioned evaluation datasets are implemented. Retrieval benchmarking is the next milestone, followed by cited answer generation and adaptive retrieval.

---

## Highlights

- Hybrid retrieval using **OpenSearch BM25 + vector search**
- Dense embeddings generated with **Cohere Embed**
- **Reciprocal Rank Fusion (RRF)** for lexical + semantic result fusion
- **Cohere Rerank** for neural reranking of hybrid candidates
- Multi-tenant data model with authorization-aware retrieval
- Asynchronous ingestion using **FastAPI, Dramatiq, and Redis**
- Deterministic parsing and chunking for PDF, Markdown, HTML, and plain text
- PostgreSQL-backed document metadata, permissions, jobs, and evaluation data
- Reproducible evaluation datasets for measuring retrieval quality
- Fully reproducible local environment using **Docker Compose**
- Deterministic fake model adapters for automated tests and load testing

---

## Architecture

```text
Files / JSON message archives
            |
            v
     FastAPI ingestion API
            |
            v
      Dramatiq workers
         via Redis
            |
            v
     parse -> chunk -> embed
            |
            v
         OpenSearch
    text + metadata + vectors
        /              \
     BM25            dense
        \              /
         \            /
        RRF hybrid ranking
               |
               v
        Cohere Rerank
               |
               v
      ranked evidence chunks
```

The next product milestone extends the retrieval pipeline with:

```text
ranked evidence chunks
          |
          v
      Cohere Chat
          |
          v
 streamed answer + citations
```

### Supporting Infrastructure

- **PostgreSQL** — users, permissions, document metadata, lifecycle, job history, retrieval configurations, and evaluation results
- **OpenSearch** — rebuildable lexical and vector search index
- **Redis** — Dramatiq broker, caching, rate limiting, and temporary coordination
- **FastAPI** — application API and authorization layer
- **Next.js / React / TypeScript** — thin user interface
- **Local filesystem** — original documents, parsed artifacts, and evaluation reports
- **Docker Compose** — reproducible six-service local environment

---

## Retrieval Pipeline

The search pipeline combines lexical and semantic retrieval rather than depending on a single retrieval strategy.

### 1. BM25 Lexical Retrieval

OpenSearch BM25 retrieves documents using exact lexical relevance.

This is particularly useful for:

- exact terminology
- identifiers
- uncommon names
- technical phrases
- queries where semantic similarity alone can miss important matches

### 2. Dense Vector Retrieval

Documents are embedded using Cohere and stored in OpenSearch alongside searchable text and metadata.

Dense retrieval provides semantic matching when the query and relevant document use different wording.

### 3. Reciprocal Rank Fusion

BM25 and dense search results are combined using **Reciprocal Rank Fusion (RRF)**.

RRF allows both retrieval strategies to contribute without requiring their raw relevance scores to be directly comparable.

### 4. Neural Reranking

The top hybrid candidates are sent to **Cohere Rerank**.

The current pipeline sends the top 20 hybrid candidates to the reranker and returns the top 10 evidence chunks.

A single request deadline covers both hybrid retrieval and reranking.

If reranking encounters a timeout, rate limit, or temporary provider failure, the system returns explicitly marked RRF fallback results instead of failing the entire retrieval request.

---

## Evaluation

Retrieval quality is treated as a first-class engineering concern.

The system is designed to compare:

- BM25
- dense retrieval
- hybrid retrieval
- hybrid + neural reranking
- adaptive retrieval strategies

using measurable retrieval and system metrics.

### Retrieval Metrics

- Recall@5
- Recall@10
- Recall@50
- MRR@10
- nDCG@10

### Answer and Citation Metrics

Planned cited-answer evaluation includes:

- citation precision
- citation recall
- answer faithfulness

### System Metrics

- p50 latency
- p95 latency
- p99 latency
- query throughput
- indexing throughput
- Cohere usage
- estimated cost per query

---

## Evaluation Datasets

Two evaluation corpora are used.

### Synthetic Enterprise Workspace

A deterministic synthetic company workspace containing:

- company policies
- engineering design documents
- incidents
- conflicting document versions
- permissions
- Discord-style message archives

The dataset is designed to test both retrieval relevance and authorization boundaries.

### BEIR SciFact

The project also uses **BEIR SciFact** as a small external retrieval benchmark with existing relevance judgments.

Both datasets share the same versioned contract for:

- corpus documents
- queries
- graded relevance judgments
- manifests

The synthetic workspace is committed directly to the repository.

SciFact is downloaded and validated explicitly:

```bash
make scifact-download
make scifact-validate
```

See [`datasets/README.md`](datasets/README.md) for the complete dataset contract.

---

## Authorization Model

The system models a multi-tenant enterprise search environment.

Development data includes seeded users with different workspace roles and collection permissions, including:

- administrator
- engineering user
- HR user
- restricted user
- user belonging to a second tenant

Identity verification is intentionally simulated because this project is focused on search and authorization rather than authentication infrastructure.

Authorization boundaries are enforced within the application and retrieval layers rather than relying on the user interface to hide inaccessible data.

The goal is to ensure authorization remains consistent across:

- document retrieval
- caches
- document access
- future generated answers
- future citations

---

## Document Ingestion

Documents pass through an asynchronous ingestion pipeline:

```text
upload
  |
  v
store original
  |
  v
create ingestion job
  |
  v
Dramatiq worker
  |
  v
parse
  |
  v
chunk
  |
  v
embed
  |
  v
index in OpenSearch
```

Original uploads are stored through a replaceable `FileStorage` interface.

PostgreSQL tracks:

- content hash
- file size
- media type
- lifecycle state
- document metadata

This separates durable application state from the rebuildable OpenSearch index.

---

## Document Parsing

The parser supports:

- PDF
- Markdown
- HTML
- plain text

All document formats are converted into a shared sequence of normalized text blocks.

Blocks retain source-location metadata where available:

- PDF page numbers
- nested heading paths

This provides stable locations for later chunking and citation generation.

Parsing is deterministic and local.

The parsing stage performs:

```text
input document
      |
      v
format-specific parser
      |
      v
normalized text blocks
      |
      v
source location metadata
```

It does **not** perform:

- Cohere calls
- database writes
- OpenSearch indexing
- worker dispatch

HTML scripts, styles, and hidden content are excluded.

Encrypted, corrupt, unsupported, or textless files produce deterministic safe errors.

OCR and complex layout reconstruction are intentionally outside the current scope.

---

## Model Providers

The backend communicates with Cohere through typed provider interfaces rather than directly coupling application logic to the vendor SDK.

Current model baselines:

```text
Embedding: embed-english-light-v3.0
Reranking: rerank-v4.0-fast
Generation: command-r7b-12-2024
```

Production adapters use the official asynchronous Cohere SDK.

Automated tests and sustained load tests use deterministic fake adapters.

This provides several benefits:

- reproducible tests
- zero API cost during routine development
- deterministic failure testing
- easier future provider replacement
- separation between application logic and model vendor code

Live Cohere calls are deliberately excluded from normal:

```bash
pytest
make test
make check
```

After configuring a Cohere key, live provider behavior can be tested explicitly:

```bash
make cohere-smoke
```

---

## Local Development

### Requirements

Supported development environment:

- Python 3.12 or 3.13
- Node.js 22 LTS
- npm 10+
- Docker Desktop

macOS on Apple Silicon is the reference environment.

Docker Desktop should have at least **4 GB of memory** available.

OpenSearch uses a configurable 512 MB JVM heap by default.

---

## Quick Start

Copy the environment configuration:

```bash
cp .env.example .env
```

Install dependencies, start the development environment, and open the frontend:

```bash
make dev
```

The committed synthetic workspace is immediately available for deterministic
tests. SciFact is an explicit, verified download: run `make scifact-download`,
then `make scifact-validate`. Both datasets use the same manifest, corpus,
queries, and graded-judgments contract described in
[`datasets/README.md`](datasets/README.md). Run `make evaluation-prepare`, then
`make evaluation-run` to compare BM25, dense, hybrid, and hybrid-plus-rerank with
Recall@5/10/50, MRR@10, and nDCG@10. The shared search boundary is also available
through `POST /debug/search`. Preparation reuses completed ingestion work in an
isolated evaluation scope. See the [Day 15 runbook](docs/tickets/day-15.md) for
reproduction, API examples, and pacing. Day 15 uses the committed
`scifact-subset` fixture (500 documents, 30 queries); full SciFact is deferred.
Run `make evaluation-prepare EVAL_DATASET=scifact-subset`, then
`make evaluation-run EVAL_DATASET=scifact-subset`. Regenerate the fixture with
`make scifact-download scifact-subset`.

```text
http://localhost:3000
```

The API runs at:

```text
http://localhost:8000
```

To run the frontend and backend separately:

```bash
make setup

make backend-dev
make frontend-dev
```

---

## Docker Compose Environment

Start the complete local stack:

```bash
make stack-up
```

The command waits until all six services pass their health checks.

| Service | Address |
|---|---|
| Frontend | `http://127.0.0.1:3000` |
| FastAPI | `http://127.0.0.1:8000` |
| PostgreSQL | `127.0.0.1:5432` |
| Redis | `127.0.0.1:6379` |
| OpenSearch | `http://127.0.0.1:9200` |
| OpenSearch Metrics | `http://127.0.0.1:9600` |

Inspect service status:

```bash
make stack-status
```

Follow logs:

```bash
make stack-logs
```

Stop the environment:

```bash
make stack-down
```

Named volumes are preserved when the stack is stopped.

To deliberately remove PostgreSQL, Redis, OpenSearch, application storage, frontend dependencies, and the Next.js cache:

```bash
make stack-reset CONFIRM=1
```

The explicit confirmation flag prevents accidental data deletion.

> OpenSearch security is disabled for this local-only environment and should not be exposed outside loopback.

---

## Database

Apply migrations and load deterministic development data:

```bash
make db-upgrade
make db-seed
```

Inspect the current schema revision:

```bash
make db-current
```

Run PostgreSQL-backed constraint tests:

```bash
make db-test
```

Reverse one migration:

```bash
make db-downgrade
```

The seed process is idempotent.

Re-running it reuses the same deterministic development entities.

---

## Testing and Quality Gates

Run the complete CI-quality suite:

```bash
make check
```

Focused commands are also available:

```bash
make format
make format-check
make lint
make typecheck
make test
make build
```

Automated tests avoid unnecessary external model calls through deterministic provider fakes.

This allows retrieval logic, failure handling, infrastructure behavior, and application contracts to be tested reproducibly.

---

## Retrieval CLI

The reranked retrieval pipeline can be inspected directly from the command line:

```bash
python -m hybrid_rag_search.reranked_retrieval \
  --tenant <uuid> \
  "query"
```

This executes the current hybrid retrieval and reranking boundary without requiring the user interface.

---

## Current Development Status

Development is organized into 30 focused implementation milestones.

Each numbered "Day" represents approximately one focused 2–4 hour engineering session rather than a calendar day.

### Completed Foundation

**Days 1–14**

Implemented work includes:

- repository and application foundation
- Docker Compose environment
- PostgreSQL schema and deterministic seed data
- document storage and lifecycle
- model-provider abstractions
- deterministic document parsing
- ingestion infrastructure
- lexical retrieval
- vector retrieval
- hybrid RRF retrieval
- neural reranking
- versioned evaluation datasets

### Next Milestone

**Day 15**

Run reproducible retrieval benchmarks across the evaluation datasets and establish measured baselines for:

- BM25
- dense retrieval
- hybrid retrieval
- hybrid + reranking

### Upcoming

**Days 16–22**

Cited-answer product:

```text
retrieval
   |
   v
reranking
   |
   v
context selection
   |
   v
generation
   |
   v
answer + citations
```

**Days 23–28**

Adaptive retrieval and production-style hardening.

The planned adaptive retrieval layer will choose between bounded search strategies based on query and retrieval confidence signals.

**Days 29–30**

Final validation, benchmark reporting, documentation, and demo preparation.

The full implementation backlog is available in the repository's GitHub issues.

---

## Planned Evaluation Results

Once retrieval benchmarking is complete, this section will report measured results rather than estimates.

| Retrieval Strategy | Recall@10 | MRR@10 | nDCG@10 | p95 Latency |
|---|---:|---:|---:|---:|
| BM25 | — | — | — | — |
| Dense | — | — | — | — |
| Hybrid RRF | — | — | — | — |
| Hybrid + Rerank | — | — | — | — |

Additional measurements will include:

- indexing throughput
- query throughput
- model usage
- estimated cost per query
- authorization/security validation

No benchmark numbers will be reported until they are produced by the reproducible evaluation pipeline.

---

## Design Philosophy

This project deliberately avoids several shortcuts common in small RAG demos.

### Retrieval Before Generation

The usefulness of a RAG system depends heavily on whether the correct evidence is retrieved before generation begins.

For that reason, retrieval quality is measured independently from answer generation.

### Search Strategies Are Evaluated, Not Assumed

BM25, dense retrieval, hybrid search, and reranking are compared using the same datasets and relevance judgments.

### Authorization Belongs Inside Retrieval

Filtering inaccessible documents only after retrieval can leak information through:

- result metadata
- caches
- generated answers
- citations

Authorization therefore needs to remain part of the retrieval and application model.

### External Models Should Be Replaceable

Embedding, reranking, and generation are accessed behind typed interfaces so application logic does not depend directly on a single provider.

### Reproducibility Matters

Routine development and load testing should not depend on:

- nondeterministic external model output
- unlimited API credits
- manually configured infrastructure

Deterministic model fakes, versioned datasets, migrations, seed data, and Docker Compose are used to make experiments reproducible.

---

## Scope Boundaries

The project intentionally does **not** include:

- cloud deployment
- Kubernetes deployment
- live Slack connectors
- live Discord connectors
- production account authentication
- source-native ACL synchronization
- autonomous agents
- proprietary model training
- OCR
- complex PDF layout reconstruction

These are deliberately excluded so the project can stay focused on retrieval systems, evaluation, authorization, and production-style application architecture.

---

## Documentation

Detailed engineering decisions and implementation notes live outside the main README so this page can remain focused on understanding and running the system.

- [`SPEC.md`](SPEC.md) — complete product and system specification
- [`datasets/README.md`](datasets/README.md) — evaluation dataset contracts
- `docs/tickets/` — implementation decisions and verification evidence
- GitHub Issues — scoped development backlog, dependencies, acceptance criteria, and verification checkpoints

---

## Tech Stack

### Backend

- Python
- FastAPI
- Dramatiq

### Search / AI

- OpenSearch
- BM25
- vector search
- Reciprocal Rank Fusion
- Cohere Embed
- Cohere Rerank
- Cohere Chat

### Data

- PostgreSQL
- Redis

### Frontend

- Next.js
- React
- TypeScript

### Infrastructure

- Docker
- Docker Compose

### Evaluation

- BEIR SciFact
- synthetic enterprise dataset
- Recall@K
- MRR@10
- nDCG@10
- citation precision / recall
- latency
- throughput
- cost tracking

---

## Project Scope

The goal of Hybrid RAG Search is not to demonstrate that an LLM can answer questions about documents.

The goal is to explore the systems engineering required to make retrieval-backed AI applications **measurable, secure, reproducible, and reliable**.

That includes:

- deciding how evidence is retrieved
- measuring whether retrieval actually improves
- handling model-provider failures
- preventing cross-tenant data leakage
- understanding latency and cost tradeoffs
- keeping the search layer observable and testable
- separating durable application state from rebuildable search infrastructure

See [`SPEC.md`](SPEC.md) for the complete system design.
