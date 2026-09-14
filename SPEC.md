# Hybrid RAG Search — Product and System Specification

Status: Draft v0.2 — product and technical choices remain proposals unless marked confirmed  
Audience: Engineering reviewers, hiring managers, and future contributors  
Initial target: Portfolio-quality single-developer build with production-grade design

### Decision convention

- **Confirmed** means agreed with the project owner.
- **Proposed** means a recommended default to discuss before implementation planning.
- This project will not be deployed. Production-grade refers to architecture, correctness, security boundaries, failure handling, measurement, and reproducible local operation—not operation of a public service.

## 1. Product Summary

Hybrid RAG Search is an enterprise evidence-retrieval and question-answering system. A workspace connects or uploads document collections, the system indexes them for lexical and semantic search, and authorized users receive grounded answers whose claims link back to exact source passages.

The differentiator is the search system around the language model: hybrid lexical/vector retrieval, neural reranking, adaptive retrieval, authorization-aware filtering, measurable relevance, and explicit latency/cost tradeoffs.

This is a “mini Cohere Search,” not a general-purpose chatbot or a thin wrapper over a retrieval framework.

## 2. Problem Statement

Knowledge workers often know that an answer exists somewhere in internal documents or messages, but cannot find the relevant passage quickly or confidently. Keyword search misses paraphrases, semantic search can miss exact identifiers and names, and naive RAG systems produce plausible answers without sufficient evidence or access-control guarantees.

The system must retrieve the best permitted evidence, explain where the answer came from, abstain when evidence is insufficient, and remain observable and responsive under realistic indexing and query load.

## 3. Users and Jobs to Be Done

### Primary personas

- Knowledge worker: finds a reliable answer across an internal collection and verifies it against cited passages.
- Workspace administrator: creates collections, controls access, monitors ingestion, and removes or reindexes sources.
- Search/relevance engineer: compares retrieval strategies, diagnoses failures, and measures quality, latency, throughput, and cost.
- Platform operator: observes service health, manages capacity, and investigates failed jobs or degraded dependencies.

### Core user stories

1. As a knowledge worker, I want to ask a natural-language question across an authorized collection so that I can find an answer without knowing the source vocabulary.
2. As a knowledge worker, I want each material claim linked to an exact source chunk so that I can verify the answer in context.
3. As a knowledge worker, I want the system to say when evidence is insufficient so that absence of evidence is not disguised as certainty.
4. As an administrator, I want to ingest files and inspect progress or failures so that I know when a collection is searchable.
5. As an administrator, I want deleted or revoked content to disappear from retrieval within a defined interval so that search does not expose stale data.
6. As an administrator, I want workspace and collection access enforced during retrieval so that users cannot obtain evidence they are not permitted to see.
7. As a relevance engineer, I want to run a versioned benchmark against BM25, dense, hybrid, and reranked configurations so that changes are supported by comparable evidence.
8. As a platform operator, I want traces and metrics for each retrieval stage so that I can locate quality, cost, or latency regressions.

## 4. Goals

### User goals

- Return a useful, evidence-grounded answer for at least 80% of answerable benchmark queries.
- Make every citation open the exact supporting passage plus document context.
- Never return evidence outside the requester’s authorized workspace and collections.
- Stream the first visible answer content within 1.5 seconds at p95 when dependencies are healthy and cached retrieval is not required.

### Engineering and portfolio goals

- Demonstrate a measurable relevance lift from hybrid retrieval and reranking over lexical and dense baselines.
- Support reproducible offline evaluation of Recall@K, MRR, nDCG@K, answer faithfulness, citation correctness, latency, throughput, and estimated cost/query.
- Sustain 20 retrieval requests/second and 20 full API requests/second with deterministic fake model adapters in the documented local test environment; measure live Cohere end-to-end latency separately at trial-compatible concurrency.
- Index at least 10,000 chunks using real Cohere embeddings with resumable, idempotent background jobs; use deterministic synthetic vectors for larger infrastructure-only scale tests.
- Make retrieval policy decisions visible in traces: query classification, candidate counts, fusion, reranking, retry/rewrite, cache use, and final evidence.

## 5. Non-Goals for v1

- Full parity with commercial enterprise search products: dozens of connectors, global scale, and enterprise procurement features are separate initiatives.
- Autonomous agents or tool execution: v1 answers questions from indexed evidence only.
- Training proprietary embedding, reranking, or generation models: the project uses Cohere through stable provider interfaces.
- Pixel-perfect collaboration UI: the frontend exists to exercise and explain the search system.
- Fine-grained permission mirroring for every external source: v1 enforces workspace and collection authorization; source-native ACL synchronization is a future extension.
- Cloud or Kubernetes deployment: the project is designed and tested locally; operating public infrastructure is outside the project goal.

## 6. Product Scope

Requirements are assigned to delivery tiers. P0 establishes measurable search, P0.5 turns it into a cited-answer product, and P1 adds the distinctive adaptive and production-style features. All three tiers remain part of the planned project; the tiers define build order and runnable checkpoints.

### P0 — search foundation

#### Ingestion and lifecycle

- Upload PDF, Markdown, plain text, and HTML files into a collection.
- Extract normalized text, document structure where available, source metadata, content hashes, and stable source locators.
- Chunk documents with configurable structure-aware rules and overlap; assign stable document and chunk IDs.
- Submit asynchronous, retryable jobs for extraction, chunking, embedding, and OpenSearch index writes.
- Expose ingestion state: queued, processing, partially indexed, ready, failed, or deleting.
- Make jobs idempotent and resumable; unchanged content must not be embedded twice.
- Deleting a document removes its chunks from OpenSearch, invalidates affected caches, and preserves an audit record.

Acceptance criteria:

- Re-uploading unchanged content creates no duplicate searchable chunks.
- A failed index write can be retried without duplicating chunks or leaving partially indexed documents permanently searchable.
- A deleted document is absent from new retrieval results within 60 seconds.
- Unsupported or corrupt files fail with a user-visible reason while other jobs continue.

#### Retrieval

- BM25 retrieval runs in OpenSearch.
- Dense retrieval runs in OpenSearch using vectors produced by Cohere Embed.
- BM25 and dense retrieval run as independently measurable paths over the same OpenSearch index.
- Metadata filters support collection, source type, author, and created/updated time ranges.
- Reciprocal Rank Fusion (RRF) combines lexical and dense candidate lists with a versioned configuration.
- A Cohere reranker scores the fused candidate set and returns the final evidence set.
- Retrieval returns scores, rank changes, source metadata, chunk text, and timing for each stage to internal observability/evaluation paths.
- Query-time dependency failures degrade explicitly: reranker failure falls back to fused results, while OpenSearch failure returns an error rather than an unsupported answer.

Acceptance criteria:

- Exact identifiers and quoted phrases can be recovered by lexical search even when dense similarity is weak.
- Paraphrased benchmark questions can recover relevant chunks through dense search.
- Metadata filters are applied consistently to both retrieval paths.
- A reranker timeout does not exceed the request deadline and records the fallback in the response metadata and trace.

#### Evaluation

- Store a versioned evaluation dataset containing corpus snapshot, query, graded relevant chunk/document IDs, expected answer or key facts, and metadata slices.
- Compare at minimum: BM25, dense, hybrid, and hybrid plus rerank. Add adaptive retrieval to the comparison when P1 is implemented.
- Compute Recall@5/10/50, MRR@10, nDCG@10, citation precision/recall, answer faithfulness, p50/p95/p99 latency, queries/second, indexing throughput, and estimated cost/query as applicable to each tier.
- Separate retrieval evaluation from generation evaluation so regressions can be localized.
- Produce machine-readable JSON plus a human-readable report with configuration and dataset version.
- CI runs a small deterministic relevance regression suite; full benchmarks run manually or on a schedule.

Acceptance criteria:

- The same dataset/configuration/seed reproduces ranking metrics within a documented tolerance.
- A pull request fails when a protected metric drops beyond its configured regression budget.
- Reports include confidence intervals or per-query deltas where practical, not only aggregate scores.

#### Local runtime and baseline diagnostics

- Docker Compose starts FastAPI, Dramatiq workers, PostgreSQL, OpenSearch, Redis, and the thin frontend with documented seed data.
- Services expose basic health/readiness checks and emit structured logs with request/job IDs.
- OpenSearch memory is configurable for the target MacBook Pro environment.

### P0.5 — cited-answer product

#### Demo identity and access model

- The local application provides pre-created demo identities and does not implement passwords, signup, or external authentication.
- A user explicitly selects a demo identity; the backend resolves it to a seeded user and enforces that user's permissions on every request.
- Demo users belong to one or more workspaces, and an administrator can adjust collection permissions.
- All persistent records and indexes carry a `tenant_id`; collections carry explicit member access.
- Every ingestion, retrieval, cache, citation, and deletion path is tenant-scoped.
- Retrieval applies access filters before candidates can reach reranking or generation.
- Security tests attempt cross-tenant and unauthorized-collection retrieval.
- Documentation clearly states that authorization is implemented and tested while identity verification is simulated.

Acceptance criteria:

- Given two tenants containing identical and unique terms, when a selected demo user queries one tenant, then no result, citation, cache entry, trace payload, or answer contains the other tenant’s content.
- Given a demo user without collection access, when they query or fetch a citation directly, then the service returns no protected content.

#### Answer generation and citations

- The generator receives only the final authorized evidence and a prompt that requires grounded answers and abstention.
- Answers stream to the client using Server-Sent Events.
- Each citation maps to immutable chunk ID, document ID, source locator, and quoted supporting span.
- Citation links open a source view centered on the chunk with surrounding context.
- The API distinguishes `answered`, `partial`, `insufficient_evidence`, and `failed` outcomes.

Acceptance criteria:

- Every material factual claim in the benchmark answer has at least one supporting citation.
- Citation IDs always resolve to evidence included in that request and remain authorization-checked when opened.
- When supplied evidence is below the sufficiency threshold, the system abstains or clearly limits its answer.
- Client cancellation stops generation and releases request resources.

#### API and thin UI

- FastAPI exposes versioned APIs for demo identity selection, workspaces, collections, documents, jobs, search/debug, answers, citations, and evaluations.
- The Next.js UI supports demo-user switching, collection management, uploads, job status, query/answer streaming, citation inspection, and an optional retrieval-debug panel.
- The default query view shows only the answer and citations; a `Show search details` control reveals retrieved chunks, lexical/vector ranks, fused scores, reranking changes, adaptive path when available, fallbacks, and stage latency.
- Public response schemas and error codes are documented with OpenAPI.
- Request IDs and idempotency keys are supported where retries could create duplicate work.

### P1 — differentiators and production-style hardening

#### Adaptive retrieval

- A lightweight policy selects one of three versioned paths:
  - Fast: lexical and/or dense retrieval with a small candidate budget and no rerank.
  - Standard: hybrid retrieval, fusion, and reranking.
  - Deep: query decomposition or reformulation followed by a second retrieval pass and reranking.
- The initial policy uses interpretable signals such as lexical specificity, retrieval score margin, agreement between retrievers, candidate diversity, and evidence sufficiency.
- Hard caps constrain retrieval passes, candidates, wall time, and provider spend.
- The chosen path and reason are logged and included in evaluation output.
- A feature flag can force a path or disable adaptation for experiments and incident mitigation.

Acceptance criteria:

- A high-confidence factual lookup can bypass reranking only when the configured confidence threshold is met.
- A low-confidence or explanatory query can perform at most one rewrite/decomposition round in v1.
- Adaptive mode is evaluated against always-rerank and never-rerank baselines for quality, latency, and cost.

#### Caching and traffic control

- Redis caches embeddings for normalized queries, retrieval results where authorization scope is part of the key, and safe reusable provider outputs.
- Cache entries include tenant, collection/access scope, index version, model version, retrieval configuration, and normalized query.
- Ingestion, deletion, access changes, and configuration changes invalidate or version out affected entries.
- Per-user and per-workspace rate limits protect query and ingestion endpoints.
- Backpressure limits queue depth and concurrent provider calls.

#### Observability and operations

- Add metrics and distributed traces correlated with the request/job IDs established in P0.
- Record stage latency, candidate counts, cache hit rate, provider errors, fallback frequency, token usage, and estimated cost. Raw content capture follows the confirmed local `CAPTURE_QUERY_CONTENT` policy.
- Expand baseline health/readiness checks with dependency-level status.
- Provide dashboards for query SLOs, indexing health, dependency errors, and adaptive-path distribution.

#### Generic message archive ingestion

- Import a generic JSON message archive containing channels, authors, timestamps, message IDs, text, and optional reply/thread relationships.
- Ship a synthetic Discord-style conversation archive for demos and evaluation; do not connect to Discord or include private user exports in the repository.
- Group messages into bounded, conversation-aware chunks while retaining message-level source locators.
- Message citations resolve to the specific message and display enough surrounding conversation for verification.

### P1 follow-ups — optional after the planned project

- One real OAuth-based connector with incremental sync and cursor checkpoints.
- Source-native document ACL synchronization and permission-change propagation tests.
- Admin relevance dashboard with side-by-side result comparison and query failure labeling.
- User feedback on answers and citations feeding an evaluation queue.
- Semantic/result caching with conservative similarity and authorization constraints.

### P2 — architectural extensions

- Multilingual collections and cross-lingual retrieval.
- Image/table-aware document parsing and multimodal evidence.
- Learning-to-rank from feedback and hard-negative mining.
- Federated search across connectors without copying all content.
- Regional data residency, customer-managed encryption keys, SSO/SCIM, retention policies, and legal holds.
- Multi-hop retrieval beyond one bounded rewrite pass.

## 7. System Architecture

```text
                         +----------------------+
Files / connectors ----> | Ingestion API        |
                         +----------+-----------+
                                    |
                              durable job queue
                                    |
                         +----------v-----------+
                         | Indexing workers     |
                         | parse/chunk/embed    |
                         +----+------------+----+
                              |            |
                    +---------v-----------------------+
                    | OpenSearch                     |
                    | text + metadata + vectors      |
                    | BM25 path + dense search path  |
                    +----------------+---------------+
                                     |
User -> Query API -> auth/filter -> parallel retrieval
                                     |
                                    v
                           RRF / adaptive policy
                                    |
                              Cohere rerank
                                    |
                           evidence sufficiency
                                    |
                           grounded generation
                                    |
                         SSE answer + citations

PostgreSQL: tenants, users, collections, documents, jobs, configs, audit/eval metadata
Redis: queue coordination, rate limits, bounded/versioned caches
OpenTelemetry: traces, metrics, logs
Local filesystem storage: originals, parsed artifacts, evaluation reports
```

### Key design decisions

1. Use one OpenSearch index for text, metadata, and vectors. BM25 and dense retrieval remain separate, measurable query paths, while a single index avoids unjustified cross-database consistency work.
2. Use Cohere for embeddings, reranking, and generation, with a separate adapter interface for each capability so tests can use deterministic fakes.
3. Use application-owned retrieval orchestration rather than a high-level RAG framework so ranking behavior, deadlines, and telemetry remain visible.
4. Store immutable, content-derived document/chunk identities where possible. Index versions and configuration versions make caches and evaluations reproducible.
5. Treat authorization as a retrieval invariant, not a post-filtering step.

## 8. Data Model

Minimum entities:

- `Tenant`: isolation and billing/rate-limit boundary.
- `User` and `Membership`: identity and workspace role.
- `Collection` and `CollectionGrant`: searchable scope and authorization.
- `Document`: source identity, filesystem storage key, hash, metadata, lifecycle state, index version.
- `Chunk`: stable ID, ordered text span, source locator, metadata, token count.
- `IngestionJob`: state, attempts, checkpoints, errors, and timing.
- `RetrievalConfig`: versioned candidate budgets, fusion weights, thresholds, deadlines, and model versions.
- `QueryTrace`: sanitized query metadata, chosen path, timings, ranks, fallbacks, and cost.
- `EvaluationDataset`, `Judgment`, and `EvaluationRun`: reproducible relevance experiments.
- `AuditEvent`: access, ingestion, deletion, and configuration changes.

Query trace content capture is controlled by `CAPTURE_QUERY_CONTENT`. It defaults to enabled in the local synthetic environment so questions, retrieved chunks, answers, and citations can be inspected during debugging and evaluation. When disabled, traces retain only IDs, ranks, scores, timings, model/configuration versions, usage, fallbacks, and errors. Secrets and credentials are never logged.

## 9. Query Contract and Latency Budget

Reference target for a healthy, warm system:

| Stage | p95 budget |
|---|---:|
| Demo identity resolution, policy, normalization | 25 ms |
| Parallel BM25 + dense retrieval | 120 ms |
| Fusion and filtering | 15 ms |
| Reranking when selected | 250 ms |
| Evidence assembly | 20 ms |
| Generation time to first token | 1,000 ms |
| End-to-end first visible content | 1,500 ms |

The request owns an absolute deadline propagated to every dependency. Adaptive paths trade quality against remaining latency and cost budgets. Retrieval-only endpoints have a 500 ms p95 target for the standard path in the reference environment.

## 10. Quality and Performance Targets

Targets are hypotheses until the first baseline run establishes achievable values.

| Metric | Ship threshold | Stretch |
|---|---:|---:|
| Recall@10, hybrid + rerank | >= 0.82 | >= 0.88 |
| nDCG@10, hybrid + rerank | >= 0.78 | >= 0.84 |
| nDCG@10 lift over dense baseline | >= 10% relative | >= 18% relative |
| Citation precision | >= 0.90 | >= 0.95 |
| Cross-tenant leakage tests | 0 failures | 0 failures |
| Retrieval p95, standard path | <= 500 ms | <= 300 ms |
| Retrieval-only availability in local load test | >= 99.5% | >= 99.9% |
| Retrieval-only throughput against OpenSearch | >= 20 QPS | >= 50 QPS |
| Full API throughput with deterministic fake model adapters | >= 20 QPS | >= 50 QPS |
| Indexing success after retries | >= 99.5% | >= 99.9% |

Live Cohere end-to-end tests run at trial-compatible low concurrency and report p50/p95 stage and time-to-first-token latency; they do not claim a sustained-QPS target. Real Cohere embeddings are used for relevance benchmarks and for the corpus up to 10,000 chunks. Larger ingestion/load tests use deterministic synthetic vectors and fake model adapters.

Benchmark reports must state corpus size, hardware, concurrency, cache state, adapter type (live or fake), provider/model versions, and whether latency includes network calls. Results from live-provider and fake-adapter tests are reported separately and never combined into a single throughput claim.

## 11. Reliability, Security, and Privacy

- TLS and secrets management are required outside local development; secrets never enter source control or logs.
- Inputs have size/type limits; extracted content is treated as untrusted data and isolated from system instructions.
- Prompt injection in documents must not grant tool access, alter authorization, or override answer/citation policy.
- Timeouts, bounded retries with jitter, circuit breakers, and bulkheads protect external providers.
- At-least-once jobs plus idempotent writes are the default processing model.
- Index state is tied to document lifecycle status so partially indexed documents cannot be treated as ready.
- Local backup/restore procedures cover PostgreSQL and original documents; search indexes are rebuildable.
- Dependency and container scanning, static analysis, unit/integration tests, and secret scanning run in CI.
- Audit events record administrative and content-lifecycle actions without exposing document bodies.

## 12. Test Strategy

- Unit: chunking, normalization, RRF, filters, policy thresholds, cache keys, citations, and metric calculations.
- Contract: provider adapters and API schemas using recorded/synthetic responses.
- Integration: PostgreSQL, Redis, OpenSearch lexical/vector search, workers, retries, and deletion.
- Security: tenant isolation, direct-object access, cache isolation, malicious metadata, and prompt-injection corpus cases.
- Relevance: curated judgments, adversarial exact-match/paraphrase queries, hard negatives, and metadata slices.
- Load: OpenSearch retrieval-only and full API with deterministic fake model adapters, covering warm/cold cache, mixed adaptive paths, ingestion/query contention, dependency slowdown, and recovery.
- Live-provider performance: trial-compatible low-concurrency Cohere runs measuring stage latency, time to first token, usage, and estimated cost without asserting provider throughput.
- End-to-end: upload to searchable state, query streaming, source opening, revoke/delete, and evaluation report generation.

## 13. Rollout Phases

1. Retrieval laboratory: corpus format, BM25/dense baselines, metrics, and reproducible evaluation CLI.
2. Search core: OpenSearch ingestion, separate BM25/dense query paths, RRF, metadata filters, reranking, and debug API.
3. Grounded answer product: authorization, generation, citations, streaming, and minimal UI.
4. Production-style hardening: caching, rate limits, deadlines, fallbacks, observability, security tests, and local load tests.
5. Adaptive retrieval: policy, rewrite path, ablation study, and tuning against quality/latency/cost.
6. Reproducibility and final report: one-command local environment, seeded demo, benchmark results, and architecture/tradeoff documentation.

Each phase must retain a runnable vertical slice and publish benchmark deltas; infrastructure work must not postpone the first relevance baseline.

## 14. Release Definition

The project is portfolio-ready when a reviewer can:

1. Start the stack from documented commands and load a representative corpus.
2. Ask exact-match, semantic, filtered, and explanatory questions and inspect cited source spans.
3. Observe at least one adaptive fast-path and one deep-path decision in a trace.
4. Run the benchmark and compare BM25, dense, hybrid, reranked, and adaptive results.
5. Run a documented load test and inspect latency, throughput, cost, caching, and fallback behavior.
6. Verify deletion and tenant-isolation tests.
7. Read an engineering report explaining at least one measured relevance-versus-latency tradeoff.

## 15. Open Questions

### Blocking before implementation planning

- None currently.

### Non-blocking during early implementation

- None currently.

## 16. Explicit Tradeoffs and Risks

- A single OpenSearch index reduces operational complexity but creates one retrieval failure domain; the project prioritizes relevance engineering over cross-database coordination.
- Provider reranking can improve relevance while adding cost, network variance, and tail latency; adaptive retrieval exists to quantify and control this tradeoff.
- A synthetic enterprise corpus enables safe reproducibility but may overstate real-world quality; include noisy, duplicated, stale, conflicting, and permissioned documents.
- LLM-based faithfulness graders are scalable but imperfect; validate a sample manually and keep retrieval judgments independent.
- Local benchmarks do not prove internet-scale operation; reports must state that limitation rather than presenting projected capacity as deployed performance.

## 17. Decision Log

| Area | Status | Current choice | Rationale |
|---|---|---|---|
| Runtime scope | **Confirmed** | Local only; no cloud deployment | Focus effort on retrieval, evaluation, and system quality |
| Evaluation corpora | **Confirmed** | Synthetic company workspace plus BEIR SciFact | Enterprise-style demo scenarios plus a small, recognized external retrieval benchmark |
| Demo identity and access | **Confirmed** | Pre-created demo users with adjustable roles and collection permissions | Demonstrates authorization and isolation without building account lifecycle features |
| Search UI detail | **Confirmed** | Clean answer/citation view plus optional search-details panel | Demonstrates both the user experience and retrieval engineering |
| Message ingestion | **Confirmed** | Generic JSON archive plus synthetic Discord-style conversations | Exercises conversational retrieval without a live external connector or private data |
| Query trace content | **Confirmed** | Configurable capture, enabled for local synthetic data | Improves debugging and evaluation while preserving a production-conscious off switch |
| Synthetic relevance labels | **Confirmed** | Assisted authoring, manual verification, version-controlled JSON; no Cohere API generation | Conserves trial usage and avoids provider-induced benchmark bias |
| Performance measurement | **Confirmed** | Real Cohere for relevance and low-concurrency latency; fake adapters/vectors for sustained load and larger scale | Respects trial limits and keeps reported claims honest |
| Background jobs | **Confirmed** | Dramatiq with Redis; durable job state in PostgreSQL | Provides retries and worker isolation with less framework overhead than Celery |
| Local environment | **Confirmed** | Docker Compose with configurable OpenSearch memory | Reproducible full-stack development on the target MacBook Pro |
| API/backend | **Confirmed** | Python + FastAPI | Strong ML ecosystem, async API support, typed schemas |
| Application system of record | **Confirmed** | PostgreSQL | Authoritative users, permissions, document metadata/lifecycle, jobs, configurations, and evaluations |
| Original file storage | **Confirmed** | Regular local filesystem behind a storage interface | Minimal local complexity while preserving clean storage-key boundaries |
| Search storage | **Confirmed** | OpenSearch for BM25 and vector retrieval | Supports hybrid search without unnecessary cross-database consistency work |
| Cache/coordination | **Confirmed** | Redis | Dramatiq broker, rate limiting, caching, and temporary coordination; never authoritative state |
| Model provider | **Confirmed** | Cohere Embed, Rerank, and Chat through adapters | Cohesive search stack, native citations, and direct relevance to project goal |
| Frontend | **Confirmed** | Thin Next.js/React/TypeScript client | Polished streaming and citation experience with all application logic retained in FastAPI |
