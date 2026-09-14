# Hybrid RAG Search

A production-style enterprise search and retrieval-augmented generation engine built to explore retrieval quality, reranking, adaptive search, citations, and performance tradeoffs.

The project is intentionally focused on the search system around the language model—not on building another generic “chat with a PDF” interface.

> Status: specification complete; implementation has not started. The system is designed for reproducible local operation and will not be deployed as a public service.

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

Development commands and prerequisites will be added by the first implementation ticket. Until then, there is no runnable application in this repository.

## Scope boundaries

This project will not include cloud deployment, Kubernetes, live Slack or Discord connectors, real account authentication, source-native ACL synchronization, autonomous agents, or proprietary model training.

The complete scope, success targets, design decisions, risks, and release definition are maintained in [SPEC.md](SPEC.md).

