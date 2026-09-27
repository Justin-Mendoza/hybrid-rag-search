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
