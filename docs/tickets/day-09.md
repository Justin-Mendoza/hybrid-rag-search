# Day 09 — OpenSearch index and indexing service

[Issue #7](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/7)

## Design

All searchable chunks share one versioned physical index. Stable read and write
aliases let a rebuild receive writes without moving live reads until validation
is complete.

```text
original file → Day 8 parse/chunk/embed → staged chunk generation
                                             ↓ complete bulk write
                                      searchable generation
                                             ↓
                                      read alias / retrieval
```

Chunk records contain typed tenant, collection, document, generation, text,
vector, and source-span fields. Retrieval must filter `visibility=ready`.
Vectors use 384 dimensions and cosine similarity for the configured
`embed-english-light-v3.0` model.

## Rebuild

A canonical rebuild intentionally uses a new logical schema version so Day 8
creates new durable jobs instead of reusing successful jobs from an older index.
Set `OPENSEARCH_INDEX_SCHEMA_VERSION` consistently for the command and worker.

```bash
make index-rebuild-start BUILD_ID=20260920
# Wait until every ingestion job finishes and inspect failures.
make index-rebuild-promote INDEX=hybrid-rag-chunks-v2-20260920
```

The start command creates the physical index, moves only the write alias, and
re-enqueues every non-deleted PostgreSQL document from its canonical local
original. Promotion refuses to move the read alias while any document is not
ready. Re-embedding may consume Cohere credits, so neither step is automatic.

Document-generation replacement retains the previous ready generation until
the new bulk write is complete. PostgreSQL readiness follows immediately after
OpenSearch promotion; this small cross-system handoff is intentionally not a
distributed transaction in the local project.
