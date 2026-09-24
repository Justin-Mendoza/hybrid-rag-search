# Day 12 — Metadata filters and reciprocal-rank fusion

[Issue #12](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/12)

Day 12 combines the existing lexical and semantic candidate lists into one
inspectable hybrid ranking. It remains a Python library and debug command;
public search APIs and authorization grants arrive later.

```text
query + one RetrievalFilters value
          ├── BM25 + filters ── lexical ranks ──┐
          └── k-NN + filters ── vector ranks  ──┤
                                                ↓
                                    rrf-v1 fused ranking
```

## Metadata contract

PostgreSQL `Document.source_metadata` is authoritative. During indexing, the
complete object is copied to each chunk's `source_metadata`, while three
supported values receive typed OpenSearch fields:

- `source_type`: normalized exact-match keyword;
- `author`: normalized exact-match keyword; and
- `source_date`: timezone-aware ISO-8601 value normalized to a UTC date.

Invalid supplied supported values fail indexing instead of silently producing
unfilterable chunks. Missing optional values remain absent. Arbitrary additional
metadata is preserved but is not part of the filter API. These mapping changes
introduce `chunks-v3`, so an existing local index must be completely rebuilt.

`RetrievalFilters` is the single clause builder used by BM25 and k-NN. Tenant
and ready visibility are always required. Collection, source type, author, and
date bounds are optional and combine with AND. The start bound is inclusive;
the end bound is exclusive. Filtering happens inside each OpenSearch query, so
each path ranks the best candidates from the same eligible scope.

## Fusion contract

`HybridRetriever` launches both existing retrievers with `asyncio.gather` and
fails the request if either path fails. Candidates are joined by `chunk_id`.
For `rrf-v1`, each candidate receives:

```text
fused score = sum(1 / (60 + rank))
```

A candidate found by both paths receives both contributions; one found by only
one remains eligible. Results sort by fused score descending and then
`chunk_id` ascending for deterministic ties. The default candidate and final
result limit is 20. Each result retains lexical rank/score, vector rank/score,
and fused rank/score. Raw scores are diagnostic only and do not enter fusion.

## Debugging locally

Rebuild and promote a `chunks-v3` index, configure Cohere, and run:

```bash
.venv/bin/python -m hybrid_rag_search.hybrid_retrieval \
  --tenant <tenant-uuid> \
  --source-type email \
  --author "Ada Lovelace" \
  --source-date-from 2026-09-01T00:00:00Z \
  --source-date-to 2026-10-01T00:00:00Z \
  "password recovery policy"
```

After `make stack-up`, `make opensearch-hybrid-test` creates a temporary index
and proves both paths apply identical metadata scope before fusion.

## Deferred

Authorization grants, Cohere reranking, public HTTP/UI search, degraded
single-path fallback, multi-value filters, caching, and evaluation-driven RRF
tuning remain owned by later tickets.
