# Retrieval baseline: scifact-subset

## Reproduction metadata

```json
{
  "adapters": {
    "embedding": "cohere",
    "rerank": "cohere",
    "retrieval": "opensearch"
  },
  "cache_state": "Prepared static compacted index; caches not deliberately flushed; provider cache unknown",
  "chunk_count": 771,
  "code_revision": "d22e6633efe352f0cd758bd5476ba1de8ca59126",
  "collection_id": "45d8edc3-ed87-5b05-92c7-178a65373e2f",
  "configuration": {
    "cohere_embed_dimensions": 384,
    "cohere_rerank_timeout_seconds": 5.0,
    "cohere_timeout_seconds": 10.0,
    "opensearch_bm25_candidate_limit": 100,
    "opensearch_bm25_content_boost": 1.0,
    "opensearch_bm25_heading_boost": 1.5,
    "opensearch_bm25_identifier_boost": 5.0,
    "opensearch_bm25_phrase_boost": 3.0,
    "opensearch_dense_candidate_limit": 100,
    "opensearch_index_schema_version": "chunks-v3",
    "rerank_candidate_limit": 100,
    "rerank_result_limit": 100,
    "retrieval_deadline_seconds": 8.0,
    "rrf_rank_constant": 60
  },
  "created_at": "2026-10-03T20:12:52.117824+00:00",
  "dataset": {
    "corpus_hash": "c05084c8828f33d70a6d7c072c9c617d65777a3c77fdc11bd4cf60a8db92a1e0",
    "dataset_hash": "c7bdf3cc997e2463ef13b531076c84a265de417599e010ca1c151c00bf7f0e18",
    "dataset_id": "scifact-subset",
    "document_count": 500,
    "query_count": 30,
    "version": "scifact-subset-v1"
  },
  "execution": "sequential queries and modes; hybrid branches concurrent; no query embedding cache",
  "hardware": {
    "logical_cpus": 8,
    "machine": "arm64",
    "processor": "arm",
    "system": "macOS-26.6.2-arm64-arm-64bit"
  },
  "index": [
    "hybrid-rag-eval-45d8edc3-ed87-5b05-92c7-178a65373e2f"
  ],
  "metric_rules": "document first occurrence; positive >0; nDCG gain 2^grade-1; macro mean; unjudged=0",
  "models": {
    "embedding": "embed-english-light-v3.0",
    "rerank": "rerank-v4.0-fast"
  },
  "pipeline_version": "pipe_ab3e29ca6b3460cb114fc135c3e942259316fa6efe6b14e135ca341abcbd495f",
  "query_delay_seconds": 7.0,
  "report_schema": "retrieval-evaluation-v1",
  "rerank_fallback_count": 0,
  "tenant_id": "00000000-0000-4000-8000-000000000201",
  "tracked_worktree_dirty": true
}
```

## Aggregate scores

| Mode | Recall@5 | Recall@10 | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.825000 | 0.866667 | 0.966667 | 0.738889 | 0.771419 |
| dense | 0.933333 | 0.933333 | 0.933333 | 0.873333 | 0.888290 |
| hybrid | 0.866667 | 0.933333 | 0.966667 | 0.794444 | 0.826226 |
| hybrid_rerank | 0.966667 | 0.966667 | 0.966667 | 0.911111 | 0.925395 |

## Per-query differences against BM25

| Query | Mode | Δ Recall@5 | Δ Recall@10 | Δ Recall@50 | Δ MRR@10 | Δ nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| scifact-query-1062 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1062 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1062 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1185 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1185 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1185 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1194 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1194 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1194 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1197 | dense | +1.000000 | +1.000000 | +0.000000 | +0.500000 | +0.630930 |
| scifact-query-1197 | hybrid | +1.000000 | +1.000000 | +0.000000 | +0.500000 | +0.630930 |
| scifact-query-1197 | hybrid_rerank | +1.000000 | +1.000000 | +0.000000 | +0.500000 | +0.630930 |
| scifact-query-124 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-124 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-124 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1241 | dense | +1.000000 | +1.000000 | +0.000000 | +1.000000 | +1.000000 |
| scifact-query-1241 | hybrid | +0.000000 | +1.000000 | +0.000000 | +0.166667 | +0.356207 |
| scifact-query-1241 | hybrid_rerank | +1.000000 | +1.000000 | +0.000000 | +1.000000 | +1.000000 |
| scifact-query-1316 | dense | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.369070 |
| scifact-query-1316 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1316 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | -0.166667 | -0.130930 |
| scifact-query-1332 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1332 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1332 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1344 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1344 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1344 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-1370 | dense | +0.000000 | +0.000000 | +0.000000 | +0.666667 | +0.500000 |
| scifact-query-1370 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.666667 | +0.500000 |
| scifact-query-1370 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.666667 | +0.500000 |
| scifact-query-179 | dense | +0.250000 | +0.000000 | +0.000000 | +0.000000 | +0.038001 |
| scifact-query-179 | hybrid | +0.250000 | +0.000000 | +0.000000 | +0.000000 | +0.038001 |
| scifact-query-179 | hybrid_rerank | +0.250000 | +0.000000 | +0.000000 | +0.000000 | +0.038001 |
| scifact-query-268 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-268 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-268 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-327 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-327 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-327 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-411 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-411 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-411 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-478 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-478 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-478 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-5 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-5 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-5 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-501 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-501 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-501 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-53 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-53 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-53 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-535 | dense | +1.000000 | +1.000000 | +0.000000 | +1.000000 | +1.000000 |
| scifact-query-535 | hybrid | +0.000000 | +1.000000 | +0.000000 | +0.166667 | +0.356207 |
| scifact-query-535 | hybrid_rerank | +1.000000 | +1.000000 | +0.000000 | +1.000000 | +1.000000 |
| scifact-query-575 | dense | +0.000000 | +0.000000 | +0.000000 | +0.666667 | +0.500000 |
| scifact-query-575 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.166667 | +0.130930 |
| scifact-query-575 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.666667 | +0.500000 |
| scifact-query-674 | dense | +0.000000 | +0.000000 | +0.000000 | -0.800000 | -0.613147 |
| scifact-query-674 | hybrid | +0.000000 | +0.000000 | +0.000000 | -0.500000 | -0.369070 |
| scifact-query-674 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-783 | dense | +0.000000 | +0.000000 | +0.000000 | +0.166667 | +0.130930 |
| scifact-query-783 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.166667 | +0.130930 |
| scifact-query-783 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.166667 | +0.130930 |
| scifact-query-793 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-793 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-793 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-834 | dense | +0.000000 | -1.000000 | -1.000000 | -0.166667 | -0.356207 |
| scifact-query-834 | hybrid | +0.000000 | -1.000000 | +0.000000 | -0.166667 | -0.356207 |
| scifact-query-834 | hybrid_rerank | +1.000000 | +0.000000 | +0.000000 | +0.833333 | +0.643793 |
| scifact-query-839 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-839 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-839 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-845 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-845 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-845 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-903 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-903 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-903 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-907 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-907 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-907 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-967 | dense | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.306574 |
| scifact-query-967 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.226294 |
| scifact-query-967 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.306574 |
| scifact-query-982 | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-982 | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| scifact-query-982 | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

JSON contains winning chunks, complete chunk rankings, timings, and traces.
Unjudged documents earn zero credit; their relevance is unknown.
