# Retrieval baseline: synthetic-workspace

## Reproduction metadata

```json
{
  "adapters": {
    "embedding": "cohere",
    "rerank": "cohere",
    "retrieval": "opensearch"
  },
  "cache_state": "Repeat on same compacted static index; caches not flushed; provider cache unknown; paced trial-key requests",
  "chunk_count": 8,
  "code_revision": "258d194fc1275078896bef75582aa641fab01693",
  "collection_id": "a05c32f7-1d8f-56f6-971a-c91fb8e44007",
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
  "created_at": "2026-10-03T17:54:52.403566+00:00",
  "dataset": {
    "corpus_hash": "f88d309e8d848c75580779b6bfaf1920fc112d64e097532bad62040be2685951",
    "dataset_hash": "77cb42d9226bbf4c0feac1171a0e6b2d2243f5ec1df01edd5b6a02dc6d7db202",
    "dataset_id": "synthetic-workspace",
    "document_count": 8,
    "query_count": 12,
    "version": "1"
  },
  "execution": "sequential queries and modes; hybrid branches concurrent; no query embedding cache",
  "hardware": {
    "logical_cpus": 8,
    "machine": "arm64",
    "processor": "arm",
    "system": "macOS-26.6.2-arm64-arm-64bit"
  },
  "index": [
    "hybrid-rag-eval-a05c32f7-1d8f-56f6-971a-c91fb8e44007"
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
| bm25 | 0.875000 | 0.875000 | 0.875000 | 0.833333 | 0.848265 |
| dense | 0.958333 | 1.000000 | 1.000000 | 0.916667 | 0.921654 |
| hybrid | 1.000000 | 1.000000 | 1.000000 | 0.895833 | 0.920371 |
| hybrid_rerank | 1.000000 | 1.000000 | 1.000000 | 0.875000 | 0.891712 |

## Per-query differences against BM25

| Query | Mode | Δ Recall@5 | Δ Recall@10 | Δ Recall@50 | Δ MRR@10 | Δ nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| q-exact-policy-id | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.166009 |
| q-exact-policy-id | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-exact-policy-id | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.166009 |
| q-exact-channel | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-exact-channel | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-exact-channel | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-exact-adr | dense | +0.500000 | +0.500000 | +0.500000 | +0.000000 | +0.082681 |
| q-exact-adr | hybrid | +0.500000 | +0.500000 | +0.500000 | +0.000000 | +0.082681 |
| q-exact-adr | hybrid_rerank | +0.500000 | +0.500000 | +0.500000 | +0.000000 | +0.082681 |
| q-paraphrase-learning | dense | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.369070 |
| q-paraphrase-learning | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.369070 |
| q-paraphrase-learning | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.500000 | +0.369070 |
| q-paraphrase-travel | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-paraphrase-travel | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-paraphrase-travel | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-paraphrase-outage | dense | -0.500000 | +0.000000 | +0.000000 | +0.000000 | -0.036001 |
| q-paraphrase-outage | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.017158 |
| q-paraphrase-outage | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.026242 |
| q-current-meal-limit | dense | +1.000000 | +1.000000 | +1.000000 | +1.000000 | +1.000000 |
| q-current-meal-limit | hybrid | +1.000000 | +1.000000 | +1.000000 | +0.250000 | +0.430677 |
| q-current-meal-limit | hybrid_rerank | +1.000000 | +1.000000 | +1.000000 | +0.500000 | +0.630930 |
| q-old-meal-limit | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-old-meal-limit | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-old-meal-limit | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-current-expense-date | dense | +0.000000 | +0.000000 | +0.000000 | -0.500000 | -0.369070 |
| q-current-expense-date | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-current-expense-date | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | -0.500000 | -0.369070 |
| q-source-of-truth | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-source-of-truth | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-source-of-truth | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-opensearch-role | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-opensearch-role | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-opensearch-role | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-incident-approval | dense | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-incident-approval | hybrid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| q-incident-approval | hybrid_rerank | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

JSON contains winning chunks, complete chunk rankings, timings, and traces.
Unjudged documents earn zero credit; their relevance is unknown.
