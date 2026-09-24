# Day 13 — Cohere Rerank and deadline-aware fallback

[Issue #11](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/11)

Day 13 adds a second-stage neural ranking step after Day 12's filtered hybrid
retrieval. It remains a Python library and debug command; later tickets expose
the ranking mode through the search API and UI.

```text
one absolute request deadline
            │
            ▼
BM25 + dense → rrf-v1 top 20
                     │
                     ▼
       Cohere Rerank (at most 5s)
              │             │
           success       retryable failure
              │             │
       reranked top 10   RRF top 10
```

## Ranking contract

`RerankedRetriever` requests 20 hybrid candidates by default and sends their
stable chunk IDs and source-faithful content text to the existing application-
owned reranking provider. Cohere returns 10 items by default. The coordinator
rejects missing, duplicate, reordered-index, or unknown chunk mappings instead
of attaching a score to the wrong evidence.

Each final result retains its complete hybrid result, final rank, Cohere
relevance score, and `rank_change = fused_rank - final_rank`. A positive rank
change means the candidate moved upward. Raw BM25, vector, RRF, and Cohere
scores remain diagnostics; only Cohere rank determines the successful final
order.

## Deadline and fallback contract

The caller may pass an absolute monotonic deadline. Without one, the configured
default is eight seconds from request start. Hybrid retrieval runs inside that
same budget. Before reranking, the coordinator computes the remaining time and
passes `min(5 seconds, remaining)` through to the Cohere adapter and SDK request.
Caller cancellation always propagates.

If hybrid retrieval consumes the complete budget, the request fails because no
candidates exist for fallback. After hybrid candidates exist, deadline
exhaustion, timeout, rate limiting, and temporary provider unavailability return
the top 10 RRF candidates in their original order. Authentication, invalid
requests/responses, and candidate mapping failures remain visible errors.

The trace gives later API/UI work an explicit signal:

```text
ranking_mode: cohere_rerank | rrf_fallback
fallback_reason: timeout | rate_limited | unavailable | deadline_exhausted | null
```

It also records the model, adapter, latency, usage/cost, rank counts, total
deadline budget, remaining rerank budget, and complete hybrid trace.

## Configuration

```text
RERANK_CANDIDATE_LIMIT=20
RERANK_RESULT_LIMIT=10
COHERE_RERANK_TIMEOUT_SECONDS=5
RETRIEVAL_DEADLINE_SECONDS=8
```

Run a local inspection with configured Cohere credentials:

```bash
.venv/bin/python -m hybrid_rag_search.reranked_retrieval \
  --tenant <tenant-uuid> \
  "password recovery policy"
```

`make cohere-smoke` remains the explicitly authorized live-provider test. All
normal reranking, fallback, deadline, cancellation, and mapping tests use
deterministic providers and consume no Cohere usage.

## Deferred

The HTTP response, visible UI fallback indicator, retry orchestration, circuit
breakers, caching, and evaluation-based limit/deadline tuning remain owned by
later tickets.
