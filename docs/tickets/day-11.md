# Day 11 — Dense vector retrieval

[Issue #13](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/13)

Day 11 adds the semantic retrieval path beside Day 10's exact-word BM25 path.
It is intentionally a library and debug command, not a public search endpoint:
hybrid fusion and authorization grants arrive in later tickets.

```text
raw query
    ↓ trim outer whitespace; collapse whitespace runs
normalized query
    ↓ Cohere Embed, input_type=search_query
one validated 384-dimensional query vector
    ↓ cosine k-NN + tenant / optional collection / ready filters
OpenSearch read alias
    ↓
RetrievalResult evidence + DenseRetrievalTrace diagnostics
```

## Contract and choices

`normalize_dense_query` preserves the query's spelling, punctuation, and case.
It only turns whitespace runs into one space and removes leading/trailing
whitespace. A blank normalized query performs neither an embedding call nor an
OpenSearch request.

`OpenSearchDenseRetriever` sends the normalized value as a single
`EmbeddingRequest` with `EmbeddingPurpose.QUERY`; Cohere translates that to
`input_type="search_query"`. This matters because Cohere's document and query
embedding spaces are purpose-specific. There is deliberately no embedding
cache in this increment: the duplicate-query cache could pre-empt the planned
P1 cache work, where a single owner and invalidation policy will be chosen.

The default candidate budget is 20 and can be changed with
`OPENSEARCH_DENSE_CANDIDATE_LIMIT`. k-NN uses the existing `embedding` field,
whose `cosinesimil` mapping was introduced in Day 9, against the stable read
alias. Every request applies `tenant_id` and `visibility=ready` filters; a
collection filter is included only when requested.

Dense hits become the existing `RetrievalResult` contract used by BM25, so Day
12 can fuse candidates without knowing OpenSearch's raw JSON shape. The
separate `DenseRetrievalTrace` records the normalized query, model, configured
dimension, provider adapter, schema version, read alias, filters, candidate
budget/count, and independent provider/OpenSearch elapsed times.

## Version and dimension safety

The configured Cohere model/dimension pair is validated at settings creation:
the baseline `embed-english-light-v3.0` must use 384 dimensions. The same
dimension config creates the OpenSearch vector mapping, and the dense config
validates the configured index schema version using the existing physical-index
naming rules. At query time, the retriever rejects a provider response unless
it is exactly one `QUERY` vector whose model and dimension match that
configuration and whose values are finite.

This is the agreed lean compatibility approach. The retriever trusts the
configured model/index pairing and records both values in its trace; it does
not make an OpenSearch mapping lookup for every query. A rebuild is still
required whenever the configured model, dimensions, or schema version changes.

## Debugging locally

After a complete index rebuild and with `COHERE_API_KEY` configured, inspect a
dense query without adding a public HTTP API:

```bash
.venv/bin/python -m hybrid_rag_search.dense_retrieval \
  --tenant <tenant-uuid> \
  --collection <collection-uuid> \
  'How long does a password recovery link last?'
```

The output is the normalized evidence and its dense trace. Normal tests use a
scripted fake adapter whose semantic/paraphrase vectors are deterministic.
After `make stack-up`, run the real-OpenSearch smoke test with:

```bash
make opensearch-dense-test
```

It indexes a known vector, queries it with the paraphrase fixture, verifies the
expected chunk returns, and verifies a different tenant receives no result.

## Deferred

This ticket does not execute BM25 alongside dense retrieval, fuse ranks,
rerank, cache embeddings, perform authorization grants, or expose search over
HTTP/UI. Those choices remain deliberately owned by their later tickets.
