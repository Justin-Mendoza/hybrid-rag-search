# Day 10 — BM25 lexical retrieval

Day 9 made complete chunk generations searchable. This increment adds the
lexical path that can recover exact words, quoted wording, and identifiers
before dense retrieval and hybrid fusion are introduced.

```text
query text
    ↓ parse ordinary terms / quoted phrases / identifiers
BM25 clauses + tenant, collection, and ready filters
    ↓
OpenSearch read alias
    ↓
normalized RetrievalResult values + separate RetrievalTrace
```

## Query contract

`content_text` is the primary BM25 field. `heading_text`, derived from the
source-span heading paths already stored by Day 9, receives a modest boost.
This retains source-faithful passage text for citations while allowing a query
to match useful section context.

The parser deliberately recognizes only three constructs:

- ordinary words, such as `password timeout`, use a `multi_match` BM25 clause;
- a complete quoted phrase, such as `"reset links expire"`, uses a phrase
  clause; and
- identifier-shaped tokens, such as `HR-104`, `RFC-9110`, and `ERR_AUTH_42`,
  use an exact `identifiers` keyword clause.

OpenSearch query-string syntax is not exposed. Every nonempty search filters
by tenant and `visibility=ready`; a collection UUID can add a further filter.
Blank or punctuation-only input makes no OpenSearch request and returns an
empty result set.

## Result and trace boundary

The retriever translates OpenSearch hits into `RetrievalResult` objects with
the chunk, tenant/collection/document IDs, BM25 rank and score, source spans,
and generation metadata. `RetrievalTrace` separately retains parsed query
parts, candidate count, elapsed time, filters, configured boosts, and optional
OpenSearch explanations. Day 11 dense retrieval and Day 12 fusion can use the
result object without depending on raw OpenSearch JSON.

The configurable local defaults are:

```text
content boost:    1.0
heading boost:    1.5
phrase boost:     3.0
identifier boost: 5.0
candidate limit:  20
```

Set the corresponding `OPENSEARCH_BM25_*` values in `.env` to experiment with
them. These mapping changes require `chunks-v2`; create and rebuild that index
before expecting a pre-Day-10 local index to support heading or identifier
search.

## Debugging locally

After the index rebuild has completed, run an inspectable lexical query without
introducing the later public Search API:

```bash
.venv/bin/python -m hybrid_rag_search.lexical_retrieval \
  --tenant <tenant-uuid> \
  --collection <collection-uuid> \
  'HR-104 "password reset" timeout'
```

The command enables OpenSearch explanations and prints the normalized results
and trace. `make opensearch-lexical-test` runs the optional real-OpenSearch
smoke test after `make stack-up`; normal tests use deterministic HTTP transports.

## Deferred

Dense query embeddings and k-NN retrieval belong to Day 11. Metadata range
filters, reciprocal-rank fusion, authorization grants, public HTTP endpoints,
and UI search remain later tickets.
