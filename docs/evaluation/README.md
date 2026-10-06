# Retrieval baselines

The selected [synthetic baseline](synthetic-workspace/baseline/report.md) and its
[JSON evidence](synthetic-workspace/baseline/report.json) use real OpenSearch,
Cohere `embed-english-light-v3.0`, and `rerank-v4.0-fast` over 8 documents and
12 authored queries. No rerank request fell back in the selected run.

The four configurations measure distinct stages; they are not four requests on
the normal product path. Each benchmark mode has a 100-chunk budget, followed
by document deduplication and fixed metric cutoffs.

| Mode | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.875000 | 0.833333 | 0.848265 |
| Dense | 0.958333 | 0.916667 | 0.921654 |
| Hybrid | 1.000000 | 0.895833 | 0.920371 |
| Hybrid + rerank | 1.000000 | 0.875000 | 0.891712 |

On this small fixture hybrid improves coverage, but reranking does not improve
ordering. This is a measured baseline, not a claim that every added stage must
win or that twelve queries establish general search quality. Recall@50 on eight
documents cannot demonstrate larger-corpus behavior. Full SciFact is deferred;
the approved pipeline baseline uses the versioned SciFact subset instead.

## SciFact subset status

`datasets/scifact-subset/v1/` contains 500 documents, 30 queries, and all 34
judgments for those queries. Selection and provenance are recorded in its
manifest and [fixture documentation](../../datasets/scifact-subset/README.md).
This is not a full BEIR benchmark; reduced-corpus scores must not be compared
directly with full SciFact results.

The [selected subset baseline](scifact-subset/baseline/report.md) and
[JSON evidence](scifact-subset/baseline/report.json) use all 500 documents and
771 indexed chunks. Preparation resumed with the owner-configured production
key, reusing 291 completed documents and finishing the remaining 209.
The full corpus's 480 completed documents remain untouched in a separate scope.

Both 30-query runs used all four live modes with zero rerank fallbacks.
Every per-query and aggregate metric matched exactly (maximum absolute drift
`0.0`). For this fixed compacted index and configuration use `1e-12` absolute
metric tolerance; latency remains descriptive. The selected baseline is the
second run. This observation is not a guarantee about future provider changes.

| Mode | Recall@5 | Recall@10 | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.825000 | 0.866667 | 0.966667 | 0.738889 | 0.771419 |
| Dense | 0.933333 | 0.933333 | 0.933333 | 0.873333 | 0.888290 |
| Hybrid | 0.866667 | 0.933333 | 0.966667 | 0.794444 | 0.826226 |
| Hybrid + rerank | 0.966667 | 0.966667 | 0.966667 | 0.911111 | 0.925395 |

On this subset reranking improves early coverage and ordering, unlike the tiny
synthetic fixture. Dense alone beats unreranked hybrid on early ordering, so
fusion is not automatically an improvement. These are observations on a fixed
30-query pipeline fixture, not universal claims or full SciFact benchmark scores.

Reproduce the comparison without provider calls:

```bash
.venv/bin/python -m hybrid_rag_search.evaluation.runner compare \
  .data/evaluation/scifact-subset-first/report.json \
  .data/evaluation/scifact-subset-repeat/report.json
```

For new live runs use `make evaluation-run EVAL_DATASET=scifact-subset` with
distinct `EVAL_OUTPUT` directories. All 807 backend tests pass with 100% coverage;
lint, formatting, and type checks also pass. The subset debug endpoint returned
HTTP 200 with live hybrid reranking and no fallback. Reports were checked not
to contain either configured API key.

## Reproducibility evidence

Two paced runs against the same completed, compacted synthetic index had zero
absolute metric difference across every query and all four modes. The selected
report is the second run. Local raw runs remain under
`.data/evaluation/synthetic-final/` and `.data/evaluation/synthetic-final-repeat/`.
For this static synthetic corpus use an absolute metric tolerance of `1e-12`
to allow floating-point representation only. Timing differences are descriptive.
Re-establish a tolerance for a larger dataset or changed model/configuration;
these two observations do not guarantee that a remote provider never changes.

```bash
.venv/bin/python -m hybrid_rag_search.evaluation.runner compare \
  .data/evaluation/synthetic-final/report.json \
  .data/evaluation/synthetic-final-repeat/report.json
```

Expected observed result:

```json
{
  "max_absolute_per_query_metric_delta": 0.0,
  "max_absolute_aggregate_metric_delta": 0.0,
  "latency": "descriptive; no equality requirement"
}
```

Ordinary runs stay local. Select an existing report without making new provider
calls with:

```bash
.venv/bin/python -m hybrid_rag_search.evaluation.runner export \
  .data/evaluation/synthetic-final-repeat/report.json \
  --output docs/evaluation/synthetic-workspace/baseline
```

The report records the parent code revision and a dirty worktree because the
implementation was being verified before its first commit. The Day 15 branch
contains the evaluated implementation and tests. Never relabel a partial SciFact
index as a full baseline; the runner checks preparation before searching.

See the [Day 15 runbook](../tickets/day-15.md) for preparation, exact metric
rules, commands, HTTP examples, and the trial-key quota constraint.
