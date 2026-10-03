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
documents cannot demonstrate large-corpus behavior; full SciFact remains pending.

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
