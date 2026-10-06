# Day 15 — Retrieval metrics and the P0 baseline

## Outcome and status

The evaluation runner prepares a reusable, isolated corpus, searches it through
four existing retrieval modes, and saves document-level quality metrics and
per-query comparisons. The debug HTTP endpoint exposes that same search boundary.

Day 15 is complete within the owner-approved revised scope: synthetic workspace
and a fixed SciFact subset, with the full benchmark explicitly deferred.
Both [selected baselines](../evaluation/README.md) pass all four live modes with
no fallback and zero observed repeat metric drift after index compaction.
The subset has 500 documents, 771 indexed chunks, 30 queries, and 34 judgments.
Its debug HTTP request also passed with real retrieval and reranking.

Trial preparation stopped after 291 subset documents with `provider_rate_limited`.
After the owner configured a production key, preparation reused completed jobs
and finished the remaining 209 documents. Both full subset runs then succeeded.
The saved trial error does not distinguish monthly quota from a short-term limit.

The full SciFact preparation remains paused and deferred: 480 completed documents
and one interrupted job are preserved in a separate scope. That job would need
lease recovery before any future resumption; no full-corpus baseline is claimed.

```text
versioned corpus → original storage + PostgreSQL documents/jobs
                           ↓ parse → chunk → embed → index → compact once
                     isolated evaluation OpenSearch index
                           ↑
dataset queries → shared search → BM25 / dense / hybrid / hybrid + rerank
                           ↓
                   ranked chunks → unique documents
                           ↓ fixed Day 14 judgments
                metrics + per-query deltas → JSON and Markdown

POST /debug/search ───────→ shared search → chunks + diagnostic traces
```

## Agreed decisions

- Each document counts once; its first chunk establishes its document rank.
  Each configuration is scored independently. Chunk-level evidence coverage
  requires new judgments and is deferred.
- Original text enters the existing durable ingestion pipeline. A fixed
  evaluation tenant and a collection derived from corpus hash and pipeline
  identity isolate each recipe. Dataset-local document IDs map to stable UUIDs.
- Completed jobs and their parse/chunk/embed artifacts are reused. Changing
  judgments or metric code does not change corpus identity. Changing corpus,
  parser, chunking, embedding, or schema configuration creates a new scope.
- Revised scope: live Cohere baselines target synthetic workspace and a fixed
  SciFact subset (500 documents, 30 queries, 34 judgments). Full SciFact is deferred.
  Selection uses salted SHA-256 ordering of query IDs, includes every judgment
  target for selected queries, then fills to 500 documents by salted document-ID
  hash ordering. The versioned manifest records the parent dataset hash and recipe.
  This is a pipeline baseline, not the standard full SciFact benchmark. Its smaller
  distractor pool changes retrieval difficulty; do not compare its scores directly
  with published full-corpus results. It uses a separate index and does not reuse
  embeddings from the partial full corpus; that corpus remains untouched.
  Controlled CI fixtures and fake providers test behavior without network calls.
- Recall@5/10/50 and MRR@10 treat grades 1–3 as relevant; grade 0 and unjudged
  documents earn no credit. nDCG@10 uses gains `2^grade - 1` and logarithmic rank
  discounts. Metrics are averaged equally across queries.
- BM25 and dense retrieve up to 100 chunks each. Hybrid fuses them and keeps
  100. Reranking uses a 100-candidate hybrid pool and returns up to 100 chunks.
  Deduplication follows final ranking. Cutoffs remain fixed for shorter lists;
  there is no automatic refill.
- The runner and HTTP endpoint share `EvaluationSearch`. The runner calls it
  directly. The endpoint accepts only a registered dataset, query, and mode;
  callers cannot supply arbitrary tenants, collections, paths, or index names.
- Each run writes JSON and Markdown files. BM25 is the comparison reference.
  Selected baseline reports belong in `docs/evaluation/`; ordinary runs remain
  under ignored `.data/evaluation/`. Experiment database tables are not needed.

## Preparation and reuse

Run from the repository root after configuring Cohere in `.env`, starting the
local services, migrating PostgreSQL, and provisioning the pinned tokenizer:

```bash
docker compose up -d postgres redis opensearch
make db-upgrade
make tokenizer-setup
make evaluation-prepare
make scifact-download
make scifact-subset
make evaluation-prepare EVAL_DATASET=scifact-subset
```

Preparation runs the existing jobs inline. It creates its own evaluation index
with the existing mapping and index manager, leaving demo aliases untouched.
One-time compaction removes replaced promotion records so background segment
cleanup does not change lexical term statistics during a static benchmark.
Re-running preparation reuses succeeded jobs rather than calling embeddings
again. For a failed/retry-scheduled job, inspect its durable error and resume
after the existing retry time. After an interrupted active job, its lease must
expire and be recovered using the existing ingestion recovery workflow.

Before evaluation, the runner checks every expected job completed and its exact
generation exists in the readable index. A missing/reset/stale index fails
closed; incomplete corpora are never silently benchmarked. The saved original
bytes and artifacts remain in configured local storage, state in PostgreSQL,
and searchable chunks/vectors in OpenSearch.

## Run and compare

```bash
make evaluation-run EVAL_OUTPUT=.data/evaluation/synthetic-first
make evaluation-run EVAL_OUTPUT=.data/evaluation/synthetic-repeat
.venv/bin/python -m hybrid_rag_search.evaluation.runner compare \
  .data/evaluation/synthetic-first/report.json \
  .data/evaluation/synthetic-repeat/report.json
```

Use `EVAL_DATASET=scifact-subset` for the agreed pipeline baseline. The committed
`datasets/scifact-subset/v1/` fixture needs no download to prepare or evaluate;
`make scifact-download scifact-subset` regenerates it from the pinned full archive.
`EVAL_DATASET=scifact` remains available for the deferred full test split.
Preparation pacing is
controlled by `EVAL_DOCUMENT_DELAY` (default 1 second before unfinished jobs),
and runs by `EVAL_QUERY_DELAY` (default 7 seconds before each query). Pacing is
explicit CLI behavior, not a retry subsystem. Trial reranking allows 10 requests
per minute; monthly quotas still apply. Verify current limits in
[Cohere's documentation](https://docs.cohere.com/v2/docs/rate-limits).

Reports state dataset version/hashes, corpus and chunk counts, hardware, cache
conditions, code revision/worktree state, physical index, models, live/fake
adapters, ranking controls, timings, and rerank fallback counts. JSON preserves
every ranked chunk reference and the winning chunk per document. Corpus bytes
are already versioned; inspect full evidence through the debug endpoint.

Fallbacks are retained explicitly and must be considered when interpreting a
reranking benchmark. The normal four-mode runner issues three query-embedding
calls and one rerank call per query. It does not cache query embeddings, so the
timings reflect each configuration's actual query-time work.

Deterministic CI metric results must match exactly. The synthetic live tolerance
is `1e-12`, with zero observed drift in two completed compacted-index runs.
The SciFact subset also has zero observed drift across two complete runs; use
`1e-12` absolute metric tolerance for this fixed index/model/configuration.
The full SciFact benchmark is explicitly deferred. Latency is
descriptive; local load and provider timings need not match. The comparison
command refuses datasets, configurations, indexes, models, or adapters that
do not match and reports the largest per-query and aggregate metric difference.

## Debug HTTP API

Historical Day 15 interface: Day 16 retires `/debug/search` and adds authorized
`/v1/search`. Use the [Day 16 runbook](day-16.md) for the current HTTP contract.
The evaluation CLI and committed baselines remain available.

Start `make backend-dev`, then:

```bash
curl --fail-with-body http://127.0.0.1:8000/debug/search \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"synthetic-workspace","query":"What does FIN-042 cover?","mode":"hybrid_rerank"}'
```

Modes are `bm25`, `dense`, `hybrid`, and `hybrid_rerank`. The response includes
ranked chunks, the deduplicated document ranking, and diagnostic traces.
Incomplete preparation returns 409; retrieval/provider failures return 503;
unsupported request fields return 422. The endpoint is disabled in production
and intended for the loopback-only local stack. Identity verification and
user-facing authorization remain Day 16 work.

The Compose API mounts `datasets/` read-only so it can validate the same corpus
contract used by the local runner.

## Verification

This machine's existing `.venv/bin/python` was unusable, so the live commands
and backend checks used `/private/tmp/hybrid-rag-day15-venv/bin/python` and the
matching pytest/ruff/mypy executables (Python 3.12.1, backend dev dependencies).
The documented Make targets assume a working project `.venv`; the temporary
environment was not committed or substituted into repository configuration.

```bash
make evaluation-test
make check
```

Tests cover hand-calculated metric answers, fixed cutoffs, duplicate documents,
unjudged results, graded relevance, equal-weight aggregation, exact deterministic
regression, fake-provider four-mode wiring, scope enforcement, preparation reuse,
stale/missing generations, endpoint validation, pacing, report comparison, and
credential exclusion from metadata.

The complete normal backend suite passes 801 tests with 100% coverage. Backend
formatting, linting, and strict types pass. Frontend formatting, linting, types,
its test, and production build pass. Actual HTTP requests for all four modes
returned 200; the live rerank request had no fallback.

During this session the existing `.venv` points to an empty Python 3.13 binary.
Verification uses an isolated Python 3.12 environment at
`/private/tmp/hybrid-rag-day15-venv`; the existing environment was not overwritten.
Until repaired, replace `.venv/bin/` in commands with that temporary path.

## Deferred

Authorization, UI, answer generation, dashboards, scheduled experiments,
database-backed run tracking, and chunk-level relevance scoring are outside
Day 15. A full SciFact report remains required unless the owner explicitly
approves a smaller versioned benchmark in response to the trial quota constraint.
