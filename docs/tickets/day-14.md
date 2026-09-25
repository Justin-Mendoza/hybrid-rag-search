# Day 14 — Versioned evaluation datasets

## Outcome

The project now has deterministic, validated inputs for the retrieval metrics
introduced on Day 15. Evaluation data is isolated from the product demo and can
later be ingested into a dedicated evaluation tenant.

```text
versioned corpus + queries + judgments
                 |
       deterministic loader
                 |
    Day 15 retrieval evaluation
```

## Contract

Each dataset has `manifest.json`, `corpus.jsonl`, `queries.jsonl`, and
`judgments.jsonl`. The manifest declares `purpose: evaluation`, the dataset
version, and SHA-256 hashes. Loading fails for modified files, malformed records,
duplicate IDs, missing query or target references, mismatched target types, or a
query without a positive judgment.

Relevance is query-specific and assigned once when the fixture is authored. A
grade of 0 is an explicit hard negative; grades 1–3 represent increasing
relevance. Evaluation compares search output with these fixed labels—it does not
regrade results after retrieval or reranking.

## Included datasets

- `synthetic-workspace/v1`: 8 small company documents and 12 queries covering
  exact matching, paraphrases, stale versions, and conflicting evidence.
- `scifact`: the official BEIR archive, pinned by SHA-256 and converted from its
  test split. Published positive qrels map to grade 3. The generated copy stays
  in the ignored `datasets/.cache/` directory.

## Verify

```bash
make scifact-download
make scifact-validate
.venv/bin/pytest backend/tests/test_evaluation_datasets.py backend/tests/test_scifact_dataset.py
make check
```

The loader and CLI output include the dataset version, corpus hash, and combined
dataset hash. Normal tests use a tiny local archive and never access the network.

## Deferred

Retrieval runs, evaluation-tenant ingestion, Recall/MRR/nDCG calculation, and
comparison reports belong to Day 15.
