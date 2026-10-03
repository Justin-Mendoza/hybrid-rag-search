# Datasets

Evaluation data is separate from the product demo and is loaded into a dedicated
evaluation tenant during explicit preparation, then reused for benchmark runs.
Every dataset uses the same four-file
contract:

- `manifest.json` pins its version and SHA-256 file hashes.
- `corpus.jsonl` contains stable, dataset-local document or chunk IDs.
- `queries.jsonl` contains the questions being evaluated.
- `judgments.jsonl` assigns query-specific relevance grades from 0 (irrelevant)
  through 3 (highly relevant).

The machine-readable contracts are in `schemas/`. The committed
`synthetic-workspace/v1/` fixture covers exact matching, paraphrases, stale
versions, and conflicting evidence.

The full SciFact corpus is public benchmark data and is not committed or downloaded
during setup or tests. Run `make scifact-download` to fetch the pinned official
BEIR archive, verify its SHA-256, and deterministically convert its test split
under `datasets/.cache/`. Run `make scifact-validate` to validate the cached copy
and print its version, record counts, corpus hash, and dataset hash.

The committed [SciFact subset](scifact-subset/README.md) is the Day 15 pipeline
fixture: 500 documents, 30 queries, and every judgment for those queries.
`make scifact-subset` regenerates it from the cached full source. Its manifest
pins the selection recipe and source; it is not a full SciFact benchmark.

Generated benchmark data and ordinary reports remain ignored. Selected baseline
JSON and Markdown reports are committed under `docs/evaluation/`. See the
[Day 15 runbook](../docs/tickets/day-15.md) for preparation and evaluation.
