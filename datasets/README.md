# Datasets

Evaluation data is separate from the product demo and is loaded into a dedicated
evaluation tenant when a benchmark runs. Every dataset uses the same four-file
contract:

- `manifest.json` pins its version and SHA-256 file hashes.
- `corpus.jsonl` contains stable, dataset-local document or chunk IDs.
- `queries.jsonl` contains the questions being evaluated.
- `judgments.jsonl` assigns query-specific relevance grades from 0 (irrelevant)
  through 3 (highly relevant).

The machine-readable contracts are in `schemas/`. The committed
`synthetic-workspace/v1/` fixture covers exact matching, paraphrases, stale
versions, and conflicting evidence.

SciFact is public benchmark data and is intentionally not committed or downloaded
during setup or tests. Run `make scifact-download` to fetch the pinned official
BEIR archive, verify its SHA-256, and deterministically convert its test split
under `datasets/.cache/`. Run `make scifact-validate` to validate the cached copy
and print its version, record counts, corpus hash, and dataset hash.

Generated benchmark data and evaluation reports do not belong in version control.
