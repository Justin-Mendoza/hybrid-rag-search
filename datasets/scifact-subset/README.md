# SciFact pipeline subset v1

This committed evaluation-only fixture contains 500 documents, 30 queries, and
34 judgments from the pinned BEIR SciFact test split. It is not a full SciFact
benchmark and its scores are not directly comparable to full-corpus scores.

Selection sorts query IDs by SHA-256 of `scifact-subset-v1:<ID>` and takes 30.
Every judgment and its target for those queries is retained. Remaining documents
are selected by the same salted hash ordering until the corpus contains 500.
Files are sorted by ID (judgments by query and target). No machine-dependent
random generator or saved ingestion state participates in selection.

The manifest records the source archive checksum, parent dataset hash, selection
recipe, version, file hashes, and inherited CC BY-NC 2.0 license. Unjudged
documents are distractors for scoring, not proven irrelevant documents.

Regenerate: `make scifact-download scifact-subset`.
Prepare: `make evaluation-prepare EVAL_DATASET=scifact-subset`.
Run: `make evaluation-run EVAL_DATASET=scifact-subset`.

The existing partially prepared full corpus is separate and remains untouched.
