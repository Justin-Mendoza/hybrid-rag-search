# Day 08 — Idempotent Dramatiq ingestion jobs

[Issue #6](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/6)

## Orientation

An ingestion request first stores the original file and creates durable document
and job rows in PostgreSQL. It then sends only the job UUID through Redis.
Dramatiq workers are already-running processes that consume those messages, load
the authoritative job and document data from PostgreSQL, execute the pipeline,
and persist progress back to PostgreSQL.

The baseline uses one Dramatiq actor invocation per document ingestion. That
invocation runs parse, chunk, embed, and index stages sequentially. Separate
documents can still execute concurrently in separate worker slots. This avoids
passing document data through Redis or coordinating a chain of stage messages;
stage-level checkpoints allow a redelivered job to resume.

Day 8 owns orchestration and an indexing boundary. Day 9 will implement the
concrete OpenSearch index and indexing service.

## Slice 1: durable next-stage column

`ingestion_jobs.stage` records the next operation the worker should execute:
`parse`, `chunk`, `embed`, `index`, or `complete`. Job `status` answers whether
work is queued, running, succeeded, failed, or cancelled; stage answers where it
should resume. Keeping this as a constrained column makes progress inspectable
and queryable without decoding checkpoint JSON.

The migration defaults existing unfinished jobs to `parse` and maps existing
successful jobs to `complete` before adding the consistency constraint. A job is
successful if and only if its stage is complete. The existing JSON checkpoint
remains available for stage-specific metadata and result references.

## Slice 2: content and pipeline identity

Each job snapshots the document content SHA-256 and a `pipe_<sha256>` pipeline
version. The pipeline version hashes every input that changes final searchable
output: parser name/version, chunk configuration, embedding model/dimensions/input
and output types, and index schema version. It is a transformation recipe, not an
application release number.

PostgreSQL permits only one job for a tenant, document, content hash, and pipeline
version. The same bytes under the same recipe therefore reuse one durable job;
changed bytes or an intentionally changed recipe produce new work. Historical
jobs are assigned unique valid legacy versions during migration, preserving job
history without incorrectly treating old work as reusable under the new recipe.

## Slice 3: queued-job recovery dispatch

Redis publication is deliberately recoverable rather than treated as a second
transactional source of truth. `RecoveryDispatcher` reads a bounded, stable-order
batch of PostgreSQL jobs still marked `queued` and republishes only their UUIDs.
It does not mark them as dispatched: only a worker claim changes durable state.

If publication fails partway through a batch, the error is visible and the jobs
remain queued for the next pass. Multiple passes may publish the same UUID. That
at-least-once behavior is intentional; the next slice makes the worker claim
atomic so duplicate Redis deliveries cannot execute the pipeline twice.

## Slice 4: atomic worker claim

`JobClaimer` uses one conditional PostgreSQL `UPDATE ... RETURNING` statement to
move a queued job to running, increment its attempt count, and return the durable
job snapshot. The update succeeds only while the job is still queued and below
its attempt limit. Competing workers cannot both satisfy that predicate: one
receives the job, while every duplicate delivery receives no row and exits
before parsing or provider work.

The claim also preserves the first start time, clears any prior finish time, and
returns the content hash, pipeline version, next stage, and checkpoint needed by
the eventual actor. Expired running-job leases are intentionally deferred to the
next slice.

## Slice 5: expiring worker ownership

An atomic claim now assigns a random lease token and a five-minute expiration.
The token is an ownership receipt: lease renewal requires the job UUID, current
token, running status, and an expiration still in the future. An old or delayed
worker therefore cannot renew ownership after another attempt takes over.

The worker will renew after each stage and embedding batch. A supporting index
on status and expiration makes expired-job recovery efficient. During migration,
pre-existing running jobs are returned to queued because they have no verifiable
live owner; inventing leases for unknown workers could strand them permanently.

## Slice 6: durable retry timing and terminal failure

Retry eligibility is stored as `next_attempt_at` rather than existing only in a
Redis delayed message. Both dispatch and claim require that this time has arrived,
so a Redis restart or early duplicate delivery cannot bypass the schedule.

The baseline permits three total attempts. Retryable failures use exponential
backoff from five seconds with 20 percent jitter; permanent failures stop on the
first attempt. After attempt three, the job becomes terminally failed, retains
its failed stage and safe error details, clears its lease, and marks the matching
document content failed. A stale worker whose lease token no longer matches is
not allowed to change either row.

Expired leases are recovered using two bounded state transitions. Jobs below the
attempt limit return to queued and become immediately eligible for publication;
jobs at the limit become terminally failed and mark only the matching document
content failed. The same lease-expiration predicate also protects normal failure
recording, so a worker cannot update state after its ownership has expired.

## Slice 7: immutable checkpoint artifacts

Large parsed, chunked, and embedded results live in replaceable shared artifact
storage rather than bloating the ingestion job row. Deterministic keys scope each
artifact to tenant, job, stage, and an input-derived SHA-256 identity. Writes are
atomic and create-only: identical bytes are safely reused after redelivery, while
different bytes under the same logical identity raise a conflict.

Every save returns a small reference containing the key, byte length, and content
SHA-256 for PostgreSQL checkpoint JSON. Reads verify that reference before data
can re-enter the pipeline. Locally the root is `.data/ingestion-artifacts`; both
API and worker containers share it through the existing data volume. The storage
contract can later be replaced by object storage without changing job logic.

## Slice 8: versioned artifact payloads

Parsed documents, chunks, and embeddings use strict version-1 JSON envelopes.
Each envelope names its artifact kind and input-derived identity, then carries a
stage-specific payload. Decoding rejects unknown fields, wrong kinds or versions,
nonfinite vectors, broken source locations, and identities that do not match the
checkpoint being resumed.

Chunk artifacts preserve both source text and exact source spans. Embedding
artifacts explicitly pair every vector with its stable chunk ID rather than
depending on array position across two files. This keeps the later indexing
boundary inspectable and prevents a valid vector from being attached to the
wrong chunk after a retry or partial write.

## Slice 9: lease-protected stage checkpoints

Artifact-producing stages advance through one conditional PostgreSQL update.
The update requires the job to remain running at the claimed stage with the
same unexpired lease token. It records the small artifact reference, clears a
prior retry error, advances to the next stage, and renews the five-minute lease.

The transition is fixed by the current stage (`parse` to `chunk`, `chunk` to
`embed`, and `embed` to `index`) rather than supplied by a caller. Artifact keys
must also match the claimed tenant, job, and stage. A late worker, duplicated
message, or stale in-memory claim therefore cannot skip a stage or overwrite a
new owner's checkpoint.

## Slice 10: deterministic artifact recovery

Artifact storage can look up a deterministic key and reconstruct its checksum
and size reference. A retry calculates the same tenant/job/stage/input key before
performing work. If the artifact exists, the worker fetches it through that
reference and runs the strict stage decoder before checkpointing it; if it is
absent, the worker performs the stage normally.

This closes the crash window after an artifact is atomically saved but before
its PostgreSQL checkpoint commits. In particular, an embeddings artifact in
that window can be reused without another Cohere request. PostgreSQL remains the
durable progress record; lookup is only recovery for a known deterministic key.

## Slice 11: recoverable parse-stage runner

The parse stage first derives its artifact identity from the claimed content hash
and pipeline version. It reuses an existing artifact only after checksum fetch
and strict payload decoding. An invalid artifact raises an integrity failure and
is never silently deleted or overwritten.

When no artifact exists, a tenant-scoped PostgreSQL read resolves the document's
canonical storage key and media type. The loader rejects deleted documents,
changed database hashes, invalid storage pointers, and source bytes whose SHA-256
does not match the claimed job. The dispatcher then parses the verified original,
the runner atomically publishes its JSON artifact, and the lease-protected
checkpointer advances the job to `chunk`.

## Slice 12: recoverable chunk-stage runner

The chunk stage reads its parsed input from the checksum-bearing PostgreSQL
checkpoint rather than rediscovering it by filename. PostgreSQL separately
resolves the document's current collection and verifies that its content hash
still matches the claim. Tenant, collection, and content hash form the stable
search-document identity; database metadata does not leak into parser output.

The chunks artifact identity covers the parsed artifact checksum, stable document
identity, and complete chunk configuration version. A retry validates and reuses
that artifact before invoking the tokenizer. Otherwise the runner fetches and
decodes the parsed artifact, executes deterministic structure-aware chunking,
publishes the versioned chunks artifact, and lease-safely advances to `embed`.

## Slice 13: batch-resumable embedding runner

The embedding stage reads and validates the chunks checkpoint, then divides its
ordered chunks into at most 96 texts per provider request. Each batch has a
deterministic artifact identity covering the chunks checksum, configured model
and dimensions, and exact ordered chunk IDs. Existing valid batches are reused;
only missing batches call the embedding provider.

After every recovered or newly saved batch, the worker renews its lease. Losing
the lease stops work before another provider request or final checkpoint. Once
all batches exist, their chunk-ID/vector pairs are combined in source order into
one final embeddings artifact. Only that final reference advances PostgreSQL to
`index`; saved batches remain crash-recovery inputs until later cleanup.

## Slice 14: idempotent document index boundary

The worker hands indexing one complete, tenant-scoped replacement request. It
contains the source document UUID, collection, content and pipeline identities,
embedding configuration, and ordered pairs of each validated chunk with its
matching vector. The boundary rejects mixed documents, configurations, vector
dimensions, duplicated chunk IDs, and broken source order.

A deterministic index generation identifies the complete replacement. The fake
reference implementation stages all records before changing the active document,
cleans staging after any partial failure, preserves the previous generation until
the replacement succeeds, and treats replaying an identical generation as a
successful no-op. Different content under the same generation is a permanent
nondeterminism conflict. Day 9 will implement these semantics with OpenSearch.

## Slice 15: lease-protected index completion

The index runner loads the chunk and embedding references from PostgreSQL,
checksum-verifies and decodes both artifacts, and pairs records by stable chunk
ID. It resolves the current tenant-scoped collection before sending one complete
replacement request to the document index boundary.

After replacement succeeds, one lease-guarded PostgreSQL transaction marks the
job `succeeded` at stage `complete`, stores the index receipt, clears ownership,
and marks only the matching document content `ready`. If the worker lost its
lease, the database transition does nothing even if the old index call returns.

## Slice 16: durable enqueue service

`IngestionJobService` is the application boundary an eventual upload endpoint
calls after storing a document. It locks and validates the tenant-scoped
document, inserts one job for the document content and pipeline recipe, and
publishes the job UUID only after the PostgreSQL transaction commits.

The uniqueness constraint makes concurrent or repeated enqueue requests reuse
the same job. Queued jobs may be safely republished, while running, successful,
failed, and cancelled jobs are returned without creating work or calling Redis.
A publication failure therefore leaves recoverable queued work in PostgreSQL.

## Slice 17: one-attempt worker orchestration

The `ingest_document` Dramatiq actor accepts only a serialized job UUID and has
Dramatiq automatic retries disabled. `IngestionPipeline` atomically claims the
job and runs parse, chunk, embed, and index from the durable current stage. A
duplicate delivery that cannot claim exits before file, tokenizer, Cohere, or
index work.

Known exceptions are translated into stable, safe error codes. The PostgreSQL
failure handler decides whether the error is retryable and whether attempts
remain. A retry stores its exact eligibility time before the worker publishes a
delayed Redis hint; a permanent or exhausted failure updates both the job and
matching document. Provider response bodies and unexpected exception details
are never persisted.

The configured runtime uses Cohere embeddings and the pinned local tokenizer.
Its index implementation is deliberately the in-memory Day 8 reference
implementation; Day 9 replaces that dependency with the durable OpenSearch
adapter without changing orchestration.

## Slice 18: bounded startup and manual recovery

Worker startup runs one bounded recovery pass: expired leases are requeued or
failed according to their attempt count, then all currently eligible queued jobs
are republished in stable order. There is no always-on recovery scheduler for
this personal-project baseline. If local infrastructure is interrupted without
a worker restart, the same safe pass can be run explicitly:

```bash
make ingestion-recover
```

Artifact retention and automatic cleanup remain deferred. Immutable artifacts
are useful for inspecting and learning from retries, and local disk cleanup can
be revisited once actual usage makes it necessary.

## Verification

```bash
# Fast unit/contract suite, including 100% backend coverage
.venv/bin/pytest backend

# Apply migrations and exercise real PostgreSQL transitions
make db-test

# Explicitly republish eligible work after a local interruption
make ingestion-recover
```

The worker integration test executes the full parse-to-index success path with
local fakes, proves redelivery does not repeat embedding work, and verifies both
durably scheduled retry and permanent-failure transitions against PostgreSQL.
