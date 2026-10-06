# Day 16 — Demo identities and authorization

GitHub: [Issue #17](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/17)

## Outcome

The local API lets a caller select one of five seeded demo users and searches
only that user's permitted collections. Original-file reads use the same database
permission policy. Identity verification is simulated: a caller can deliberately
select any listed demo identity without a password. Authorization is real and
tested; changing the workspace or document ID does not grant access.

```text
X-Demo-User-ID + requested workspace
              ↓
PostgreSQL user → membership → readable collections
              ↓
       tenant + collection filters
              ↓
       BM25 / dense → RRF → rerank → chunks and optional trace

workspace + document ID → same permission policy → original-file read
```

## Decisions and scope

- Owners and admins read every collection in their own tenant. Membership in a
  different tenant is still required; no role grants global access.
- Members need an explicit collection grant. `read`, `write`, and `manage` each
  permit reading. Administrative mutations remain Day 17 work.
- `X-Demo-User-ID` selects the user consistently for POST and GET requests.
  The workspace is supplied in the search body or document URL.
- Default search covers all readable collections; optional `collection_id`
  narrows this set. Callers cannot submit an authorization allowlist or index.
- PostgreSQL is consulted for every protected request. Once a grant change
  commits, the next request observes it. An in-progress request uses its starting
  permission scope; revocation does not cancel already-running work.
- BM25 and dense queries both filter by tenant, ready visibility, and explicit
  readable collection IDs before fetching candidates. An empty set returns no
  results without calling OpenSearch or providers. Unexpected out-of-scope hits
  fail closed before their evidence or explanations can leave retrieval.
- Direct document reads authorize the database document's tenant and collection
  before touching storage. Forbidden, deleted, and missing documents return the
  same `404` response. Original bytes download as an attachment.
- The unauthenticated Day 15 `/debug/search` HTTP endpoint is retired. User-scoped
  `/v1/search` optionally returns diagnostics. The benchmark CLI still uses its
  isolated evaluation corpus and remains an explicit local engineering tool.
- Citation creation, request-bound citation mapping, and citation HTTP routes
  remain Day 18 work. Their source-document access must use `document_scope` on
  every open; a previously generated citation never grants continuing access.
  This citation work is explicitly deferred, not claimed as implemented.
- No new service, dependency, migration, permission cache, frontend, or real login.
  Low-level ingestion/storage/retrieval libraries remain trusted internal tools;
  they are not public authorization entry points.

## Seed matrix

`make db-seed` creates fixed IDs and can be repeated. It restores the seeded
roles/grants to this matrix, so reseeding intentionally restores a removed demo
grant. The original Day 3 tenant, owner, collection, and grant IDs are retained.
Seed data creates identities and collection metadata, not documents or embeddings.

| User | User ID ending | Workspace | Readable collections |
| --- | --- | --- | --- |
| Acme Admin (owner) | `0102` | Acme Demo (`0101`) | Every Acme collection |
| Acme Engineering | `0112` | Acme Demo | Handbook, Engineering |
| Acme HR | `0122` | Acme Demo | Handbook, HR |
| Acme Restricted | `0132` | Acme Demo | Handbook |
| Other Member | `0142` | Other Demo (`0141`) | Other Handbook |

All IDs start with `00000000-0000-4000-8000-00000000`.
Collection endings: Handbook `0104`, Engineering `0114`, HR `0124`, Other Handbook `0144`.

## API and local inspection

Start PostgreSQL and OpenSearch, apply migrations, seed, and start the API:

```bash
docker desktop start
docker compose up -d postgres redis opensearch
make db-upgrade db-seed
make backend-dev
```

List identities and the selected user's workspaces:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/demo-users
curl --fail-with-body http://127.0.0.1:8000/v1/workspaces \
  -H 'X-Demo-User-ID: 00000000-0000-4000-8000-000000000112'
```

The directory returns `id` and `display_name`; workspaces return `id`, `name`,
and `role`. Only the five configured, database-backed demo users can be selected;
benchmark/internal users are not selectable identities.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-User-ID: 00000000-0000-4000-8000-000000000112' \
  -d '{"workspace_id":"00000000-0000-4000-8000-000000000101","query":"deployment policy","mode":"bm25","debug":true}'
```

Search accepts `workspace_id`, `query`, optional `collection_id`, `mode`, and
`debug`. Modes: `bm25`, `dense`, `hybrid`, `hybrid_rerank` (default). Query length
is 1–4096 characters. Responses contain `mode`, ranked `chunks`, and `trace`
(`null` unless requested). Dense modes use live configured embedding providers;
reranked search also uses the configured reranking provider. Ordinary verification
uses fakes instead. Seed-only collections contain no evidence: ingestion must
populate the application's normal read index before meaningful demo search.
Day 15 evaluation indexes are separate and are not searched by this API.

Try changing `workspace_id` to `00000000-0000-4000-8000-000000000141` while keeping
the Engineering header: expect `403` before any retrieval/provider call.
Selecting HR collection `00000000-0000-4000-8000-000000000124` as Engineering
returns `404`. Missing/unrecognized identity returns `401`; malformed IDs and
unsupported search fields return `422`; provider/retrieval failure returns `503`
with a safe error code.

Original-file access:

```bash
curl --fail-with-body \
  http://127.0.0.1:8000/v1/workspaces/WORKSPACE_UUID/documents/DOCUMENT_UUID/content \
  -H 'X-Demo-User-ID: 00000000-0000-4000-8000-000000000112' \
  -o original.bin
```

## Verification

```bash
make check
make db-test
make evaluation-test
```

The current project's `.venv/bin/python` is unusable. This session uses Python
3.12.1 in `/private/tmp/hybrid-rag-day16-venv`, without overwriting `.venv`.
The exact focused authorization command used here is:

```bash
/private/tmp/hybrid-rag-day16-venv/bin/pytest \
  backend/tests/test_authorization_integration.py -m integration --no-cov -q
```

For normal backend checks, use that environment's `ruff`, `mypy`, and `pytest`
executables in place of `.venv/bin/`. Frontend checks use the normal npm scripts.

Verification proves:

- Five fixed identities and repeatable seed records; both owner and admin access
  without collection grants; `read`, `write`, and `manage` read permissions.
- Real PostgreSQL/OpenSearch retrieval through all four modes for each identity,
  with protected text absent from results, diagnostic traces, and reranker input.
- Optional narrowing, empty grants, invalid identities, cross-tenant requests,
  guessed IDs, and authorized original-file reads.
- Grant removal takes effect on the next independently connected HTTP request
  after a real database commit, for both search and document access.
- Existing evaluation fixture, four-mode wiring, and metric regressions still pass.

The main isolation test uses rollback-only database edits, temporary originals,
and a unique OpenSearch index that is removed afterward. The commit/freshness
check creates its own temporary workspace and deletes only its fixture records.
Neither resets the demo or evaluation corpora. No live Cohere calls are needed.

## Recorded results

- Backend formatting, lint, and strict types pass.
- Ordinary backend suite: 816 tests pass with 100% statement coverage.
- Existing integration suite including the first Day 16 isolation test: 9 tests
  pass. Both Day 16 integration tests also pass together, including the added
  committed-grant/independent-request check.
- Focused evaluation regression command: 22 tests pass.
- Frontend formatting, lint, types, its test, and production build pass.
- A running Uvicorn API returned five users (`200`), only the selected user's
  workspace (`200`), cross-tenant search denial (`403`), guessed-document denial
  (`404`), and retired debug-route absence (`404`).
- Existing `.venv` prevented invoking `make check` directly; its backend checks
  were run with the temporary environment and its frontend checks with the
  unchanged npm commands. No live provider calls were made.

## Acceptance and next work

- [x] Identity verification is documented as simulated.
- [x] Cross-tenant and unauthorized-collection requests return no protected content.
- [x] Committed grant changes affect the next request.
- [x] Direct document IDs cannot bypass authorization.
- [x] All planned checks pass; actual citation routes explicitly deferred to Day 18.

Day 17 builds collection, document, permission, and job management APIs on this
policy. Day 18 implements generation and request-bound citations using authorized
evidence and rechecks source-document access when citations are opened.
