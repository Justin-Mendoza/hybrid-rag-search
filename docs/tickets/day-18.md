# Day 18 — Grounded answers, citations, and private history

[Issue #16](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/16)

This increment turns authorized search evidence into a bounded answer with
request-bound citations. Answer creation is an internal Python service. Day 19
will expose it through the streaming endpoint and implement cancellation.
History reads are available now so ownership and permission checks can be tested
before the frontend arrives.

```text
selected demo user + workspace + question + optional collection
                         |
               current authorization
                         |
          existing hybrid search + reranking
                         |
      whole ranked chunks within evidence budgets
                         |
        one Cohere Chat call with chunk IDs
                         |
         validate outcome + native citations
                         |
      saved question/result owned by its author
                         |
 history list/detail -> recheck author + current source access
```

## Decisions and scope

- Questions are independent. Saved messages are not sent as conversation context.
- History belongs only to its author within its workspace. Workspace owners and
  admins cannot read another user's history. Demo identity verification remains
  simulated using `X-Demo-User-ID`; this is not production authentication.
- Use the existing PostgreSQL database, generation adapter, hybrid/reranking
  pipeline, and pinned local tokenizer. No new service or dependency.
- Use Cohere Chat's native document citations. No additional model call to
  classify or repair answers, and no automatic Chat retries.
- Persist the validated result, including evidence snapshots, configuration,
  model, usage/cost metadata when available, and creation time. Failed responses
  retain safe error codes rather than raw rejected prose or provider exceptions.
- Streaming, cancellation, frontend screens, multi-turn context, sharing, editing,
  and retention/purge controls remain deferred.

## Evidence budgets and cost

| Setting | Default | Meaning |
|---|---:|---|
| `ANSWER_MAX_CHUNKS` | 5 | Maximum selected chunks |
| `ANSWER_EVIDENCE_TOKEN_BUDGET` | 2000 | Evidence selection token estimate |
| `ANSWER_MAX_OUTPUT_TOKENS` | 512 | Chat output-token limit, including status header |

Selection preserves reranked order, deduplicates identical chunk IDs, and skips
whole chunks that cannot fit. It never truncates a chunk or changes its source
locations. Conflicting records with the same chunk ID fail closed. All candidate
identities are checked against the authorized scope before anything reaches Chat.

The evidence estimate uses the already-provisioned, pinned embedding tokenizer
locally. It is **not an exact Chat token count or a spending cap**: the Chat
model's tokenizer, document formatting, system prompt, and question add input
cost. Output is bounded through Chat's `max_tokens`. The question is limited to
4096 characters. Actual Chat usage and estimated cost are retained when supplied.
Provision the local tokenizer with `make tokenizer-setup` if it is absent; the
answer service never downloads one implicitly.

Empty evidence, including all chunks exceeding the selection budget, produces
`insufficient_evidence` without invoking the generation provider.

## Outcomes and grounding

The grounding prompt treats documents as evidence, never instructions. It asks
for a concise supported answer, an explicit limitation when evidence is partial,
and citations to both sides of a disagreement. It requires a first line of
`STATUS: answered`, `STATUS: partial`, or `STATUS: insufficient_evidence`.
This status is obtained in the same Chat call, without combining native document
citations with a separate JSON-output contract.

The service validates native citation offsets against the original provider
text, then removes the status line and adjusts offsets to the returned answer.
The status line cannot itself be cited.

| Outcome | Behavior |
|---|---|
| `answered` | Provider reports sufficient evidence and supplies valid citations |
| `partial` | Provider reports a limited answer, or output reached the token limit |
| `insufficient_evidence` | No usable evidence, or provider abstention; return fixed safe text |
| `failed` | Retrieval/provider failure, invalid status, blank answer, missing or invalid citations |

Token-limited answers include a visible output-limit notice. Invalid citations
fail the entire answer, without publishing its text or evidence. There is no
paid repair attempt. Arbitrary provider prose under an abstention label is
replaced with a fixed abstention message.

Structural validation does **not** prove semantic faithfulness, completeness, or
that every material claim has a citation. The prompt sets that policy; reviewed
multi-source/conflict fixtures verify the mapping behavior without paid calls.
The model's assessment of sufficiency is not a calibrated confidence score.

## Citation mapping

[Cohere's native citation contract](https://docs.cohere.com/v2/docs/rag-citations)
links answer character ranges to the custom document IDs supplied in a request.
Each supplied document here is one chunk, using its immutable `chunk_id`.

For example:

```text
supplied chunk-a: "Receipts last 24 hours."
answer:           "Receipts last 24 hours."
citation:         start=0, end=23, source_ids=["chunk-a"]
```

The citation's `text` must equal the exact `answer[start:end]` slice, with an
exclusive end offset. Each source ID must exist in the selected evidence for
that request. Unknown IDs, invalid ranges, mismatched text, empty references,
and malformed provider citation payloads are rejected.

The result keeps two separate structures:

- `citations`: answer offsets, cited answer text, and source IDs.
- `evidence`: chunk/document/collection/workspace IDs, exact supplied chunk text,
  immutable document-content ID, generation/pipeline versions, and source spans.

Source spans contain the original block and optional PDF page/heading, plus
character offsets within the normalized parser block. They are not answer
positions. The exact chunk text is the quoted supporting passage available for
inspection; native citations do not identify a narrower verbatim source quote.
Provider-echoed source content is ignored; all displayed evidence comes from our
own selected snapshot. A citation inspection view can resolve its source ID in
the returned evidence array without a provider lookup.

## Internal creation and history API

The runtime entry point for the next ticket is:

```python
async with configured_answer_service(settings) as service:
    answer_id = await AnswerHistory(policy).answer_and_save(
        service, demo_user_id, workspace_id, question, collection_id=None
    )
```

`configured_answer_service` is in `hybrid_rag_search.answers`; `AnswerHistory`
is in `hybrid_rag_search.answer_history`. This runtime path makes paid calls to
Embed, Rerank, and Chat when evidence is available. Ordinary tests inject fakes.
History is saved by server code; clients cannot POST fabricated answers.

| Method | Route | Response |
|---|---|---|
| GET | `/v1/workspaces/{workspace_id}/answers` | Author's summaries, newest first |
| GET | `/v1/workspaces/{workspace_id}/answers/{answer_id}` | Question and authorized saved result |

Both use `X-Demo-User-ID`. List pagination uses `limit=50&offset=0`, with limits
1–100, nonnegative offsets, and an ID ordering tie-breaker. Summaries return
`id`, `question`, `created_at`, and `available`; they never contain answer text
or evidence. Detail adds `result`, which contains the outcome, text, citations,
evidence, model, usage, error code, and configuration.

Every read verifies current workspace membership and author ownership. Direct
IDs return 404 for another author's answer; missing identity returns 401 and
workspace denial returns 403. Database failures use the existing safe 503 error
contract. OpenAPI includes the typed history response schemas.

Every source document in the saved evidence is authorization-checked again on
list/detail reads. Revoked grants and deleting/deleted documents cause
`available=false`; detail returns `result=null`, withholding the entire answer,
citations, evidence, and usage. The author's own question and timestamp remain
visible. Snapshots stay in PostgreSQL; this is access withholding, not a content
purge. Restoring access makes the original result available again. Membership
deletion cascades to that author's saved answers in that workspace.

Permission checks have the same request-time semantics as the existing policy;
they do not retroactively erase text already delivered to a client.

## Verification

```bash
make answer-test
make check
make db-test
```

All these tests use fakes or mocked SDK HTTP responses: **zero paid provider
calls**. There is no automatic live smoke test in this increment.

The history integration test uses a disposable PostgreSQL database, runs
upgrade/downgrade/reapply for the newest migration, and drops that test database
in cleanup. It tests author-only reads (including owner/admin denial), tenant
isolation, direct IDs, pagination, membership foreign keys, grant revocation and
restoration, source deletion, and membership removal. The broader integration
suite also requires local OpenSearch. Existing workspace content is preserved.

The repository's `.venv` Python points to a broken interpreter. Verification for
this increment uses the existing working Python 3.13.16 environment at
`/private/tmp/hybrid-rag-day17-py313`, with a temporary Makefile that only replaces
`.venv/bin/` paths. No system interpreter or existing environment was modified.
Validation passed:

- `make check` through `/private/tmp/day18-Makefile`: 848 default backend tests,
  100% line coverage, Ruff formatting/lint, strict mypy, and frontend
  formatting/lint/type checks, test, and production build.
- `make db-test` through that Makefile: all 12 integration tests passed.
- `make answer-test`: all 57 focused tests passed.
- Alembic schema comparison: no pending model/schema changes.
- No live Cohere calls were made.
- `make stack-down` succeeded; named-volume data was preserved.

## Acceptance

- [x] Citations resolve only to evidence supplied for their request.
- [x] Invalid provider citations fail safely without exposing the answer.
- [x] Empty evidence skips Chat; insufficient evidence abstains and partial answers
  are explicitly classified.
- [x] Multi-source and conflicting-evidence fixtures pass.
- [x] Independent exchanges are saved with author/workspace ownership.
- [x] History APIs enforce author-only access and recheck source permissions.
- [x] Migration round-trip and all planned checks pass.

Next: Day 19 streaming answer creation and cancellation.
