# Day 05 — Model adapters, fakes, and usage accounting

[Issue #5](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/5)

## Slice 1: embedding contract and deterministic fake

Implemented on `day5`; the complete Day 5 ticket is still in progress.

- `EmbeddingProvider` describes an async `embed(request)` operation. Callers can
  use this interface without depending on a provider SDK.
- `EmbeddingRequest` contains an immutable tuple of nonblank texts and an explicit
  `DOCUMENT` or `QUERY` purpose. Text is preserved exactly; normalization belongs
  to the calling ingestion/retrieval layer.
- `EmbeddingResult` contains one vector per input in input order, model identity,
  dimensions, purpose, and an explicit fake/live adapter label.
- `FakeEmbeddingProvider` derives repeatable vectors from a versioned hash of the
  exact text, purpose, and dimension. It defaults to eight dimensions for readable
  examples and accepts other positive dimensions for index tests. Each vector is
  normalized to length one. These are synthetic values, not learned embeddings:
  similar phrases need not have similar vectors, and document/query vectors do
  not align for relevance evaluation.

The fake has no network client, credentials, or provider dependency. It lets later
code exercise vector plumbing without spending Cohere quota. Model identity is
`fake-embedding-v1`; changing the vector algorithm requires changing that version.

## Try the first slice

From the repository root:

```bash
.venv/bin/pytest backend/tests/test_embeddings.py --no-cov
```

Expected: 16 tests pass. The full backend suite has 63 passing unit tests with
100% statement coverage; Ruff and strict mypy pass as well.

In an async Python caller:

```python
from hybrid_rag_search.providers.embeddings import EmbeddingPurpose, EmbeddingRequest
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider

async def example():
    provider = FakeEmbeddingProvider(dimensions=3)
    request = EmbeddingRequest(("Hello", "Hello"), EmbeddingPurpose.DOCUMENT)
    result = await provider.embed(request)
    print(result)
    assert result.vectors[0] == result.vectors[1]
```

Run with `asyncio.run(example())`. Two inputs produce two identical three-number
vectors. Switching purpose to `QUERY` changes the vectors, so tests can notice
when a caller accidentally uses the wrong purpose. This does not simulate
semantic matching between questions and documents.

## Slice 2: Cohere SDK embedding adapter

Decision: use the official Cohere Python SDK rather than constructing HTTP
requests ourselves. The application still depends on `EmbeddingProvider`, and
the fake remains unchanged. Added `cohere>=5.21.1,<6`; locally verified with 5.21.1.

`CohereEmbeddingProvider` receives an async SDK client, model name, expected vector
dimension, and timeout. It maps `DOCUMENT` to `search_document` and `QUERY` to
`search_query`, requests floating-point embeddings, and rejects batches over 96
texts. `truncate="NONE"` asks the provider to reject oversized text rather than
silently dropping part of a passage. These follow the
[Cohere Embed API](https://docs.cohere.com/reference/embed).

The current adapter checks the model's returned dimension; it does not request
a custom output dimension. The selected baseline is documented below.

The SDK handles the HTTP request and response types. Our code adds:

- An overall async timeout (10 seconds by default), plus SDK I/O timeouts.
- Disabled automatic SDK retries, leaving retry decisions to later workers and
  retrieval orchestration. External cancellation propagates normally.
- Stable errors: `authentication`, `rate_limited`, `timeout`, `unavailable`,
  `invalid_request`, and `invalid_response`. Errors contain no raw provider body
  or headers. Retryable errors are marked, not automatically retried.
- Explicit vector count, dimension, numeric-type, and finite-number validation.
  SDK parsing alone does not fully validate the provider's data.
- An `open_cohere_embeddings(...)` async context manager for explicit live client
  construction and cleanup. No client is created on module import.

Tests use the actual SDK with HTTPX's mock transport underneath it. HTTPX here
simulates the network in tests; application requests still go through the SDK.
Fixtures use synthetic vectors and test credentials, not captured live responses.

```bash
.venv/bin/pytest backend/tests/test_cohere_embeddings.py --no-cov
```

Expected: 25 tests pass. The full backend suite has 88 passing unit tests at 100%
statement coverage; strict mypy and Ruff pass. No live Cohere request was made.

## Confirmed embedding baseline

Use `embed-english-light-v3.0`: English text, 384-dimensional vectors, and a
512-token input limit. This is a compact starting point for our evaluation; we
have not yet measured its relevance or established a cost advantage over v4.
See [Cohere's model table](https://docs.cohere.com/docs/cohere-embed).

`Settings` and `.env.example` define `COHERE_EMBED_MODEL`,
`COHERE_EMBED_DIMENSIONS`, and `COHERE_TIMEOUT_SECONDS`. API keys use `SecretStr`
and are omitted from settings repr. `configured_cohere_embeddings(settings)`
explicitly opens the live adapter and closes it on exit; missing credentials or
a non-384 dimension with the chosen light model fail before client creation.
Nothing switches to live mode merely because a key is present.

The 512-token limit is a requirement for the Day 7 token-aware chunker, not a
character or word limit. The current adapter sends `truncate="NONE"` so Cohere
rejects overlong inputs instead of silently truncating them. Local token counting
is not implemented yet. Both documents and queries must use the same model;
changing models later requires re-embedding documents and rebuilding the index.

## Slice 3: embedding usage accounting

Successful embedding results include `usage`: reported input tokens, billed input
tokens, elapsed milliseconds, optional estimated USD cost, and the rate used.
Reported and billed counts are kept separate. Missing counts remain `None`;
they are never guessed from text length. Invalid negative/nonfinite counts fail
response validation. Timing uses a monotonic clock and covers the SDK call plus
response validation, excluding client setup.

Set `COHERE_EMBED_USD_PER_MILLION_TOKENS` only after verifying the selected model's
rate. No price is assumed. Estimated text-embedding cost is billed input tokens
times that rate divided by one million, using decimal arithmetic. Missing pricing
or billed usage yields `None`, not zero. This estimates list cost, not an invoice
or trial-credit balance. Change the rate along with the model when experimenting.

Fake results report zero billed tokens and zero provider cost; token counts and
latency remain unknown. This preserves repeatable fake results and avoids claiming
fake runtime is Cohere latency. Measure fake pipeline runtime externally in load
tests. Failed calls still return provider errors without usage; missing error-path
usage must not be counted as zero billed usage in future aggregates.

## Slice 4: reranking contract and deterministic fake

`RerankingProvider` accepts a nonblank query, uniquely identified candidate
passages, and `top_n`. Results contain the stable candidate ID, its original
input index, its new one-based rank, and a finite relevance score. Keeping both
positions will let the search-details view explain how reranking changed the
earlier fused order without trusting provider-owned identifiers.

`FakeRerankingProvider` scores the fraction of distinct query terms present in
each candidate. Matching is Unicode-aware and case-insensitive. Scores sort
descending, ties keep original candidate order, and only `top_n` results return.
Its model ID is `fake-rerank-v1`; it reports zero provider search units and cost,
with unknown latency. This transparent lexical rule tests plumbing, fallbacks,
and load behavior. It is not a neural reranker and cannot establish relevance
quality.

For example, query `remote work policy` gives a passage containing all three
terms score `1.0`, a passage containing only `remote` score `1/3`, and unrelated
text score `0`. Stable IDs follow each passage through the reorder.

```bash
.venv/bin/pytest backend/tests/test_reranking.py --no-cov
```

Expected: 19 tests pass. The full backend suite now has 113 passing unit tests at
100% statement coverage; strict mypy and Ruff pass.

## Slice 5: Cohere SDK reranking adapter

The selected baseline is `rerank-v4.0-fast`, Cohere's low-latency/high-throughput
Rerank 4 variant. Reranking does not create durable vectors, so changing to the
Pro model later does not require rebuilding the OpenSearch index. Model quality
still needs evaluation on our fixtures.

`CohereRerankingProvider` sends candidate text through the async SDK and translates
provider result indexes back to the stable candidate IDs in the request. Results
must contain exactly `top_n` unique, in-range indexes in descending score order,
with finite scores from zero to one. Malformed responses fail as
`invalid_response` before they can corrupt retrieval traces or citations.

The adapter enforces the provider's 10,000-document maximum, applies an overall
timeout plus SDK I/O timeout, disables SDK retries, and preserves caller
cancellation. Authentication, rate-limit, timeout, unavailable, invalid-request,
and invalid-response failures use the same stable application errors as Embed.
Later retrieval orchestration owns fallback to fused results and decides whether
a retry fits the request deadline.

Successful calls record elapsed milliseconds and Cohere's billed search units.
`COHERE_RERANK_USD_PER_SEARCH_UNIT` remains unset until a current API price is
verified; missing pricing produces unknown cost rather than zero. Trial usage
must still report billed units even when the provider charges no invoice amount.

Tests use the actual Cohere SDK over a simulated HTTP transport and make no live
calls. They cover request translation, stable-ID mapping, usage/cost arithmetic,
malformed results, every stable error class, disabled retries, timeout,
cancellation, candidate limits, configuration, and SDK cleanup.

```bash
.venv/bin/pytest backend/tests/test_cohere_reranking.py --no-cov
```

Expected: 30 tests pass.

## Slice 6: generation contract and deterministic fake

`GenerationProvider` accepts an ordered, immutable message history with `system`,
`user`, and `assistant` roles plus a positive output-token limit. At least one
user message is required. A successful result contains nonblank text, a stable
finish reason (`complete` or `max_tokens`), model and adapter identity, reported
and billed token counts, latency, configured input/output rates, and estimated
cost.

This boundary deliberately does not contain documents, evidence IDs, citations,
or prompt templates. Day 18 owns the grounded prompt and citation contract and
will translate that richer application request into these provider messages.
Day 19 will define streaming events; this Day 5 interface represents a complete
non-streaming response and does not constrain the later event shape.

`FakeGenerationProvider` hashes the exact ordered messages and token limit into a
short synthetic response. Identical requests produce identical text, while a
changed message or limit changes the result. The fake never echoes prompt text,
does not answer questions, and reports zero provider-billed tokens and cost with
unknown token counts and latency. It is suitable for control-flow and load tests,
not answer-quality evaluation.

```bash
.venv/bin/pytest backend/tests/test_generation.py --no-cov
```

Expected: 19 tests pass.

## Slice 7: Cohere SDK generation adapter

The selected baseline is `command-r7b-12-2024`, Cohere's smallest and fastest
RAG-oriented Command model. `CohereGenerationProvider` converts our ordered
messages into Chat V2 messages and makes a non-streaming request with the caller's
output limit and a configurable temperature (zero by default).

Successful responses may contain multiple text blocks, which are joined in order.
`COMPLETE` and `STOP_SEQUENCE` map to our `complete` result; `MAX_TOKENS` remains
distinct so the later answer flow can report truncation. Tool-call, empty,
thinking-only, malformed, provider-error, and provider-timeout responses do not
masquerade as successful text. The adapter uses the same stable errors, total
timeout, disabled retries, cancellation behavior, validation, and explicit SDK
lifecycle as Embed and Rerank.

Reported and billed input/output tokens remain separate. The configured rates,
verified from Cohere's Command R7B model page on 2026-09-17, are $0.0375 per
million input tokens and $0.15 per million output tokens. Estimates use billed
tokens and decimal arithmetic. These versioned configuration values must be
rechecked when changing the model or preparing a report.

Tests use the real SDK over simulated HTTP and make no live calls. They cover all
message roles, request translation, text assembly, finish reasons, usage and cost,
malformed responses, stable errors, disabled retries, timeout, cancellation,
configuration, and cleanup.

```bash
.venv/bin/pytest backend/tests/test_cohere_generation.py --no-cov
```

Expected: 33 tests pass.

## Agreed query-routing behavior for later tickets

The generation provider is not automatically called for every input. The later
query API will return a fixed application response for greetings/help requests,
without retrieval or provider cost. Substantive workspace questions retrieve
authorized evidence before generation. Unsupported general-knowledge questions
follow the insufficient-evidence path rather than silently becoming an ungrounded
general chatbot. Implementation belongs with the query and grounded-answer flow.

## Slice 8: explicitly gated live smoke tests

Three small live checks cover one query embedding, one two-candidate rerank, and
one short generation. Normal `pytest`, `make test`, `make check`, and CI exclude
the `live` marker. An API key alone cannot enable calls: the tests additionally
require `RUN_LIVE_COHERE_TESTS=1`. Enabling the flag without a real key fails
clearly instead of silently skipping a requested live verification.

Run all three authorized calls with:

```bash
make cohere-smoke
```

The command reads `COHERE_API_KEY` (or the supported `COHERE_TRIAL_KEY` alias)
and model configuration through `Settings`, including a local `.env`. If both
key names exist, `COHERE_API_KEY` takes precedence. It makes three provider calls
and may consume trial allowance or billed usage. The smoke checks transport,
credentials, configured model availability, response validation,
dimensions/source mapping, and latency; it does not establish retrieval or answer
quality.

The disabled-gate check produced three skips, and the normal suite selected 196
tests while deselecting the two PostgreSQL integration tests and three live tests.

### First authorized live run

On 2026-09-17, `make cohere-smoke` passed all three adapters using the local
`COHERE_TRIAL_KEY` alias. The suite completed in 3.17 seconds overall. It verified
that `embed-english-light-v3.0` returned one 384-dimensional query vector,
`rerank-v4.0-fast` returned a valid stable candidate mapping, and
`command-r7b-12-2024` returned validated nonblank text. The initial test version
captured per-call metrics for assertions but did not print them; subsequent runs
print model, latency, billed units, and generation estimated cost without printing
prompts, responses, or credentials.

## Acceptance criteria

- [x] Normal tests and CI make zero live Cohere calls by default.
- [x] Live smoke tests require both an explicit flag and an API key.
- [x] Provider failures map to stable application errors without leaking raw
  provider responses.
- [x] Contract tests cover deterministic fakes and SDK response fixtures; the
  explicitly authorized smoke run also validated one live response per adapter.

## Final verification

`make check` passed formatting, linting, strict type checking, 196 backend tests
with 100% statement coverage, the frontend test, and the production frontend
build. Pytest deselected two PostgreSQL integration tests and all three live tests.
`.venv/bin/pip check`, `docker compose config --quiet`, and `git diff --check`
also passed.

The paid live smoke suite was intentionally not repeated during this final pass:
its three calls had already succeeded once, while normal verification must remain
offline and repeatable.
