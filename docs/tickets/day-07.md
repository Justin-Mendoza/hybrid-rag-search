# Day 07 — Structure-aware chunking and stable IDs

[Issue #9](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/9)

## Slice 1: explicit local tokenizer provisioning

The selected Cohere Embed model has a 512-token input limit. Chunking therefore
needs the model's tokenizer, but it must not make a provider request for every
document. `make tokenizer-setup` explicitly asks the official Cohere SDK for the
`embed-english-light-v3.0` tokenizer once and saves it beneath the ignored
`.data/tokenizers` directory.

The tokenizer is pinned to CohereLabs model revision
`af99964c9f9b5356e7c690bec577743df457fcb2` and SHA-256
`243655762254d2a961c98b2b8b9d5783dfdb66ff0be4dee94207fb94b35938a0`.
Downloaded bytes are serialized deterministically, checked before publication,
flushed to disk, and linked into place without overwriting an existing asset. An
existing file must pass the same checksum before it can be reused.

This is one explicit provisioning operation, not necessarily one HTTP request:
the SDK may obtain model metadata and then download its tokenizer configuration.
Afterward, chunking reads only the local file. Missing, tampered, unsupported, or
unfetchable assets fail clearly and never fall back to live tokenization or an
approximate counter. No tokenizer file is redistributed in this repository.

Normal tests use a tiny in-memory tokenizer and make no network calls. The local
asset path is configurable with `TOKENIZER_ROOT`; it defaults to
`.data/tokenizers`.

At runtime, `load_configured_tokenizer` verifies the pinned checksum again and
loads the file directly with the local `tokenizers` library. Its narrow
`TextTokenizer` contract returns model token IDs and exact character offsets,
which lets chunking enforce token limits while retaining source ranges. There is
no SDK call or online tokenization fallback on this path.

```bash
.venv/bin/pytest backend/tests/test_tokenization.py backend/tests/test_tokenizer_setup.py --no-cov
```

## Slice 2: scoped content-derived document identity

The PostgreSQL document UUID continues to identify an upload and its lifecycle.
Search content instead uses `DocumentContentIdentity`, derived from the tenant ID,
collection ID, and Day 4's SHA-256 of the exact original bytes. Reprocessing or
rebuilding the same content in the same authorization scope therefore produces
the same `doc_<sha256>` identifier.

Tenant and collection scope are part of the digest so identical bytes belonging
to different authorization boundaries cannot overwrite one another in a shared
OpenSearch index. The canonical input requires UUID values and a lowercase
64-character SHA-256 digest.

Hash fields use length-prefixed UTF-8 bytes rather than delimiter concatenation,
preventing different field combinations from producing the same byte stream. A
versioned identity namespace makes any future hashing-rule change explicit.

```bash
.venv/bin/pytest backend/tests/test_chunk_identity.py --no-cov
```

## Slice 3: chunk configuration and exact source spans

The baseline chunking configuration uses a 400-token maximum with 50 tokens of
overlap. Chunks may be smaller when the algorithm can stop at a structural
boundary. The maximum stays below the embedding model's 512-token limit so later
heading context can be included without relying on provider truncation.

The configuration version hashes every boundary-affecting input: algorithm,
maximum, overlap, embedding limit, tokenizer model, pinned revision, and tokenizer
checksum. Changing any field produces a different `chunkcfg_<sha256>` identity,
making index and evaluation versions distinguishable.

Each `Chunk` is one future OpenSearch record, not separate content and heading
records. `content_text` contains only the normalized source text used for display
and citation. `embedding_text` contains that content enriched with its heading
context and is the single value later sent to Cohere Embed. The separately named
`embedding_token_count` makes clear which value must fit the model token limit.

A change in heading path is a hard chunk boundary because separate sections are
assumed to represent separate ideas. Heading context is rendered as lightweight
Markdown (`# Parent`, `## Child`) so hierarchy is retained with little token
overhead. Markdown and HTML parsers also emit a heading as a normal source block;
when that block leads a chunk, it remains in `content_text` and provenance but is
removed from the body of `embedding_text` to avoid embedding it twice.

Ordered `SourceSpan` values retain each Day 6 block, page or heading locator plus
a half-open character range within the normalized block text. Multiple spans
allow one chunk to combine adjacent blocks or pages; character ranges preserve
exact provenance when an oversized block is split. Heading metadata remains on
these locators rather than becoming a duplicate search record.

Chunk IDs hash the scoped document ID, configuration version, one-based order,
exact content text, embedding text, and canonical source spans. They are
reproducible for the same input and configuration and change when any searchable
or provenance-bearing value changes.

```bash
.venv/bin/pytest backend/tests/test_chunk_contracts.py --no-cov
```

## Slice 4: structure-aware splitting

`chunk_document` groups consecutive blocks by their complete heading path and
never carries content or overlap across a heading change. Within one section it
first tries to keep complete parser blocks together. When a single block cannot
fit, it uses local tokenizer character offsets to split that block without
losing its original locator or character range.

Every chunk repeats its Markdown heading context. The heading tokens count
toward the configured maximum, while overlap copies up to the configured number
of trailing source-content tokens into the next chunk. If a long heading leaves
less room, overlap is reduced as needed so at least one new source token can make
progress. The splitter fails clearly rather than truncating when the heading
context and one source token cannot fit together.

```bash
.venv/bin/pytest backend/tests/test_chunker.py --no-cov
```
