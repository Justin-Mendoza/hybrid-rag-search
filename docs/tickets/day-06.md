# Day 06 — Document parsers

[Issue #10](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/10)

## Slice 1: shared block-oriented contract

Day 4 stores immutable original bytes. This ticket translates those bytes into
normalized, source-locatable text that Day 7 can chunk. It does not save parsed
output, invoke Cohere, index OpenSearch, or run through Dramatiq; Day 8 will
orchestrate those stages.

All supported formats return the same `ParsedDocument` shape rather than leaking
PDF pages, Markdown syntax trees, or HTML elements into later pipeline stages. A
document contains ordered `ParsedBlock` values. Every block has normalized text
and a one-based block number; parsers may additionally retain a PDF page number
or nested heading path.

`ParseRequest` preserves the exact original bytes and declared media type. The
result records parser name and version so indexed content can later be traced to
the extraction behavior that produced it. The contract rejects empty documents,
blank text, invalid locations, missing parser identity, and reordered or duplicate
block numbers.

The block number is a universal fallback locator, not an array implementation
detail. Page numbers are also one-based to match what a user sees in a PDF viewer.
Heading paths retain hierarchy, such as `("Benefits", "Remote work")`, rather
than only the nearest heading.

```bash
.venv/bin/pytest backend/tests/test_parsing_contract.py --no-cov
```

The format-specific extraction rules, stable parsing errors, and representative
fixture files are added incrementally in later slices.

## Slice 2: deterministic plain-text parser

`PlainTextParser` accepts `text/plain` bytes encoded as strict UTF-8, including an
optional UTF-8 byte-order mark. It deliberately avoids automatic encoding
detection: guessing could silently produce different text across library versions
and would make indexing difficult to reproduce. Invalid bytes fail with the stable
`invalid_encoding` code without including document content in the error.

Line endings normalize to `\n`, trailing spaces and tabs are removed, and any run
of blank lines becomes a paragraph boundary. A paragraph becomes one block.
Single line breaks, leading indentation, and internal spacing are preserved so
normalization does not flatten potentially meaningful structure.

Whitespace-only files fail with `no_extractable_text`; other media types fail with
`unsupported_media_type`. These join `corrupt_document` in the shared stable error
vocabulary that later format parsers and Dramatiq jobs can handle without knowing
library-specific exceptions.

```bash
.venv/bin/pytest backend/tests/test_plain_text_parser.py --no-cov
```

## Slice 3: CommonMark parser with heading paths

Markdown uses `markdown-it-py` rather than a handwritten recognizer. Its token
stream handles headings, paragraphs, lists, links, inline code, images, fenced
code, and CommonMark escaping without coupling our application to rendered HTML.
The dependency is constrained to the current major version.

Each visible Markdown block becomes a `ParsedBlock`. Heading blocks update and
retain their nested path; later paragraphs, list items, and code blocks inherit
that path. Link destinations and Markdown punctuation are excluded, while link
labels, inline code, and image alternative text remain searchable. Raw HTML is
recognized as its own token type and excluded rather than rendered or indexed.

Like plain text, Markdown is strict UTF-8 with optional BOM support. It uses
stable errors for invalid encoding, unsupported media types, and files with no
extractable content.

```bash
.venv/bin/pytest backend/tests/test_markdown_parser.py --no-cov
```

## Slice 4: standard-library HTML parser

HTML extraction uses Python's standard `html.parser`, avoiding another dependency
for the deliberately narrow requirement. It never renders or executes the input.
Visible headings, paragraphs, list items, quotes, definition entries, table cells,
preformatted blocks, and loose body text become ordered blocks. Character
references decode to their visible form, and heading hierarchy becomes the same
locator path used by Markdown.

The parser explicitly suppresses `head`, `script`, `style`, `noscript`, and
`template` subtrees, plus elements marked `hidden` or `aria-hidden="true"`.
Ordinary HTML whitespace collapses as a browser would; preformatted content keeps
line breaks while dropping trailing spaces. Strict UTF-8, optional BOM support,
stable media-type and encoding errors, and no-extractable-text behavior match the
other text formats.

```bash
.venv/bin/pytest backend/tests/test_html_parser.py --no-cov
```

## Slice 5: page-aware PDF extraction

PDF extraction uses pure-Python `pypdf`, constrained to its current major
version. Embedded text is extracted page by page and normalized with the same
paragraph rules as plain text. Each resulting block retains the original
one-based PDF page number even when intervening pages contain no text, allowing
later citations to match the page a user sees.

Encrypted and structurally unreadable PDFs fail with `corrupt_document` without
exposing document content or passwords. A valid PDF with no embedded text fails
with `no_extractable_text` and explains that OCR is unsupported. This is an
intentional scope boundary: scanned documents, layout reconstruction, images,
and tables need a later parsing/OCR evaluation rather than silent low-quality
output.

```bash
.venv/bin/pytest backend/tests/test_pdf_parser.py --no-cov
```

## Slice 6: central media-type dispatcher

`ParserDispatcher` is the single parsing entry point for later ingestion code.
It removes the media-type parameters, normalizes case, and selects the registered
parser while passing the original immutable request through unchanged. The
default registry supports PDF, plain text, standard and legacy Markdown media
types, and HTML/XHTML aliases.

Unknown formats produce the stable `unsupported_media_type` error. Errors from a
selected parser propagate unchanged, so a future worker can handle one shared
error vocabulary without importing any format library or parser implementation.
The dispatcher stores its own registration copy, allowing tests or future
deployments to inject parsers without mutating global state.

```bash
.venv/bin/pytest backend/tests/test_parser_dispatcher.py --no-cov
```

## Slice 7: acceptance integration and failure isolation

Representative plain-text, Markdown, and HTML files now pass through the default
dispatcher, proving selection and format extraction together rather than only in
isolated unit tests. PDF tests use deterministic in-memory files with real pypdf
writing and reading, including blank intervening pages so citation page numbers
cannot accidentally collapse.

A corrupt PDF followed by a valid plain-text parse on the same dispatcher proves
that parser failures do not poison later work. At this layer there are no jobs or
shared persistence to damage; Day 8 will separately verify that a worker records
only its own job failure.

## Acceptance criteria

- [x] Representative fixture content produces deterministic parsed output.
- [x] PDF blocks retain their original one-based page numbers when extraction
  succeeds.
- [x] HTML scripts and styles are excluded, along with other non-visible content.
- [x] Corrupt input returns a stable safe error and does not affect the next parse.

## Final verification

`make check` passed formatting, linting, strict type checking, 293 backend tests
with 100% statement coverage, the frontend test, and the production frontend
build. Pytest deselected two PostgreSQL integration tests and three live Cohere
tests. `.venv/bin/pip check`, `docker compose config --quiet`, and
`git diff --check` also passed.

Day 6 does not persist parsed output, create chunks, enqueue Dramatiq jobs, call
Cohere, or write to OpenSearch.
