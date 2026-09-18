"""Deterministic structure-aware splitting of parsed documents."""

from dataclasses import dataclass
from itertools import groupby

from hybrid_rag_search.chunking.contracts import (
    DEFAULT_CHUNKING_CONFIG,
    Chunk,
    ChunkingConfig,
    SourceSpan,
)
from hybrid_rag_search.chunking.embedding_text import build_embedding_text
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument
from hybrid_rag_search.tokenization import TextTokenizer


class ChunkingError(RuntimeError):
    """The document cannot be chunked without violating the configured limit."""


@dataclass(frozen=True)
class _Fragment:
    block: ParsedBlock
    start_char: int
    end_char: int

    @property
    def text(self) -> str:
        return self.block.text[self.start_char : self.end_char]


@dataclass(frozen=True)
class _Candidate:
    fragments: tuple[_Fragment, ...]
    content_text: str
    embedding_text: str
    token_count: int


def _append_fragment(fragments: list[_Fragment], fragment: _Fragment) -> None:
    if (
        fragments
        and fragments[-1].block is fragment.block
        and fragments[-1].end_char <= fragment.start_char
        and (
            fragments[-1].end_char == fragment.start_char
            or fragment.block.text[fragments[-1].end_char : fragment.start_char].isspace()
        )
    ):
        previous = fragments[-1]
        fragments[-1] = _Fragment(previous.block, previous.start_char, fragment.end_char)
    else:
        fragments.append(fragment)


def _candidate(fragments: list[_Fragment], tokenizer: TextTokenizer) -> _Candidate:
    if not fragments:
        raise ChunkingError("A chunk candidate requires source text")
    heading_path = fragments[0].block.locator.heading_path
    if any(fragment.block.locator.heading_path != heading_path for fragment in fragments):
        raise ChunkingError("A chunk cannot cross a heading boundary")

    content_text = "\n\n".join(fragment.text for fragment in fragments)
    first = fragments[0]
    leading_heading = None
    if (
        heading_path
        and first.start_char == 0
        and first.end_char == len(first.block.text)
        and first.block.text == heading_path[-1]
    ):
        leading_heading = first.block.text
    embedding_text = build_embedding_text(
        content_text=content_text,
        heading_path=heading_path,
        leading_heading_block=leading_heading,
    )
    return _Candidate(
        fragments=tuple(fragments),
        content_text=content_text,
        embedding_text=embedding_text,
        token_count=len(tokenizer.tokenize(embedding_text)),
    )


def _whole_fragment(block: ParsedBlock) -> _Fragment:
    return _Fragment(block=block, start_char=0, end_char=len(block.text))


def _is_heading_only(fragments: list[_Fragment]) -> bool:
    if len(fragments) != 1:
        return False
    fragment = fragments[0]
    headings = fragment.block.locator.heading_path
    return (
        bool(headings)
        and fragment.start_char == 0
        and fragment.end_char == len(fragment.block.text)
        and fragment.block.text == headings[-1]
    )


def _largest_fitting_prefix(
    *,
    carry: list[_Fragment],
    fragment: _Fragment,
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> _Fragment | None:
    token_spans = tokenizer.tokenize(fragment.text)
    if not token_spans:
        return None

    low = 0
    high = len(token_spans)
    best_end: int | None = None
    while low < high:
        middle = (low + high) // 2
        relative_end = token_spans[middle].end_char
        prefix = _Fragment(fragment.block, fragment.start_char, fragment.start_char + relative_end)
        trial = list(carry)
        _append_fragment(trial, prefix)
        if _candidate(trial, tokenizer).token_count <= max_tokens:
            best_end = prefix.end_char
            low = middle + 1
        else:
            high = middle

    if best_end is None:
        return None
    if not tokenizer.tokenize(fragment.block.text[best_end:]):
        best_end = fragment.end_char
    return _Fragment(fragment.block, fragment.start_char, best_end)


def _content_layout(fragments: tuple[_Fragment, ...]) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, fragment in enumerate(fragments):
        if index:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(fragment.text)
        cursor += len(fragment.text)
        ranges.append((start, cursor))
    return "".join(parts), tuple(ranges)


def _trailing_overlap(
    candidate: _Candidate, tokenizer: TextTokenizer, overlap_tokens: int
) -> list[_Fragment]:
    if overlap_tokens == 0:
        return []
    content_text, ranges = _content_layout(candidate.fragments)
    tokens = tokenizer.tokenize(content_text)
    if not tokens:
        return []
    start_offset = tokens[max(0, len(tokens) - overlap_tokens)].start_char

    overlap: list[_Fragment] = []
    for fragment, (global_start, global_end) in zip(candidate.fragments, ranges, strict=True):
        if global_end <= start_offset:
            continue
        local_start = max(0, start_offset - global_start)
        _append_fragment(
            overlap,
            _Fragment(fragment.block, fragment.start_char + local_start, fragment.end_char),
        )
    return overlap


def _make_chunk(
    *, document_id: str, config: ChunkingConfig, order: int, candidate: _Candidate
) -> Chunk:
    if candidate.token_count <= 0:
        raise ChunkingError("Tokenizer produced no tokens for nonblank chunk text")
    return Chunk(
        document_id=document_id,
        config_version=config.version,
        order=order,
        content_text=candidate.content_text,
        embedding_text=candidate.embedding_text,
        embedding_token_count=candidate.token_count,
        source_spans=tuple(
            SourceSpan(
                locator=fragment.block.locator,
                start_char=fragment.start_char,
                end_char=fragment.end_char,
            )
            for fragment in candidate.fragments
        ),
    )


def _chunk_section(
    *,
    document_id: str,
    blocks: tuple[ParsedBlock, ...],
    tokenizer: TextTokenizer,
    config: ChunkingConfig,
    first_order: int,
) -> tuple[Chunk, ...]:
    pending = [_whole_fragment(block) for block in blocks]
    carry: list[_Fragment] = []
    chunks: list[Chunk] = []

    while pending:
        current = list(carry)
        added_whole_block = False
        while pending:
            trial = list(current)
            _append_fragment(trial, pending[0])
            if _candidate(trial, tokenizer).token_count > config.max_tokens:
                break
            current = trial
            pending.pop(0)
            added_whole_block = True

        if pending and (not added_whole_block or _is_heading_only(current)):
            prefix = _largest_fitting_prefix(
                carry=current,
                fragment=pending[0],
                tokenizer=tokenizer,
                max_tokens=config.max_tokens,
            )
            while prefix is None and current:
                current_candidate = _candidate(current, tokenizer)
                current = _trailing_overlap(
                    current_candidate,
                    tokenizer,
                    max(0, len(tokenizer.tokenize(current_candidate.content_text)) - 1),
                )
                prefix = _largest_fitting_prefix(
                    carry=current,
                    fragment=pending[0],
                    tokenizer=tokenizer,
                    max_tokens=config.max_tokens,
                )
            if prefix is None:
                raise ChunkingError(
                    "Heading context and one source token exceed the chunk token limit"
                )
            _append_fragment(current, prefix)
            if prefix.end_char == pending[0].end_char:
                pending.pop(0)
            else:
                remainder = pending[0].block.text[prefix.end_char : pending[0].end_char]
                next_start = prefix.end_char + len(remainder) - len(remainder.lstrip())
                pending[0] = _Fragment(
                    pending[0].block,
                    next_start,
                    pending[0].end_char,
                )

        candidate = _candidate(current, tokenizer)
        chunks.append(
            _make_chunk(
                document_id=document_id,
                config=config,
                order=first_order + len(chunks),
                candidate=candidate,
            )
        )
        carry = _trailing_overlap(candidate, tokenizer, config.overlap_tokens) if pending else []

    return tuple(chunks)


def chunk_document(
    *,
    document_id: str,
    document: ParsedDocument,
    tokenizer: TextTokenizer,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> tuple[Chunk, ...]:
    """Create stable chunks, never carrying content across a heading change."""

    chunks: list[Chunk] = []
    grouped = groupby(document.blocks, key=lambda block: block.locator.heading_path)
    for _, section in grouped:
        section_chunks = _chunk_section(
            document_id=document_id,
            blocks=tuple(section),
            tokenizer=tokenizer,
            config=config,
            first_order=len(chunks) + 1,
        )
        chunks.extend(section_chunks)
    return tuple(chunks)
