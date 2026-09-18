import re

import pytest

from hybrid_rag_search.chunking import chunker
from hybrid_rag_search.chunking.chunker import ChunkingError, chunk_document
from hybrid_rag_search.chunking.contracts import ChunkingConfig
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.tokenization import TokenSpan

DOCUMENT_ID = "doc_" + "a" * 64


class WordTokenizer:
    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(
            TokenSpan(index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\S+", text))
        )


class AlphanumericTokenizer:
    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(
            TokenSpan(index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\w+", text))
        )


def parsed(*blocks: tuple[str, tuple[str, ...]]) -> ParsedDocument:
    return ParsedDocument(
        blocks=tuple(
            ParsedBlock(text, SourceLocator(index, heading_path=headings))
            for index, (text, headings) in enumerate(blocks, start=1)
        ),
        media_type="text/markdown",
        parser_name="test-parser",
        parser_version="1",
    )


def config(max_tokens: int = 400, overlap_tokens: int = 50) -> ChunkingConfig:
    return ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def test_tiny_document_becomes_one_traceable_chunk() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(("A tiny document", ())),
        tokenizer=WordTokenizer(),
    )
    assert len(chunks) == 1
    assert chunks[0].content_text == chunks[0].embedding_text == "A tiny document"
    assert chunks[0].embedding_token_count == 3
    assert chunks[0].source_spans[0].start_char == 0
    assert chunks[0].source_spans[0].end_char == len("A tiny document")


def test_heading_change_is_a_hard_boundary_without_cross_section_overlap() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(
            ("Benefits", ("Benefits",)),
            ("Health coverage", ("Benefits",)),
            ("Security", ("Security",)),
            ("Use MFA", ("Security",)),
        ),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=20, overlap_tokens=2),
    )
    assert tuple(chunk.content_text for chunk in chunks) == (
        "Benefits\n\nHealth coverage",
        "Security\n\nUse MFA",
    )
    assert chunks[0].embedding_text == "# Benefits\n\nHealth coverage"
    assert chunks[1].embedding_text == "# Security\n\nUse MFA"
    assert chunks[0].source_spans[-1].locator.block_number == 2
    assert chunks[1].source_spans[0].locator.block_number == 3


def test_whole_blocks_are_preferred_and_overlap_retains_source_locations() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(
            ("one two three", ()),
            ("four five six", ()),
            ("seven eight", ()),
        ),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=5, overlap_tokens=1),
    )
    assert tuple(chunk.content_text for chunk in chunks) == (
        "one two three",
        "three\n\nfour five six",
        "six\n\nseven eight",
    )
    assert tuple(span.locator.block_number for span in chunks[1].source_spans) == (1, 2)
    assert chunks[1].source_spans[0].start_char == len("one two ")


def test_oversized_block_splits_on_token_offsets_with_overlap() -> None:
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed((text, ())),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=6, overlap_tokens=2),
    )
    assert tuple(chunk.content_text for chunk in chunks) == (
        "one two three four five six",
        "five six seven eight nine ten",
    )
    assert all(chunk.embedding_token_count <= 6 for chunk in chunks)
    assert chunks[0].source_spans[0].end_char == len("one two three four five six")
    assert chunks[1].source_spans[0].start_char == len("one two three four ")
    assert chunks[1].source_spans[0].end_char == len(text)


def test_heading_prefix_repeats_and_counts_toward_each_chunk_limit() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(
            ("Policy", ("Policy",)),
            ("one two three four five six", ("Policy",)),
        ),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=5, overlap_tokens=1),
    )
    assert len(chunks) == 3
    assert all(chunk.embedding_text.startswith("# Policy") for chunk in chunks)
    assert all(chunk.embedding_token_count <= 5 for chunk in chunks)
    assert chunks[0].embedding_text.count("Policy") == 1
    assert chunks[0].content_text == "Policy\n\none two three"


def test_same_input_and_configuration_produce_same_chunks_and_ids() -> None:
    document = parsed(("one two three four five six seven", ()))
    options = config(max_tokens=4, overlap_tokens=1)
    first = chunk_document(
        document_id=DOCUMENT_ID,
        document=document,
        tokenizer=WordTokenizer(),
        config=options,
    )
    second = chunk_document(
        document_id=DOCUMENT_ID,
        document=document,
        tokenizer=WordTokenizer(),
        config=options,
    )
    assert first == second
    assert tuple(chunk.chunk_id for chunk in first) == tuple(chunk.chunk_id for chunk in second)


def test_fails_when_heading_and_one_source_token_cannot_fit() -> None:
    with pytest.raises(ChunkingError, match="Heading context"):
        chunk_document(
            document_id=DOCUMENT_ID,
            document=parsed(("content", ("very long heading",))),
            tokenizer=WordTokenizer(),
            config=config(max_tokens=2, overlap_tokens=1),
        )


def test_overlap_shrinks_when_repeated_heading_leaves_no_room() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(
            ("H", ("H",)),
            ("one", ("H",)),
            ("two", ("H",)),
        ),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=3, overlap_tokens=2),
    )
    assert tuple(chunk.content_text for chunk in chunks) == ("H\n\none", "two")
    assert all(chunk.embedding_token_count == 3 for chunk in chunks)


def test_zero_overlap_does_not_repeat_source_text() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(("one two three four", ())),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=2, overlap_tokens=0),
    )
    assert tuple(chunk.content_text for chunk in chunks) == ("one two", "three four")


def test_trailing_untokenized_characters_remain_in_source_span() -> None:
    text = "one two   "
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed((text, ())),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=1, overlap_tokens=0),
    )
    assert chunks[-1].content_text == "two   "
    assert chunks[-1].source_spans[0].end_char == len(text)


def test_unicode_splits_retain_python_character_offsets() -> None:
    text = "café 東京 🚀 résumé naïve"
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed((text, ())),
        tokenizer=WordTokenizer(),
        config=config(max_tokens=3, overlap_tokens=1),
    )
    assert tuple(chunk.content_text for chunk in chunks) == (
        "café 東京 🚀",
        "🚀 résumé naïve",
    )
    for result in chunks:
        span = result.source_spans[0]
        assert text[span.start_char : span.end_char] == result.content_text


def test_chunks_retain_page_transition_provenance() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock("page one", SourceLocator(1, page_number=1)),
            ParsedBlock("page two", SourceLocator(2, page_number=2)),
        ),
        media_type="application/pdf",
        parser_name="test-parser",
        parser_version="1",
    )
    result = chunk_document(
        document_id=DOCUMENT_ID,
        document=document,
        tokenizer=WordTokenizer(),
        config=config(max_tokens=10, overlap_tokens=1),
    )
    assert len(result) == 1
    assert tuple(span.locator.page_number for span in result[0].source_spans) == (1, 2)
    assert result[0].content_text == "page one\n\npage two"


def test_punctuation_only_content_with_heading_can_precede_tokenized_content() -> None:
    chunks = chunk_document(
        document_id=DOCUMENT_ID,
        document=parsed(("...", ("H",)), ("one two", ("H",))),
        tokenizer=AlphanumericTokenizer(),
        config=config(max_tokens=2, overlap_tokens=1),
    )
    assert chunks[0].content_text == "..."
    assert chunks[0].embedding_token_count == 1
    assert chunks[1].content_text == "one"


def test_fails_when_tokenizer_produces_no_tokens_for_a_chunk() -> None:
    with pytest.raises(ChunkingError, match="produced no tokens"):
        chunk_document(
            document_id=DOCUMENT_ID,
            document=parsed(("...", ())),
            tokenizer=AlphanumericTokenizer(),
        )


def test_candidate_defensively_rejects_empty_or_mixed_heading_fragments() -> None:
    with pytest.raises(ChunkingError, match="requires source text"):
        chunker._candidate([], WordTokenizer())

    first, second = parsed(("one", ("A",)), ("two", ("B",))).blocks
    fragments = [chunker._whole_fragment(first), chunker._whole_fragment(second)]
    with pytest.raises(ChunkingError, match="heading boundary"):
        chunker._candidate(fragments, WordTokenizer())


def test_unsplittable_text_without_tokens_has_no_fitting_prefix() -> None:
    block = parsed(("...", ())).blocks[0]
    assert (
        chunker._largest_fitting_prefix(
            carry=[],
            fragment=chunker._whole_fragment(block),
            tokenizer=AlphanumericTokenizer(),
            max_tokens=1,
        )
        is None
    )
