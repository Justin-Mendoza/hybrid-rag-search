from dataclasses import replace
from typing import Any

import pytest

from hybrid_rag_search.chunking.contracts import (
    DEFAULT_CHUNKING_CONFIG,
    Chunk,
    ChunkingConfig,
    SourceSpan,
)
from hybrid_rag_search.parsers.contracts import SourceLocator

DOCUMENT_ID = "doc_" + "a" * 64
CONFIG_VERSION = DEFAULT_CHUNKING_CONFIG.version


def span(
    block_number: int = 1,
    start: int = 0,
    end: int = 10,
    page: int | None = 1,
    headings: tuple[str, ...] = (),
) -> SourceSpan:
    return SourceSpan(SourceLocator(block_number, page, headings), start, end)


def chunk(**changes: Any) -> Chunk:
    values: dict[str, Any] = {
        "document_id": DOCUMENT_ID,
        "config_version": CONFIG_VERSION,
        "order": 1,
        "content_text": "Employees may work remotely.",
        "embedding_text": "Workplace policy\nRemote work\n\nEmployees may work remotely.",
        "embedding_token_count": 10,
        "source_spans": (span(),),
    }
    values.update(changes)
    return Chunk(**values)


def test_default_configuration_reserves_embedding_capacity() -> None:
    config = DEFAULT_CHUNKING_CONFIG
    assert config.max_tokens == 400
    assert config.overlap_tokens == 50
    assert config.embedding_limit_tokens == 512
    assert config.tokenizer_model == "embed-english-light-v3.0"
    assert config.version.startswith("chunkcfg_")
    assert len(config.version) == 73


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_configuration_requires_positive_maximum(max_tokens: Any) -> None:
    with pytest.raises(ValueError, match="maximum tokens"):
        ChunkingConfig(max_tokens=max_tokens)


@pytest.mark.parametrize("overlap_tokens", [-1, True, 1.5])
def test_configuration_requires_nonnegative_overlap(overlap_tokens: Any) -> None:
    with pytest.raises(ValueError, match="overlap tokens"):
        ChunkingConfig(overlap_tokens=overlap_tokens)


@pytest.mark.parametrize("overlap_tokens", [400, 401])
def test_overlap_must_be_smaller_than_maximum(overlap_tokens: int) -> None:
    with pytest.raises(ValueError, match="smaller"):
        ChunkingConfig(overlap_tokens=overlap_tokens)


@pytest.mark.parametrize("embedding_limit", [0, -1, True, 1.5])
def test_configuration_requires_positive_embedding_limit(embedding_limit: Any) -> None:
    with pytest.raises(ValueError, match="Embedding token limit"):
        ChunkingConfig(embedding_limit_tokens=embedding_limit)


def test_maximum_cannot_exceed_embedding_limit() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ChunkingConfig(max_tokens=513)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokenizer_model", "", "tokenizer model"),
        ("tokenizer_model", 42, "tokenizer model"),
        ("tokenizer_revision", " ", "tokenizer revision"),
        ("tokenizer_revision", 42, "tokenizer revision"),
        ("algorithm", "", "algorithm"),
        ("algorithm", 42, "algorithm"),
        ("tokenizer_sha256", "a" * 63, "tokenizer hash"),
        ("tokenizer_sha256", "A" * 64, "tokenizer hash"),
        ("tokenizer_sha256", 42, "tokenizer hash"),
    ],
)
def test_configuration_rejects_invalid_identity_fields(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**{field: value})


@pytest.mark.parametrize(
    "changed",
    [
        {"max_tokens": 399},
        {"overlap_tokens": 49},
        {"embedding_limit_tokens": 513},
        {"tokenizer_model": "other-model"},
        {"tokenizer_revision": "other-revision"},
        {"tokenizer_sha256": "b" * 64},
        {"algorithm": "structure-aware-v2"},
    ],
)
def test_every_configuration_field_changes_version(changed: dict[str, Any]) -> None:
    assert replace(DEFAULT_CHUNKING_CONFIG, **changed).version != CONFIG_VERSION


def test_source_span_retains_exact_range_and_locator() -> None:
    source = span(2, 5, 12, page=3, headings=("Policy",))
    assert source.start_char == 5 and source.end_char == 12
    assert source.locator.page_number == 3
    assert source.locator.heading_path == ("Policy",)
    assert source.canonical_value() == (
        '{"block_number":2,"end_char":12,"heading_path":["Policy"],"page_number":3,"start_char":5}'
    )


def test_source_span_validation() -> None:
    with pytest.raises(ValueError, match="SourceLocator"):
        SourceSpan("block", 0, 1)  # type: ignore[arg-type]
    for start in (-1, True, 1.5):
        with pytest.raises(ValueError, match="start"):
            SourceSpan(SourceLocator(1), start, 2)  # type: ignore[arg-type]
    for start, end in ((0, 0), (1, 1), (2, 1), (0, True), (0, 1.5)):
        with pytest.raises(ValueError, match="end"):
            SourceSpan(SourceLocator(1), start, end)  # type: ignore[arg-type]


def test_chunk_id_is_stable_and_sensitive_to_every_meaningful_field() -> None:
    base = chunk()
    assert base.chunk_id == chunk().chunk_id
    assert base.chunk_id.startswith("chk_") and len(base.chunk_id) == 68
    variants = (
        chunk(document_id="doc_" + "b" * 64),
        chunk(config_version="chunkcfg_" + "b" * 64),
        chunk(order=2),
        chunk(content_text="Different source text"),
        chunk(embedding_text="Different heading context\n\nEmployees may work remotely."),
        chunk(source_spans=(span(start=1),)),
        chunk(source_spans=(span(), span(2, 0, 5, page=2))),
    )
    assert all(candidate.chunk_id != base.chunk_id for candidate in variants)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_id", "random", "document ID"),
        ("config_version", "random", "configuration version"),
        ("order", 0, "order"),
        ("order", True, "order"),
        ("content_text", "", "content text"),
        ("content_text", 42, "content text"),
        ("embedding_text", "", "embedding text"),
        ("embedding_text", 42, "embedding text"),
        ("embedding_token_count", 0, "embedding token count"),
        ("embedding_token_count", True, "embedding token count"),
        ("source_spans", (), "source spans"),
        ("source_spans", [], "source spans"),
        ("source_spans", ("span",), "SourceSpan"),
    ],
)
def test_chunk_validation(field: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chunk(**{field: value})


def test_chunk_spans_must_preserve_source_order() -> None:
    with pytest.raises(ValueError, match="source order"):
        chunk(source_spans=(span(2), span(1)))


def test_chunk_spans_within_one_block_must_preserve_character_order() -> None:
    with pytest.raises(ValueError, match="source order"):
        chunk(source_spans=(span(start=5, end=10), span(start=0, end=5)))


@pytest.mark.parametrize("field", ["document_id", "config_version"])
def test_chunk_identity_fields_reject_non_strings(field: str) -> None:
    with pytest.raises(ValueError, match="identity|version"):
        chunk(**{field: 42})
