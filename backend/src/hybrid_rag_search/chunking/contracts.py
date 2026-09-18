"""Immutable chunking configuration, source spans, and chunk output."""

import json
import re
from dataclasses import dataclass

from hybrid_rag_search.chunking.identity import stable_digest
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.tokenization import COHERE_EMBED_ENGLISH_LIGHT_V3

CHUNKING_ALGORITHM = "structure-aware-v1"
EMBEDDING_LIMIT_TOKENS = 512


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class ChunkingConfig:
    """Every value that can change chunk boundaries or token interpretation."""

    max_tokens: int = 400
    overlap_tokens: int = 50
    embedding_limit_tokens: int = EMBEDDING_LIMIT_TOKENS
    tokenizer_model: str = COHERE_EMBED_ENGLISH_LIGHT_V3.model
    tokenizer_revision: str = COHERE_EMBED_ENGLISH_LIGHT_V3.revision
    tokenizer_sha256: str = COHERE_EMBED_ENGLISH_LIGHT_V3.sha256
    algorithm: str = CHUNKING_ALGORITHM

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.max_tokens):
            raise ValueError("Chunk maximum tokens must be a positive integer")
        if not _is_nonnegative_integer(self.overlap_tokens):
            raise ValueError("Chunk overlap tokens must be a nonnegative integer")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("Chunk overlap tokens must be smaller than maximum tokens")
        if not _is_positive_integer(self.embedding_limit_tokens):
            raise ValueError("Embedding token limit must be a positive integer")
        if self.max_tokens > self.embedding_limit_tokens:
            raise ValueError("Chunk maximum tokens cannot exceed the embedding token limit")
        for label, value in (
            ("tokenizer model", self.tokenizer_model),
            ("tokenizer revision", self.tokenizer_revision),
            ("algorithm", self.algorithm),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Chunk {label} must be a nonblank string")
        if not isinstance(self.tokenizer_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.tokenizer_sha256
        ):
            raise ValueError("Chunk tokenizer hash must be a lowercase SHA-256 hex digest")

    @property
    def version(self) -> str:
        canonical = json.dumps(
            {
                "algorithm": self.algorithm,
                "embedding_limit_tokens": self.embedding_limit_tokens,
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "tokenizer_model": self.tokenizer_model,
                "tokenizer_revision": self.tokenizer_revision,
                "tokenizer_sha256": self.tokenizer_sha256,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"chunkcfg_{stable_digest('chunk-config', canonical)}"


@dataclass(frozen=True)
class SourceSpan:
    """Exact half-open character range within one normalized parser block."""

    locator: SourceLocator
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if not isinstance(self.locator, SourceLocator):
            raise ValueError("Chunk source span locator must be a SourceLocator")
        if not _is_nonnegative_integer(self.start_char):
            raise ValueError("Chunk source span start must be a nonnegative integer")
        if not _is_positive_integer(self.end_char) or self.end_char <= self.start_char:
            raise ValueError("Chunk source span end must be greater than its start")

    def canonical_value(self) -> str:
        return json.dumps(
            {
                "block_number": self.locator.block_number,
                "end_char": self.end_char,
                "heading_path": self.locator.heading_path,
                "page_number": self.locator.page_number,
                "start_char": self.start_char,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class Chunk:
    """One search record with exact source text and enriched embedding input."""

    document_id: str
    config_version: str
    order: int
    content_text: str
    embedding_text: str
    embedding_token_count: int
    source_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not re.fullmatch(
            r"doc_[0-9a-f]{64}", self.document_id
        ):
            raise ValueError("Chunk document ID must be a stable document identity")
        if not isinstance(self.config_version, str) or not re.fullmatch(
            r"chunkcfg_[0-9a-f]{64}", self.config_version
        ):
            raise ValueError("Chunk configuration version must be a stable configuration identity")
        if not _is_positive_integer(self.order):
            raise ValueError("Chunk order must be a positive integer")
        if not isinstance(self.content_text, str) or not self.content_text.strip():
            raise ValueError("Chunk content text must be a nonblank string")
        if not isinstance(self.embedding_text, str) or not self.embedding_text.strip():
            raise ValueError("Chunk embedding text must be a nonblank string")
        if not _is_positive_integer(self.embedding_token_count):
            raise ValueError("Chunk embedding token count must be a positive integer")
        if not isinstance(self.source_spans, tuple) or not self.source_spans:
            raise ValueError("Chunk source spans must be a nonempty tuple")
        if any(not isinstance(span, SourceSpan) for span in self.source_spans):
            raise ValueError("Chunk source spans must contain SourceSpan values")
        positions = tuple(
            (span.locator.block_number, span.start_char, span.end_char)
            for span in self.source_spans
        )
        if positions != tuple(sorted(positions)):
            raise ValueError("Chunk source spans must preserve source order")

    @property
    def chunk_id(self) -> str:
        digest = stable_digest(
            "chunk",
            self.document_id,
            self.config_version,
            str(self.order),
            self.content_text,
            self.embedding_text,
            *(span.canonical_value() for span in self.source_spans),
        )
        return f"chk_{digest}"


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
