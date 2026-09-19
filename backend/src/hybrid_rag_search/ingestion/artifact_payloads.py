"""Versioned JSON contracts for resumable ingestion-stage artifacts."""

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.ingestion.artifacts import ArtifactKind, canonical_json_bytes
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.providers.embeddings import EmbeddingPurpose, EmbeddingResult

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactContractError(ValueError):
    """Stored artifact bytes do not satisfy the expected versioned contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class _LocatorValue(_StrictModel):
    block_number: int = Field(gt=0)
    page_number: int | None = Field(default=None, gt=0)
    heading_path: tuple[str, ...] = ()


class _BlockValue(_StrictModel):
    text: str = Field(min_length=1)
    locator: _LocatorValue


class _ParsedPayload(_StrictModel):
    media_type: str = Field(min_length=1)
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    blocks: tuple[_BlockValue, ...] = Field(min_length=1)


class _ParsedEnvelope(_StrictModel):
    schema_version: Literal[1]
    kind: Literal["parsed"]
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: _ParsedPayload


class _SpanValue(_StrictModel):
    locator: _LocatorValue
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class _ChunkValue(_StrictModel):
    chunk_id: str = Field(pattern=r"^chk_[0-9a-f]{64}$")
    order: int = Field(gt=0)
    content_text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    embedding_token_count: int = Field(gt=0)
    source_spans: tuple[_SpanValue, ...] = Field(min_length=1)


class _ChunksPayload(_StrictModel):
    document_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    config_version: str = Field(pattern=r"^chunkcfg_[0-9a-f]{64}$")
    chunks: tuple[_ChunkValue, ...] = Field(min_length=1)


class _ChunksEnvelope(_StrictModel):
    schema_version: Literal[1]
    kind: Literal["chunks"]
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: _ChunksPayload


class _EmbeddingValue(_StrictModel):
    chunk_id: str = Field(pattern=r"^chk_[0-9a-f]{64}$")
    vector: tuple[float, ...] = Field(min_length=1)


class _EmbeddingsPayload(_StrictModel):
    document_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    adapter: str = Field(min_length=1)
    purpose: Literal["document"]
    embeddings: tuple[_EmbeddingValue, ...] = Field(min_length=1)


class _EmbeddingsEnvelope(_StrictModel):
    schema_version: Literal[1]
    kind: Literal["embeddings"]
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: _EmbeddingsPayload


@dataclass(frozen=True)
class ChunkEmbedding:
    """One vector explicitly paired with the stable chunk it represents."""

    chunk_id: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not re.fullmatch(
            r"chk_[0-9a-f]{64}", self.chunk_id
        ):
            raise ValueError("Embedding chunk ID must be a stable chunk identity")
        if not isinstance(self.vector, tuple) or not self.vector:
            raise ValueError("Embedding vector must be a nonempty tuple")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in self.vector
        ):
            raise ValueError("Embedding vector values must be finite numbers")


@dataclass(frozen=True)
class EmbeddingsArtifact:
    """Index-ready document vectors with stable chunk associations."""

    document_id: str
    model: str
    dimensions: int
    adapter: str
    embeddings: tuple[ChunkEmbedding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not re.fullmatch(
            r"doc_[0-9a-f]{64}", self.document_id
        ):
            raise ValueError("Embeddings document ID must be a stable document identity")
        for label, value in (("model", self.model), ("adapter", self.adapter)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Embeddings {label} must be a nonblank string")
        if (
            not isinstance(self.dimensions, int)
            or isinstance(self.dimensions, bool)
            or self.dimensions <= 0
        ):
            raise ValueError("Embedding dimensions must be a positive integer")
        if not isinstance(self.embeddings, tuple) or not self.embeddings:
            raise ValueError("Embeddings must be a nonempty tuple")
        if any(not isinstance(item, ChunkEmbedding) for item in self.embeddings):
            raise ValueError("Embeddings must contain ChunkEmbedding values")
        chunk_ids = tuple(item.chunk_id for item in self.embeddings)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Embedding chunk IDs must be unique")
        if any(len(item.vector) != self.dimensions for item in self.embeddings):
            raise ValueError("Embedding vectors must match the declared dimensions")


def _locator_value(locator: SourceLocator) -> dict[str, object]:
    return {
        "block_number": locator.block_number,
        "page_number": locator.page_number,
        "heading_path": locator.heading_path,
    }


def _span_value(span: SourceSpan) -> dict[str, object]:
    return {
        "locator": _locator_value(span.locator),
        "start_char": span.start_char,
        "end_char": span.end_char,
    }


def _envelope(kind: ArtifactKind, identity: str, payload: object) -> bytes:
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("Artifact payload identity must be a lowercase SHA-256 digest")
    return canonical_json_bytes(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": kind.value,
            "identity": identity,
            "payload": payload,
        }
    )


def _parse_envelope(model: type[_StrictModel], content: bytes, label: str) -> _StrictModel:
    if not isinstance(content, bytes):
        raise ArtifactContractError(f"Invalid {label} artifact")
    try:
        return model.model_validate_json(content)
    except ValidationError:
        raise ArtifactContractError(f"Invalid {label} artifact") from None


def _require_identity(actual: str, expected: str) -> None:
    if actual != expected:
        raise ArtifactContractError("Artifact identity does not match its checkpoint")


def encode_parsed_artifact(document: ParsedDocument, identity: str) -> bytes:
    if not isinstance(document, ParsedDocument):
        raise ValueError("Parsed artifact requires a ParsedDocument")
    return _envelope(
        ArtifactKind.PARSED,
        identity,
        {
            "media_type": document.media_type,
            "parser_name": document.parser_name,
            "parser_version": document.parser_version,
            "blocks": [
                {"text": block.text, "locator": _locator_value(block.locator)}
                for block in document.blocks
            ],
        },
    )


def decode_parsed_artifact(content: bytes, expected_identity: str) -> ParsedDocument:
    envelope = _parse_envelope(_ParsedEnvelope, content, "parsed")
    assert isinstance(envelope, _ParsedEnvelope)
    _require_identity(envelope.identity, expected_identity)
    try:
        return ParsedDocument(
            blocks=tuple(
                ParsedBlock(
                    block.text,
                    SourceLocator(
                        block.locator.block_number,
                        block.locator.page_number,
                        block.locator.heading_path,
                    ),
                )
                for block in envelope.payload.blocks
            ),
            media_type=envelope.payload.media_type,
            parser_name=envelope.payload.parser_name,
            parser_version=envelope.payload.parser_version,
        )
    except ValueError:
        raise ArtifactContractError("Invalid parsed artifact") from None


def _validate_chunks(
    chunks: tuple[Chunk, ...], *, require_first_order: bool = True
) -> tuple[str, str]:
    if not isinstance(chunks, tuple) or not chunks:
        raise ValueError("Chunks artifact requires a nonempty tuple")
    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise ValueError("Chunks artifact requires Chunk values")
    document_id, config_version = chunks[0].document_id, chunks[0].config_version
    if any(
        chunk.document_id != document_id or chunk.config_version != config_version
        for chunk in chunks
    ):
        raise ValueError("Chunks artifact must use one document and configuration")
    first_order = 1 if require_first_order else chunks[0].order
    if tuple(chunk.order for chunk in chunks) != tuple(
        range(first_order, first_order + len(chunks))
    ):
        raise ValueError("Chunks artifact order must be consecutive")
    return document_id, config_version


def encode_chunks_artifact(chunks: tuple[Chunk, ...], identity: str) -> bytes:
    document_id, config_version = _validate_chunks(chunks)
    return _envelope(
        ArtifactKind.CHUNKS,
        identity,
        {
            "document_id": document_id,
            "config_version": config_version,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "order": chunk.order,
                    "content_text": chunk.content_text,
                    "embedding_text": chunk.embedding_text,
                    "embedding_token_count": chunk.embedding_token_count,
                    "source_spans": [_span_value(span) for span in chunk.source_spans],
                }
                for chunk in chunks
            ],
        },
    )


def decode_chunks_artifact(content: bytes, expected_identity: str) -> tuple[Chunk, ...]:
    envelope = _parse_envelope(_ChunksEnvelope, content, "chunks")
    assert isinstance(envelope, _ChunksEnvelope)
    _require_identity(envelope.identity, expected_identity)
    try:
        chunks = tuple(
            Chunk(
                document_id=envelope.payload.document_id,
                config_version=envelope.payload.config_version,
                order=value.order,
                content_text=value.content_text,
                embedding_text=value.embedding_text,
                embedding_token_count=value.embedding_token_count,
                source_spans=tuple(
                    SourceSpan(
                        SourceLocator(
                            span.locator.block_number,
                            span.locator.page_number,
                            span.locator.heading_path,
                        ),
                        span.start_char,
                        span.end_char,
                    )
                    for span in value.source_spans
                ),
            )
            for value in envelope.payload.chunks
        )
        _validate_chunks(chunks)
        if any(
            value.chunk_id != chunk.chunk_id
            for value, chunk in zip(envelope.payload.chunks, chunks, strict=True)
        ):
            raise ValueError
        return chunks
    except ValueError:
        raise ArtifactContractError("Invalid chunks artifact") from None


def build_embeddings_artifact(
    chunks: tuple[Chunk, ...], result: EmbeddingResult
) -> EmbeddingsArtifact:
    document_id, _ = _validate_chunks(chunks, require_first_order=False)
    if not isinstance(result, EmbeddingResult):
        raise ValueError("Embeddings artifact requires an EmbeddingResult")
    if result.purpose != EmbeddingPurpose.DOCUMENT:
        raise ValueError("Ingestion requires document-purpose embeddings")
    if len(result.vectors) != len(chunks):
        raise ValueError("Embedding count must match the chunk count")
    return EmbeddingsArtifact(
        document_id=document_id,
        model=result.model,
        dimensions=result.dimensions,
        adapter=result.adapter,
        embeddings=tuple(
            ChunkEmbedding(chunk.chunk_id, vector)
            for chunk, vector in zip(chunks, result.vectors, strict=True)
        ),
    )


def encode_embeddings_artifact(artifact: EmbeddingsArtifact, identity: str) -> bytes:
    if not isinstance(artifact, EmbeddingsArtifact):
        raise ValueError("Embeddings artifact requires an EmbeddingsArtifact")
    return _envelope(
        ArtifactKind.EMBEDDINGS,
        identity,
        {
            "document_id": artifact.document_id,
            "model": artifact.model,
            "dimensions": artifact.dimensions,
            "adapter": artifact.adapter,
            "purpose": EmbeddingPurpose.DOCUMENT.value,
            "embeddings": [
                {"chunk_id": item.chunk_id, "vector": item.vector} for item in artifact.embeddings
            ],
        },
    )


def decode_embeddings_artifact(content: bytes, expected_identity: str) -> EmbeddingsArtifact:
    envelope = _parse_envelope(_EmbeddingsEnvelope, content, "embeddings")
    assert isinstance(envelope, _EmbeddingsEnvelope)
    _require_identity(envelope.identity, expected_identity)
    try:
        return EmbeddingsArtifact(
            document_id=envelope.payload.document_id,
            model=envelope.payload.model,
            dimensions=envelope.payload.dimensions,
            adapter=envelope.payload.adapter,
            embeddings=tuple(
                ChunkEmbedding(value.chunk_id, value.vector)
                for value in envelope.payload.embeddings
            ),
        )
    except ValueError:
        raise ArtifactContractError("Invalid embeddings artifact") from None
