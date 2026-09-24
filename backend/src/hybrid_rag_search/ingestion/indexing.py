"""Application-owned document replacement boundary for search indexes."""

import json
import re
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from hybrid_rag_search.chunking.contracts import Chunk
from hybrid_rag_search.chunking.identity import DocumentContentIdentity, stable_digest
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.retrieval_filters import IndexedSourceMetadata


class IndexWriteError(RuntimeError):
    """A safe index replacement failure with retry classification."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", code):
            raise ValueError("Index error code must use lowercase snake_case")
        if not isinstance(retryable, bool):
            raise ValueError("Index retryable flag must be boolean")
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True)
class IndexRecord:
    """One validated chunk and its matching vector."""

    chunk: Chunk
    embedding: ChunkEmbedding

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise ValueError("Index record chunk must be a Chunk")
        if not isinstance(self.embedding, ChunkEmbedding):
            raise ValueError("Index record embedding must be a ChunkEmbedding")
        if self.chunk.chunk_id != self.embedding.chunk_id:
            raise ValueError("Index record chunk and embedding IDs must match")


@dataclass(frozen=True)
class ReplaceDocumentRequest:
    """Complete searchable generation for one tenant-scoped source document."""

    tenant_id: UUID
    collection_id: UUID
    document_id: UUID
    content_hash: str
    pipeline_version: str
    embedding_model: str
    embedding_dimensions: int
    embedding_adapter: str
    source_metadata: dict[str, object]
    records: tuple[IndexRecord, ...]

    def __post_init__(self) -> None:
        for id_label, id_value in (
            ("tenant", self.tenant_id),
            ("collection", self.collection_id),
            ("document", self.document_id),
        ):
            if not isinstance(id_value, UUID):
                raise ValueError(f"Index {id_label} ID must be a UUID")
        if not isinstance(self.content_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.content_hash
        ):
            raise ValueError("Index content hash must be a lowercase SHA-256 digest")
        if not isinstance(self.pipeline_version, str) or not re.fullmatch(
            r"pipe_[0-9a-f]{64}", self.pipeline_version
        ):
            raise ValueError("Index pipeline version must be a stable pipeline identity")
        for text_label, text_value in (
            ("embedding model", self.embedding_model),
            ("embedding adapter", self.embedding_adapter),
        ):
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(f"Index {text_label} must be a nonblank string")
        if (
            not isinstance(self.embedding_dimensions, int)
            or isinstance(self.embedding_dimensions, bool)
            or self.embedding_dimensions <= 0
        ):
            raise ValueError("Index embedding dimensions must be a positive integer")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("Index records must be a nonempty tuple")
        if any(not isinstance(record, IndexRecord) for record in self.records):
            raise ValueError("Index records must contain IndexRecord values")

        expected_content_id = DocumentContentIdentity(
            self.tenant_id,
            self.collection_id,
            self.content_hash,
        ).document_id
        if any(record.chunk.document_id != expected_content_id for record in self.records):
            raise ValueError("Index records must match the document content identity")
        if tuple(record.chunk.order for record in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise ValueError("Index records must preserve consecutive chunk order")
        config_versions = {record.chunk.config_version for record in self.records}
        if len(config_versions) != 1:
            raise ValueError("Index records must use one chunk configuration")
        if any(
            len(record.embedding.vector) != self.embedding_dimensions for record in self.records
        ):
            raise ValueError("Index vectors must match the declared dimensions")
        IndexedSourceMetadata.from_document(self.source_metadata)

    @property
    def generation_id(self) -> str:
        digest = stable_digest(
            "index-generation",
            str(self.tenant_id),
            str(self.collection_id),
            str(self.document_id),
            self.content_hash,
            self.pipeline_version,
            self.embedding_model,
            str(self.embedding_dimensions),
            json.dumps(
                self.source_metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ),
        )
        return f"idxgen_{digest}"


@dataclass(frozen=True)
class IndexReceipt:
    generation_id: str
    tenant_id: UUID
    document_id: UUID
    pipeline_version: str
    chunk_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, str) or not re.fullmatch(
            r"idxgen_[0-9a-f]{64}", self.generation_id
        ):
            raise ValueError("Index receipt generation must be a stable identity")
        if not isinstance(self.tenant_id, UUID) or not isinstance(self.document_id, UUID):
            raise ValueError("Index receipt tenant and document IDs must be UUIDs")
        if not isinstance(self.pipeline_version, str) or not re.fullmatch(
            r"pipe_[0-9a-f]{64}", self.pipeline_version
        ):
            raise ValueError("Index receipt pipeline version must be a stable identity")
        if (
            not isinstance(self.chunk_count, int)
            or isinstance(self.chunk_count, bool)
            or self.chunk_count <= 0
        ):
            raise ValueError("Index receipt chunk count must be a positive integer")

    @classmethod
    def from_request(cls, request: ReplaceDocumentRequest) -> "IndexReceipt":
        if not isinstance(request, ReplaceDocumentRequest):
            raise ValueError("Index receipt requires a replacement request")
        return cls(
            generation_id=request.generation_id,
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            pipeline_version=request.pipeline_version,
            chunk_count=len(request.records),
        )

    def checkpoint_value(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "tenant_id": str(self.tenant_id),
            "document_id": str(self.document_id),
            "pipeline_version": self.pipeline_version,
            "chunk_count": self.chunk_count,
        }


class DocumentIndex(Protocol):
    async def replace_document(self, request: ReplaceDocumentRequest) -> IndexReceipt: ...


class FakeDocumentIndex:
    """In-memory reference implementation with atomic active-generation replacement."""

    def __init__(self, *, fail_after_records: int | None = None) -> None:
        if fail_after_records is not None and (
            not isinstance(fail_after_records, int)
            or isinstance(fail_after_records, bool)
            or fail_after_records <= 0
        ):
            raise ValueError("Fake index failure point must be a positive integer")
        self.fail_after_records = fail_after_records
        self._active: dict[tuple[UUID, UUID], ReplaceDocumentRequest] = {}
        self._staging: dict[str, dict[str, IndexRecord]] = {}
        self.replacement_attempts = 0

    async def replace_document(self, request: ReplaceDocumentRequest) -> IndexReceipt:
        if not isinstance(request, ReplaceDocumentRequest):
            raise ValueError("Document replacement requires a ReplaceDocumentRequest")
        self.replacement_attempts += 1
        active_key = (request.tenant_id, request.document_id)
        existing = self._active.get(active_key)
        if existing is not None and existing.generation_id == request.generation_id:
            if existing != request:
                raise IndexWriteError("generation_conflict", retryable=False)
            return IndexReceipt.from_request(request)

        staged: dict[str, IndexRecord] = {}
        self._staging[request.generation_id] = staged
        try:
            for record in request.records:
                staged[record.chunk.chunk_id] = record
                if self.fail_after_records == len(staged):
                    raise IndexWriteError("partial_write", retryable=True)
            self._active[active_key] = request
        finally:
            self._staging.pop(request.generation_id, None)
        return IndexReceipt.from_request(request)

    def active_document(self, tenant_id: UUID, document_id: UUID) -> ReplaceDocumentRequest | None:
        return self._active.get((tenant_id, document_id))

    @property
    def staged_count(self) -> int:
        return sum(len(records) for records in self._staging.values())
