from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import (
    DocumentIndex,
    FakeDocumentIndex,
    IndexReceipt,
    IndexRecord,
    IndexWriteError,
    ReplaceDocumentRequest,
)
from hybrid_rag_search.parsers.contracts import SourceLocator

CONTENT_HASH = "a" * 64
PIPELINE_VERSION = "pipe_" + "b" * 64
CONFIG_VERSION = "chunkcfg_" + "c" * 64


def chunk(content_id: str, order: int, text: str | None = None) -> Chunk:
    content = text or f"chunk {order}"
    return Chunk(
        document_id=content_id,
        config_version=CONFIG_VERSION,
        order=order,
        content_text=content,
        embedding_text=content,
        embedding_token_count=2,
        source_spans=(SourceSpan(SourceLocator(order), 0, len(content)),),
    )


def request(
    *,
    tenant_id=None,
    collection_id=None,
    document_id=None,
    content_hash: str = CONTENT_HASH,
    pipeline_version: str = PIPELINE_VERSION,
    count: int = 2,
    text_prefix: str = "chunk",
) -> ReplaceDocumentRequest:
    tenant_id = tenant_id or uuid4()
    collection_id = collection_id or uuid4()
    document_id = document_id or uuid4()
    content_id = DocumentContentIdentity(tenant_id, collection_id, content_hash).document_id
    chunks = tuple(
        chunk(content_id, order, f"{text_prefix} {order}") for order in range(1, count + 1)
    )
    return ReplaceDocumentRequest(
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_id=document_id,
        content_hash=content_hash,
        pipeline_version=pipeline_version,
        embedding_model="embed-test",
        embedding_dimensions=2,
        embedding_adapter="fake",
        source_metadata={},
        records=tuple(
            IndexRecord(item, ChunkEmbedding(item.chunk_id, (float(item.order), 0.0)))
            for item in chunks
        ),
    )


def test_request_has_stable_generation_and_preserves_search_data() -> None:
    value = request()
    assert value.generation_id == replace(value).generation_id
    assert value.generation_id.startswith("idxgen_")
    assert len(value.generation_id) == 71
    assert value.records[0].chunk.content_text == "chunk 1"
    assert value.records[0].chunk.source_spans[0].locator.block_number == 1
    assert value.records[0].embedding.vector == (1.0, 0.0)


def test_record_requires_matching_domain_values() -> None:
    value = request().records[0]
    with pytest.raises(ValueError, match="must be a Chunk"):
        IndexRecord("chunk", value.embedding)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ChunkEmbedding"):
        IndexRecord(value.chunk, "embedding")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="IDs must match"):
        IndexRecord(value.chunk, ChunkEmbedding("chk_" + "f" * 64, (1.0, 0.0)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tenant_id", "tenant", "tenant ID"),
        ("collection_id", "collection", "collection ID"),
        ("document_id", "document", "document ID"),
        ("content_hash", "bad", "content hash"),
        ("pipeline_version", "bad", "pipeline version"),
        ("embedding_model", " ", "embedding model"),
        ("embedding_adapter", 42, "embedding adapter"),
        ("embedding_dimensions", 0, "dimensions"),
        ("embedding_dimensions", True, "dimensions"),
        ("records", (), "nonempty tuple"),
        ("records", [], "nonempty tuple"),
        ("records", ("record",), "IndexRecord"),
    ],
)
def test_request_validates_scalar_and_collection_fields(
    field: str, value: object, message: str
) -> None:
    values: dict[str, Any] = request().__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ReplaceDocumentRequest(**values)


def test_request_requires_matching_content_identity() -> None:
    value = request()
    wrong_chunk = replace(value.records[0].chunk, document_id="doc_" + "f" * 64)
    records = (IndexRecord(wrong_chunk, ChunkEmbedding(wrong_chunk.chunk_id, (1.0, 0.0))),)
    with pytest.raises(ValueError, match="content identity"):
        replace(value, records=records)


def test_request_requires_order_and_one_configuration() -> None:
    value = request()
    first, second = value.records
    reordered_chunk = replace(first.chunk, order=2)
    reordered = (
        IndexRecord(
            reordered_chunk,
            ChunkEmbedding(reordered_chunk.chunk_id, first.embedding.vector),
        ),
        second,
    )
    with pytest.raises(ValueError, match="consecutive"):
        replace(value, records=reordered)

    other_chunk = replace(second.chunk, config_version="chunkcfg_" + "d" * 64)
    other = IndexRecord(
        other_chunk,
        ChunkEmbedding(other_chunk.chunk_id, second.embedding.vector),
    )
    with pytest.raises(ValueError, match="one chunk configuration"):
        replace(value, records=(first, other))


def test_request_requires_declared_vector_dimensions() -> None:
    value = request()
    with pytest.raises(ValueError, match="declared dimensions"):
        replace(value, embedding_dimensions=3)


def test_receipt_round_trip_and_checkpoint_value() -> None:
    value = request()
    receipt = IndexReceipt.from_request(value)
    assert receipt == IndexReceipt(
        value.generation_id,
        value.tenant_id,
        value.document_id,
        value.pipeline_version,
        len(value.records),
    )
    assert receipt.checkpoint_value() == {
        "generation_id": value.generation_id,
        "tenant_id": str(value.tenant_id),
        "document_id": str(value.document_id),
        "pipeline_version": value.pipeline_version,
        "chunk_count": 2,
    }
    with pytest.raises(ValueError, match="replacement request"):
        IndexReceipt.from_request("request")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_id", "bad", "generation"),
        ("tenant_id", "bad", "tenant and document"),
        ("document_id", "bad", "tenant and document"),
        ("pipeline_version", "bad", "pipeline version"),
        ("chunk_count", 0, "chunk count"),
        ("chunk_count", True, "chunk count"),
    ],
)
def test_receipt_validation(field: str, value: object, message: str) -> None:
    values: dict[str, Any] = IndexReceipt.from_request(request()).__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        IndexReceipt(**values)


@pytest.mark.parametrize("code", ["Bad-Code", "", "has space"])
def test_index_error_requires_stable_code(code: str) -> None:
    with pytest.raises(ValueError, match="snake_case"):
        IndexWriteError(code, retryable=True)


def test_index_error_requires_boolean_retry_classification() -> None:
    with pytest.raises(ValueError, match="boolean"):
        IndexWriteError("partial_write", retryable=1)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_fake_index_is_idempotent_and_structurally_implements_contract() -> None:
    index: DocumentIndex = FakeDocumentIndex()
    value = request()
    first = await index.replace_document(value)
    second = await index.replace_document(value)
    assert first == second == IndexReceipt.from_request(value)
    assert isinstance(index, FakeDocumentIndex)
    assert index.active_document(value.tenant_id, value.document_id) == value
    assert index.staged_count == 0
    assert index.replacement_attempts == 2


@pytest.mark.anyio
async def test_partial_failure_cleans_staging_and_preserves_old_generation() -> None:
    index = FakeDocumentIndex()
    old = request()
    await index.replace_document(old)
    replacement = request(
        tenant_id=old.tenant_id,
        collection_id=old.collection_id,
        document_id=old.document_id,
        content_hash="d" * 64,
        pipeline_version="pipe_" + "e" * 64,
        count=3,
        text_prefix="new",
    )
    index.fail_after_records = 2
    with pytest.raises(IndexWriteError) as caught:
        await index.replace_document(replacement)
    assert caught.value.code == "partial_write"
    assert caught.value.retryable is True
    assert index.staged_count == 0
    assert index.active_document(old.tenant_id, old.document_id) == old

    index.fail_after_records = None
    assert await index.replace_document(replacement) == IndexReceipt.from_request(replacement)
    assert index.active_document(old.tenant_id, old.document_id) == replacement


@pytest.mark.anyio
async def test_same_generation_with_different_data_is_permanent_conflict() -> None:
    index = FakeDocumentIndex()
    original = request()
    await index.replace_document(original)
    changed_chunk = replace(original.records[0].chunk, content_text="nondeterministic")
    changed_record = IndexRecord(
        changed_chunk,
        ChunkEmbedding(changed_chunk.chunk_id, original.records[0].embedding.vector),
    )
    changed = replace(original, records=(changed_record, *original.records[1:]))
    assert changed.generation_id == original.generation_id
    with pytest.raises(IndexWriteError) as caught:
        await index.replace_document(changed)
    assert caught.value.code == "generation_conflict"
    assert caught.value.retryable is False
    assert index.active_document(original.tenant_id, original.document_id) == original


@pytest.mark.anyio
async def test_tenants_and_documents_have_independent_active_generations() -> None:
    index = FakeDocumentIndex()
    first = request()
    other_tenant = request(document_id=first.document_id)
    other_document = request(tenant_id=first.tenant_id, collection_id=first.collection_id)
    for value in (first, other_tenant, other_document):
        await index.replace_document(value)
    assert index.active_document(first.tenant_id, first.document_id) == first
    assert index.active_document(other_tenant.tenant_id, first.document_id) == other_tenant
    assert index.active_document(first.tenant_id, other_document.document_id) == other_document
    assert index.active_document(uuid4(), uuid4()) is None


@pytest.mark.anyio
async def test_fake_index_validates_inputs_and_failure_configuration() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            FakeDocumentIndex(fail_after_records=value)  # type: ignore[arg-type]
    index = FakeDocumentIndex()
    with pytest.raises(ValueError, match="ReplaceDocumentRequest"):
        await index.replace_document("request")  # type: ignore[arg-type]
