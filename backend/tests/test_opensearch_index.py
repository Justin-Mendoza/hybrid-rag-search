import json
from uuid import uuid4

import httpx
import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import (
    IndexRecord,
    IndexWriteError,
    ReplaceDocumentRequest,
)
from hybrid_rag_search.opensearch_index import (
    OpenSearchDocumentIndex,
    OpenSearchIndexManager,
    index_mapping,
    physical_index_name,
)
from hybrid_rag_search.parsers.contracts import SourceLocator


def replacement_request() -> ReplaceDocumentRequest:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    content_hash = "a" * 64
    content_id = DocumentContentIdentity(tenant_id, collection_id, content_hash).document_id
    chunk = Chunk(
        content_id,
        "chunkcfg_" + "b" * 64,
        1,
        "A reset token expires after fifteen minutes.",
        "# Login\n\nA reset token expires after fifteen minutes.",
        10,
        (SourceSpan(SourceLocator(1, page_number=2, heading_path=("Login",)), 0, 43),),
    )
    return ReplaceDocumentRequest(
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_id=document_id,
        content_hash=content_hash,
        pipeline_version="pipe_" + "c" * 64,
        embedding_model="embed-test",
        embedding_dimensions=2,
        embedding_adapter="fake",
        records=(IndexRecord(chunk, ChunkEmbedding(chunk.chunk_id, (0.1, 0.2))),),
    )


def test_index_mapping_and_name_are_versioned() -> None:
    mapping = index_mapping(384)
    properties = mapping["mappings"]["properties"]  # type: ignore[index]
    assert properties["embedding"]["method"]["space_type"] == "cosinesimil"  # type: ignore[index]
    assert properties["embedding"]["dimension"] == 384  # type: ignore[index]
    assert properties["source_spans"]["type"] == "nested"  # type: ignore[index]
    assert physical_index_name("chunks-v1", "20260919") == "hybrid-rag-chunks-v1-20260919"
    for value in (0, True):
        with pytest.raises(ValueError, match="dimensions"):
            index_mapping(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema version"):
        physical_index_name("v1", "valid")
    with pytest.raises(ValueError, match="build ID"):
        physical_index_name("chunks-v1", "Invalid")


@pytest.mark.anyio
async def test_manager_creates_and_switches_aliases() -> None:
    requests: list[httpx.Request] = []

    async def responder(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"acknowledged": True})

    manager = OpenSearchIndexManager(
        "http://search.test",
        "chunks-read",
        "chunks-write",
        2,
        httpx.MockTransport(responder),
    )
    await manager.create("hybrid-rag-chunks-v1-test")
    await manager.point_write_alias("hybrid-rag-chunks-v1-test")
    await manager.promote_read_alias("hybrid-rag-chunks-v1-test")
    assert requests[0].method == "PUT"
    assert requests[0].url.path == "/hybrid-rag-chunks-v1-test"
    assert json.loads(requests[0].content)["mappings"]["properties"]["embedding"]["dimension"] == 2
    write_actions = json.loads(requests[1].content)["actions"]
    assert write_actions[-1]["add"]["alias"] == "chunks-write"
    read_actions = json.loads(requests[2].content)["actions"]
    assert read_actions[-1]["add"]["alias"] == "chunks-read"
    assert "is_write_index" not in read_actions[-1]["add"]


@pytest.mark.anyio
async def test_replace_stages_promotes_and_removes_prior_generation() -> None:
    requests: list[httpx.Request] = []

    async def responder(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("_bulk"):
            return httpx.Response(200, json={"errors": False, "items": []})
        return httpx.Response(200, json={"updated": 1, "deleted": 1})

    request = replacement_request()
    index = OpenSearchDocumentIndex(
        "http://search.test", "chunks-write", 2, transport=httpx.MockTransport(responder)
    )
    receipt = await index.replace_document(request)
    assert receipt.generation_id == request.generation_id
    assert [item.url.path for item in requests] == [
        "/chunks-write/_bulk",
        "/chunks-write/_update_by_query",
        "/chunks-write/_delete_by_query",
    ]
    staged = requests[0].content.decode().splitlines()
    document = json.loads(staged[1])
    assert document["visibility"] == "staged"
    assert document["source_spans"][0]["page_number"] == 2
    promotion = json.loads(requests[1].content)
    assert promotion["script"]["source"] == "ctx._source.visibility = 'ready'"
    cleanup = json.loads(requests[2].content)
    assert cleanup["query"]["bool"]["must_not"]


@pytest.mark.anyio
async def test_index_errors_are_classified_and_deletion_is_scoped() -> None:
    request = replacement_request()

    async def failure(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    index = OpenSearchDocumentIndex(
        "http://search.test", "chunks-write", 2, transport=httpx.MockTransport(failure)
    )
    with pytest.raises(IndexWriteError, match="opensearch_unavailable") as error:
        await index.replace_document(request)
    assert error.value.retryable

    requests: list[httpx.Request] = []

    async def success(value: httpx.Request) -> httpx.Response:
        requests.append(value)
        return httpx.Response(200, json={"deleted": 0})

    index = OpenSearchDocumentIndex(
        "http://search.test", "chunks-write", 2, transport=httpx.MockTransport(success)
    )
    await index.delete_document(str(request.tenant_id), str(request.document_id))
    query = json.loads(requests[0].content)["query"]["bool"]["filter"]
    assert {next(iter(item["term"])) for item in query} == {"tenant_id", "document_id"}


@pytest.mark.anyio
async def test_bulk_item_errors_are_retryable() -> None:
    async def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("_bulk")
        return httpx.Response(200, json={"errors": True})

    index = OpenSearchDocumentIndex(
        "http://search.test", "chunks-write", 2, transport=httpx.MockTransport(responder)
    )
    with pytest.raises(IndexWriteError, match="bulk_write_failed") as error:
        await index.replace_document(replacement_request())
    assert error.value.retryable


@pytest.mark.anyio
async def test_dimension_promotion_and_transport_failures_are_safe() -> None:
    request = replacement_request()
    index = OpenSearchDocumentIndex("http://search.test", "chunks-write", 3)
    with pytest.raises(IndexWriteError, match="vector_dimensions_mismatch") as mismatch:
        await index.replace_document(request)
    assert not mismatch.value.retryable

    async def incomplete(value: httpx.Request) -> httpx.Response:
        if value.url.path.endswith("_bulk"):
            return httpx.Response(200, json={"errors": False})
        return httpx.Response(200, json={"updated": True})

    index = OpenSearchDocumentIndex(
        "http://search.test", "chunks-write", 2, transport=httpx.MockTransport(incomplete)
    )
    with pytest.raises(IndexWriteError, match="generation_incomplete"):
        await index.replace_document(request)

    async def disconnected(value: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=value)

    transport = httpx.MockTransport(disconnected)
    index = OpenSearchDocumentIndex("http://search.test", "chunks-write", 2, transport=transport)
    with pytest.raises(IndexWriteError) as unavailable:
        await index.delete_document(str(request.tenant_id), str(request.document_id))
    assert unavailable.value.retryable

    manager = OpenSearchIndexManager("http://search.test", "read", "write", 2, transport=transport)
    with pytest.raises(IndexWriteError) as manager_unavailable:
        await manager.create("hybrid-rag-chunks-v1-test")
    assert manager_unavailable.value.retryable


@pytest.mark.anyio
async def test_http_client_errors_distinguish_retryability() -> None:
    for status, retryable in ((400, False), (408, True), (429, True), (500, True)):

        async def responder(_: httpx.Request, status: int = status) -> httpx.Response:
            return httpx.Response(status)

        manager = OpenSearchIndexManager(
            "http://search.test", "read", "write", 2, httpx.MockTransport(responder)
        )
        with pytest.raises(IndexWriteError) as error:
            await manager.create("hybrid-rag-chunks-v1-test")
        assert error.value.retryable is retryable
