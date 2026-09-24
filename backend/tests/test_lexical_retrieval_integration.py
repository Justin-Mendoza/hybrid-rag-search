"""Optional real-OpenSearch checks for the Day 10 lexical retrieval contract."""

from uuid import uuid4

import httpx
import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.config import Settings
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import IndexRecord, ReplaceDocumentRequest
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever
from hybrid_rag_search.opensearch_index import OpenSearchDocumentIndex, OpenSearchIndexManager
from hybrid_rag_search.parsers.contracts import SourceLocator


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_bm25_recovers_heading_phrase_and_identifier_from_real_opensearch() -> None:
    settings = Settings()
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    content_hash = "a" * 64
    content_id = DocumentContentIdentity(tenant_id, collection_id, content_hash).document_id
    chunk = Chunk(
        content_id,
        "chunkcfg_" + "b" * 64,
        1,
        "HR-104 says password reset links expire after fifteen minutes.",
        "# Authentication\n\nHR-104 says password reset links expire after fifteen minutes.",
        12,
        (SourceSpan(SourceLocator(1, heading_path=("Authentication",)), 0, 62),),
    )
    request = ReplaceDocumentRequest(
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_id=document_id,
        content_hash=content_hash,
        pipeline_version="pipe_" + "c" * 64,
        embedding_model="integration-test",
        embedding_dimensions=2,
        embedding_adapter="fake",
        source_metadata={},
        records=(IndexRecord(chunk, ChunkEmbedding(chunk.chunk_id, (0.1, 0.2))),),
    )
    index_name = f"hybrid-rag-chunks-v3-lexical-{uuid4().hex[:12]}"
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        "unused-read-alias",
        "unused-write-alias",
        2,
    )
    try:
        await manager.create(index_name)
        await OpenSearchDocumentIndex(settings.opensearch_url, index_name, 2).replace_document(
            request
        )
        retriever = OpenSearchBM25Retriever(settings.opensearch_url, index_name)
        response = await retriever.search(
            'authentication "password reset" HR-104',
            tenant_id=tenant_id,
            collection_id=collection_id,
            debug=True,
        )
        assert [result.chunk_id for result in response.results] == [chunk.chunk_id]
        assert response.trace.candidate_count == 1
        assert response.trace.explanations
        denied = await retriever.search("HR-104", tenant_id=uuid4())
        assert denied.results == ()
    finally:
        async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=10) as client:
            response = await client.delete(f"/{index_name}")
            if response.status_code != 404:
                response.raise_for_status()
