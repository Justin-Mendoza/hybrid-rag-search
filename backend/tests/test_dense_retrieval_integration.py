"""Optional real-OpenSearch checks for the Day 11 dense retrieval contract."""

from uuid import uuid4

import httpx
import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import DenseRetrievalConfig, OpenSearchDenseRetriever
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import IndexRecord, ReplaceDocumentRequest
from hybrid_rag_search.opensearch_index import OpenSearchDocumentIndex, OpenSearchIndexManager
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.providers.embeddings import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ParaphraseFixtureProvider:
    """A deterministic fake adapter whose query vector matches the target chunk."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        assert request == EmbeddingRequest(
            ("How long does a password recovery link last?",), EmbeddingPurpose.QUERY
        )
        return EmbeddingResult(
            vectors=((1.0, 0.0, 0.0),),
            model="semantic-fixture-v1",
            dimensions=3,
            purpose=EmbeddingPurpose.QUERY,
            adapter="fake",
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_paraphrase_fixture_recovers_ready_tenant_scoped_chunk_from_real_opensearch() -> None:
    settings = Settings()
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    content_hash = "a" * 64
    content_id = DocumentContentIdentity(tenant_id, collection_id, content_hash).document_id
    chunk = Chunk(
        content_id,
        "chunkcfg_" + "b" * 64,
        1,
        "Password reset links expire after fifteen minutes.",
        "# Authentication\n\nPassword reset links expire after fifteen minutes.",
        8,
        (SourceSpan(SourceLocator(1, heading_path=("Authentication",)), 0, 50),),
    )
    request = ReplaceDocumentRequest(
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_id=document_id,
        content_hash=content_hash,
        pipeline_version="pipe_" + "c" * 64,
        embedding_model="semantic-fixture-v1",
        embedding_dimensions=3,
        embedding_adapter="fake",
        records=(IndexRecord(chunk, ChunkEmbedding(chunk.chunk_id, (1.0, 0.0, 0.0))),),
    )
    index_name = f"hybrid-rag-chunks-v2-dense-{uuid4().hex[:12]}"
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        "unused-read-alias",
        "unused-write-alias",
        3,
    )
    try:
        await manager.create(index_name)
        await OpenSearchDocumentIndex(settings.opensearch_url, index_name, 3).replace_document(
            request
        )
        retriever = OpenSearchDenseRetriever(
            settings.opensearch_url,
            index_name,
            ParaphraseFixtureProvider(),
            DenseRetrievalConfig("semantic-fixture-v1", 3, "chunks-v2", 20),
        )
        response = await retriever.search(
            "  How long does a password recovery link last?  ",
            tenant_id=tenant_id,
            collection_id=collection_id,
            debug=True,
        )
        assert [result.chunk_id for result in response.results] == [chunk.chunk_id]
        assert response.trace.normalized_query == "How long does a password recovery link last?"
        assert response.trace.embedding_adapter == "fake"
        assert response.trace.embedding_model == "semantic-fixture-v1"
        assert response.trace.embedding_dimensions == 3
        assert response.trace.index_schema_version == "chunks-v2"
        assert response.trace.candidate_count == 1
        denied = await retriever.search(
            "How long does a password recovery link last?", tenant_id=uuid4()
        )
        assert denied.results == ()
    finally:
        async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=10) as client:
            response = await client.delete(f"/{index_name}")
            if response.status_code != 404:
                response.raise_for_status()
