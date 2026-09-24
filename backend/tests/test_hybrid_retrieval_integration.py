"""Optional real-OpenSearch checks for Day 12 filters and hybrid fusion."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest

from hybrid_rag_search.chunking.contracts import Chunk, SourceSpan
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import DenseRetrievalConfig, OpenSearchDenseRetriever
from hybrid_rag_search.hybrid_retrieval import HybridRetriever
from hybrid_rag_search.ingestion.artifact_payloads import ChunkEmbedding
from hybrid_rag_search.ingestion.indexing import IndexRecord, ReplaceDocumentRequest
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever
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


class HybridFixtureProvider:
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=((1.0, 0.0, 0.0),),
            model="semantic-fixture-v1",
            dimensions=3,
            purpose=EmbeddingPurpose.QUERY,
            adapter="fake",
        )


def indexed_document(
    tenant_id: UUID,
    collection_id: UUID,
    *,
    author: str,
    suffix: str,
) -> ReplaceDocumentRequest:
    document_id = uuid4()
    content_hash = suffix * 64
    content_id = DocumentContentIdentity(tenant_id, collection_id, content_hash).document_id
    content = "Password recovery links expire after fifteen minutes."
    chunk = Chunk(
        content_id,
        "chunkcfg_" + "b" * 64,
        1,
        content,
        content,
        8,
        (SourceSpan(SourceLocator(1), 0, len(content)),),
    )
    return ReplaceDocumentRequest(
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_id=document_id,
        content_hash=content_hash,
        pipeline_version="pipe_" + "c" * 64,
        embedding_model="semantic-fixture-v1",
        embedding_dimensions=3,
        embedding_adapter="fake",
        source_metadata={
            "source_type": "Email",
            "author": author,
            "source_date": "2026-09-15T12:00:00Z",
        },
        records=(IndexRecord(chunk, ChunkEmbedding(chunk.chunk_id, (1.0, 0.0, 0.0))),),
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_hybrid_paths_apply_identical_metadata_scope_in_real_opensearch() -> None:
    settings = Settings()
    tenant_id, collection_id = uuid4(), uuid4()
    allowed = indexed_document(tenant_id, collection_id, author="Ada Lovelace", suffix="a")
    denied = indexed_document(tenant_id, collection_id, author="Grace Hopper", suffix="d")
    index_name = f"hybrid-rag-chunks-v3-hybrid-{uuid4().hex[:12]}"
    manager = OpenSearchIndexManager(
        settings.opensearch_url,
        "unused-read-alias",
        "unused-write-alias",
        3,
    )
    try:
        await manager.create(index_name)
        index = OpenSearchDocumentIndex(settings.opensearch_url, index_name, 3)
        await index.replace_document(allowed)
        await index.replace_document(denied)
        retriever = HybridRetriever(
            OpenSearchBM25Retriever(settings.opensearch_url, index_name),
            OpenSearchDenseRetriever(
                settings.opensearch_url,
                index_name,
                HybridFixtureProvider(),
                DenseRetrievalConfig("semantic-fixture-v1", 3, "chunks-v3", 20),
            ),
        )

        response = await retriever.search(
            "password recovery",
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_type="email",
            author="ada lovelace",
            source_date_from=datetime(2026, 9, 1, tzinfo=UTC),
            source_date_to=datetime(2026, 10, 1, tzinfo=UTC),
        )

        assert [item.evidence.document_id for item in response.results] == [allowed.document_id]
        assert response.results[0].lexical_rank == 1
        assert response.results[0].vector_rank == 1
        assert response.trace.lexical.filters == response.trace.dense.filters
    finally:
        async with httpx.AsyncClient(base_url=settings.opensearch_url, timeout=10) as client:
            response = await client.delete(f"/{index_name}")
            if response.status_code != 404:
                response.raise_for_status()
