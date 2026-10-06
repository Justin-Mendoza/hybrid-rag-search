import json
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from test_lexical_retrieval import hit

from hybrid_rag_search import search as module
from hybrid_rag_search.authorization import AuthorizedScope
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import OpenSearchDenseRetriever
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever, RetrievalError
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider, FakeRerankingProvider
from hybrid_rag_search.retrieval_filters import RetrievalFilters
from hybrid_rag_search.search import AuthorizedSearch


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_collection_allowlist_filters_are_explicit_and_intersect_selection():
    tenant, a, b = uuid4(), uuid4(), uuid4()
    filters = RetrievalFilters(tenant, a, collection_ids=(b, a, a))
    assert filters.collection_ids == tuple(sorted((a, b)))
    assert {"term": {"collection_id": str(a)}} in filters.clauses()
    assert {"terms": {"collection_id": [str(v) for v in sorted((a, b))]}} in filters.clauses()
    assert filters.trace_values()["collection_ids"] == ",".join(map(str, sorted((a, b))))
    assert {"match_none": {}} in RetrievalFilters(tenant, collection_ids=()).clauses()
    for invalid in ([a], ("bad",)):
        with pytest.raises(ValueError, match="tuple of UUIDs"):
            RetrievalFilters(tenant, collection_ids=invalid)


@pytest.mark.anyio
async def test_all_modes_scope_candidates_and_diagnostics_before_reranking(monkeypatch):
    tenant, collection, document = uuid4(), uuid4(), uuid4()
    scope = AuthorizedScope(uuid4(), tenant, (collection,))
    settings = Settings(
        _env_file=None, cohere_embed_model=FakeEmbeddingProvider.MODEL, cohere_embed_dimensions=3
    )
    bodies = []
    corrupt = False

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        item = hit(tenant, uuid4() if corrupt else collection, document)
        return httpx.Response(200, json={"hits": {"total": {"value": 1}, "hits": [item]}})

    transport = httpx.MockTransport(respond)
    original_lexical = OpenSearchBM25Retriever.from_settings
    original_dense = OpenSearchDenseRetriever.from_settings
    monkeypatch.setattr(
        module.OpenSearchBM25Retriever,
        "from_settings",
        lambda s: original_lexical(s, transport=transport),
    )
    monkeypatch.setattr(
        module.OpenSearchDenseRetriever,
        "from_settings",
        lambda s, p: original_dense(s, p, transport=transport),
    )

    @asynccontextmanager
    async def embeddings(settings):
        yield FakeEmbeddingProvider(3)

    @asynccontextmanager
    async def reranking(settings):
        yield FakeRerankingProvider()

    monkeypatch.setattr(module, "configured_cohere_embeddings", embeddings)
    monkeypatch.setattr(module, "configured_cohere_reranking", reranking)
    async with AuthorizedSearch(settings) as search:
        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            result = await search.search(scope, "password reset", mode, debug=True)
            assert result["chunks"][0]["document_id"] == document
            assert result["trace"]
        assert (await search.search(scope, "password reset", "bm25"))["trace"] is None
        assert (await search.search(replace(scope, collection_ids=()), "query", "hybrid_rerank"))[
            "chunks"
        ] == []
        assert (
            await search.search(replace(scope, collection_ids=()), "query", "bm25", debug=True)
        )["trace"] == {"candidate_count": 0}
        with pytest.raises(ValueError, match="Unknown"):
            await search.search(scope, "query", "invalid")
        corrupt = True
        for mode in ("bm25", "dense", "hybrid_rerank"):
            with pytest.raises(RetrievalError, match="scope_violation"):
                await search.search(scope, "password reset", mode, debug=True)
    assert bodies
    for body in bodies:
        serialized = json.dumps(body)
        assert str(tenant) in serialized
        assert '"terms": {"collection_id": ["' + str(collection) + '"]}' in serialized
    lexical = original_lexical(settings, transport=transport)
    dense = original_dense(settings, FakeEmbeddingProvider(3), transport=transport)
    for retriever in (lexical, dense):
        assert not (await retriever.search("query", tenant_id=tenant, collection_ids=())).results


@pytest.mark.anyio
async def test_public_search_rejects_results_from_a_broken_retriever(monkeypatch):
    tenant, collection = uuid4(), uuid4()
    scope = AuthorizedScope(uuid4(), tenant, (collection,))
    retriever = OpenSearchBM25Retriever("http://unused", "unused")
    empty = await retriever.search("", tenant_id=tenant)
    denied = OpenSearchBM25Retriever._result_from_hit(hit(tenant, uuid4(), uuid4()), 1)

    async def broken_search(*args, **kwargs):
        return replace(empty, results=(denied,))

    monkeypatch.setattr(retriever, "search", broken_search)
    monkeypatch.setattr(module.OpenSearchBM25Retriever, "from_settings", lambda _: retriever)
    async with AuthorizedSearch(Settings(_env_file=None)) as search:
        with pytest.raises(RetrievalError, match="scope_violation"):
            await search.search(scope, "password reset", "bm25", debug=True)
