import json
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import DenseRetrievalConfig, OpenSearchDenseRetriever
from hybrid_rag_search.evaluation import api
from hybrid_rag_search.evaluation import search as module
from hybrid_rag_search.evaluation.corpus import evaluation_scope, selected_dataset
from hybrid_rag_search.evaluation.search import MODES, EvaluationSearch
from hybrid_rag_search.lexical_retrieval import BM25Config, OpenSearchBM25Retriever, RetrievalError
from hybrid_rag_search.main import create_app
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider, FakeRerankingProvider


@pytest.mark.anyio
async def test_four_modes_use_fake_adapters_and_scope_before_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None, cohere_embed_model=FakeEmbeddingProvider.MODEL, cohere_embed_dimensions=8
    )
    dataset = selected_dataset("synthetic-workspace")
    scope = evaluation_scope(dataset, settings)
    document = next(iter(scope.document_ids))
    requests: list[dict[str, Any]] = []
    corrupt = False

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        hits = []
        for number in (1, 2):
            hits.append(
                {
                    "_id": f"chunk-{number}",
                    "_score": 1.0,
                    "_source": {
                        "chunk_id": f"chunk-{number}",
                        "tenant_id": str(uuid4() if corrupt else scope.tenant_id),
                        "collection_id": str(scope.collection_id),
                        "document_id": document,
                        "document_content_id": "doc_" + "a" * 64,
                        "content_text": "expense policy",
                        "source_spans": [
                            {
                                "block_number": 1,
                                "page_number": None,
                                "heading_path": [],
                                "start_char": 0,
                                "end_char": 14,
                            }
                        ],
                        "generation_id": "idxgen_" + "b" * 64,
                        "pipeline_version": scope.pipeline_version,
                        "source_metadata": {},
                    },
                }
            )
        return httpx.Response(200, json={"hits": {"hits": hits, "total": {"value": 2}}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        module.OpenSearchBM25Retriever,
        "from_settings",
        lambda s: OpenSearchBM25Retriever(
            "http://search.test",
            s.opensearch_read_alias,
            BM25Config(candidate_limit=100),
            transport=transport,
        ),
    )
    monkeypatch.setattr(
        module.OpenSearchDenseRetriever,
        "from_settings",
        lambda s, p: OpenSearchDenseRetriever(
            "http://search.test",
            s.opensearch_read_alias,
            p,
            DenseRetrievalConfig(s.cohere_embed_model, 8, "chunks-v3", 100),
            transport=transport,
        ),
    )

    @asynccontextmanager
    async def embedding(_settings: Settings):
        yield FakeEmbeddingProvider()

    @asynccontextmanager
    async def reranking(_settings: Settings):
        yield FakeRerankingProvider()

    monkeypatch.setattr(module, "configured_cohere_embeddings", embedding)
    monkeypatch.setattr(module, "configured_cohere_reranking", reranking)
    async with EvaluationSearch(settings) as search:
        for mode in MODES:
            result = await search.search(
                dataset,
                "expense policy",
                mode,
                {"source_date_from": "2026-01-01T00:00:00Z", "author": "Finance"},
            )
            assert len(result["chunks"]) == 2
            assert result["documents"] == [
                {"document_id": scope.document_ids[document], "chunk_id": "chunk-1", "rank": 1}
            ]
        await search.search(dataset, "expense policy", "hybrid_rerank")
        with pytest.raises(ValueError, match="Unknown search"):
            await search.search(dataset, "query", "other")
        with pytest.raises(ValueError, match="Unsupported"):
            await search.search(dataset, "query", "bm25", {"tenant_id": str(uuid4())})
        corrupt = True
        with pytest.raises(ValueError, match="outside"):
            await search.search(dataset, "query", "bm25")
    assert requests
    for body in requests:
        serialized = json.dumps(body)
        assert str(scope.tenant_id) in serialized and str(scope.collection_id) in serialized


def test_debug_endpoint_validates_scope_and_all_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = selected_dataset("synthetic-workspace")
    monkeypatch.setattr(api, "selected_dataset", lambda _: dataset)
    settings = Settings(_env_file=None)
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    async def prepared(*args: object) -> int:
        return 8

    class Search:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def search(self, dataset, query, mode, **kwargs):
            return {"mode": mode, "chunks": [], "trace": {}}

    monkeypatch.setattr(api, "verify_prepared", prepared)
    monkeypatch.setattr(api, "EvaluationSearch", lambda _: Search())
    with TestClient(create_app()) as client:
        request = {"dataset": "synthetic-workspace", "query": "expense policy", "mode": "bm25"}
        for mode in MODES:
            response = client.post("/debug/search", json={**request, "mode": mode})
            assert response.status_code == 200 and response.json()["mode"] == mode
        assert (
            client.post(
                "/debug/search", json={**request, "tenant_id": str(UUID(int=1))}
            ).status_code
            == 422
        )
        assert (
            client.post("/debug/search", json={**request, "dataset": "../notes.txt"}).status_code
            == 422
        )
        settings = settings.model_copy(update={"app_env": "production"})
        assert client.post("/debug/search", json=request).status_code == 404
        settings = settings.model_copy(update={"app_env": "development"})

        async def fail(*args: object) -> int:
            raise ValueError("not prepared")

        monkeypatch.setattr(api, "verify_prepared", fail)
        assert client.post("/debug/search", json=request).status_code == 409
        for error in (
            RetrievalError("opensearch_unavailable", retryable=True),
            ProviderError("unavailable"),
        ):

            async def unavailable(*args: object, error=error) -> int:
                raise error

            monkeypatch.setattr(api, "verify_prepared", unavailable)
            assert client.post("/debug/search", json=request).status_code == 503
