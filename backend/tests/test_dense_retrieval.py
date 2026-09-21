import json
import sys
from uuid import UUID, uuid4

import httpx
import pytest

from hybrid_rag_search import dense_retrieval
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import (
    DenseRetrievalConfig,
    OpenSearchDenseRetriever,
    normalize_dense_query,
)
from hybrid_rag_search.lexical_retrieval import RetrievalError
from hybrid_rag_search.providers.embeddings import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedEmbeddingProvider:
    def __init__(self, result: EmbeddingResult) -> None:
        self.result = result
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.requests.append(request)
        return self.result


def embedding_result(
    *,
    vectors: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0),),
    model: str = "semantic-fixture-v1",
    dimensions: int = 3,
    purpose: EmbeddingPurpose = EmbeddingPurpose.QUERY,
) -> EmbeddingResult:
    return EmbeddingResult(
        vectors=vectors,
        model=model,
        dimensions=dimensions,
        purpose=purpose,
        adapter="fake",
    )


def hit(
    tenant_id: UUID,
    collection_id: UUID,
    document_id: UUID,
    *,
    score: float = 0.92,
    metadata: object = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "chunk_id": "chk_" + "a" * 64,
        "tenant_id": str(tenant_id),
        "collection_id": str(collection_id),
        "document_id": str(document_id),
        "document_content_id": "doc_" + "b" * 64,
        "content_text": "Password reset links expire after fifteen minutes.",
        "source_spans": [
            {
                "block_number": 2,
                "page_number": 3,
                "heading_path": ["Authentication", "Password reset"],
                "start_char": 0,
                "end_char": 49,
            }
        ],
        "generation_id": "idxgen_" + "c" * 64,
        "pipeline_version": "pipe_" + "d" * 64,
    }
    if metadata is not None:
        source["source_metadata"] = metadata
    return {"_id": "internal-hit-id", "_score": score, "_source": source}


def retriever(
    provider: ScriptedEmbeddingProvider,
    responder: httpx.MockTransport | None = None,
) -> OpenSearchDenseRetriever:
    return OpenSearchDenseRetriever(
        "http://search.test",
        "chunks-read",
        provider,
        DenseRetrievalConfig("semantic-fixture-v1", 3, "chunks-v2", 3),
        transport=responder,
    )


def test_normalization_and_configuration_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert normalize_dense_query("  Reset\n\tLink?  ") == "Reset Link?"
    assert normalize_dense_query("  HR-104  ") == "HR-104"
    with pytest.raises(ValueError, match="string"):
        normalize_dense_query(None)  # type: ignore[arg-type]

    for kwargs in (
        {"embedding_model": " "},
        {"embedding_dimensions": True},
        {"candidate_limit": 0},
        {"index_schema_version": "not-a-version"},
    ):
        with pytest.raises(ValueError):
            DenseRetrievalConfig(**kwargs)  # type: ignore[arg-type]
    monkeypatch.setenv("OPENSEARCH_DENSE_CANDIDATE_LIMIT", "7")
    settings = Settings()
    config = DenseRetrievalConfig.from_settings(settings)
    assert config.candidate_limit == 7
    assert config.embedding_model == settings.cohere_embed_model
    with pytest.raises(ValueError, match="384"):
        Settings(cohere_embed_dimensions=385)


def test_retriever_constructor_and_factory_validate_boundaries() -> None:
    provider = ScriptedEmbeddingProvider(embedding_result())
    for args in (("", "read", provider), ("http://search.test", "", provider)):
        with pytest.raises(ValueError):
            OpenSearchDenseRetriever(*args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="provider"):
        OpenSearchDenseRetriever("http://search.test", "read", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration"):
        OpenSearchDenseRetriever(
            "http://search.test",
            "read",
            provider,
            config="wrong",  # type: ignore[arg-type]
        )
    assert OpenSearchDenseRetriever.from_settings(Settings(), provider).read_alias == (
        "hybrid-rag-chunks-read"
    )


@pytest.mark.anyio
async def test_search_embeds_normalized_query_builds_scoped_knn_and_normalizes_response() -> None:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    requests: list[httpx.Request] = []

    async def responder(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "hits": {
                    "total": {"value": 7},
                    "hits": [
                        hit(tenant_id, collection_id, document_id, metadata={"author": "Ada"})
                    ],
                }
            },
        )

    provider = ScriptedEmbeddingProvider(embedding_result())
    response = await retriever(provider, httpx.MockTransport(responder)).search(
        "  How long\n does a reset link last?  ",
        tenant_id=tenant_id,
        collection_id=collection_id,
        limit=99,
        debug=True,
    )

    assert provider.requests == [
        EmbeddingRequest(("How long does a reset link last?",), EmbeddingPurpose.QUERY)
    ]
    assert len(response.results) == 1
    assert response.results[0].rank == 1
    assert response.results[0].score == 0.92
    assert response.results[0].tenant_id == tenant_id
    assert response.results[0].source_metadata == {"author": "Ada"}
    assert response.trace.normalized_query == "How long does a reset link last?"
    assert response.trace.embedding_model == "semantic-fixture-v1"
    assert response.trace.embedding_dimensions == 3
    assert response.trace.embedding_adapter == "fake"
    assert response.trace.index_schema_version == "chunks-v2"
    assert response.trace.read_alias == "chunks-read"
    assert response.trace.candidate_limit == 3
    assert response.trace.candidate_count == 7
    assert response.trace.embedding_elapsed_ms >= 0
    assert response.trace.opensearch_elapsed_ms >= 0
    assert response.trace.filters == {
        "tenant_id": str(tenant_id),
        "collection_id": str(collection_id),
        "visibility": "ready",
    }

    assert requests[0].url.path == "/chunks-read/_search"
    body = json.loads(requests[0].content)
    assert body["size"] == 3
    assert body["explain"] is True
    knn = body["query"]["knn"]["embedding"]
    assert knn["vector"] == [1.0, 0.0, 0.0]
    assert knn["k"] == 3
    assert {next(iter(item["term"])) for item in knn["filter"]["bool"]["filter"]} == {
        "tenant_id",
        "collection_id",
        "visibility",
    }


@pytest.mark.anyio
async def test_search_skips_blank_query_and_validates_request_arguments() -> None:
    tenant_id = uuid4()
    provider = ScriptedEmbeddingProvider(embedding_result())
    calls = 0

    async def responder(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"hits": {"total": 0, "hits": []}})

    value = retriever(provider, httpx.MockTransport(responder))
    empty = await value.search(" \n\t ", tenant_id=tenant_id)
    assert empty.results == ()
    assert empty.trace.embedding_adapter is None
    assert empty.trace.candidate_count == 0
    assert provider.requests == []
    assert calls == 0
    no_results = await value.search("password", tenant_id=tenant_id, limit=1)
    assert no_results.results == ()
    assert no_results.trace.filters == {"tenant_id": str(tenant_id), "visibility": "ready"}
    assert calls == 1
    with pytest.raises(ValueError, match="result limit"):
        await value.search("password", tenant_id=tenant_id, limit=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tenant"):
        await value.search("password", tenant_id="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection"):
        await value.search("password", tenant_id=tenant_id, collection_id="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="debug"):
        await value.search("password", tenant_id=tenant_id, debug=1)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_search_classifies_opensearch_and_payload_errors() -> None:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()

    async def disconnected(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(RetrievalError, match="opensearch_unavailable") as unavailable:
        await retriever(
            ScriptedEmbeddingProvider(embedding_result()), httpx.MockTransport(disconnected)
        ).search("password", tenant_id=tenant_id)
    assert unavailable.value.retryable

    for status, retryable in ((400, False), (503, True)):

        async def bad_status(_: httpx.Request, status: int = status) -> httpx.Response:
            return httpx.Response(status)

        with pytest.raises(RetrievalError, match="opensearch_unavailable") as error:
            await retriever(
                ScriptedEmbeddingProvider(embedding_result()), httpx.MockTransport(bad_status)
            ).search("password", tenant_id=tenant_id)
        assert error.value.retryable is retryable

    invalid_payloads: list[object] = [[], {}, {"hits": {}}, {"hits": {"hits": {}}}]
    for payload in invalid_payloads:

        async def bad_payload(_: httpx.Request, payload: object = payload) -> httpx.Response:
            return httpx.Response(200, json=payload)

        with pytest.raises(RetrievalError, match="opensearch_invalid_response") as error:
            await retriever(
                ScriptedEmbeddingProvider(embedding_result()), httpx.MockTransport(bad_payload)
            ).search("password", tenant_id=tenant_id)
        assert not error.value.retryable

    malformed = hit(tenant_id, collection_id, document_id, metadata=[])

    async def malformed_hit(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": {"total": 1, "hits": [malformed]}})

    with pytest.raises(RetrievalError, match="opensearch_invalid_response"):
        await retriever(
            ScriptedEmbeddingProvider(embedding_result()), httpx.MockTransport(malformed_hit)
        ).search("password", tenant_id=tenant_id)


@pytest.mark.anyio
async def test_query_embedding_response_must_match_configured_model_and_dimension() -> None:
    tenant_id = uuid4()
    invalid_results = (
        embedding_result(vectors=((1.0, 0.0, 0.0),), purpose=EmbeddingPurpose.DOCUMENT),
        embedding_result(vectors=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        embedding_result(model="wrong-model"),
        embedding_result(dimensions=2),
        embedding_result(vectors=((1.0, 0.0),)),
        embedding_result(vectors=((float("nan"), 0.0, 0.0),)),
    )
    for result in invalid_results:
        with pytest.raises(RetrievalError) as error:
            await retriever(ScriptedEmbeddingProvider(result)).search(
                "password", tenant_id=tenant_id
            )
        assert not error.value.retryable


@pytest.mark.anyio
async def test_debug_query(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    provider = ScriptedEmbeddingProvider(embedding_result())

    class ProviderContext:
        async def __aenter__(self) -> ScriptedEmbeddingProvider:
            return provider

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakeRetriever:
        async def search(self, *args: object, **kwargs: object) -> object:
            return expected

    monkeypatch.setattr(
        dense_retrieval, "configured_cohere_embeddings", lambda _: ProviderContext()
    )
    monkeypatch.setattr(
        dense_retrieval.OpenSearchDenseRetriever,
        "from_settings",
        lambda *_args, **_kwargs: FakeRetriever(),
    )
    result = await dense_retrieval._debug_query(
        Settings(),
        type(
            "Args", (), {"query": "password", "tenant": uuid4(), "collection": None, "limit": None}
        )(),
    )
    assert result is expected


def test_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    expected = object()

    async def fake_debug_query(*_args: object) -> object:
        return expected

    monkeypatch.setattr(dense_retrieval, "_debug_query", fake_debug_query)
    tenant_id = uuid4()
    monkeypatch.setattr(sys, "argv", ["dense", "password", "--tenant", str(tenant_id)])
    dense_retrieval.main()
    assert repr(expected) in capsys.readouterr().out
