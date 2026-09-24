import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import cohere
import httpx
import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.cohere_reranking import (
    CohereRerankingProvider,
    configured_cohere_reranking,
    open_cohere_reranking,
)
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.reranking import RerankCandidate, RerankRequest

REQUEST = RerankRequest(
    "remote policy",
    (RerankCandidate("chunk-a", "cafeteria"), RerankCandidate("chunk-b", "remote work")),
    1,
)


@pytest.mark.anyio
@pytest.mark.parametrize("price,expected", [(None, None), (Decimal("0.002"), Decimal("0.003"))])
async def test_sdk_request_identity_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch, price: Decimal | None, expected: Decimal | None
) -> None:
    clock = iter([10.0, 10.125])
    monkeypatch.setattr(
        "hybrid_rag_search.providers.cohere_reranking.perf_counter", lambda: next(clock)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/rerank"
        assert request.headers["authorization"] == "Bearer test-only"
        assert json.loads(request.content) == {
            "model": "rerank-v4.0-fast",
            "query": "remote policy",
            "documents": ["cafeteria", "remote work"],
            "top_n": 1,
        }
        return httpx.Response(
            200,
            json={
                "id": "fixture-id",
                "results": [{"index": 1, "relevance_score": 0.9}],
                "meta": {"billed_units": {"search_units": 1.5}},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = cohere.AsyncClientV2(api_key="test-only", httpx_client=http)
        result = await CohereRerankingProvider(
            client,
            model="rerank-v4.0-fast",
            usd_per_search_unit=price,
        ).rerank(REQUEST)
    assert [(item.candidate_id, item.original_index, item.rank) for item in result.items] == [
        ("chunk-b", 1, 1)
    ]
    assert result.items[0].relevance_score == 0.9
    assert result.model == "rerank-v4.0-fast" and result.adapter == "cohere"
    assert result.usage.billed_search_units == 1.5
    assert result.usage.latency_ms == 125
    assert result.usage.estimated_cost_usd == expected
    assert result.usage.usd_per_search_unit == price


@pytest.mark.anyio
async def test_caller_timeout_budget_caps_configured_timeout() -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    response = MagicMock()
    response.model_dump.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
    client.rerank.return_value = response
    provider = CohereRerankingProvider(client, model="test", timeout_seconds=5)
    await provider.rerank(REQUEST, timeout_seconds=0.25)
    assert client.rerank.await_args.kwargs["request_options"] == {
        "timeout_in_seconds": 1,
        "max_retries": 0,
    }
    for invalid in (0, float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            await provider.rerank(REQUEST, timeout_seconds=invalid)


@pytest.mark.anyio
async def test_missing_usage_stays_unknown() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"id": "fixture", "results": [{"index": 0, "relevance_score": 0.5}]}
            )
        )
    ) as http:
        provider = CohereRerankingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http),
            model="test",
        )
        result = await provider.rerank(REQUEST)
    assert result.usage.billed_search_units is None
    assert result.usage.estimated_cost_usd is None
    assert result.usage.latency_ms is not None and result.usage.latency_ms >= 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (400, "invalid_request", False),
        (401, "authentication", False),
        (403, "authentication", False),
        (429, "rate_limited", True),
        (498, "authentication", False),
        (500, "unavailable", True),
        (503, "unavailable", True),
    ],
)
async def test_http_errors_are_sanitized_without_retries(
    status: int, code: str, retryable: bool
) -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"message": "secret provider detail"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = CohereRerankingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError) as caught:
            await provider.rerank(REQUEST)
    assert caught.value.code == code and caught.value.retryable == retryable
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"results": []},
        {"results": [{"index": 0, "relevance_score": "bad"}]},
        {"results": [{"index": -1, "relevance_score": 0.5}]},
        {"results": [{"index": 2, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": 1.5}]},
        {
            "results": [{"index": 0, "relevance_score": 0.5}],
            "meta": {"billed_units": {"search_units": -1}},
        },
    ],
)
async def test_malformed_response(body: dict[str, object]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"id": "fixture", **body})
        )
    ) as http:
        provider = CohereRerankingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError, match="invalid_response"):
            await provider.rerank(REQUEST)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "results",
    [
        [
            {"index": 0, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.7},
        ],
        [
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.8},
        ],
    ],
)
async def test_duplicate_or_unsorted_results(results: list[dict[str, object]]) -> None:
    request = RerankRequest("query", REQUEST.candidates, 2)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"id": "fixture", "results": results})
        )
    ) as http:
        provider = CohereRerankingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError, match="invalid_response"):
            await provider.rerank(request)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error,code",
    [
        (httpx.ReadTimeout("private"), "timeout"),
        (httpx.ConnectError("private"), "unavailable"),
        (ValueError("invalid JSON"), "invalid_response"),
    ],
)
async def test_transport_and_parse_errors(error: Exception, code: str) -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.rerank.side_effect = error
    provider = CohereRerankingProvider(client, model="test")
    with pytest.raises(ProviderError) as caught:
        await provider.rerank(REQUEST)
    assert caught.value.code == code


@pytest.mark.anyio
async def test_total_timeout_and_caller_cancellation() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever(**_kwargs: object) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.rerank.side_effect = wait_forever
    provider = CohereRerankingProvider(client, model="test", timeout_seconds=0.01)
    with pytest.raises(ProviderError, match="timeout"):
        await provider.rerank(REQUEST)
    assert cancelled.is_set()
    entered.clear()
    provider.timeout_seconds = 10
    task = asyncio.create_task(provider.rerank(REQUEST))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_oversized_candidate_set_never_calls_sdk() -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    candidates = tuple(RerankCandidate(f"chunk-{index}", "text") for index in range(10_001))
    provider = CohereRerankingProvider(client, model="test")
    with pytest.raises(ProviderError, match="invalid_request"):
        await provider.rerank(RerankRequest("query", candidates, 1))
    client.rerank.assert_not_called()


@pytest.mark.parametrize(
    "model,timeout,price",
    [
        (" ", 2, None),
        ("test", 0, None),
        ("test", float("inf"), None),
        ("test", 2, Decimal("-1")),
        ("test", 2, Decimal("NaN")),
    ],
)
def test_configuration_validation(model: str, timeout: float, price: Decimal | None) -> None:
    with pytest.raises(ValueError):
        CohereRerankingProvider(
            AsyncMock(), model=model, timeout_seconds=timeout, usd_per_search_unit=price
        )


@pytest.mark.anyio
async def test_factories_close_sdk_and_use_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.__aenter__.return_value = client
    monkeypatch.setattr(cohere, "AsyncClientV2", lambda **_kwargs: client)
    async with open_cohere_reranking(api_key="test-only", model="test") as provider:
        assert provider.client is client
    client.__aexit__.assert_awaited_once()
    with pytest.raises(ValueError, match="API key"):
        async with open_cohere_reranking(api_key="", model="test"):
            pytest.fail("Blank key must fail")

    monkeypatch.setenv("COHERE_API_KEY", "test-secret-only")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
    settings = Settings()
    async with configured_cohere_reranking(settings) as provider:
        assert provider.model == "rerank-v4.0-fast"
        assert provider.timeout_seconds == 5
    settings.cohere_api_key = None
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        async with configured_cohere_reranking(settings):
            pytest.fail("Missing credentials must fail")
