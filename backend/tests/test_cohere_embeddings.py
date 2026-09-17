import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import cohere
import httpx
import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.cohere_embeddings import (
    CohereEmbeddingProvider,
    configured_cohere_embeddings,
    open_cohere_embeddings,
)
from hybrid_rag_search.providers.embeddings import EmbeddingPurpose, EmbeddingRequest
from hybrid_rag_search.providers.errors import ProviderError

REQUEST = EmbeddingRequest(("hello",), EmbeddingPurpose.DOCUMENT)


@pytest.mark.anyio
@pytest.mark.parametrize("price,expected", [(None, None), (Decimal("0.10"), Decimal("0.000002"))])
async def test_reported_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch, price: Decimal | None, expected: Decimal | None
) -> None:
    clock = iter([10.0, 10.125])
    monkeypatch.setattr(
        "hybrid_rag_search.providers.cohere_embeddings.perf_counter", lambda: next(clock)
    )

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "usage-fixture",
                "embeddings": {"float": [[1.0, 0.0]]},
                "meta": {"tokens": {"input_tokens": 25}, "billed_units": {"input_tokens": 20}},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = CohereEmbeddingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http),
            model="test",
            dimensions=2,
            usd_per_million_tokens=price,
        )
        result = await provider.embed(REQUEST)
    assert result.usage.input_tokens == 25
    assert result.usage.billed_input_tokens == 20
    assert result.usage.latency_ms == 125
    assert result.usage.estimated_cost_usd == expected
    assert result.usage.usd_per_million_tokens == price


@pytest.mark.parametrize("price", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_price(price: Decimal) -> None:
    with pytest.raises(ValueError, match="price"):
        CohereEmbeddingProvider(
            AsyncMock(), model="test", dimensions=2, usd_per_million_tokens=price
        )


@pytest.mark.anyio
@pytest.mark.parametrize("purpose", list(EmbeddingPurpose))
async def test_sdk_request_and_response(purpose: EmbeddingPurpose) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/embed"
        assert request.headers["authorization"] == "Bearer test-only"
        assert json.loads(request.content) == {
            "model": "test-model",
            "texts": ["hello", "world"],
            "input_type": "search_document"
            if purpose == EmbeddingPurpose.DOCUMENT
            else "search_query",
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        return httpx.Response(
            200, json={"id": "fixture-id", "embeddings": {"float": [[1.0, 0.0], [0.0, 1.0]]}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = cohere.AsyncClientV2(api_key="test-only", httpx_client=http)
        result = await CohereEmbeddingProvider(client, model="test-model", dimensions=2).embed(
            EmbeddingRequest(("hello", "world"), purpose)
        )
    assert result.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert result.adapter == "cohere" and result.model == "test-model"
    assert result.dimensions == 2 and result.purpose == purpose
    assert result.usage.input_tokens is None
    assert result.usage.billed_input_tokens is None
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
        provider = CohereEmbeddingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test", dimensions=2
        )
        with pytest.raises(ProviderError) as caught:
            await provider.embed(REQUEST)
    assert caught.value.code == code and caught.value.retryable == retryable
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"embeddings": {}},
        {"embeddings": {"float": None}},
        {"embeddings": {"float": [[1.0]]}},
        {"embeddings": {"float": []}},
        {"embeddings": {"float": [[1.0, "bad"]]}},
    ],
)
async def test_malformed_vectors(body: dict[str, object]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"id": "fixture", **body})
        )
    ) as http:
        provider = CohereEmbeddingProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test", dimensions=2
        )
        with pytest.raises(ProviderError, match="invalid_response"):
            await provider.embed(REQUEST)


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
    client.embed.side_effect = error
    provider = CohereEmbeddingProvider(client, model="test", dimensions=2)
    with pytest.raises(ProviderError) as caught:
        await provider.embed(REQUEST)
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
    client.embed.side_effect = wait_forever
    provider = CohereEmbeddingProvider(client, model="test", dimensions=2, timeout_seconds=0.01)
    with pytest.raises(ProviderError, match="timeout"):
        await provider.embed(REQUEST)
    assert cancelled.is_set()
    entered.clear()
    provider.timeout_seconds = 10
    task = asyncio.create_task(provider.embed(REQUEST))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_oversized_batch_never_calls_sdk() -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    provider = CohereEmbeddingProvider(client, model="test", dimensions=2)
    with pytest.raises(ProviderError, match="invalid_request"):
        await provider.embed(EmbeddingRequest(("hello",) * 97, EmbeddingPurpose.DOCUMENT))
    client.embed.assert_not_called()


@pytest.mark.parametrize(
    "model,dimensions,timeout",
    [
        (" ", 2, 10),
        ("test", 0, 10),
        ("test", True, 10),
        ("test", 2, 0),
        ("test", 2, float("inf")),
    ],
)
def test_configuration_validation(model: str, dimensions: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        CohereEmbeddingProvider(
            AsyncMock(), model=model, dimensions=dimensions, timeout_seconds=timeout
        )


@pytest.mark.anyio
async def test_live_factory_closes_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.__aenter__.return_value = client
    monkeypatch.setattr(cohere, "AsyncClientV2", lambda **_kwargs: client)
    async with open_cohere_embeddings(api_key="test-only", model="test", dimensions=2) as provider:
        assert provider.client is client
    client.__aexit__.assert_awaited_once()
    with pytest.raises(ValueError, match="API key"):
        async with open_cohere_embeddings(api_key="", model="test", dimensions=2):
            pytest.fail("Should reject blank key")


@pytest.mark.anyio
async def test_configured_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit environment values isolate this test from the developer's .env.
    monkeypatch.setenv("COHERE_API_KEY", "test-secret-only")
    monkeypatch.setenv("COHERE_EMBED_MODEL", "embed-english-light-v3.0")
    monkeypatch.setenv("COHERE_EMBED_DIMENSIONS", "384")
    monkeypatch.setenv("COHERE_TIMEOUT_SECONDS", "10")
    settings = Settings()
    assert "test-secret-only" not in repr(settings)
    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.__aenter__.return_value = client
    monkeypatch.setattr(cohere, "AsyncClientV2", lambda **_kwargs: client)
    async with configured_cohere_embeddings(settings) as provider:
        assert provider.model == "embed-english-light-v3.0"
        assert provider.dimensions == 384
        assert provider.timeout_seconds == 10
    client.__aexit__.assert_awaited_once()
    settings.cohere_embed_dimensions = 1024
    with pytest.raises(ValueError, match="384"):
        async with configured_cohere_embeddings(settings):
            pytest.fail("Wrong model dimension must fail")
    settings.cohere_api_key = None
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        async with configured_cohere_embeddings(settings):
            pytest.fail("Missing credentials must fail")


def test_trial_key_alias_and_primary_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_TRIAL_KEY", "trial-test-only")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    trial_key = Settings().cohere_api_key
    assert trial_key is not None and trial_key.get_secret_value() == "trial-test-only"
    monkeypatch.setenv("COHERE_API_KEY", "primary-test-only")
    primary_key = Settings().cohere_api_key
    assert primary_key is not None and primary_key.get_secret_value() == "primary-test-only"
