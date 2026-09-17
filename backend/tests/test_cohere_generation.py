import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import cohere
import httpx
import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.cohere_generation import (
    CohereGenerationProvider,
    configured_cohere_generation,
    open_cohere_generation,
)
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.generation import (
    GenerationFinishReason,
    GenerationMessage,
    GenerationRequest,
    MessageRole,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:The `__fields__` attribute is deprecated:DeprecationWarning:cohere.*"
)

REQUEST = GenerationRequest(
    (
        GenerationMessage(MessageRole.SYSTEM, "Be concise."),
        GenerationMessage(MessageRole.USER, "Hello"),
        GenerationMessage(MessageRole.ASSISTANT, "Hi"),
        GenerationMessage(MessageRole.USER, "Answer this"),
    ),
    100,
)


def response(
    *, finish_reason: str = "COMPLETE", content: object = None, usage: object = None
) -> dict[str, object]:
    if content is None:
        content = [{"type": "text", "text": "Grounded answer."}]
    result: dict[str, object] = {
        "id": "fixture-id",
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content},
    }
    if usage is not None:
        result["usage"] = usage
    return result


@pytest.mark.anyio
async def test_sdk_request_response_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([10.0, 10.2])
    monkeypatch.setattr(
        "hybrid_rag_search.providers.cohere_generation.perf_counter", lambda: next(clock)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/chat"
        assert request.headers["authorization"] == "Bearer test-only"
        assert json.loads(request.content) == {
            "model": "command-r7b-12-2024",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Answer this"},
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "stream": False,
        }
        return httpx.Response(
            200,
            json=response(
                content=[
                    {"type": "text", "text": "Grounded "},
                    {"type": "text", "text": "answer."},
                ],
                usage={
                    "tokens": {"input_tokens": 120, "output_tokens": 30},
                    "billed_units": {"input_tokens": 100, "output_tokens": 20},
                },
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = CohereGenerationProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http),
            model="command-r7b-12-2024",
        )
        result = await provider.generate(REQUEST)
    assert result.text == "Grounded answer."
    assert result.finish_reason == GenerationFinishReason.COMPLETE
    assert result.model == "command-r7b-12-2024" and result.adapter == "cohere"
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 30
    assert result.usage.billed_input_tokens == 100
    assert result.usage.billed_output_tokens == 20
    assert result.usage.latency_ms == pytest.approx(200)
    assert result.usage.estimated_cost_usd == Decimal("0.00000675")
    assert result.usage.input_usd_per_million_tokens == Decimal("0.0375")
    assert result.usage.output_usd_per_million_tokens == Decimal("0.15")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider_reason,expected",
    [
        ("COMPLETE", GenerationFinishReason.COMPLETE),
        ("STOP_SEQUENCE", GenerationFinishReason.COMPLETE),
        ("MAX_TOKENS", GenerationFinishReason.MAX_TOKENS),
    ],
)
async def test_finish_reason_mapping(
    provider_reason: str, expected: GenerationFinishReason
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response(finish_reason=provider_reason))
        )
    ) as http:
        provider = CohereGenerationProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        result = await provider.generate(REQUEST)
    assert result.finish_reason == expected
    assert result.usage.input_tokens is None
    assert result.usage.billed_input_tokens is None
    assert result.usage.estimated_cost_usd is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider_reason,code,retryable",
    [
        ("TIMEOUT", "timeout", True),
        ("ERROR", "unavailable", True),
        ("TOOL_CALL", "invalid_response", False),
    ],
)
async def test_non_text_finish_reasons(provider_reason: str, code: str, retryable: bool) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response(finish_reason=provider_reason))
        )
    ) as http:
        provider = CohereGenerationProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError) as caught:
            await provider.generate(REQUEST)
    assert caught.value.code == code and caught.value.retryable == retryable


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
        provider = CohereGenerationProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError) as caught:
            await provider.generate(REQUEST)
    assert caught.value.code == code and caught.value.retryable == retryable
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        response(content=[]),
        response(content=[{"type": "text", "text": " "}]),
        response(content=[{"type": "thinking", "thinking": "private"}]),
        response(finish_reason="UNKNOWN"),
        response(usage={"tokens": {"input_tokens": -1}}),
        response(usage={"billed_units": {"output_tokens": "bad"}}),
    ],
)
async def test_malformed_response(body: dict[str, object]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
    ) as http:
        provider = CohereGenerationProvider(
            cohere.AsyncClientV2(api_key="test-only", httpx_client=http), model="test"
        )
        with pytest.raises(ProviderError, match="invalid_response"):
            await provider.generate(REQUEST)


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
    client.chat.side_effect = error
    provider = CohereGenerationProvider(client, model="test")
    with pytest.raises(ProviderError) as caught:
        await provider.generate(REQUEST)
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
    client.chat.side_effect = wait_forever
    provider = CohereGenerationProvider(client, model="test", timeout_seconds=0.01)
    with pytest.raises(ProviderError, match="timeout"):
        await provider.generate(REQUEST)
    assert cancelled.is_set()
    entered.clear()
    provider.timeout_seconds = 10
    task = asyncio.create_task(provider.generate(REQUEST))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "model,timeout,temperature,input_rate,output_rate",
    [
        (" ", 30, 0, Decimal("0.1"), Decimal("0.2")),
        ("test", 0, 0, Decimal("0.1"), Decimal("0.2")),
        ("test", 30, float("inf"), Decimal("0.1"), Decimal("0.2")),
        ("test", 30, -0.1, Decimal("0.1"), Decimal("0.2")),
        ("test", 30, 1.1, Decimal("0.1"), Decimal("0.2")),
        ("test", 30, 0, Decimal("-1"), Decimal("0.2")),
        ("test", 30, 0, Decimal("0.1"), Decimal("NaN")),
    ],
)
def test_configuration_validation(
    model: str,
    timeout: float,
    temperature: float,
    input_rate: Decimal,
    output_rate: Decimal,
) -> None:
    with pytest.raises(ValueError):
        CohereGenerationProvider(
            AsyncMock(),
            model=model,
            timeout_seconds=timeout,
            temperature=temperature,
            input_usd_per_million_tokens=input_rate,
            output_usd_per_million_tokens=output_rate,
        )


@pytest.mark.anyio
async def test_factories_close_sdk_and_use_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncMock(spec=cohere.AsyncClientV2)
    client.__aenter__.return_value = client
    monkeypatch.setattr(cohere, "AsyncClientV2", lambda **_kwargs: client)
    async with open_cohere_generation(api_key="test-only", model="test") as provider:
        assert provider.client is client
    client.__aexit__.assert_awaited_once()
    with pytest.raises(ValueError, match="API key"):
        async with open_cohere_generation(api_key="", model="test"):
            pytest.fail("Blank key must fail")

    monkeypatch.setenv("COHERE_API_KEY", "test-secret-only")
    monkeypatch.setenv("COHERE_GENERATION_MODEL", "command-r7b-12-2024")
    settings = Settings()
    async with configured_cohere_generation(settings) as provider:
        assert provider.model == "command-r7b-12-2024"
        assert provider.input_rate == Decimal("0.0375")
        assert provider.output_rate == Decimal("0.15")
    settings.cohere_api_key = None
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        async with configured_cohere_generation(settings):
            pytest.fail("Missing credentials must fail")
