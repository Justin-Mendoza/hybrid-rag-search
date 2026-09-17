"""Translate our text-generation contract to Cohere's async Chat SDK."""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from time import perf_counter
from typing import Literal

import cohere
import httpx
from cohere.core import ApiError
from cohere.core.request_options import RequestOptions
from cohere.types.chat_message_v2 import (
    AssistantChatMessageV2,
    ChatMessageV2,
    SystemChatMessageV2,
    UserChatMessageV2,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.generation import (
    GenerationFinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    MessageRole,
)


class _TextContent(BaseModel):
    model_config = ConfigDict(strict=True)
    type: Literal["text"]
    text: str


class _AssistantMessage(BaseModel):
    content: list[_TextContent]


class _TokenUsage(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    input_tokens: float | None = Field(default=None, ge=0)
    output_tokens: float | None = Field(default=None, ge=0)


class _Usage(BaseModel):
    tokens: _TokenUsage | None = None
    billed_units: _TokenUsage | None = None


class _ChatResponse(BaseModel):
    finish_reason: Literal[
        "COMPLETE", "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_CALL", "ERROR", "TIMEOUT"
    ]
    message: _AssistantMessage
    usage: _Usage | None = None


def _estimate_cost(
    billed_input_tokens: float | None,
    billed_output_tokens: float | None,
    input_rate: Decimal,
    output_rate: Decimal,
) -> Decimal | None:
    if billed_input_tokens is None or billed_output_tokens is None:
        return None
    million = Decimal(1_000_000)
    return (
        Decimal(str(billed_input_tokens)) * input_rate
        + Decimal(str(billed_output_tokens)) * output_rate
    ) / million


class CohereGenerationProvider:
    """Non-streaming Chat adapter; the injected SDK client owns its lifetime."""

    def __init__(
        self,
        client: cohere.AsyncClientV2,
        *,
        model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        input_usd_per_million_tokens: Decimal = Decimal("0.0375"),
        output_usd_per_million_tokens: Decimal = Decimal("0.15"),
    ) -> None:
        if not model.strip():
            raise ValueError("Generation model must be nonblank")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Generation timeout must be positive and finite")
        if not math.isfinite(temperature) or not 0 <= temperature <= 1:
            raise ValueError("Generation temperature must be between zero and one")
        for rate in (input_usd_per_million_tokens, output_usd_per_million_tokens):
            if not rate.is_finite() or rate < 0:
                raise ValueError("Generation prices must be nonnegative and finite")
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.input_rate = input_usd_per_million_tokens
        self.output_rate = output_usd_per_million_tokens

    @staticmethod
    def _messages(request: GenerationRequest) -> list[ChatMessageV2]:
        converted: list[ChatMessageV2] = []
        for message in request.messages:
            if message.role == MessageRole.SYSTEM:
                converted.append(SystemChatMessageV2(content=message.content))
            elif message.role == MessageRole.USER:
                converted.append(UserChatMessageV2(content=message.content))
            else:
                converted.append(AssistantChatMessageV2(content=message.content))
        return converted

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        options: RequestOptions = {
            "timeout_in_seconds": math.ceil(self.timeout_seconds),
            "max_retries": 0,
        }
        started = perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.chat(
                    model=self.model,
                    messages=self._messages(request),
                    max_tokens=request.max_output_tokens,
                    temperature=self.temperature,
                    request_options=options,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise ProviderError("timeout", retryable=True) from None
        except httpx.TransportError:
            raise ProviderError("unavailable", retryable=True) from None
        except ApiError as error:
            if error.status_code in (401, 403, 498):
                raise ProviderError("authentication") from None
            if error.status_code == 429:
                raise ProviderError("rate_limited", retryable=True) from None
            if error.status_code is not None and error.status_code >= 500:
                raise ProviderError("unavailable", retryable=True) from None
            raise ProviderError("invalid_request") from None
        except (ValueError, TypeError):
            raise ProviderError("invalid_response") from None

        try:
            parsed = _ChatResponse.model_validate(
                response.model_dump(by_alias=True, warnings=False)
            )
        except ValidationError:
            raise ProviderError("invalid_response") from None
        if parsed.finish_reason == "TIMEOUT":
            raise ProviderError("timeout", retryable=True)
        if parsed.finish_reason == "ERROR":
            raise ProviderError("unavailable", retryable=True)
        if parsed.finish_reason == "TOOL_CALL":
            raise ProviderError("invalid_response")

        text = "".join(item.text for item in parsed.message.content)
        if not text.strip():
            raise ProviderError("invalid_response")
        tokens = parsed.usage.tokens if parsed.usage else None
        billed = parsed.usage.billed_units if parsed.usage else None
        billed_input = billed.input_tokens if billed else None
        billed_output = billed.output_tokens if billed else None
        return GenerationResult(
            text=text,
            finish_reason=(
                GenerationFinishReason.MAX_TOKENS
                if parsed.finish_reason == "MAX_TOKENS"
                else GenerationFinishReason.COMPLETE
            ),
            model=self.model,
            adapter="cohere",
            usage=GenerationUsage(
                input_tokens=tokens.input_tokens if tokens else None,
                output_tokens=tokens.output_tokens if tokens else None,
                billed_input_tokens=billed_input,
                billed_output_tokens=billed_output,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=_estimate_cost(
                    billed_input, billed_output, self.input_rate, self.output_rate
                ),
                input_usd_per_million_tokens=self.input_rate,
                output_usd_per_million_tokens=self.output_rate,
            ),
        )


@asynccontextmanager
async def open_cohere_generation(
    *,
    api_key: str,
    model: str,
    timeout_seconds: float = 30.0,
    temperature: float = 0.0,
    input_usd_per_million_tokens: Decimal = Decimal("0.0375"),
    output_usd_per_million_tokens: Decimal = Decimal("0.15"),
) -> AsyncIterator[CohereGenerationProvider]:
    if not api_key.strip():
        raise ValueError("Cohere API key must be nonblank")
    async with cohere.AsyncClientV2(api_key=api_key, timeout=timeout_seconds) as client:
        yield CohereGenerationProvider(
            client,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            input_usd_per_million_tokens=input_usd_per_million_tokens,
            output_usd_per_million_tokens=output_usd_per_million_tokens,
        )


@asynccontextmanager
async def configured_cohere_generation(
    settings: Settings,
) -> AsyncIterator[CohereGenerationProvider]:
    """Opt-in live generator; ordinary tests and load runs use the fake."""
    if settings.cohere_api_key is None:
        raise ValueError("COHERE_API_KEY is required for live generation")
    async with open_cohere_generation(
        api_key=settings.cohere_api_key.get_secret_value(),
        model=settings.cohere_generation_model,
        timeout_seconds=settings.cohere_generation_timeout_seconds,
        temperature=settings.cohere_generation_temperature,
        input_usd_per_million_tokens=settings.cohere_generation_input_usd_per_million_tokens,
        output_usd_per_million_tokens=settings.cohere_generation_output_usd_per_million_tokens,
    ) as provider:
        yield provider
