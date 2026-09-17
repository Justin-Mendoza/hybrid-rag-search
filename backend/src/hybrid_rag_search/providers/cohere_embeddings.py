"""Translate our embedding contract to Cohere's async SDK."""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from time import perf_counter

import cohere
import httpx
from cohere.core import ApiError
from cohere.core.request_options import RequestOptions
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.embeddings import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
)
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.usage import EmbeddingUsage, estimate_embedding_cost


class _FloatEmbeddings(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    values: list[list[float]] = Field(alias="float")


class _TokenUsage(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    input_tokens: float | None = Field(default=None, ge=0)


class _Meta(BaseModel):
    tokens: _TokenUsage | None = None
    billed_units: _TokenUsage | None = None


class _EmbedResponse(BaseModel):
    embeddings: _FloatEmbeddings
    meta: _Meta | None = None


class CohereEmbeddingProvider:
    """One SDK request per batch; the caller owns the injected client's lifetime.

    No automatic retries: workers/retrieval will decide whether another attempt
    fits their budgets. Model and dimension are required, not index defaults.
    """

    def __init__(
        self,
        client: cohere.AsyncClientV2,
        *,
        model: str,
        dimensions: int,
        timeout_seconds: float = 10.0,
        usd_per_million_tokens: Decimal | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Embedding model must be nonblank")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Embedding dimensions must be a positive integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Embedding timeout must be positive and finite")
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        if usd_per_million_tokens is not None and (
            not usd_per_million_tokens.is_finite() or usd_per_million_tokens < 0
        ):
            raise ValueError("Embedding price must be nonnegative and finite")
        self.usd_per_million_tokens = usd_per_million_tokens

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        if len(request.texts) > 96:
            raise ProviderError("invalid_request")
        options: RequestOptions = {
            "timeout_in_seconds": math.ceil(self.timeout_seconds),
            "max_retries": 0,
        }
        started = perf_counter()
        try:
            # Bound total elapsed call time as well as the SDK's I/O timeouts.
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.embed(
                    model=self.model,
                    texts=list(request.texts),
                    input_type=(
                        "search_document"
                        if request.purpose == EmbeddingPurpose.DOCUMENT
                        else "search_query"
                    ),
                    embedding_types=["float"],
                    truncate="NONE",
                    request_options=options,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise ProviderError("timeout", retryable=True) from None
        except httpx.TransportError:
            raise ProviderError("unavailable", retryable=True) from None
        except ApiError as error:
            if error.status_code in (401, 403):
                raise ProviderError("authentication") from None
            if error.status_code == 429:
                raise ProviderError("rate_limited", retryable=True) from None
            if error.status_code is not None and error.status_code >= 500:
                raise ProviderError("unavailable", retryable=True) from None
            raise ProviderError("invalid_request") from None
        except (ValueError, TypeError):
            raise ProviderError("invalid_response") from None

        try:
            # SDK responses are constructed without full validation; enforce our
            # boundary before malformed or nonfinite vectors reach the index.
            parsed = _EmbedResponse.model_validate(
                response.model_dump(by_alias=True, warnings=False)
            )
        except ValidationError:
            raise ProviderError("invalid_response") from None
        vectors = tuple(tuple(vector) for vector in parsed.embeddings.values)
        if len(vectors) != len(request.texts) or any(
            len(vector) != self.dimensions for vector in vectors
        ):
            raise ProviderError("invalid_response")
        meta = parsed.meta
        input_tokens = meta.tokens.input_tokens if meta and meta.tokens else None
        billed = meta.billed_units.input_tokens if meta and meta.billed_units else None
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=self.dimensions,
            purpose=request.purpose,
            adapter="cohere",
            usage=EmbeddingUsage(
                input_tokens=input_tokens,
                billed_input_tokens=billed,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=estimate_embedding_cost(billed, self.usd_per_million_tokens),
                usd_per_million_tokens=self.usd_per_million_tokens,
            ),
        )


@asynccontextmanager
async def open_cohere_embeddings(
    *,
    api_key: str,
    model: str,
    dimensions: int,
    timeout_seconds: float = 10.0,
    usd_per_million_tokens: Decimal | None = None,
) -> AsyncIterator[CohereEmbeddingProvider]:
    """Explicit live-client construction; close SDK resources on success or failure."""
    if not api_key.strip():
        raise ValueError("Cohere API key must be nonblank")
    async with cohere.AsyncClientV2(api_key=api_key, timeout=timeout_seconds) as client:
        yield CohereEmbeddingProvider(
            client,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            usd_per_million_tokens=usd_per_million_tokens,
        )


@asynccontextmanager
async def configured_cohere_embeddings(
    settings: Settings,
) -> AsyncIterator[CohereEmbeddingProvider]:
    """Opt-in live adapter using project settings; ordinary tests use the fake."""
    if settings.cohere_api_key is None:
        raise ValueError("COHERE_API_KEY is required for live embeddings")
    if (
        settings.cohere_embed_model == "embed-english-light-v3.0"
        and settings.cohere_embed_dimensions != 384
    ):
        raise ValueError("embed-english-light-v3.0 requires 384 dimensions")
    async with open_cohere_embeddings(
        api_key=settings.cohere_api_key.get_secret_value(),
        model=settings.cohere_embed_model,
        dimensions=settings.cohere_embed_dimensions,
        timeout_seconds=settings.cohere_timeout_seconds,
        usd_per_million_tokens=settings.cohere_embed_usd_per_million_tokens,
    ) as provider:
        yield provider
