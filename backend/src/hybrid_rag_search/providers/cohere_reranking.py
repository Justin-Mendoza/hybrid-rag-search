"""Translate our reranking contract to Cohere's async SDK."""

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
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.reranking import (
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankUsage,
)


class _RerankResponseItem(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)


class _BilledUnits(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    search_units: float | None = Field(default=None, ge=0)


class _Meta(BaseModel):
    billed_units: _BilledUnits | None = None


class _RerankResponse(BaseModel):
    results: list[_RerankResponseItem]
    meta: _Meta | None = None


def _estimate_cost(
    billed_search_units: float | None, usd_per_search_unit: Decimal | None
) -> Decimal | None:
    if billed_search_units is None or usd_per_search_unit is None:
        return None
    return Decimal(str(billed_search_units)) * usd_per_search_unit


class CohereRerankingProvider:
    """One SDK request per candidate set; the injected client owns its lifetime."""

    def __init__(
        self,
        client: cohere.AsyncClientV2,
        *,
        model: str,
        timeout_seconds: float = 2.0,
        usd_per_search_unit: Decimal | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Rerank model must be nonblank")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Rerank timeout must be positive and finite")
        if usd_per_search_unit is not None and (
            not usd_per_search_unit.is_finite() or usd_per_search_unit < 0
        ):
            raise ValueError("Rerank price must be nonnegative and finite")
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.usd_per_search_unit = usd_per_search_unit

    async def rerank(self, request: RerankRequest) -> RerankResult:
        if len(request.candidates) > 10_000:
            raise ProviderError("invalid_request")
        options: RequestOptions = {
            "timeout_in_seconds": math.ceil(self.timeout_seconds),
            "max_retries": 0,
        }
        started = perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.rerank(
                    model=self.model,
                    query=request.query,
                    documents=[candidate.text for candidate in request.candidates],
                    top_n=request.top_n,
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
            parsed = _RerankResponse.model_validate(
                response.model_dump(by_alias=True, warnings=False)
            )
        except ValidationError:
            raise ProviderError("invalid_response") from None

        indexes = [item.index for item in parsed.results]
        scores = [item.relevance_score for item in parsed.results]
        if (
            len(parsed.results) != request.top_n
            or len(set(indexes)) != len(indexes)
            or any(index >= len(request.candidates) for index in indexes)
            or scores != sorted(scores, reverse=True)
        ):
            raise ProviderError("invalid_response")
        billed = (
            parsed.meta.billed_units.search_units
            if parsed.meta and parsed.meta.billed_units
            else None
        )
        return RerankResult(
            items=tuple(
                RerankItem(
                    candidate_id=request.candidates[item.index].id,
                    original_index=item.index,
                    rank=rank,
                    relevance_score=item.relevance_score,
                )
                for rank, item in enumerate(parsed.results, start=1)
            ),
            model=self.model,
            adapter="cohere",
            usage=RerankUsage(
                billed_search_units=billed,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=_estimate_cost(billed, self.usd_per_search_unit),
                usd_per_search_unit=self.usd_per_search_unit,
            ),
        )


@asynccontextmanager
async def open_cohere_reranking(
    *,
    api_key: str,
    model: str,
    timeout_seconds: float = 2.0,
    usd_per_search_unit: Decimal | None = None,
) -> AsyncIterator[CohereRerankingProvider]:
    if not api_key.strip():
        raise ValueError("Cohere API key must be nonblank")
    async with cohere.AsyncClientV2(api_key=api_key, timeout=timeout_seconds) as client:
        yield CohereRerankingProvider(
            client,
            model=model,
            timeout_seconds=timeout_seconds,
            usd_per_search_unit=usd_per_search_unit,
        )


@asynccontextmanager
async def configured_cohere_reranking(settings: Settings) -> AsyncIterator[CohereRerankingProvider]:
    """Opt-in live reranker; ordinary tests and load runs use the fake."""
    if settings.cohere_api_key is None:
        raise ValueError("COHERE_API_KEY is required for live reranking")
    async with open_cohere_reranking(
        api_key=settings.cohere_api_key.get_secret_value(),
        model=settings.cohere_rerank_model,
        timeout_seconds=settings.cohere_rerank_timeout_seconds,
        usd_per_search_unit=settings.cohere_rerank_usd_per_search_unit,
    ) as provider:
        yield provider
