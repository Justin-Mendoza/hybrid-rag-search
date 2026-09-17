"""Small live-provider checks; excluded from every default test command."""

import os

import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.providers.cohere_generation import configured_cohere_generation
from hybrid_rag_search.providers.cohere_reranking import configured_cohere_reranking
from hybrid_rag_search.providers.embeddings import EmbeddingPurpose, EmbeddingRequest
from hybrid_rag_search.providers.generation import (
    GenerationMessage,
    GenerationRequest,
    MessageRole,
)
from hybrid_rag_search.providers.reranking import RerankCandidate, RerankRequest

pytestmark = [pytest.mark.anyio, pytest.mark.live]
pytestmark.append(
    pytest.mark.filterwarnings(
        "ignore:The `__fields__` attribute is deprecated:DeprecationWarning:cohere.*"
    )
)


def live_settings() -> Settings:
    if os.getenv("RUN_LIVE_COHERE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_COHERE_TESTS=1 to authorize live Cohere calls")
    settings = Settings()
    if settings.cohere_api_key is None:
        pytest.fail("COHERE_API_KEY or COHERE_TRIAL_KEY is required when live tests are authorized")
    if settings.cohere_api_key.get_secret_value().startswith("replace-with"):
        pytest.fail("Replace the placeholder COHERE_API_KEY before running live tests")
    return settings


async def test_live_embedding_smoke() -> None:
    settings = live_settings()
    async with configured_cohere_embeddings(settings) as provider:
        result = await provider.embed(
            EmbeddingRequest(("Where is the vacation policy?",), EmbeddingPurpose.QUERY)
        )
    assert result.adapter == "cohere"
    assert result.model == settings.cohere_embed_model
    assert result.dimensions == settings.cohere_embed_dimensions
    assert len(result.vectors) == 1
    assert result.usage.latency_ms is not None and result.usage.latency_ms > 0
    print(
        f"embed model={result.model} latency_ms={result.usage.latency_ms:.1f} "
        f"billed_input_tokens={result.usage.billed_input_tokens}"
    )


async def test_live_reranking_smoke() -> None:
    settings = live_settings()
    request = RerankRequest(
        "Where is the vacation policy?",
        (
            RerankCandidate("cafeteria", "The cafeteria opens at 8 AM."),
            RerankCandidate("vacation", "The vacation policy grants paid time off."),
        ),
        top_n=1,
    )
    async with configured_cohere_reranking(settings) as provider:
        result = await provider.rerank(request)
    assert result.adapter == "cohere"
    assert result.model == settings.cohere_rerank_model
    assert len(result.items) == 1
    assert result.items[0].candidate_id in {"cafeteria", "vacation"}
    assert result.usage.latency_ms is not None and result.usage.latency_ms > 0
    print(
        f"rerank model={result.model} latency_ms={result.usage.latency_ms:.1f} "
        f"billed_search_units={result.usage.billed_search_units}"
    )


async def test_live_generation_smoke() -> None:
    settings = live_settings()
    request = GenerationRequest(
        (
            GenerationMessage(
                MessageRole.SYSTEM,
                "Reply briefly. This checks transport, not answer quality.",
            ),
            GenerationMessage(MessageRole.USER, "Say that the provider connection works."),
        ),
        max_output_tokens=30,
    )
    async with configured_cohere_generation(settings) as provider:
        result = await provider.generate(request)
    assert result.adapter == "cohere"
    assert result.model == settings.cohere_generation_model
    assert result.text.strip()
    assert result.usage.latency_ms is not None and result.usage.latency_ms > 0
    print(
        f"generation model={result.model} latency_ms={result.usage.latency_ms:.1f} "
        f"billed_input_tokens={result.usage.billed_input_tokens} "
        f"billed_output_tokens={result.usage.billed_output_tokens} "
        f"estimated_cost_usd={result.usage.estimated_cost_usd}"
    )
