import math
from typing import Any

import pytest

from hybrid_rag_search.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingRequest,
)
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider


@pytest.mark.parametrize("texts", [(), ("",), (" \n\t",), ("valid", ""), (42,), ["text"]])
def test_invalid_text_batches(texts: Any) -> None:
    with pytest.raises(ValueError, match="texts"):
        EmbeddingRequest(texts=texts, purpose=EmbeddingPurpose.DOCUMENT)


def test_invalid_purpose() -> None:
    invalid: Any = "unknown"
    with pytest.raises(ValueError, match="purpose"):
        EmbeddingRequest(texts=("text",), purpose=invalid)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5])
def test_invalid_dimensions(dimensions: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FakeEmbeddingProvider(dimensions=dimensions)


@pytest.mark.anyio
async def test_repeatable_ordered_vectors_independent_of_batch() -> None:
    provider: EmbeddingProvider = FakeEmbeddingProvider()
    request = EmbeddingRequest(("Hello", "世界", "Hello"), EmbeddingPurpose.DOCUMENT)
    result = await provider.embed(request)
    assert result == await FakeEmbeddingProvider().embed(request)
    assert result.adapter == "fake"
    assert result.usage.billed_input_tokens == 0
    assert result.usage.estimated_cost_usd == 0
    assert result.usage.input_tokens is None
    assert result.usage.latency_ms is None
    assert result.model == "fake-embedding-v1"
    assert result.purpose == EmbeddingPurpose.DOCUMENT
    assert result.dimensions == 8
    assert len(result.vectors) == 3
    assert result.vectors[0] == result.vectors[2]
    assert result.vectors[0] != result.vectors[1]
    single = await provider.embed(EmbeddingRequest(("世界",), EmbeddingPurpose.DOCUMENT))
    assert single.vectors[0] == result.vectors[1]
    for vector in result.vectors:
        assert len(vector) == 8
        assert all(math.isfinite(value) for value in vector)
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


@pytest.mark.anyio
async def test_purpose_and_exact_text_affect_vectors() -> None:
    provider = FakeEmbeddingProvider()
    document = await provider.embed(EmbeddingRequest(("Hello",), EmbeddingPurpose.DOCUMENT))
    query = await provider.embed(EmbeddingRequest(("Hello",), EmbeddingPurpose.QUERY))
    spaced = await provider.embed(EmbeddingRequest((" Hello ",), EmbeddingPurpose.DOCUMENT))
    assert query.purpose == EmbeddingPurpose.QUERY
    assert document.vectors != query.vectors
    assert document.vectors != spaced.vectors


@pytest.mark.anyio
@pytest.mark.parametrize("dimensions", [1, 3, 1024])
async def test_configurable_dimensions(dimensions: int) -> None:
    result = await FakeEmbeddingProvider(dimensions).embed(
        EmbeddingRequest(("text",), EmbeddingPurpose.QUERY)
    )
    assert result.dimensions == dimensions
    assert len(result.vectors[0]) == dimensions
    assert sum(value * value for value in result.vectors[0]) == pytest.approx(1.0)
