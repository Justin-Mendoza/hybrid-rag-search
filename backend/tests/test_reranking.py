from typing import Any

import pytest

from hybrid_rag_search.providers.fake import FakeRerankingProvider
from hybrid_rag_search.providers.reranking import (
    RerankCandidate,
    RerankingProvider,
    RerankItem,
    RerankRequest,
)


def candidate(candidate_id: str = "chunk-1", text: str = "vacation policy") -> RerankCandidate:
    return RerankCandidate(candidate_id, text)


@pytest.mark.parametrize(
    "candidate_id,text", [("", "text"), (" ", "text"), ("id", ""), ("id", "\n")]
)
def test_candidate_requires_nonblank_fields(candidate_id: str, text: str) -> None:
    with pytest.raises(ValueError):
        RerankCandidate(candidate_id, text)


@pytest.mark.parametrize(
    "query,candidates,top_n,message",
    [
        ("", (candidate(),), 1, "query"),
        ("query", (), 1, "candidates"),
        ("query", [candidate()], 1, "candidates"),
        ("query", (candidate(), candidate()), 1, "unique"),
        ("query", (candidate(),), 0, "top_n"),
        ("query", (candidate(),), 2, "top_n"),
        ("query", (candidate(),), True, "top_n"),
    ],
)
def test_request_validation(query: str, candidates: Any, top_n: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RerankRequest(query, candidates, top_n)


@pytest.mark.parametrize(
    "item",
    [
        ("", 0, 1, 0.0),
        ("id", -1, 1, 0.0),
        ("id", 0, 0, 0.0),
        ("id", 0, 1, float("nan")),
        ("id", 0, 1, float("inf")),
    ],
)
def test_result_item_validation(item: tuple[str, int, int, float]) -> None:
    with pytest.raises(ValueError):
        RerankItem(*item)


@pytest.mark.anyio
async def test_fake_reranks_and_preserves_source_identity() -> None:
    provider: RerankingProvider = FakeRerankingProvider()
    request = RerankRequest(
        query="remote work policy",
        candidates=(
            candidate("chunk-a", "Cafeteria opening hours"),
            candidate("chunk-b", "The remote work policy allows two days"),
            candidate("chunk-c", "Remote employees submit expenses"),
            candidate("chunk-d", "POLICY for remote work"),
        ),
        top_n=3,
    )
    result = await provider.rerank(request)
    assert [(item.candidate_id, item.original_index, item.rank) for item in result.items] == [
        ("chunk-b", 1, 1),
        ("chunk-d", 3, 2),
        ("chunk-c", 2, 3),
    ]
    assert [item.relevance_score for item in result.items] == [1.0, 1.0, 1 / 3]
    assert result.model == "fake-rerank-v1" and result.adapter == "fake"
    assert result.usage.billed_search_units == 0
    assert result.usage.estimated_cost_usd == 0
    assert result.usage.latency_ms is None


@pytest.mark.anyio
async def test_fake_is_repeatable_and_ties_keep_input_order() -> None:
    request = RerankRequest(
        "missing terms",
        (candidate("first", "alpha"), candidate("second", "beta")),
        2,
    )
    first = await FakeRerankingProvider().rerank(request)
    second = await FakeRerankingProvider().rerank(request)
    assert first == second
    assert [item.candidate_id for item in first.items] == ["first", "second"]
    assert all(item.relevance_score == 0 for item in first.items)


@pytest.mark.anyio
async def test_unicode_terms_are_case_insensitive() -> None:
    result = await FakeRerankingProvider().rerank(
        RerankRequest("CAFÉ benefits", (candidate("match", "Café BENEFITS"),), 1)
    )
    assert result.items[0].relevance_score == 1.0
