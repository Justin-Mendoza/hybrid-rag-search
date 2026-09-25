import asyncio
import sys
import time
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from hybrid_rag_search import reranked_retrieval
from hybrid_rag_search.chunking.contracts import SourceSpan
from hybrid_rag_search.config import Settings
from hybrid_rag_search.hybrid_retrieval import (
    HybridRetrievalResponse,
    HybridRetrievalResult,
)
from hybrid_rag_search.lexical_retrieval import RetrievalError, RetrievalResult
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.reranking import (
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankUsage,
)
from hybrid_rag_search.reranked_retrieval import (
    RerankedRetrievalConfig,
    RerankedRetriever,
)


def hybrid_item(chunk_id: str, fused_rank: int) -> HybridRetrievalResult:
    evidence = RetrievalResult(
        chunk_id=chunk_id,
        tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
        collection_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_id=uuid4(),
        document_content_id="doc_" + "a" * 64,
        rank=fused_rank,
        score=1.0,
        content_text=f"evidence for {chunk_id}",
        source_spans=(SourceSpan(SourceLocator(fused_rank), 0, 4),),
        generation_id="idxgen_" + "b" * 64,
        pipeline_version="pipe_" + "c" * 64,
        source_metadata={},
    )
    return HybridRetrievalResult(
        evidence=evidence,
        lexical_rank=fused_rank,
        lexical_score=1.0,
        vector_rank=None,
        vector_score=None,
        fused_rank=fused_rank,
        fused_score=1 / (60 + fused_rank),
    )


def hybrid_response(count: int = 3) -> HybridRetrievalResponse:
    return HybridRetrievalResponse(
        tuple(hybrid_item(f"chunk-{index}", index) for index in range(1, count + 1)),
        None,  # type: ignore[arg-type]
    )


class FakeHybrid:
    def __init__(self, response: HybridRetrievalResponse, action=None) -> None:
        self.response = response
        self.action = action
        self.kwargs: dict[str, object] = {}

    async def search(self, query_text: str, **kwargs: object) -> HybridRetrievalResponse:
        self.kwargs = kwargs
        if self.action is not None:
            value = self.action()
            if asyncio.iscoroutine(value):
                await value
        return self.response


class ScriptedProvider:
    def __init__(self, result: RerankResult | Exception) -> None:
        self.result = result
        self.request: RerankRequest | None = None
        self.timeout_seconds: float | None = None

    async def rerank(
        self, request: RerankRequest, *, timeout_seconds: float | None = None
    ) -> RerankResult:
        self.request = request
        self.timeout_seconds = timeout_seconds
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def provider_result(*ids: tuple[str, int, float]) -> RerankResult:
    return RerankResult(
        tuple(
            RerankItem(candidate_id, original_index, rank, score)
            for rank, (candidate_id, original_index, score) in enumerate(ids, start=1)
        ),
        "rerank-test-v1",
        "fake",
        RerankUsage(
            billed_search_units=1,
            latency_ms=25,
            estimated_cost_usd=Decimal("0.001"),
        ),
    )


@pytest.mark.anyio
async def test_reranks_candidates_and_preserves_rank_changes_and_usage() -> None:
    hybrid = FakeHybrid(hybrid_response())
    provider = ScriptedProvider(provider_result(("chunk-3", 2, 0.9), ("chunk-1", 0, 0.8)))
    retriever = RerankedRetriever(
        hybrid,
        provider,
        RerankedRetrievalConfig(candidate_limit=3, result_limit=2),
    )

    response = await retriever.search("policy", tenant_id=uuid4())

    assert [item.hybrid.evidence.chunk_id for item in response.results] == [
        "chunk-3",
        "chunk-1",
    ]
    assert [
        (item.final_rank, item.rank_change, item.relevance_score) for item in response.results
    ] == [
        (1, 2, 0.9),
        (2, -1, 0.8),
    ]
    assert provider.request is not None
    assert provider.request.top_n == 2
    assert [item.id for item in provider.request.candidates] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert provider.timeout_seconds is not None and 0 < provider.timeout_seconds <= 5
    assert hybrid.kwargs["limit"] == 3
    assert response.trace.ranking_mode == "cohere_rerank"
    assert response.trace.fallback_reason is None
    assert response.trace.model == "rerank-test-v1"
    assert response.trace.usage.billed_search_units == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error,reason",
    [
        (ProviderError("timeout", retryable=True), "timeout"),
        (ProviderError("rate_limited", retryable=True), "rate_limited"),
        (ProviderError("unavailable", retryable=True), "unavailable"),
    ],
)
async def test_retryable_provider_errors_return_marked_rrf_fallback(
    error: ProviderError, reason: str
) -> None:
    response = await RerankedRetriever(
        FakeHybrid(hybrid_response()),
        ScriptedProvider(error),
        RerankedRetrievalConfig(candidate_limit=3, result_limit=2),
    ).search("policy", tenant_id=uuid4())

    assert [item.hybrid.evidence.chunk_id for item in response.results] == [
        "chunk-1",
        "chunk-2",
    ]
    assert all(item.relevance_score is None and item.rank_change == 0 for item in response.results)
    assert response.trace.ranking_mode == "rrf_fallback"
    assert response.trace.fallback_reason == reason
    assert response.trace.model is None and response.trace.adapter is None


@pytest.mark.anyio
async def test_nonretryable_provider_error_remains_visible() -> None:
    retriever = RerankedRetriever(
        FakeHybrid(hybrid_response()), ScriptedProvider(ProviderError("authentication"))
    )
    with pytest.raises(ProviderError, match="authentication"):
        await retriever.search("policy", tenant_id=uuid4())


@pytest.mark.anyio
async def test_empty_candidates_skip_provider_and_return_explicit_trace() -> None:
    provider = ScriptedProvider(provider_result(("unused", 0, 1.0)))
    response = await RerankedRetriever(FakeHybrid(hybrid_response(0)), provider).search(
        "policy", tenant_id=uuid4()
    )
    assert response.results == ()
    assert response.trace.fallback_reason == "no_candidates"
    assert provider.request is None


class MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.anyio
async def test_exhausted_deadline_after_hybrid_skips_provider() -> None:
    clock = MutableClock()
    provider = ScriptedProvider(provider_result(("chunk-1", 0, 1.0)))

    def exhaust() -> None:
        clock.now = 8.0

    response = await RerankedRetriever(
        FakeHybrid(hybrid_response(), exhaust), provider, clock=clock
    ).search("policy", tenant_id=uuid4())
    assert response.trace.fallback_reason == "deadline_exhausted"
    assert provider.request is None


@pytest.mark.anyio
async def test_provider_timeout_and_deadline_timeout_have_distinct_reasons() -> None:
    class SlowProvider:
        async def rerank(self, request: RerankRequest, **kwargs: object) -> RerankResult:
            await asyncio.sleep(1)
            raise AssertionError("timeout should cancel the provider")

    provider_timeout = await RerankedRetriever(
        FakeHybrid(hybrid_response()),
        SlowProvider(),
        RerankedRetrievalConfig(provider_timeout_seconds=0.01, request_deadline_seconds=1),
    ).search("policy", tenant_id=uuid4())
    assert provider_timeout.trace.fallback_reason == "timeout"

    deadline_timeout = await RerankedRetriever(
        FakeHybrid(hybrid_response()),
        SlowProvider(),
        RerankedRetrievalConfig(provider_timeout_seconds=1, request_deadline_seconds=1),
    ).search("policy", tenant_id=uuid4(), deadline=time.monotonic() + 0.01)
    assert deadline_timeout.trace.fallback_reason == "deadline_exhausted"


@pytest.mark.anyio
async def test_hybrid_timeout_fails_when_no_candidates_exist_for_fallback() -> None:
    async def slow() -> None:
        await asyncio.sleep(1)

    retriever = RerankedRetriever(
        FakeHybrid(hybrid_response(), slow), ScriptedProvider(Exception())
    )
    with pytest.raises(RetrievalError, match="retrieval_deadline_exceeded"):
        await retriever.search("policy", tenant_id=uuid4(), deadline=time.monotonic() + 0.01)


@pytest.mark.anyio
async def test_caller_cancellation_propagates() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingProvider:
        async def rerank(self, request: RerankRequest, **kwargs: object) -> RerankResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError

    task = asyncio.create_task(
        RerankedRetriever(FakeHybrid(hybrid_response()), WaitingProvider()).search(
            "policy", tenant_id=uuid4()
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reranked",
    [
        provider_result(("missing", 0, 1.0), ("chunk-1", 0, 0.5)),
        provider_result(("chunk-1", 0, 1.0)),
        provider_result(("chunk-1", 1, 1.0), ("chunk-2", 1, 0.5)),
        RerankResult(
            (
                RerankItem("chunk-1", 0, 2, 1.0),
                RerankItem("chunk-2", 1, 1, 0.5),
            ),
            "test",
            "fake",
        ),
    ],
)
async def test_invalid_provider_mapping_fails_visibly(reranked: RerankResult) -> None:
    retriever = RerankedRetriever(
        FakeHybrid(hybrid_response()),
        ScriptedProvider(reranked),
        RerankedRetrievalConfig(candidate_limit=3, result_limit=2),
    )
    with pytest.raises(RetrievalError, match="rerank_candidate_mismatch"):
        await retriever.search("policy", tenant_id=uuid4())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_limit": 0},
        {"result_limit": 0},
        {"candidate_limit": 1, "result_limit": 2},
        {"provider_timeout_seconds": 0},
        {"provider_timeout_seconds": "slow"},
        {"provider_timeout_seconds": True},
        {"provider_timeout_seconds": float("inf")},
        {"request_deadline_seconds": 0},
        {"provider_timeout_seconds": 9, "request_deadline_seconds": 8},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RerankedRetrievalConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_constructor_and_deadline_validation() -> None:
    hybrid = FakeHybrid(hybrid_response())
    provider = ScriptedProvider(provider_result(("chunk-1", 0, 1.0)))
    with pytest.raises(ValueError, match="dependencies"):
        RerankedRetriever(object(), provider)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clock"):
        RerankedRetriever(hybrid, provider, clock=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration"):
        RerankedRetriever(hybrid, provider, config="bad")  # type: ignore[arg-type]
    retriever = RerankedRetriever(hybrid, provider)
    for deadline in (float("inf"), "later", True):
        with pytest.raises(ValueError, match="deadline"):
            await retriever.search("policy", tenant_id=uuid4(), deadline=deadline)  # type: ignore[arg-type]


def test_settings_mapping_and_cross_field_validation() -> None:
    config = RerankedRetrievalConfig.from_settings(Settings())
    assert config == RerankedRetrievalConfig(20, 10, 5, 8)
    with pytest.raises(ValueError, match="result limit"):
        Settings(rerank_candidate_limit=1, rerank_result_limit=2)
    with pytest.raises(ValueError, match="retrieval deadline"):
        Settings(cohere_rerank_timeout_seconds=9, retrieval_deadline_seconds=8)


def test_unknown_retryable_error_cannot_be_silently_classified() -> None:
    with pytest.raises(RetrievalError, match="rerank_fallback_classification_failed"):
        reranked_retrieval._fallback_reason(ProviderError("authentication", retryable=True))


@pytest.mark.anyio
async def test_debug_query(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()

    class Context:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    class DebugRetriever:
        async def search(self, *args: object, **kwargs: object) -> object:
            return expected

    monkeypatch.setattr(reranked_retrieval, "configured_cohere_embeddings", lambda _: Context())
    monkeypatch.setattr(reranked_retrieval, "configured_cohere_reranking", lambda _: Context())
    monkeypatch.setattr(
        reranked_retrieval.OpenSearchDenseRetriever,
        "from_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(reranked_retrieval, "HybridRetriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        reranked_retrieval, "RerankedRetriever", lambda *_args, **_kwargs: DebugRetriever()
    )
    args = type(
        "Args",
        (),
        {
            "query": "policy",
            "tenant": uuid4(),
            "collection": None,
            "source_type": None,
            "author": None,
            "source_date_from": "2026-09-01T00:00:00Z",
            "source_date_to": None,
        },
    )()
    assert await reranked_retrieval._debug_query(Settings(), args) is expected


def test_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    expected = object()

    async def fake_debug(*_args: object) -> object:
        return expected

    monkeypatch.setattr(reranked_retrieval, "_debug_query", fake_debug)
    monkeypatch.setattr(sys, "argv", ["rerank", "policy", "--tenant", str(uuid4())])
    reranked_retrieval.main()
    assert repr(expected) in capsys.readouterr().out
