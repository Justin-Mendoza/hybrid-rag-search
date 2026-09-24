import asyncio
import sys
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from hybrid_rag_search import hybrid_retrieval
from hybrid_rag_search.chunking.contracts import SourceSpan
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import DenseRetrievalResponse
from hybrid_rag_search.hybrid_retrieval import HybridRetriever, RRFConfig
from hybrid_rag_search.lexical_retrieval import (
    LexicalRetrievalResponse,
    RetrievalError,
    RetrievalResult,
)
from hybrid_rag_search.parsers.contracts import SourceLocator


def result(chunk_id: str, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
        collection_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_id=UUID("33333333-3333-3333-3333-333333333333"),
        document_content_id="doc_" + "a" * 64,
        rank=rank,
        score=score,
        content_text=f"content {chunk_id}",
        source_spans=(SourceSpan(SourceLocator(rank), 0, 4),),
        generation_id="idxgen_" + "b" * 64,
        pipeline_version="pipe_" + "c" * 64,
        source_metadata={},
    )


class FakeRetriever:
    def __init__(self, response: object, entered: asyncio.Event, peer: asyncio.Event) -> None:
        self.response = response
        self.entered = entered
        self.peer = peer
        self.kwargs: dict[str, object] = {}

    async def search(self, query_text: str, **kwargs: object) -> object:
        self.kwargs = kwargs
        self.entered.set()
        await asyncio.wait_for(self.peer.wait(), timeout=1)
        return self.response


@pytest.mark.anyio
async def test_hybrid_runs_concurrently_and_fuses_one_or_both_paths() -> None:
    lexical_entered, dense_entered = asyncio.Event(), asyncio.Event()
    shared = result("shared", 1, 8.2)
    lexical_only = result("lexical", 2, 6.1)
    dense_shared = replace(shared, rank=2, score=0.91)
    dense_only = result("dense", 1, 0.95)
    lexical = FakeRetriever(
        LexicalRetrievalResponse((shared, lexical_only), "lexical-trace"),
        lexical_entered,
        dense_entered,
    )
    dense = FakeRetriever(
        DenseRetrievalResponse((dense_only, dense_shared), "dense-trace"),
        dense_entered,
        lexical_entered,
    )
    tenant_id = uuid4()

    response = await HybridRetriever(lexical, dense).search(
        "password policy",
        tenant_id=tenant_id,
        source_type=" Email ",
        author=" Ada  Lovelace ",
    )

    assert [item.evidence.chunk_id for item in response.results] == [
        "shared",
        "dense",
        "lexical",
    ]
    shared_result = response.results[0]
    assert (shared_result.lexical_rank, shared_result.vector_rank, shared_result.fused_rank) == (
        1,
        2,
        1,
    )
    assert (shared_result.lexical_score, shared_result.vector_score) == (8.2, 0.91)
    assert response.trace.fusion_version == "rrf-v1"
    assert response.trace.filters["source_type"] == "email"
    assert lexical.kwargs == dense.kwargs


@pytest.mark.anyio
async def test_rrf_ties_break_by_chunk_id_and_limit_is_applied_after_fusion() -> None:
    first, second = asyncio.Event(), asyncio.Event()
    lexical = FakeRetriever(LexicalRetrievalResponse((result("z", 1, 5.0),), None), first, second)
    dense = FakeRetriever(DenseRetrievalResponse((result("a", 1, 0.9),), None), second, first)

    response = await HybridRetriever(lexical, dense).search("query", tenant_id=uuid4(), limit=1)

    assert [item.evidence.chunk_id for item in response.results] == ["a"]


@pytest.mark.anyio
async def test_hybrid_propagates_path_failure() -> None:
    class Failure:
        async def search(self, query_text: str, **kwargs: object) -> object:
            raise RetrievalError("opensearch_unavailable", retryable=True)

    class Waiting:
        async def search(self, query_text: str, **kwargs: object) -> object:
            await asyncio.sleep(1)

    with pytest.raises(RetrievalError, match="opensearch_unavailable"):
        await HybridRetriever(Failure(), Waiting()).search("query", tenant_id=uuid4())


def test_rrf_config_and_limit_are_validated() -> None:
    with pytest.raises(ValueError, match="version"):
        RRFConfig(version="rrf-v2")
    with pytest.raises(ValueError, match="rank constant"):
        RRFConfig(rank_constant=0)

    first, second = asyncio.Event(), asyncio.Event()
    valid = FakeRetriever(LexicalRetrievalResponse((), None), first, second)
    with pytest.raises(ValueError, match="implement search"):
        HybridRetriever(object(), valid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration"):
        HybridRetriever(valid, valid, config="wrong")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_hybrid_validates_request_and_candidate_identity() -> None:
    first, second = asyncio.Event(), asyncio.Event()
    lexical = FakeRetriever(LexicalRetrievalResponse((), None), first, second)
    dense = FakeRetriever(DenseRetrievalResponse((), None), second, first)
    retriever = HybridRetriever(lexical, dense)
    with pytest.raises(ValueError, match="query text"):
        await retriever.search(7, tenant_id=uuid4())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="debug"):
        await retriever.search("query", tenant_id=uuid4(), debug=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="result limit"):
        await retriever.search("query", tenant_id=uuid4(), limit=0)

    shared = result("shared", 1, 1.0)
    with pytest.raises(RetrievalError, match="hybrid_candidate_mismatch"):
        retriever._fuse((shared,), (replace(shared, content_text="different"),), 20)
    with pytest.raises(RetrievalError, match="hybrid_candidate_mismatch"):
        hybrid_retrieval._evidence({})


@pytest.mark.anyio
async def test_debug_query(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()

    class ProviderContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    class DebugRetriever:
        async def search(self, *args: object, **kwargs: object) -> object:
            return expected

    monkeypatch.setattr(
        hybrid_retrieval, "configured_cohere_embeddings", lambda _: ProviderContext()
    )
    monkeypatch.setattr(
        hybrid_retrieval.OpenSearchBM25Retriever,
        "from_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        hybrid_retrieval.OpenSearchDenseRetriever,
        "from_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        hybrid_retrieval,
        "HybridRetriever",
        lambda *_args, **_kwargs: DebugRetriever(),
    )
    args = type(
        "Args",
        (),
        {
            "query": "password",
            "tenant": uuid4(),
            "collection": None,
            "source_type": None,
            "author": None,
            "source_date_from": "2026-09-01T00:00:00Z",
            "source_date_to": "2026-10-01T00:00:00Z",
            "limit": None,
        },
    )()
    assert await hybrid_retrieval._debug_query(Settings(), args) is expected


def test_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    expected = object()

    async def fake_debug_query(*_args: object) -> object:
        return expected

    monkeypatch.setattr(hybrid_retrieval, "_debug_query", fake_debug_query)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid", "password", "--tenant", str(uuid4()), "--source-type", "email"],
    )
    hybrid_retrieval.main()
    assert repr(expected) in capsys.readouterr().out
