"""User-scoped search through the existing four retrieval modes."""

from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Any, Literal

from hybrid_rag_search.authorization import AuthorizedScope
from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import OpenSearchDenseRetriever
from hybrid_rag_search.hybrid_retrieval import HybridRetriever
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever, RetrievalError
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.providers.cohere_reranking import configured_cohere_reranking
from hybrid_rag_search.reranked_retrieval import RerankedRetrievalConfig, RerankedRetriever

SearchMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class AuthorizedSearch:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stack = AsyncExitStack()

    async def __aenter__(self) -> "AuthorizedSearch":
        await self.stack.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stack.__aexit__(*args)

    async def search(
        self, scope: AuthorizedScope, query: str, mode: SearchMode, *, debug: bool = False
    ) -> dict[str, Any]:
        if not scope.collection_ids:
            return {"mode": mode, "chunks": [], "trace": {"candidate_count": 0} if debug else None}
        options: dict[str, Any] = {
            "tenant_id": scope.tenant_id,
            "collection_ids": scope.collection_ids,
            "debug": debug,
        }
        lexical = OpenSearchBM25Retriever.from_settings(self.settings)
        if mode == "bm25":
            lexical_response = await lexical.search(query, **options)
            evidence = lexical_response.results
            trace = asdict(lexical_response.trace)
        else:
            provider = await self.stack.enter_async_context(
                configured_cohere_embeddings(self.settings)
            )
            dense = OpenSearchDenseRetriever.from_settings(self.settings, provider)
            if mode == "dense":
                dense_response = await dense.search(query, **options)
                evidence = dense_response.results
                trace = asdict(dense_response.trace)
            else:
                hybrid = HybridRetriever(lexical, dense)
                if mode == "hybrid":
                    hybrid_response = await hybrid.search(query, **options)
                    evidence = tuple(item.evidence for item in hybrid_response.results)
                    trace = asdict(hybrid_response.trace)
                elif mode == "hybrid_rerank":
                    reranker = await self.stack.enter_async_context(
                        configured_cohere_reranking(self.settings)
                    )
                    reranked = RerankedRetriever(
                        hybrid, reranker, RerankedRetrievalConfig.from_settings(self.settings)
                    )
                    reranked_response = await reranked.search(query, **options)
                    evidence = tuple(item.hybrid.evidence for item in reranked_response.results)
                    trace = asdict(reranked_response.trace)
                else:
                    raise ValueError("Unknown search mode")
        if any(
            item.tenant_id != scope.tenant_id or item.collection_id not in scope.collection_ids
            for item in evidence
        ):
            raise RetrievalError("retrieval_scope_violation", retryable=False)
        return {
            "mode": mode,
            "chunks": [asdict(item) for item in evidence],
            "trace": trace if debug else None,
        }
