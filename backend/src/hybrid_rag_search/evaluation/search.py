"""Isolated evaluation search boundary for the local benchmark runner."""

from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Any, Literal

from hybrid_rag_search.config import Settings
from hybrid_rag_search.dense_retrieval import OpenSearchDenseRetriever
from hybrid_rag_search.evaluation.corpus import evaluation_index_settings, evaluation_scope
from hybrid_rag_search.evaluation.datasets import EvaluationDataset
from hybrid_rag_search.hybrid_retrieval import HybridRetriever, RRFConfig
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever, RetrievalResult
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.providers.cohere_reranking import configured_cohere_reranking
from hybrid_rag_search.reranked_retrieval import RerankedRetrievalConfig, RerankedRetriever
from hybrid_rag_search.retrieval_filters import parse_source_date

SearchMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")
BUDGET = 100


def benchmark_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "opensearch_bm25_candidate_limit": BUDGET,
            "opensearch_dense_candidate_limit": BUDGET,
            "rerank_candidate_limit": BUDGET,
            "rerank_result_limit": BUDGET,
        }
    )


class EvaluationSearch:
    """Share provider resources across queries without caching query embeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = benchmark_settings(settings)
        self.stack = AsyncExitStack()
        self.lexical = OpenSearchBM25Retriever.from_settings(self.settings)
        self.dense: OpenSearchDenseRetriever | None = None
        self.hybrid: HybridRetriever | None = None
        self.reranked: RerankedRetriever | None = None

    async def __aenter__(self) -> "EvaluationSearch":
        await self.stack.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stack.__aexit__(*args)

    async def search(
        self,
        dataset: EvaluationDataset,
        query: str,
        mode: SearchMode,
        filters: dict[str, Any] | None = None,
        *,
        debug: bool = False,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError("Unknown search mode")
        scope = evaluation_scope(dataset, self.settings)
        configured = evaluation_index_settings(dataset, self.settings)
        self.lexical.read_alias = configured.opensearch_read_alias
        if self.dense is not None:
            self.dense.read_alias = configured.opensearch_read_alias
        options: dict[str, Any] = {
            "tenant_id": scope.tenant_id,
            "collection_id": scope.collection_id,
            "debug": debug,
        }
        filters = filters or {}
        allowed = {"source_type", "author", "source_date_from", "source_date_to"}
        if set(filters) - allowed:
            raise ValueError("Unsupported evaluation metadata filter")
        for key, value in filters.items():
            options[key] = parse_source_date(value) if key.startswith("source_date_") else value
        evidence: list[RetrievalResult]
        if mode == "bm25":
            lexical_response = await self.lexical.search(query, **options)
            evidence = list(lexical_response.results)
            trace = asdict(lexical_response.trace)
        else:
            if self.dense is None:
                provider = await self.stack.enter_async_context(
                    configured_cohere_embeddings(self.settings)
                )
                self.dense = OpenSearchDenseRetriever.from_settings(configured, provider)
                self.hybrid = HybridRetriever(
                    self.lexical, self.dense, RRFConfig(candidate_limit=BUDGET)
                )
            if mode == "dense":
                dense_response = await self.dense.search(query, **options)
                evidence = list(dense_response.results)
                trace = asdict(dense_response.trace)
            elif mode == "hybrid":
                assert self.hybrid is not None
                hybrid_response = await self.hybrid.search(query, **options)
                evidence = [item.evidence for item in hybrid_response.results]
                trace = asdict(hybrid_response.trace)
            else:
                if self.reranked is None:
                    reranker = await self.stack.enter_async_context(
                        configured_cohere_reranking(self.settings)
                    )
                    assert self.hybrid is not None
                    self.reranked = RerankedRetriever(
                        self.hybrid, reranker, RerankedRetrievalConfig.from_settings(self.settings)
                    )
                reranked_response = await self.reranked.search(query, **options)
                evidence = [item.hybrid.evidence for item in reranked_response.results]
                trace = asdict(reranked_response.trace)
        chunks = []
        for rank, item in enumerate(evidence, 1):
            if (
                item.tenant_id != scope.tenant_id
                or item.collection_id != scope.collection_id
                or str(item.document_id) not in scope.document_ids
                or item.pipeline_version != scope.pipeline_version
            ):
                raise ValueError("Search returned evidence outside the prepared evaluation scope")
            chunks.append(
                {
                    **asdict(item),
                    "rank": rank,
                    "evaluation_document_id": scope.document_ids[str(item.document_id)],
                }
            )
        documents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = chunk["evaluation_document_id"]
            if key not in seen:
                documents.append(
                    {"document_id": key, "chunk_id": chunk["chunk_id"], "rank": len(documents) + 1}
                )
                seen.add(key)
        return {"mode": mode, "chunks": chunks, "documents": documents, "trace": trace}
