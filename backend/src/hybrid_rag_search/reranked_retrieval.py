"""Deadline-aware Cohere reranking over hybrid retrieval candidates."""

import argparse
import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.dense_retrieval import OpenSearchDenseRetriever
from hybrid_rag_search.hybrid_retrieval import (
    HybridRetrievalResponse,
    HybridRetrievalResult,
    HybridRetrievalTrace,
    HybridRetriever,
    RRFConfig,
)
from hybrid_rag_search.lexical_retrieval import OpenSearchBM25Retriever, RetrievalError
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.providers.cohere_reranking import configured_cohere_reranking
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.reranking import (
    RerankCandidate,
    RerankingProvider,
    RerankRequest,
    RerankResult,
    RerankUsage,
)
from hybrid_rag_search.retrieval_filters import parse_source_date

RankingMode = Literal["cohere_rerank", "rrf_fallback"]
FallbackReason = Literal[
    "timeout",
    "rate_limited",
    "unavailable",
    "deadline_exhausted",
    "no_candidates",
]


class HybridSearch(Protocol):
    async def search(
        self,
        query_text: str,
        *,
        tenant_id: UUID,
        collection_id: UUID | None = None,
        source_type: str | None = None,
        author: str | None = None,
        source_date_from: datetime | None = None,
        source_date_to: datetime | None = None,
        limit: int | None = None,
        debug: bool = False,
    ) -> HybridRetrievalResponse: ...


@dataclass(frozen=True)
class RerankedRetrievalConfig:
    candidate_limit: int = 20
    result_limit: int = 10
    provider_timeout_seconds: float = 5.0
    request_deadline_seconds: float = 8.0

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate limit", self.candidate_limit),
            ("result limit", self.result_limit),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Reranked retrieval {label} must be a positive integer")
        if self.result_limit > self.candidate_limit:
            raise ValueError("Reranked retrieval result limit cannot exceed candidate limit")
        for label, numeric_value in (
            ("provider timeout", self.provider_timeout_seconds),
            ("request deadline", self.request_deadline_seconds),
        ):
            if not isinstance(numeric_value, (float, int)) or isinstance(numeric_value, bool):
                raise ValueError(f"Reranked retrieval {label} must be a positive finite number")
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"Reranked retrieval {label} must be a positive finite number")
        if self.provider_timeout_seconds > self.request_deadline_seconds:
            raise ValueError("Reranked retrieval provider timeout cannot exceed request deadline")

    @classmethod
    def from_settings(cls, settings: Settings) -> "RerankedRetrievalConfig":
        return cls(
            candidate_limit=settings.rerank_candidate_limit,
            result_limit=settings.rerank_result_limit,
            provider_timeout_seconds=settings.cohere_rerank_timeout_seconds,
            request_deadline_seconds=settings.retrieval_deadline_seconds,
        )


@dataclass(frozen=True)
class RerankedRetrievalResult:
    hybrid: HybridRetrievalResult
    final_rank: int
    relevance_score: float | None
    rank_change: int


@dataclass(frozen=True)
class RerankedRetrievalTrace:
    ranking_mode: RankingMode
    fallback_reason: FallbackReason | None
    candidate_limit: int
    candidate_count: int
    result_limit: int
    result_count: int
    deadline_budget_ms: float
    remaining_before_rerank_ms: float
    rerank_elapsed_ms: float
    model: str | None
    adapter: Literal["fake", "cohere"] | None
    usage: RerankUsage
    hybrid: HybridRetrievalTrace


@dataclass(frozen=True)
class RerankedRetrievalResponse:
    results: tuple[RerankedRetrievalResult, ...]
    trace: RerankedRetrievalTrace


class RerankedRetriever:
    """Retrieve hybrid candidates, then rerank or explicitly fall back."""

    def __init__(
        self,
        hybrid: HybridSearch,
        provider: RerankingProvider,
        config: RerankedRetrievalConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hasattr(hybrid, "search") or not hasattr(provider, "rerank"):
            raise ValueError("Reranked retrieval dependencies must implement their protocols")
        if not callable(clock):
            raise ValueError("Reranked retrieval clock must be callable")
        self.hybrid = hybrid
        self.provider = provider
        self.config = config or RerankedRetrievalConfig()
        if not isinstance(self.config, RerankedRetrievalConfig):
            raise ValueError("Reranked retrieval configuration is invalid")
        self.clock = clock

    async def search(
        self,
        query_text: str,
        *,
        tenant_id: UUID,
        collection_id: UUID | None = None,
        source_type: str | None = None,
        author: str | None = None,
        source_date_from: datetime | None = None,
        source_date_to: datetime | None = None,
        deadline: float | None = None,
        debug: bool = False,
    ) -> RerankedRetrievalResponse:
        started = self.clock()
        absolute_deadline = self._deadline(started, deadline)
        try:
            async with asyncio.timeout(max(absolute_deadline - self.clock(), 0.0)):
                hybrid = await self.hybrid.search(
                    query_text,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    source_type=source_type,
                    author=author,
                    source_date_from=source_date_from,
                    source_date_to=source_date_to,
                    limit=self.config.candidate_limit,
                    debug=debug,
                )
        except TimeoutError:
            raise RetrievalError("retrieval_deadline_exceeded", retryable=True) from None

        candidates = hybrid.results[: self.config.candidate_limit]
        remaining = max(absolute_deadline - self.clock(), 0.0)
        if not candidates:
            return self._fallback(
                hybrid, candidates, "no_candidates", started, absolute_deadline, remaining, 0.0
            )
        if remaining <= 0:
            return self._fallback(
                hybrid,
                candidates,
                "deadline_exhausted",
                started,
                absolute_deadline,
                remaining,
                0.0,
            )

        provider_timeout = min(self.config.provider_timeout_seconds, remaining)
        rerank_started = self.clock()
        try:
            async with asyncio.timeout(provider_timeout):
                reranked = await self.provider.rerank(
                    RerankRequest(
                        query_text,
                        tuple(
                            RerankCandidate(item.evidence.chunk_id, item.evidence.content_text)
                            for item in candidates
                        ),
                        min(self.config.result_limit, len(candidates)),
                    ),
                    timeout_seconds=provider_timeout,
                )
        except TimeoutError:
            elapsed = (self.clock() - rerank_started) * 1_000
            reason: FallbackReason = (
                "deadline_exhausted" if provider_timeout == remaining else "timeout"
            )
            return self._fallback(
                hybrid, candidates, reason, started, absolute_deadline, remaining, elapsed
            )
        except ProviderError as error:
            if not error.retryable:
                raise
            elapsed = (self.clock() - rerank_started) * 1_000
            return self._fallback(
                hybrid,
                candidates,
                _fallback_reason(error),
                started,
                absolute_deadline,
                remaining,
                elapsed,
            )

        rerank_elapsed_ms = (self.clock() - rerank_started) * 1_000
        results = self._map_results(candidates, reranked)
        return RerankedRetrievalResponse(
            results,
            self._trace(
                hybrid,
                ranking_mode="cohere_rerank",
                fallback_reason=None,
                candidate_count=len(candidates),
                result_count=len(results),
                started=started,
                deadline=absolute_deadline,
                remaining=remaining,
                rerank_elapsed_ms=rerank_elapsed_ms,
                reranked=reranked,
            ),
        )

    def _deadline(self, started: float, deadline: float | None) -> float:
        if deadline is None:
            return started + self.config.request_deadline_seconds
        if not isinstance(deadline, (float, int)) or isinstance(deadline, bool):
            raise ValueError("Retrieval deadline must be a finite monotonic timestamp")
        if not math.isfinite(deadline):
            raise ValueError("Retrieval deadline must be a finite monotonic timestamp")
        return float(deadline)

    def _map_results(
        self,
        candidates: tuple[HybridRetrievalResult, ...],
        reranked: RerankResult,
    ) -> tuple[RerankedRetrievalResult, ...]:
        expected_count = min(self.config.result_limit, len(candidates))
        by_id = {item.evidence.chunk_id: item for item in candidates}
        if len(reranked.items) != expected_count or len(by_id) != len(candidates):
            raise RetrievalError("rerank_candidate_mismatch", retryable=False)
        results: list[RerankedRetrievalResult] = []
        seen: set[str] = set()
        for final_rank, item in enumerate(reranked.items, start=1):
            candidate = by_id.get(item.candidate_id)
            if (
                candidate is None
                or item.candidate_id in seen
                or item.rank != final_rank
                or item.original_index >= len(candidates)
                or candidates[item.original_index].evidence.chunk_id != item.candidate_id
            ):
                raise RetrievalError("rerank_candidate_mismatch", retryable=False)
            seen.add(item.candidate_id)
            results.append(
                RerankedRetrievalResult(
                    hybrid=candidate,
                    final_rank=final_rank,
                    relevance_score=item.relevance_score,
                    rank_change=candidate.fused_rank - final_rank,
                )
            )
        return tuple(results)

    def _fallback(
        self,
        hybrid: HybridRetrievalResponse,
        candidates: tuple[HybridRetrievalResult, ...],
        reason: FallbackReason,
        started: float,
        deadline: float,
        remaining: float,
        rerank_elapsed_ms: float,
    ) -> RerankedRetrievalResponse:
        selected = candidates[: self.config.result_limit]
        results = tuple(
            RerankedRetrievalResult(item, rank, None, item.fused_rank - rank)
            for rank, item in enumerate(selected, start=1)
        )
        return RerankedRetrievalResponse(
            results,
            self._trace(
                hybrid,
                ranking_mode="rrf_fallback",
                fallback_reason=reason,
                candidate_count=len(candidates),
                result_count=len(results),
                started=started,
                deadline=deadline,
                remaining=remaining,
                rerank_elapsed_ms=rerank_elapsed_ms,
                reranked=None,
            ),
        )

    def _trace(
        self,
        hybrid: HybridRetrievalResponse,
        *,
        ranking_mode: RankingMode,
        fallback_reason: FallbackReason | None,
        candidate_count: int,
        result_count: int,
        started: float,
        deadline: float,
        remaining: float,
        rerank_elapsed_ms: float,
        reranked: RerankResult | None,
    ) -> RerankedRetrievalTrace:
        return RerankedRetrievalTrace(
            ranking_mode=ranking_mode,
            fallback_reason=fallback_reason,
            candidate_limit=self.config.candidate_limit,
            candidate_count=candidate_count,
            result_limit=self.config.result_limit,
            result_count=result_count,
            deadline_budget_ms=max(deadline - started, 0.0) * 1_000,
            remaining_before_rerank_ms=remaining * 1_000,
            rerank_elapsed_ms=rerank_elapsed_ms,
            model=reranked.model if reranked else None,
            adapter=reranked.adapter if reranked else None,
            usage=reranked.usage if reranked else RerankUsage(),
            hybrid=hybrid.trace,
        )


def _fallback_reason(error: ProviderError) -> FallbackReason:
    if error.code == "timeout":
        return "timeout"
    if error.code == "rate_limited":
        return "rate_limited"
    if error.code == "unavailable":
        return "unavailable"
    raise RetrievalError("rerank_fallback_classification_failed", retryable=False)


async def _debug_query(settings: Settings, args: argparse.Namespace) -> RerankedRetrievalResponse:
    async with (
        configured_cohere_embeddings(settings) as embeddings,
        configured_cohere_reranking(settings) as reranker,
    ):
        hybrid = HybridRetriever(
            OpenSearchBM25Retriever.from_settings(settings),
            OpenSearchDenseRetriever.from_settings(settings, embeddings),
            RRFConfig(candidate_limit=settings.rerank_candidate_limit),
        )
        retriever = RerankedRetriever(
            hybrid,
            reranker,
            RerankedRetrievalConfig.from_settings(settings),
        )
        return await retriever.search(
            args.query,
            tenant_id=args.tenant,
            collection_id=args.collection,
            source_type=args.source_type,
            author=args.author,
            source_date_from=parse_source_date(args.source_date_from)
            if args.source_date_from
            else None,
            source_date_to=parse_source_date(args.source_date_to) if args.source_date_to else None,
            debug=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, type=UUID)
    parser.add_argument("--collection", type=UUID)
    parser.add_argument("--source-type")
    parser.add_argument("--author")
    parser.add_argument("--source-date-from")
    parser.add_argument("--source-date-to")
    args = parser.parse_args()
    print(asyncio.run(_debug_query(get_settings(), args)))


if __name__ == "__main__":  # pragma: no cover
    main()
