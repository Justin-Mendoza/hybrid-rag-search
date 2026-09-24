"""Concurrent BM25 and dense retrieval with deterministic reciprocal-rank fusion."""

import argparse
import asyncio
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol
from uuid import UUID

from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.dense_retrieval import (
    DenseRetrievalResponse,
    DenseRetrievalTrace,
    OpenSearchDenseRetriever,
)
from hybrid_rag_search.lexical_retrieval import (
    LexicalRetrievalResponse,
    OpenSearchBM25Retriever,
    RetrievalError,
    RetrievalResult,
    RetrievalTrace,
)
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.retrieval_filters import RetrievalFilters, parse_source_date

RRF_VERSION = "rrf-v1"


class LexicalRetriever(Protocol):
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
    ) -> LexicalRetrievalResponse: ...


class DenseRetriever(Protocol):
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
    ) -> DenseRetrievalResponse: ...


@dataclass(frozen=True)
class RRFConfig:
    """Versioned fusion controls kept small until evaluation can tune them."""

    version: str = RRF_VERSION
    rank_constant: int = 60
    candidate_limit: int = 20

    def __post_init__(self) -> None:
        if self.version != RRF_VERSION:
            raise ValueError(f"RRF version must be {RRF_VERSION}")
        for label, value in (
            ("rank constant", self.rank_constant),
            ("candidate limit", self.candidate_limit),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"RRF {label} must be a positive integer")


@dataclass(frozen=True)
class HybridRetrievalResult:
    """One fused candidate with both original rankings left inspectable."""

    evidence: RetrievalResult
    lexical_rank: int | None
    lexical_score: float | None
    vector_rank: int | None
    vector_score: float | None
    fused_rank: int
    fused_score: float


@dataclass(frozen=True)
class HybridRetrievalTrace:
    """Fusion diagnostics plus each underlying retriever's trace."""

    fusion_version: str
    rank_constant: int
    candidate_limit: int
    candidate_count: int
    elapsed_ms: float
    filters: dict[str, str]
    lexical: RetrievalTrace
    dense: DenseRetrievalTrace


@dataclass(frozen=True)
class HybridRetrievalResponse:
    results: tuple[HybridRetrievalResult, ...]
    trace: HybridRetrievalTrace


class HybridRetriever:
    """Run both retrieval paths concurrently and fuse their candidate ranks."""

    def __init__(
        self,
        lexical: LexicalRetriever,
        dense: DenseRetriever,
        config: RRFConfig | None = None,
    ) -> None:
        if not hasattr(lexical, "search") or not hasattr(dense, "search"):
            raise ValueError("Hybrid retrievers must implement search")
        self.lexical = lexical
        self.dense = dense
        self.config = config or RRFConfig()
        if not isinstance(self.config, RRFConfig):
            raise ValueError("Hybrid configuration must be an RRFConfig")

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
    ) -> HybridRetrievalResponse:
        if not isinstance(query_text, str):
            raise ValueError("Hybrid query text must be a string")
        if not isinstance(debug, bool):
            raise ValueError("Hybrid debug flag must be boolean")
        result_limit = self._result_limit(limit)
        filters = RetrievalFilters(
            tenant_id,
            collection_id,
            source_type,
            author,
            source_date_from,
            source_date_to,
        )
        started = time.perf_counter()
        lexical, dense = await asyncio.gather(
            self.lexical.search(
                query_text,
                tenant_id=filters.tenant_id,
                collection_id=filters.collection_id,
                source_type=filters.source_type,
                author=filters.author,
                source_date_from=filters.source_date_from,
                source_date_to=filters.source_date_to,
                limit=self.config.candidate_limit,
                debug=debug,
            ),
            self.dense.search(
                query_text,
                tenant_id=filters.tenant_id,
                collection_id=filters.collection_id,
                source_type=filters.source_type,
                author=filters.author,
                source_date_from=filters.source_date_from,
                source_date_to=filters.source_date_to,
                limit=self.config.candidate_limit,
                debug=debug,
            ),
        )
        results = self._fuse(lexical.results, dense.results, result_limit)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        return HybridRetrievalResponse(
            results,
            HybridRetrievalTrace(
                fusion_version=self.config.version,
                rank_constant=self.config.rank_constant,
                candidate_limit=self.config.candidate_limit,
                candidate_count=len(results),
                elapsed_ms=elapsed_ms,
                filters=filters.trace_values(),
                lexical=lexical.trace,
                dense=dense.trace,
            ),
        )

    def _result_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.config.candidate_limit
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("Hybrid result limit must be a positive integer when present")
        return min(limit, self.config.candidate_limit)

    def _fuse(
        self,
        lexical: tuple[RetrievalResult, ...],
        dense: tuple[RetrievalResult, ...],
        limit: int,
    ) -> tuple[HybridRetrievalResult, ...]:
        candidates: dict[str, dict[str, object]] = {}
        for path, results in (("lexical", lexical), ("vector", dense)):
            for item in results:
                candidate = candidates.setdefault(item.chunk_id, {"evidence": item})
                evidence = candidate["evidence"]
                if not isinstance(evidence, RetrievalResult) or replace(
                    evidence, rank=0, score=0.0
                ) != replace(item, rank=0, score=0.0):
                    raise RetrievalError("hybrid_candidate_mismatch", retryable=False)
                candidate[f"{path}_rank"] = item.rank
                candidate[f"{path}_score"] = item.score

        scored: list[tuple[float, str, dict[str, object]]] = []
        for chunk_id, candidate in candidates.items():
            score = sum(
                1.0 / (self.config.rank_constant + rank)
                for rank in (candidate.get("lexical_rank"), candidate.get("vector_rank"))
                if isinstance(rank, int)
            )
            scored.append((score, chunk_id, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))

        return tuple(
            HybridRetrievalResult(
                evidence=_evidence(candidate),
                lexical_rank=_optional_int(candidate.get("lexical_rank")),
                lexical_score=_optional_float(candidate.get("lexical_score")),
                vector_rank=_optional_int(candidate.get("vector_rank")),
                vector_score=_optional_float(candidate.get("vector_score")),
                fused_rank=rank,
                fused_score=score,
            )
            for rank, (score, _chunk_id, candidate) in enumerate(scored[:limit], start=1)
        )


def _evidence(candidate: dict[str, object]) -> RetrievalResult:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, RetrievalResult):
        raise RetrievalError("hybrid_candidate_mismatch", retryable=False)
    return evidence


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) else None


async def _debug_query(settings: Settings, args: argparse.Namespace) -> HybridRetrievalResponse:
    async with configured_cohere_embeddings(settings) as provider:
        retriever = HybridRetriever(
            OpenSearchBM25Retriever.from_settings(settings),
            OpenSearchDenseRetriever.from_settings(settings, provider),
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
            limit=args.limit,
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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(asyncio.run(_debug_query(get_settings(), args)))


if __name__ == "__main__":  # pragma: no cover
    main()
