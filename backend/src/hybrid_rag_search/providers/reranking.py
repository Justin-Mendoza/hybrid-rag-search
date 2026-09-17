"""Application-owned reranking contract, independent of provider SDKs."""

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol


@dataclass(frozen=True)
class RerankCandidate:
    """A stable application ID paired with the text shown to the reranker."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Rerank candidate ID must be nonblank")
        if not self.text.strip():
            raise ValueError("Rerank candidate text must be nonblank")


@dataclass(frozen=True)
class RerankRequest:
    query: str
    candidates: tuple[RerankCandidate, ...]
    top_n: int

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Rerank query must be nonblank")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("Rerank candidates must be a nonempty tuple")
        if len({candidate.id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Rerank candidate IDs must be unique")
        if type(self.top_n) is not int or not 1 <= self.top_n <= len(self.candidates):
            raise ValueError("Rerank top_n must be between 1 and the candidate count")


@dataclass(frozen=True)
class RerankItem:
    candidate_id: str
    original_index: int
    rank: int
    relevance_score: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("Rerank item candidate ID must be nonblank")
        if self.original_index < 0 or self.rank < 1:
            raise ValueError("Rerank item positions must be nonnegative index and positive rank")
        if not math.isfinite(self.relevance_score):
            raise ValueError("Rerank relevance score must be finite")


@dataclass(frozen=True)
class RerankUsage:
    billed_search_units: float | None = None
    latency_ms: float | None = None
    estimated_cost_usd: Decimal | None = None
    usd_per_search_unit: Decimal | None = None


@dataclass(frozen=True)
class RerankResult:
    items: tuple[RerankItem, ...]
    model: str
    adapter: Literal["fake", "cohere"]
    usage: RerankUsage = RerankUsage()


class RerankingProvider(Protocol):
    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Return at most top_n candidates in descending relevance order."""
        ...
