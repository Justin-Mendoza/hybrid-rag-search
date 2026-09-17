"""Deterministic synthetic vectors with no network or credential dependencies."""

import hashlib
import json
import math
import re
from decimal import Decimal

from hybrid_rag_search.providers.embeddings import EmbeddingRequest, EmbeddingResult
from hybrid_rag_search.providers.generation import (
    GenerationFinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from hybrid_rag_search.providers.reranking import (
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankUsage,
)
from hybrid_rag_search.providers.usage import EmbeddingUsage


class FakeEmbeddingProvider:
    """Versioned hash vectors, stable across instances and batch composition.

    Eight dimensions keep examples readable; tests of an index can request its
    dimension. Purpose and dimension are part of the hash input. Similar text
    does NOT imply similar vectors, and query/document vectors are not aligned.
    """

    MODEL = "fake-embedding-v1"

    def __init__(self, dimensions: int = 8) -> None:
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Embedding dimensions must be a positive integer")
        self.dimensions = dimensions

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for text in request.texts:
            payload = json.dumps(
                [self.MODEL, self.dimensions, request.purpose.value, text],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.shake_256(payload).digest(self.dimensions * 2)
            # Odd signed integers avoid an all-zero vector before normalization.
            values = [
                2 * int.from_bytes(digest[index : index + 2], "big") - 65535
                for index in range(0, len(digest), 2)
            ]
            length = math.sqrt(sum(value * value for value in values))
            vectors.append(tuple(value / length for value in values))
        return EmbeddingResult(
            vectors=tuple(vectors),
            model=self.MODEL,
            dimensions=self.dimensions,
            purpose=request.purpose,
            adapter="fake",
            usage=EmbeddingUsage(billed_input_tokens=0, estimated_cost_usd=Decimal(0)),
        )


class FakeRerankingProvider:
    """Transparent lexical scorer for plumbing, fallback, and load tests.

    The score is the fraction of distinct query terms present in a candidate.
    Ties retain input order. This is deliberately not a neural relevance model.
    """

    MODEL = "fake-rerank-v1"
    _TERM = re.compile(r"\w+", re.UNICODE)

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {match.group().casefold() for match in cls._TERM.finditer(text)}

    async def rerank(self, request: RerankRequest) -> RerankResult:
        query_terms = self._terms(request.query)
        scored = []
        for index, candidate in enumerate(request.candidates):
            score = len(query_terms & self._terms(candidate.text)) / len(query_terms)
            scored.append((index, candidate.id, score))
        scored.sort(key=lambda item: (-item[2], item[0]))
        return RerankResult(
            items=tuple(
                RerankItem(
                    candidate_id=candidate_id,
                    original_index=original_index,
                    rank=rank,
                    relevance_score=score,
                )
                for rank, (original_index, candidate_id, score) in enumerate(
                    scored[: request.top_n], start=1
                )
            ),
            model=self.MODEL,
            adapter="fake",
            usage=RerankUsage(billed_search_units=0, estimated_cost_usd=Decimal(0)),
        )


class FakeGenerationProvider:
    """Stable synthetic text for plumbing and load tests; it does not answer questions."""

    MODEL = "fake-generation-v1"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = json.dumps(
            [
                self.MODEL,
                request.max_output_tokens,
                [[message.role.value, message.content] for message in request.messages],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()[:16]
        return GenerationResult(
            text=f"Synthetic response {fingerprint}.",
            finish_reason=GenerationFinishReason.COMPLETE,
            model=self.MODEL,
            adapter="fake",
            usage=GenerationUsage(
                billed_input_tokens=0,
                billed_output_tokens=0,
                estimated_cost_usd=Decimal(0),
            ),
        )
