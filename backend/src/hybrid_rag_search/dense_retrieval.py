"""Cohere query embeddings and filtered cosine k-NN retrieval over ready chunks."""

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.lexical_retrieval import (
    OpenSearchBM25Retriever,
    RetrievalError,
    RetrievalResult,
)
from hybrid_rag_search.opensearch_index import physical_index_name
from hybrid_rag_search.providers.cohere_embeddings import configured_cohere_embeddings
from hybrid_rag_search.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
)


@dataclass(frozen=True)
class DenseRetrievalConfig:
    """Configured query-model and index pairing for the dense retrieval path."""

    embedding_model: str = "embed-english-light-v3.0"
    embedding_dimensions: int = 384
    index_schema_version: str = "chunks-v2"
    candidate_limit: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("Dense embedding model must be a nonblank string")
        if (
            not isinstance(self.embedding_dimensions, int)
            or isinstance(self.embedding_dimensions, bool)
            or self.embedding_dimensions <= 0
        ):
            raise ValueError("Dense embedding dimensions must be a positive integer")
        if (
            not isinstance(self.candidate_limit, int)
            or isinstance(self.candidate_limit, bool)
            or self.candidate_limit <= 0
        ):
            raise ValueError("Dense candidate limit must be a positive integer")
        # The physical-index naming rules are the existing schema-version
        # contract. Dense retrieval trusts this configured model/index pairing;
        # it intentionally does not add a mapping request for every query.
        physical_index_name(self.index_schema_version, "dense-query")

    @classmethod
    def from_settings(cls, settings: Settings) -> "DenseRetrievalConfig":
        return cls(
            embedding_model=settings.cohere_embed_model,
            embedding_dimensions=settings.cohere_embed_dimensions,
            index_schema_version=settings.opensearch_index_schema_version,
            candidate_limit=settings.opensearch_dense_candidate_limit,
        )


@dataclass(frozen=True)
class DenseRetrievalTrace:
    """Non-public embedding and k-NN diagnostics, separate from evidence."""

    normalized_query: str
    embedding_model: str
    embedding_dimensions: int
    embedding_adapter: str | None
    index_schema_version: str
    read_alias: str
    candidate_limit: int
    candidate_count: int
    embedding_elapsed_ms: float
    opensearch_elapsed_ms: float
    tenant_id: UUID
    collection_id: UUID | None
    filters: dict[str, str]


@dataclass(frozen=True)
class DenseRetrievalResponse:
    """Dense evidence uses the same normalized result contract as BM25."""

    results: tuple[RetrievalResult, ...]
    trace: DenseRetrievalTrace


def normalize_dense_query(query_text: str) -> str:
    """Trim outer whitespace and collapse internal whitespace without rewriting text."""

    if not isinstance(query_text, str):
        raise ValueError("Dense query text must be a string")
    return " ".join(query_text.split())


class OpenSearchDenseRetriever:
    """Embed one normalized query and run a tenant-scoped cosine k-NN search."""

    def __init__(
        self,
        base_url: str,
        read_alias: str,
        embedding_provider: EmbeddingProvider,
        config: DenseRetrievalConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OpenSearch base URL must be a nonblank string")
        if not isinstance(read_alias, str) or not read_alias.strip():
            raise ValueError("OpenSearch read alias must be a nonblank string")
        if not hasattr(embedding_provider, "embed"):
            raise ValueError("Dense embedding provider must implement embed")
        self.base_url = base_url
        self.read_alias = read_alias
        self.embedding_provider = embedding_provider
        self.config = config or DenseRetrievalConfig()
        if not isinstance(self.config, DenseRetrievalConfig):
            raise ValueError("Dense configuration must be a DenseRetrievalConfig")
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "OpenSearchDenseRetriever":
        return cls(
            settings.opensearch_url,
            settings.opensearch_read_alias,
            embedding_provider,
            DenseRetrievalConfig.from_settings(settings),
            transport=transport,
        )

    async def search(
        self,
        query_text: str,
        *,
        tenant_id: UUID,
        collection_id: UUID | None = None,
        limit: int | None = None,
        debug: bool = False,
    ) -> DenseRetrievalResponse:
        if not isinstance(tenant_id, UUID):
            raise ValueError("Dense tenant ID must be a UUID")
        if collection_id is not None and not isinstance(collection_id, UUID):
            raise ValueError("Dense collection ID must be a UUID when present")
        if not isinstance(debug, bool):
            raise ValueError("Dense debug flag must be boolean")

        normalized_query = normalize_dense_query(query_text)
        candidate_limit = self._candidate_limit(limit)
        filters = OpenSearchBM25Retriever._filter_values(tenant_id, collection_id)
        if not normalized_query:
            return DenseRetrievalResponse(
                (),
                self._trace(
                    normalized_query,
                    candidate_limit,
                    0,
                    0.0,
                    0.0,
                    tenant_id,
                    collection_id,
                    filters,
                    adapter=None,
                ),
            )

        embedding_started = time.perf_counter()
        embedding = await self.embedding_provider.embed(
            EmbeddingRequest((normalized_query,), EmbeddingPurpose.QUERY)
        )
        embedding_elapsed_ms = (time.perf_counter() - embedding_started) * 1_000
        vector = self._query_vector(embedding)

        opensearch_started = time.perf_counter()
        response = await self._request(
            "POST",
            f"/{self.read_alias}/_search",
            json=self._search_body(vector, tenant_id, collection_id, candidate_limit, debug),
        )
        opensearch_elapsed_ms = (time.perf_counter() - opensearch_started) * 1_000
        payload = response.json()
        if not isinstance(payload, dict):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        hits = payload.get("hits")
        if not isinstance(hits, dict):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        raw_hits = hits.get("hits")
        if not isinstance(raw_hits, list):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        results = tuple(
            OpenSearchBM25Retriever._result_from_hit(hit, rank)
            for rank, hit in enumerate(raw_hits, start=1)
        )
        return DenseRetrievalResponse(
            results,
            self._trace(
                normalized_query,
                candidate_limit,
                OpenSearchBM25Retriever._candidate_count(hits, len(results)),
                embedding_elapsed_ms,
                opensearch_elapsed_ms,
                tenant_id,
                collection_id,
                filters,
                adapter=embedding.adapter,
            ),
        )

    def _candidate_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.config.candidate_limit
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("Dense result limit must be a positive integer when present")
        return min(limit, self.config.candidate_limit)

    def _query_vector(self, result: EmbeddingResult) -> tuple[float, ...]:
        if result.purpose != EmbeddingPurpose.QUERY or len(result.vectors) != 1:
            raise RetrievalError("embedding_invalid_response", retryable=False)
        if result.model != self.config.embedding_model:
            raise RetrievalError("embedding_model_mismatch", retryable=False)
        if result.dimensions != self.config.embedding_dimensions:
            raise RetrievalError("query_vector_dimensions_mismatch", retryable=False)
        vector = result.vectors[0]
        if len(vector) != self.config.embedding_dimensions:
            raise RetrievalError("query_vector_dimensions_mismatch", retryable=False)
        if any(
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in vector
        ):
            raise RetrievalError("embedding_invalid_response", retryable=False)
        return tuple(float(value) for value in vector)

    def _search_body(
        self,
        vector: tuple[float, ...],
        tenant_id: UUID,
        collection_id: UUID | None,
        candidate_limit: int,
        debug: bool,
    ) -> dict[str, object]:
        filters = [
            {"term": {field: value}}
            for field, value in OpenSearchBM25Retriever._filter_values(
                tenant_id, collection_id
            ).items()
        ]
        return {
            "size": candidate_limit,
            "track_total_hits": True,
            "explain": debug,
            "_source": [
                "chunk_id",
                "tenant_id",
                "collection_id",
                "document_id",
                "document_content_id",
                "content_text",
                "source_spans",
                "generation_id",
                "pipeline_version",
                "source_metadata",
            ],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": list(vector),
                        "k": candidate_limit,
                        "filter": {"bool": {"filter": filters}},
                    }
                }
            },
        }

    def _trace(
        self,
        normalized_query: str,
        candidate_limit: int,
        candidate_count: int,
        embedding_elapsed_ms: float,
        opensearch_elapsed_ms: float,
        tenant_id: UUID,
        collection_id: UUID | None,
        filters: dict[str, str],
        *,
        adapter: str | None,
    ) -> DenseRetrievalTrace:
        return DenseRetrievalTrace(
            normalized_query=normalized_query,
            embedding_model=self.config.embedding_model,
            embedding_dimensions=self.config.embedding_dimensions,
            embedding_adapter=adapter,
            index_schema_version=self.config.index_schema_version,
            read_alias=self.read_alias,
            candidate_limit=candidate_limit,
            candidate_count=candidate_count,
            embedding_elapsed_ms=embedding_elapsed_ms,
            opensearch_elapsed_ms=opensearch_elapsed_ms,
            tenant_id=tenant_id,
            collection_id=collection_id,
            filters=filters,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self.base_url, transport=self.transport, timeout=10
        ) as client:
            try:
                response = await client.request(method, path, **kwargs)
            except httpx.HTTPError as error:
                raise RetrievalError("opensearch_unavailable", retryable=True) from error
        if response.is_error:
            retryable = response.status_code >= 500 or response.status_code in (408, 429)
            raise RetrievalError("opensearch_unavailable", retryable=retryable)
        return response


async def _debug_query(settings: Settings, args: argparse.Namespace) -> DenseRetrievalResponse:
    async with configured_cohere_embeddings(settings) as provider:
        retriever = OpenSearchDenseRetriever.from_settings(settings, provider)
        return await retriever.search(
            args.query,
            tenant_id=args.tenant,
            collection_id=args.collection,
            limit=args.limit,
            debug=True,
        )


def main() -> None:
    """Run a local dense debug query without exposing a public search API yet."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, type=UUID)
    parser.add_argument("--collection", type=UUID)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(asyncio.run(_debug_query(get_settings(), args)))


if __name__ == "__main__":  # pragma: no cover
    main()
