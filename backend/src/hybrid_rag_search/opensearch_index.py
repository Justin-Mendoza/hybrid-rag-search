"""Versioned OpenSearch storage for complete document chunk generations."""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from hybrid_rag_search.ingestion.indexing import (
    DocumentIndex,
    IndexReceipt,
    IndexWriteError,
    ReplaceDocumentRequest,
)

_SCHEMA_VERSION = re.compile(r"^[a-z][a-z0-9-]*v[0-9]+$")
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9])"
)


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Extract stable, case-normalized identifier tokens from chunk text."""

    return tuple(dict.fromkeys(match.group(0).casefold() for match in _IDENTIFIER.finditer(text)))


def physical_index_name(schema_version: str, build_id: str) -> str:
    """Return a bounded physical-index name, separate from stable aliases."""

    if not _SCHEMA_VERSION.fullmatch(schema_version):
        raise ValueError("Index schema version must look like chunks-v1")
    if not re.fullmatch(r"[a-z0-9-]{1,40}", build_id):
        raise ValueError("Index build ID must be lowercase letters, numbers, or hyphens")
    return f"hybrid-rag-{schema_version}-{build_id}"


def index_mapping(dimensions: int) -> dict[str, object]:
    """Return the complete, versioned mapping for one chunks index."""

    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise ValueError("Index vector dimensions must be a positive integer")
    return {
        "settings": {"index.knn": True},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "content_text": {"type": "text"},
                "heading_text": {"type": "text"},
                "identifiers": {"type": "keyword"},
                "document_content_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "collection_id": {"type": "keyword"},
                "content_hash": {"type": "keyword"},
                "pipeline_version": {"type": "keyword"},
                "generation_id": {"type": "keyword"},
                "visibility": {"type": "keyword"},
                "chunk_order": {"type": "integer"},
                "chunk_config_version": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding_dimensions": {"type": "integer"},
                "embedding_adapter": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "cosinesimil",
                    },
                },
                "source_spans": {
                    "type": "nested",
                    "properties": {
                        "block_number": {"type": "integer"},
                        "page_number": {"type": "integer"},
                        "heading_path": {"type": "keyword"},
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                    },
                },
                "source_metadata": {"type": "flat_object"},
            },
        },
    }


@dataclass(frozen=True)
class OpenSearchIndexManager:
    """Create physical indexes and switch the stable read/write aliases."""

    base_url: str
    read_alias: str
    write_alias: str
    dimensions: int
    transport: httpx.AsyncBaseTransport | None = None

    async def create(self, index_name: str) -> None:
        await self._request("PUT", f"/{index_name}", json=index_mapping(self.dimensions))

    async def point_write_alias(self, index_name: str) -> None:
        """Send rebuild writes to a new index without changing live reads."""

        actions: list[dict[str, dict[str, object]]] = [
            {"remove": {"alias": self.write_alias, "index": "*", "must_exist": False}},
            {
                "add": {
                    "alias": self.write_alias,
                    "index": index_name,
                    "is_write_index": True,
                }
            },
        ]
        await self._request("POST", "/_aliases", json={"actions": actions})

    async def promote_read_alias(self, index_name: str) -> None:
        """Atomically make a completely rebuilt physical index readable."""

        actions: list[dict[str, dict[str, object]]] = [
            {"remove": {"alias": self.read_alias, "index": "*", "must_exist": False}},
            {"add": {"alias": self.read_alias, "index": index_name}},
        ]
        await self._request("POST", "/_aliases", json={"actions": actions})

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self.base_url, transport=self.transport, timeout=10
        ) as client:
            try:
                response = await client.request(method, path, **kwargs)
            except httpx.HTTPError as error:
                raise IndexWriteError("opensearch_unavailable", retryable=True) from error
        if response.is_error:
            retryable = response.status_code >= 500 or response.status_code in (408, 429)
            raise IndexWriteError("opensearch_unavailable", retryable=retryable)
        return response


class OpenSearchDocumentIndex(DocumentIndex):
    """Stage, promote, and delete complete document generations through OpenSearch."""

    def __init__(
        self,
        base_url: str,
        write_alias: str,
        expected_dimensions: int,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.write_alias = write_alias
        self.expected_dimensions = expected_dimensions
        self.transport = transport

    async def replace_document(self, request: ReplaceDocumentRequest) -> IndexReceipt:
        if request.embedding_dimensions != self.expected_dimensions:
            raise IndexWriteError("vector_dimensions_mismatch", retryable=False)
        body = self._bulk_body(request)
        response = await self._request(
            "POST",
            f"/{self.write_alias}/_bulk?refresh=wait_for",
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )
        payload = response.json()
        if payload.get("errors"):
            raise IndexWriteError("bulk_write_failed", retryable=True)
        updated = await self._promote(request)
        if updated != len(request.records):
            raise IndexWriteError("generation_incomplete", retryable=True)
        await self._remove_prior_generations(request)
        return IndexReceipt.from_request(request)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        await self._request(
            "POST",
            f"/{self.write_alias}/_delete_by_query?refresh=true&conflicts=proceed",
            json={"query": {"bool": {"filter": self._document_filters(tenant_id, document_id)}}},
        )

    def _bulk_body(self, request: ReplaceDocumentRequest) -> bytes:
        lines: list[str] = []
        for record in request.records:
            chunk = record.chunk
            heading_paths = tuple(
                dict.fromkeys(
                    " / ".join(span.locator.heading_path)
                    for span in chunk.source_spans
                    if span.locator.heading_path
                )
            )
            heading_text = "\n".join(heading_paths)
            lines.append(
                json.dumps(
                    {
                        "index": {
                            "_id": f"{request.generation_id}:{chunk.chunk_id}",
                            "routing": str(request.document_id),
                        }
                    },
                    separators=(",", ":"),
                )
            )
            lines.append(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_text": chunk.content_text,
                        "heading_text": heading_text,
                        "identifiers": extract_identifiers(
                            "\n".join((chunk.content_text, heading_text))
                        ),
                        "document_content_id": chunk.document_id,
                        "document_id": str(request.document_id),
                        "tenant_id": str(request.tenant_id),
                        "collection_id": str(request.collection_id),
                        "content_hash": request.content_hash,
                        "pipeline_version": request.pipeline_version,
                        "generation_id": request.generation_id,
                        "visibility": "staged",
                        "chunk_order": chunk.order,
                        "chunk_config_version": chunk.config_version,
                        "embedding_model": request.embedding_model,
                        "embedding_dimensions": request.embedding_dimensions,
                        "embedding_adapter": request.embedding_adapter,
                        "embedding": record.embedding.vector,
                        "source_spans": [
                            {
                                "block_number": span.locator.block_number,
                                "page_number": span.locator.page_number,
                                "heading_path": list(span.locator.heading_path),
                                "start_char": span.start_char,
                                "end_char": span.end_char,
                            }
                            for span in chunk.source_spans
                        ],
                        "source_metadata": {},
                    },
                    separators=(",", ":"),
                )
            )
        return ("\n".join(lines) + "\n").encode()

    async def _promote(self, request: ReplaceDocumentRequest) -> int:
        response = await self._request(
            "POST",
            f"/{self.write_alias}/_update_by_query?refresh=true&conflicts=proceed",
            json={
                "script": {"source": "ctx._source.visibility = 'ready'", "lang": "painless"},
                "query": {
                    "bool": {
                        "filter": self._document_filters(
                            str(request.tenant_id), str(request.document_id)
                        )
                        + [{"term": {"generation_id": request.generation_id}}],
                    }
                },
            },
        )
        updated = response.json().get("updated")
        return updated if isinstance(updated, int) and not isinstance(updated, bool) else -1

    async def _remove_prior_generations(self, request: ReplaceDocumentRequest) -> None:
        await self._request(
            "POST",
            f"/{self.write_alias}/_delete_by_query?refresh=true&conflicts=proceed",
            json={
                "query": {
                    "bool": {
                        "filter": self._document_filters(
                            str(request.tenant_id), str(request.document_id)
                        ),
                        "must_not": [{"term": {"generation_id": request.generation_id}}],
                    }
                }
            },
        )

    @staticmethod
    def _document_filters(tenant_id: str, document_id: str) -> list[dict[str, dict[str, str]]]:
        return [{"term": {"tenant_id": tenant_id}}, {"term": {"document_id": document_id}}]

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self.base_url, transport=self.transport, timeout=20
        ) as client:
            try:
                response = await client.request(method, path, **kwargs)
            except httpx.HTTPError as error:
                raise IndexWriteError("opensearch_unavailable", retryable=True) from error
        if response.is_error:
            retryable = response.status_code >= 500 or response.status_code in (408, 429)
            raise IndexWriteError("opensearch_unavailable", retryable=retryable)
        return response
