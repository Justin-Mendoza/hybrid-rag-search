"""Predictable BM25 retrieval over the ready OpenSearch chunks alias."""

import argparse
import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx

from hybrid_rag_search.chunking.contracts import SourceSpan
from hybrid_rag_search.config import Settings, get_settings
from hybrid_rag_search.opensearch_index import extract_identifiers
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.retrieval_filters import RetrievalFilters

_QUOTED_PHRASE = re.compile(r'"([^\"]*)"')
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9])"
)
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


class RetrievalError(RuntimeError):
    """A retrieval failure with an explicit retry classification."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", code):
            raise ValueError("Retrieval error code must use lowercase snake_case")
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True)
class BM25Config:
    """The small set of ranking controls owned by the lexical baseline."""

    content_boost: float = 1.0
    heading_boost: float = 1.5
    phrase_boost: float = 3.0
    identifier_boost: float = 5.0
    candidate_limit: int = 20

    def __post_init__(self) -> None:
        for label, value in (
            ("content boost", self.content_boost),
            ("heading boost", self.heading_boost),
            ("phrase boost", self.phrase_boost),
            ("identifier boost", self.identifier_boost),
        ):
            if not isinstance(value, (float, int)) or isinstance(value, bool) or float(value) <= 0:
                raise ValueError(f"BM25 {label} must be a positive number")
        if (
            not isinstance(self.candidate_limit, int)
            or isinstance(self.candidate_limit, bool)
            or self.candidate_limit <= 0
        ):
            raise ValueError("BM25 candidate limit must be a positive integer")

    @classmethod
    def from_settings(cls, settings: Settings) -> "BM25Config":
        return cls(
            content_boost=settings.opensearch_bm25_content_boost,
            heading_boost=settings.opensearch_bm25_heading_boost,
            phrase_boost=settings.opensearch_bm25_phrase_boost,
            identifier_boost=settings.opensearch_bm25_identifier_boost,
            candidate_limit=settings.opensearch_bm25_candidate_limit,
        )


@dataclass(frozen=True)
class ParsedLexicalQuery:
    """The deliberately small query language understood by the BM25 baseline."""

    ordinary_terms: tuple[str, ...]
    phrases: tuple[str, ...]
    identifiers: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.ordinary_terms or self.phrases or self.identifiers)


def parse_lexical_query(query_text: str) -> ParsedLexicalQuery:
    """Separate ordinary words, quoted phrases, and identifier-shaped terms."""

    if not isinstance(query_text, str):
        raise ValueError("Lexical query text must be a string")

    phrases: list[str] = []

    def remove_phrase(match: re.Match[str]) -> str:
        phrase = " ".join(word.casefold() for word in _WORD.findall(match.group(1)))
        if phrase:
            phrases.append(phrase)
        return " "

    remainder = _QUOTED_PHRASE.sub(remove_phrase, query_text)
    identifiers = extract_identifiers(remainder)
    remainder = _IDENTIFIER.sub(" ", remainder)
    ordinary_terms = tuple(dict.fromkeys(word.casefold() for word in _WORD.findall(remainder)))
    return ParsedLexicalQuery(
        ordinary_terms=ordinary_terms,
        phrases=tuple(dict.fromkeys(phrases)),
        identifiers=identifiers,
    )


@dataclass(frozen=True)
class RetrievalResult:
    """OpenSearch-independent evidence used by later lexical, dense, and fusion stages."""

    chunk_id: str
    tenant_id: UUID
    collection_id: UUID
    document_id: UUID
    document_content_id: str
    rank: int
    score: float
    content_text: str
    source_spans: tuple[SourceSpan, ...]
    generation_id: str
    pipeline_version: str
    source_metadata: dict[str, object]


@dataclass(frozen=True)
class RetrievalTrace:
    """Non-public diagnostic data kept separate from the evidence results."""

    parsed_query: ParsedLexicalQuery
    candidate_count: int
    elapsed_ms: float
    tenant_id: UUID
    collection_id: UUID | None
    filters: dict[str, str]
    boosts: BM25Config
    explanations: dict[str, object]


@dataclass(frozen=True)
class LexicalRetrievalResponse:
    """The lexical results and their independently inspectable execution trace."""

    results: tuple[RetrievalResult, ...]
    trace: RetrievalTrace


class OpenSearchBM25Retriever:
    """Run structured lexical queries only against ready, tenant-scoped chunks."""

    def __init__(
        self,
        base_url: str,
        read_alias: str,
        config: BM25Config | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OpenSearch base URL must be a nonblank string")
        if not isinstance(read_alias, str) or not read_alias.strip():
            raise ValueError("OpenSearch read alias must be a nonblank string")
        self.base_url = base_url
        self.read_alias = read_alias
        self.config = config or BM25Config()
        if not isinstance(self.config, BM25Config):
            raise ValueError("BM25 configuration must be a BM25Config")
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "OpenSearchBM25Retriever":
        return cls(
            settings.opensearch_url,
            settings.opensearch_read_alias,
            BM25Config.from_settings(settings),
            transport=transport,
        )

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
    ) -> LexicalRetrievalResponse:
        if not isinstance(tenant_id, UUID):
            raise ValueError("BM25 tenant ID must be a UUID")
        if collection_id is not None and not isinstance(collection_id, UUID):
            raise ValueError("BM25 collection ID must be a UUID when present")
        if not isinstance(debug, bool):
            raise ValueError("BM25 debug flag must be boolean")
        search_filters = RetrievalFilters(
            tenant_id,
            collection_id,
            source_type,
            author,
            source_date_from,
            source_date_to,
        )
        parsed = parse_lexical_query(query_text)
        candidate_limit = self._candidate_limit(limit)
        filters = search_filters.trace_values()
        if parsed.is_empty:
            return LexicalRetrievalResponse(
                (),
                RetrievalTrace(parsed, 0, 0.0, tenant_id, collection_id, filters, self.config, {}),
            )

        started = time.perf_counter()
        response = await self._request(
            "POST",
            f"/{self.read_alias}/_search",
            json=self._search_body(parsed, search_filters, candidate_limit, debug),
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000
        payload = response.json()
        if not isinstance(payload, dict):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        hits_value = payload.get("hits")
        if not isinstance(hits_value, dict):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        raw_hits = hits_value.get("hits")
        if not isinstance(raw_hits, list):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        results = tuple(
            self._result_from_hit(hit, rank) for rank, hit in enumerate(raw_hits, start=1)
        )
        return LexicalRetrievalResponse(
            results,
            RetrievalTrace(
                parsed,
                self._candidate_count(hits_value, len(results)),
                elapsed_ms,
                tenant_id,
                collection_id,
                filters,
                self.config,
                self._explanations(raw_hits) if debug else {},
            ),
        )

    def _candidate_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.config.candidate_limit
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("BM25 result limit must be a positive integer when present")
        return min(limit, self.config.candidate_limit)

    def _search_body(
        self,
        parsed: ParsedLexicalQuery,
        filters: RetrievalFilters,
        candidate_limit: int,
        debug: bool,
    ) -> dict[str, object]:
        clauses: list[dict[str, object]] = []
        if parsed.ordinary_terms:
            clauses.append(
                {
                    "multi_match": {
                        "query": " ".join(parsed.ordinary_terms),
                        "type": "best_fields",
                        "fields": [
                            f"content_text^{self.config.content_boost}",
                            f"heading_text^{self.config.heading_boost}",
                        ],
                    }
                }
            )
        for phrase in parsed.phrases:
            clauses.append(
                {
                    "multi_match": {
                        "query": phrase,
                        "type": "phrase",
                        "fields": [
                            f"content_text^{self.config.content_boost}",
                            f"heading_text^{self.config.heading_boost}",
                        ],
                        "boost": self.config.phrase_boost,
                    }
                }
            )
        if parsed.identifiers:
            clauses.append(
                {
                    "terms": {
                        "identifiers": list(parsed.identifiers),
                        "boost": self.config.identifier_boost,
                    }
                }
            )
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
                "bool": {
                    "filter": filters.clauses(),
                    "should": clauses,
                    "minimum_should_match": 1,
                }
            },
        }

    @staticmethod
    def _candidate_count(hits: dict[str, object], fallback: int) -> int:
        total = hits.get("total")
        if isinstance(total, dict):
            value = total.get("value")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
        return fallback

    @staticmethod
    def _explanations(raw_hits: list[object]) -> dict[str, object]:
        explanations: dict[str, object] = {}
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = hit.get("_id")
            explanation = hit.get("_explanation")
            if isinstance(chunk_id, str) and explanation is not None:
                explanations[chunk_id] = explanation
        return explanations

    @staticmethod
    def _result_from_hit(hit: object, rank: int) -> RetrievalResult:
        if not isinstance(hit, dict):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        source = hit.get("_source")
        score = hit.get("_score")
        if (
            not isinstance(source, dict)
            or not isinstance(score, (float, int))
            or isinstance(score, bool)
        ):
            raise RetrievalError("opensearch_invalid_response", retryable=False)
        try:
            source_spans = tuple(_source_span(span) for span in _source_spans(source))
            metadata = source.get("source_metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("source metadata must be an object")
            return RetrievalResult(
                chunk_id=_source_string(source, "chunk_id"),
                tenant_id=UUID(_source_string(source, "tenant_id")),
                collection_id=UUID(_source_string(source, "collection_id")),
                document_id=UUID(_source_string(source, "document_id")),
                document_content_id=_source_string(source, "document_content_id"),
                rank=rank,
                score=float(score),
                content_text=_source_string(source, "content_text"),
                source_spans=source_spans,
                generation_id=_source_string(source, "generation_id"),
                pipeline_version=_source_string(source, "pipeline_version"),
                source_metadata=metadata,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RetrievalError("opensearch_invalid_response", retryable=False) from error

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


def _source_string(source: dict[str, object], field: str) -> str:
    value = source[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def _source_spans(source: dict[str, object]) -> list[dict[str, object]]:
    values = source["source_spans"]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, dict) for value in values)
    ):
        raise ValueError("source spans must be a nonempty array")
    return values


def _source_span(value: dict[str, object]) -> SourceSpan:
    headings = value["heading_path"]
    if not isinstance(headings, list) or any(not isinstance(heading, str) for heading in headings):
        raise ValueError("source span heading path must be an array of strings")
    return SourceSpan(
        SourceLocator(
            block_number=_source_integer(value, "block_number"),
            page_number=_source_optional_integer(value, "page_number"),
            heading_path=tuple(headings),
        ),
        start_char=_source_integer(value, "start_char"),
        end_char=_source_integer(value, "end_char"),
    )


def _source_integer(source: dict[str, object], field: str) -> int:
    value = source[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _source_optional_integer(source: dict[str, object], field: str) -> int | None:
    value = source.get(field)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{field} must be an integer when present")
    return value


def main() -> None:
    """Run a local debug query without exposing a public search API yet."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, type=UUID)
    parser.add_argument("--collection", type=UUID)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    response = asyncio.run(
        OpenSearchBM25Retriever.from_settings(get_settings()).search(
            args.query,
            tenant_id=args.tenant,
            collection_id=args.collection,
            limit=args.limit,
            debug=True,
        )
    )
    print(response)


if __name__ == "__main__":  # pragma: no cover
    main()
