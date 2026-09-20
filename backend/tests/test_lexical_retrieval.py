import json
import sys
from uuid import UUID, uuid4

import httpx
import pytest

from hybrid_rag_search import lexical_retrieval
from hybrid_rag_search.config import Settings
from hybrid_rag_search.lexical_retrieval import (
    BM25Config,
    OpenSearchBM25Retriever,
    RetrievalError,
    parse_lexical_query,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def hit(
    tenant_id: UUID,
    collection_id: UUID,
    document_id: UUID,
    *,
    score: float = 4.5,
    metadata: object = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "chunk_id": "chk_" + "a" * 64,
        "tenant_id": str(tenant_id),
        "collection_id": str(collection_id),
        "document_id": str(document_id),
        "document_content_id": "doc_" + "b" * 64,
        "content_text": "Password reset links expire after fifteen minutes.",
        "source_spans": [
            {
                "block_number": 2,
                "page_number": 3,
                "heading_path": ["Authentication", "Password reset"],
                "start_char": 0,
                "end_char": 49,
            }
        ],
        "generation_id": "idxgen_" + "c" * 64,
        "pipeline_version": "pipe_" + "d" * 64,
    }
    if metadata is not None:
        source["source_metadata"] = metadata
    return {"_id": "internal-hit-id", "_score": score, "_source": source}


def test_parser_separates_terms_phrases_and_identifiers() -> None:
    parsed = parse_lexical_query('HR-104 "Password reset" timeout ERR_AUTH_42 timeout')
    assert parsed.ordinary_terms == ("timeout",)
    assert parsed.phrases == ("password reset",)
    assert parsed.identifiers == ("hr-104", "err_auth_42")
    assert parse_lexical_query('"" !!!').is_empty
    assert parse_lexical_query('unclosed "quote').ordinary_terms == ("unclosed", "quote")
    with pytest.raises(ValueError, match="string"):
        parse_lexical_query(None)  # type: ignore[arg-type]


def test_config_and_retriever_validate_inputs() -> None:
    for kwargs in (
        {"content_boost": 0},
        {"heading_boost": True},
        {"phrase_boost": -1},
        {"identifier_boost": "high"},
        {"candidate_limit": 0},
    ):
        with pytest.raises(ValueError):
            BM25Config(**kwargs)  # type: ignore[arg-type]
    assert BM25Config.from_settings(Settings()).heading_boost == 1.5
    with pytest.raises(ValueError, match="base URL"):
        OpenSearchBM25Retriever("", "read")
    with pytest.raises(ValueError, match="read alias"):
        OpenSearchBM25Retriever("http://search.test", "")
    with pytest.raises(ValueError, match="configuration"):
        OpenSearchBM25Retriever("http://search.test", "read", config="wrong")  # type: ignore[arg-type]
    assert OpenSearchBM25Retriever.from_settings(Settings()).read_alias == "hybrid-rag-chunks-read"
    with pytest.raises(ValueError, match="snake_case"):
        RetrievalError("Not snake case", retryable=True)


@pytest.mark.anyio
async def test_search_builds_scoped_structured_query_and_normalizes_response() -> None:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    requests: list[httpx.Request] = []

    async def responder(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = hit(tenant_id, collection_id, document_id, metadata={"author": "Ada"})
        item["_explanation"] = {"description": "test explanation"}
        return httpx.Response(200, json={"hits": {"total": {"value": 7}, "hits": [item]}})

    retriever = OpenSearchBM25Retriever(
        "http://search.test",
        "chunks-read",
        BM25Config(candidate_limit=3),
        transport=httpx.MockTransport(responder),
    )
    response = await retriever.search(
        'HR-104 "password reset" timeout',
        tenant_id=tenant_id,
        collection_id=collection_id,
        limit=99,
        debug=True,
    )

    assert len(response.results) == 1
    result = response.results[0]
    assert result.rank == 1
    assert result.score == 4.5
    assert result.tenant_id == tenant_id
    assert result.source_spans[0].locator.heading_path == ("Authentication", "Password reset")
    assert result.source_metadata == {"author": "Ada"}
    assert response.trace.candidate_count == 7
    assert response.trace.elapsed_ms >= 0
    assert response.trace.filters == {
        "tenant_id": str(tenant_id),
        "collection_id": str(collection_id),
        "visibility": "ready",
    }
    assert response.trace.explanations == {"internal-hit-id": {"description": "test explanation"}}

    assert requests[0].url.path == "/chunks-read/_search"
    body = json.loads(requests[0].content)
    assert body["size"] == 3
    assert body["explain"] is True
    filters = body["query"]["bool"]["filter"]
    assert {next(iter(item["term"])) for item in filters} == {
        "tenant_id",
        "collection_id",
        "visibility",
    }
    clauses = body["query"]["bool"]["should"]
    assert clauses[0]["multi_match"]["fields"] == ["content_text^1.0", "heading_text^1.5"]
    assert clauses[1]["multi_match"]["type"] == "phrase"
    assert clauses[2]["terms"]["identifiers"] == ["hr-104"]


@pytest.mark.anyio
async def test_search_handles_empty_query_no_collection_and_limits() -> None:
    tenant_id = uuid4()
    calls = 0

    async def responder(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"hits": {"total": 0, "hits": []}})

    retriever = OpenSearchBM25Retriever(
        "http://search.test", "chunks-read", transport=httpx.MockTransport(responder)
    )
    empty = await retriever.search("   ", tenant_id=tenant_id)
    assert empty.results == ()
    assert empty.trace.candidate_count == 0
    assert calls == 0

    no_results = await retriever.search("password", tenant_id=tenant_id, limit=1)
    assert no_results.results == ()
    assert no_results.trace.candidate_count == 0
    assert calls == 1
    with pytest.raises(ValueError, match="result limit"):
        await retriever.search("password", tenant_id=tenant_id, limit=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tenant"):
        await retriever.search("password", tenant_id="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection"):
        await retriever.search("password", tenant_id=tenant_id, collection_id="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="debug"):
        await retriever.search("password", tenant_id=tenant_id, debug=1)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_search_classifies_transport_response_and_payload_errors() -> None:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()

    async def disconnected(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    retriever = OpenSearchBM25Retriever(
        "http://search.test", "read", transport=httpx.MockTransport(disconnected)
    )
    with pytest.raises(RetrievalError, match="opensearch_unavailable") as unavailable:
        await retriever.search("password", tenant_id=tenant_id)
    assert unavailable.value.retryable

    for status, retryable in ((400, False), (503, True)):

        async def bad_status(_: httpx.Request, status: int = status) -> httpx.Response:
            return httpx.Response(status)

        retriever = OpenSearchBM25Retriever(
            "http://search.test", "read", transport=httpx.MockTransport(bad_status)
        )
        with pytest.raises(RetrievalError, match="opensearch_unavailable") as error:
            await retriever.search("password", tenant_id=tenant_id)
        assert error.value.retryable is retryable

    invalid_payloads: list[object] = [[], {}, {"hits": {}}, {"hits": {"hits": {}}}]
    for payload in invalid_payloads:

        async def bad_payload(_: httpx.Request, payload: object = payload) -> httpx.Response:
            return httpx.Response(200, json=payload)

        retriever = OpenSearchBM25Retriever(
            "http://search.test", "read", transport=httpx.MockTransport(bad_payload)
        )
        with pytest.raises(RetrievalError, match="opensearch_invalid_response") as error:
            await retriever.search("password", tenant_id=tenant_id)
        assert not error.value.retryable

    malformed = hit(tenant_id, collection_id, document_id, metadata=[])

    async def malformed_hit(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": {"total": "unknown", "hits": [malformed]}})

    retriever = OpenSearchBM25Retriever(
        "http://search.test", "read", transport=httpx.MockTransport(malformed_hit)
    )
    with pytest.raises(RetrievalError, match="opensearch_invalid_response"):
        await retriever.search("password", tenant_id=tenant_id)


def test_response_helpers_cover_fallbacks_and_malformed_hits() -> None:
    assert OpenSearchBM25Retriever._candidate_count({"total": 4}, 1) == 4
    assert OpenSearchBM25Retriever._candidate_count({"total": {"value": 2}}, 1) == 2
    assert OpenSearchBM25Retriever._candidate_count({"total": True}, 3) == 3
    assert OpenSearchBM25Retriever._explanations(["bad", {"_id": 2}, {"_id": "ok"}]) == {}
    with pytest.raises(RetrievalError, match="opensearch_invalid_response"):
        OpenSearchBM25Retriever._result_from_hit("bad", 1)
    with pytest.raises(RetrievalError, match="opensearch_invalid_response"):
        OpenSearchBM25Retriever._result_from_hit({"_source": {}, "_score": True}, 1)


def test_hit_decoding_rejects_invalid_source_values() -> None:
    tenant_id, collection_id, document_id = uuid4(), uuid4(), uuid4()
    for mutate in (
        lambda value: value["_source"].update(chunk_id=""),  # type: ignore[union-attr]
        lambda value: value["_source"].update(source_spans=[]),  # type: ignore[union-attr]
        lambda value: value["_source"]["source_spans"][0].update(heading_path=[1]),  # type: ignore[index,union-attr]
        lambda value: value["_source"]["source_spans"][0].update(block_number=True),  # type: ignore[index,union-attr]
        lambda value: value["_source"]["source_spans"][0].update(page_number=True),  # type: ignore[index,union-attr]
    ):
        value = hit(tenant_id, collection_id, document_id)
        mutate(value)
        with pytest.raises(RetrievalError, match="opensearch_invalid_response"):
            OpenSearchBM25Retriever._result_from_hit(value, 1)


def test_debug_cli_prints_response(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tenant_id = uuid4()
    expected = object()

    class FakeRetriever:
        async def search(self, *args: object, **kwargs: object) -> object:
            return expected

    monkeypatch.setattr(
        lexical_retrieval.OpenSearchBM25Retriever, "from_settings", lambda _: FakeRetriever()
    )
    monkeypatch.setattr(sys, "argv", ["bm25", "password", "--tenant", str(tenant_id)])
    lexical_retrieval.main()
    assert repr(expected) in capsys.readouterr().out
