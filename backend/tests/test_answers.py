"""Grounding fixtures use deterministic provider responses, never paid calls."""

from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hybrid_rag_search import answers
from hybrid_rag_search.answers import (
    AnswerEvidence,
    AnswerService,
    select_evidence,
    validate_answer,
)
from hybrid_rag_search.authorization import AuthorizedScope
from hybrid_rag_search.chunking.contracts import SourceSpan
from hybrid_rag_search.config import Settings
from hybrid_rag_search.lexical_retrieval import RetrievalError
from hybrid_rag_search.parsers.contracts import SourceLocator
from hybrid_rag_search.providers.errors import ProviderError
from hybrid_rag_search.providers.generation import (
    GenerationCitation,
    GenerationDocument,
    GenerationFinishReason,
    GenerationMessage,
    GenerationRequest,
    GenerationResult,
    MessageRole,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def scope():
    return AuthorizedScope(uuid4(), uuid4(), (uuid4(),))


def evidence(scope, chunk_id="chunk-a", text="Upload retries reuse the same key."):
    return AnswerEvidence(
        chunk_id=chunk_id,
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_ids[0],
        document_id=uuid4(),
        document_content_id="doc_" + "a" * 64,
        content_text=text,
        source_spans=(SourceSpan(SourceLocator(1, 2, ("Retries",)), 0, len(text)),),
        generation_id="gen-a",
        pipeline_version="pipeline-a",
    )


class WordTokenizer:
    def tokenize(self, text):
        return tuple(text.split())


def generated(body="Retries reuse the same key.", status="answered", sources=("chunk-a",)):
    prefix = f"STATUS: {status}\n"
    return GenerationResult(
        prefix + body,
        GenerationFinishReason.COMPLETE,
        "fixture-chat",
        "fake",
        citations=(GenerationCitation(len(prefix), len(prefix) + len(body), body, sources),),
    )


def test_evidence_budget_ranking_deduplication_and_scope(scope):
    a, b, c = (
        evidence(scope, name, text)
        for name, text in [
            ("chunk-a", "one two three"),
            ("chunk-b", "four five"),
            ("chunk-c", "six"),
        ]
    )
    selected = select_evidence([a, a, b, c], scope, WordTokenizer(), max_chunks=2, max_tokens=4)
    assert selected == (a, c)
    assert select_evidence([a, b, c], scope, WordTokenizer(), max_chunks=1, max_tokens=9) == (a,)
    for invalid in [
        a.model_copy(update={"tenant_id": uuid4()}),
        a.model_copy(update={"collection_id": uuid4()}),
    ]:
        with pytest.raises(ValueError, match="scope_violation"):
            select_evidence([invalid], scope, WordTokenizer(), max_chunks=1, max_tokens=9)
    with pytest.raises(ValueError, match="scope_violation"):
        select_evidence(
            [a],
            replace(scope, excluded_document_ids=(a.document_id,)),
            WordTokenizer(),
            max_chunks=1,
            max_tokens=9,
        )
    with pytest.raises(ValueError, match="ambiguous"):
        select_evidence(
            [a, a.model_copy(update={"content_text": "different"})],
            scope,
            WordTokenizer(),
            max_chunks=1,
            max_tokens=9,
        )
    with pytest.raises(ValueError, match="empty_evidence"):
        select_evidence(
            [a.model_copy(update={"content_text": " "})],
            scope,
            WordTokenizer(),
            max_chunks=1,
            max_tokens=9,
        )


def test_native_citation_resolution_and_json_snapshot(scope):
    item = evidence(scope)
    result = validate_answer(generated(), (item,))
    assert result.outcome == "answered"
    assert result.citations[0].start == 0
    assert (
        result.text[result.citations[0].start : result.citations[0].end] == result.citations[0].text
    )
    assert result.evidence[0].source_spans[0].locator.page_number == 2
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_multi_source_and_conflicting_evidence(scope, status):
    a = evidence(scope, "chunk-a", "Receipts last 24 hours.")
    b = evidence(scope, "chunk-b", "Receipts last 48 hours.")
    text = "The sources disagree: one says 24 hours and the other says 48 hours."
    result = validate_answer(generated(text, status, ("chunk-a", "chunk-b")), (a, b))
    assert result.outcome == status
    assert result.citations[0].source_ids == ("chunk-a", "chunk-b")
    assert result.evidence == (a, b)


@pytest.mark.parametrize(
    "citation",
    [
        GenerationCitation(0, 3, "STA", ("chunk-a",)),
        GenerationCitation(17, 9999, "x", ("chunk-a",)),
        GenerationCitation(17, 18, "wrong", ("chunk-a",)),
        GenerationCitation(17, 18, "R", ()),
        GenerationCitation(17, 18, "R", ("unsupplied",)),
        GenerationCitation(True, 18, "R", ("chunk-a",)),
        GenerationCitation(18, 17, "", ("chunk-a",)),
        GenerationCitation(24, 25, " ", ("chunk-a",)),
    ],
)
def test_invalid_provider_citations_never_expose_answer(scope, citation):
    result = validate_answer(replace(generated(), citations=(citation,)), (evidence(scope),))
    assert result.outcome == "failed"
    assert result.error_code == "invalid_citation"
    assert result.text == "" and result.citations == () and result.evidence == ()


def test_missing_status_empty_answer_missing_citations_abstention_and_truncation(scope):
    item = evidence(scope)
    assert (
        validate_answer(replace(generated(), text="No header"), (item,)).error_code
        == "invalid_answer_status"
    )
    assert (
        validate_answer(replace(generated(), text="STATUS: answered\n "), (item,)).error_code
        == "empty_answer"
    )
    assert (
        validate_answer(replace(generated(), citations=()), (item,)).error_code
        == "missing_citations"
    )
    abstain = validate_answer(
        replace(generated(status="insufficient_evidence"), citations=()), (item,)
    )
    assert abstain.outcome == "insufficient_evidence" and abstain.citations == ()
    assert "Retries" not in abstain.text
    partial = validate_answer(
        replace(generated(), finish_reason=GenerationFinishReason.MAX_TOKENS), (item,)
    )
    assert partial.outcome == "partial"
    assert partial.text.endswith("This answer was cut short by the output limit.")
    duplicate = validate_answer(generated(sources=("chunk-a", "chunk-a")), (item,))
    assert duplicate.citations[0].source_ids == ("chunk-a",)


@pytest.mark.anyio
async def test_answer_service_request_budgets_zero_call_abstention_and_failures(scope):
    item = evidence(scope)
    retrieval = SimpleNamespace(search=AsyncMock(return_value={"chunks": [item.model_dump()]}))
    provider = SimpleNamespace(generate=AsyncMock(return_value=generated()))
    settings = Settings(_env_file=None)
    service = AnswerService(settings, retrieval, provider, WordTokenizer())
    result = await service.answer(scope, "How do retries work?")
    assert result.outcome == "answered"
    request = provider.generate.call_args.args[0]
    assert request.documents == (GenerationDocument(item.chunk_id, item.content_text),)
    assert request.max_output_tokens == 512
    assert "untrusted" in request.messages[0].content
    assert result.configuration["evidence_token_budget"] == 2000
    assert retrieval.search.call_args.args == (scope, "How do retries work?", "hybrid_rerank")
    provider.generate.side_effect = ProviderError("rate_limited")
    assert (await service.answer(scope, "Question")).error_code == "rate_limited"
    provider.generate.reset_mock()
    retrieval.search.return_value = {"chunks": []}
    assert (await service.answer(scope, "Question")).outcome == "insufficient_evidence"
    provider.generate.assert_not_called()
    retrieval.search.side_effect = RetrievalError("index_unavailable", retryable=True)
    assert (await service.answer(scope, "Question")).error_code == "index_unavailable"
    provider.generate.assert_not_called()
    for invalid in [" ", "x" * 4097]:
        with pytest.raises(ValueError, match="Question"):
            await service.answer(scope, invalid)


@pytest.mark.anyio
async def test_configured_answer_service_owns_runtime_resources(monkeypatch):
    settings = Settings(_env_file=None)
    resources = []

    @asynccontextmanager
    async def resource(value):
        resources.append("enter")
        yield value
        resources.append("exit")

    monkeypatch.setattr(answers, "load_local_tokenizer", lambda **kwargs: WordTokenizer())
    monkeypatch.setattr(answers, "AuthorizedSearch", lambda settings: resource("retrieval"))
    monkeypatch.setattr(
        answers, "configured_cohere_generation", lambda settings: resource("provider")
    )
    async with answers.configured_answer_service(settings) as service:
        assert service.provider == "provider" and service.retrieval == "retrieval"
    assert resources == ["enter", "enter", "exit", "exit"]


@pytest.mark.parametrize(
    "documents",
    [
        [GenerationDocument("a", "text")],
        ("bad",),
        (GenerationDocument("a", "one"), GenerationDocument("a", "two")),
    ],
)
def test_generation_document_contract_rejects_invalid_evidence(documents):
    with pytest.raises(ValueError):
        GenerationRequest((GenerationMessage(MessageRole.USER, "question"),), 20, documents)


@pytest.mark.parametrize("id,text", [("", "text"), ("a", " ")])
def test_generation_document_requires_nonblank_values(id, text):
    with pytest.raises(ValueError):
        GenerationDocument(id, text)
