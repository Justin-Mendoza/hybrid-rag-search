import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4, uuid5

import httpx
import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.evaluation import corpus
from hybrid_rag_search.ingestion.pipeline import JobRunResult
from hybrid_rag_search.models.content import IngestionJob


def database_mocks(monkeypatch: pytest.MonkeyPatch):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    session.add = MagicMock()
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    sessions.return_value.__aenter__.return_value = session
    monkeypatch.setattr(corpus, "create_async_engine", lambda _: engine)
    monkeypatch.setattr(corpus, "async_sessionmaker", lambda *args, **kwargs: sessions)
    return engine, session


@pytest.mark.anyio
async def test_prepare_reuses_durable_jobs_and_detects_document_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = corpus.selected_dataset("synthetic-workspace")
    dataset = dataset.model_copy(update={"corpus": dataset.corpus[:1]})
    settings = Settings(_env_file=None, storage_root=tmp_path)
    engine, session = database_mocks(monkeypatch)
    session.get.return_value = None
    monkeypatch.setattr(corpus, "ensure_evaluation_index", AsyncMock())
    monkeypatch.setattr(corpus, "compact_corpus", AsyncMock())
    verified = AsyncMock(return_value=1)
    monkeypatch.setattr(corpus, "verify_prepared", verified)
    jobs = MagicMock()
    jobs.enqueue_document = AsyncMock(return_value=MagicMock(status="queued", job_id=uuid4()))
    monkeypatch.setattr(corpus, "IngestionJobService", lambda *args: jobs)
    run = AsyncMock(return_value=JobRunResult.SUCCEEDED)
    monkeypatch.setattr(corpus, "run_configured_job", run)
    monkeypatch.setattr(corpus.asyncio, "sleep", AsyncMock())
    scope = await corpus.prepare_corpus(dataset, settings, document_delay=1)
    saved = session.add.call_args.args[0]
    assert saved.id == uuid5(scope.collection_id, dataset.corpus[0].id)
    assert saved.source_metadata["evaluation_id"] == dataset.corpus[0].id
    assert (
        corpus.LocalFileStorage(tmp_path).fetch(saved.storage_key).startswith(b"Expense Policy v3")
    )
    engine.dispose.assert_awaited_once()
    session.get.return_value = saved
    jobs.enqueue_document.return_value.status = "succeeded"
    run.reset_mock()
    session.add.reset_mock()
    await corpus.prepare_corpus(dataset, settings)
    run.assert_not_awaited()
    session.add.assert_not_called()
    saved.content_hash = "wrong"
    with pytest.raises(ValueError, match="differs"):
        await corpus.prepare_corpus(dataset, settings)
    saved.content_hash = hashlib.sha256(
        corpus.LocalFileStorage(tmp_path).fetch(saved.storage_key)
    ).hexdigest()
    jobs.enqueue_document.return_value.status = "queued"
    run.return_value = JobRunResult.RETRY_SCHEDULED
    with pytest.raises(ValueError, match="retry_scheduled"):
        await corpus.prepare_corpus(dataset, settings)
    with pytest.raises(ValueError, match="nonnegative"):
        await corpus.prepare_corpus(dataset, settings, document_delay=-1)
    actor = corpus.ManualActor()
    assert actor.send("job") == "job"
    assert actor.send_with_options(args=("job",), delay=1) == ("job",)
    await corpus.ManualPublisher().publish(uuid4())


@pytest.mark.anyio
async def test_verification_requires_complete_jobs_and_matching_index_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = corpus.selected_dataset("synthetic-workspace")
    settings = Settings(_env_file=None)
    scope = corpus.evaluation_scope(dataset, settings)
    engine, session = database_mocks(monkeypatch)
    result = MagicMock()
    session.execute.return_value = result
    jobs = [
        IngestionJob(
            document_id=UUID(key),
            checkpoint={"index": {"generation_id": "generation", "chunk_count": 2}},
        )
        for key in scope.document_ids
    ]
    result.scalars.return_value.all.return_value = []
    with pytest.raises(ValueError, match="not prepared"):
        await corpus.verify_prepared(dataset, settings)
    result.scalars.return_value.all.return_value = jobs
    jobs[0].checkpoint = {}
    with pytest.raises(ValueError, match="receipt"):
        await corpus.verify_prepared(dataset, settings)
    jobs[0].checkpoint = {"index": {"generation_id": "generation", "chunk_count": 2}}
    buckets = [
        {"key": key, "doc_count": 2, "generations": {"buckets": [{"key": "generation"}]}}
        for key in scope.document_ids
    ]
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(scope.collection_id) in request.content.decode()
        return httpx.Response(200, json={"aggregations": {"documents": {"buckets": buckets}}})

    monkeypatch.setattr(
        corpus.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    assert await corpus.verify_prepared(dataset, settings) == 16
    buckets[0]["generations"]["buckets"] = [{"key": "stale"}]
    with pytest.raises(ValueError, match="differs"):
        await corpus.verify_prepared(dataset, settings)
    assert engine.dispose.await_count == 4


@pytest.mark.anyio
async def test_evaluation_index_creation_is_isolated_and_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = corpus.selected_dataset("synthetic-workspace")
    settings = corpus.evaluation_index_settings(dataset, Settings(_env_file=None))
    manager = MagicMock()
    manager.create = AsyncMock()
    manager.point_write_alias = AsyncMock()
    manager.promote_read_alias = AsyncMock()
    monkeypatch.setattr(corpus, "OpenSearchIndexManager", lambda *args: manager)
    response_code = 404
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        corpus.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(response_code))
        ),
    )
    await corpus.ensure_evaluation_index(dataset, settings)
    assert manager.create.await_count == 1
    response_code = 200
    await corpus.ensure_evaluation_index(dataset, settings)
    assert manager.create.await_count == 1
    assert settings.opensearch_read_alias.startswith("hybrid-rag-eval-")
    await corpus.compact_corpus(dataset, settings)
