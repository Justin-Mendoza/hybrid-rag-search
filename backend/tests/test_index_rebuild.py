import sys
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search import index_rebuild
from hybrid_rag_search.config import Settings
from hybrid_rag_search.index_rebuild import RebuildStartResult


def dependencies(monkeypatch: pytest.MonkeyPatch, *, documents=(), unfinished=0):
    manager = MagicMock()
    manager.create = AsyncMock()
    manager.point_write_alias = AsyncMock()
    manager.promote_read_alias = AsyncMock()
    monkeypatch.setattr(index_rebuild, "OpenSearchIndexManager", MagicMock(return_value=manager))
    engine = MagicMock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr(index_rebuild, "create_async_engine", MagicMock(return_value=engine))
    rows = MagicMock()
    rows.all.return_value = documents
    session = AsyncMock()
    session.execute.return_value = rows
    session.scalar.return_value = unfinished
    sessions = MagicMock()
    sessions.return_value.__aenter__.return_value = session
    monkeypatch.setattr(index_rebuild, "async_sessionmaker", MagicMock(return_value=sessions))
    jobs = MagicMock()
    jobs.enqueue_document = AsyncMock()
    monkeypatch.setattr(index_rebuild, "IngestionJobService", MagicMock(return_value=jobs))
    monkeypatch.setattr(index_rebuild, "DramatiqJobPublisher", MagicMock())
    monkeypatch.setattr(
        index_rebuild, "configured_pipeline_version", MagicMock(return_value="pipe_" + "a" * 64)
    )
    return manager, engine, jobs


@pytest.mark.anyio
async def test_start_rebuild_creates_target_and_enqueues_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = ((uuid4(), uuid4()), (uuid4(), uuid4()))
    manager, engine, jobs = dependencies(monkeypatch, documents=documents)
    result = await index_rebuild.start_rebuild(Settings(), "build-1")
    assert result == RebuildStartResult("hybrid-rag-chunks-v2-build-1", 2)
    manager.create.assert_awaited_once_with(result.index_name)
    manager.point_write_alias.assert_awaited_once_with(result.index_name)
    assert jobs.enqueue_document.await_count == 2
    engine.dispose.assert_awaited_once_with()


@pytest.mark.anyio
async def test_promote_requires_all_documents_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, engine, _ = dependencies(monkeypatch, unfinished=1)
    with pytest.raises(RuntimeError, match="1 documents"):
        await index_rebuild.promote_rebuild(Settings(), "hybrid-rag-chunks-v1-build-1")
    manager.promote_read_alias.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()

    manager, _, _ = dependencies(monkeypatch, unfinished=0)
    await index_rebuild.promote_rebuild(Settings(), "hybrid-rag-chunks-v1-build-1")
    manager.promote_read_alias.assert_awaited_once_with("hybrid-rag-chunks-v1-build-1")


def test_cli_start_and_promote(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = iter((RebuildStartResult("index", 3), None))

    def finish(coroutine):
        coroutine.close()
        return next(results)

    run = MagicMock(side_effect=finish)
    monkeypatch.setattr(index_rebuild.asyncio, "run", run)
    monkeypatch.setattr(sys, "argv", ["index-rebuild", "start", "build"])
    index_rebuild.main()
    assert "index=index enqueued=3" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["index-rebuild", "promote", "index"])
    index_rebuild.main()
    assert "index=index" in capsys.readouterr().out
