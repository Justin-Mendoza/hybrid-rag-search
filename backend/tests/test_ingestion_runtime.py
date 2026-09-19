from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.ingestion.indexing import FakeDocumentIndex
from hybrid_rag_search.ingestion.pipeline import JobRunResult
from hybrid_rag_search.ingestion.recovery import RecoveryRunResult
from hybrid_rag_search.ingestion.runtime import (
    configured_pipeline_version,
    run_configured_job,
    run_configured_recovery,
)


def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql://user:pass@localhost/database",
        storage_root=tmp_path / "originals",
        tokenizer_root=tmp_path / "tokenizers",
        ingestion_artifact_root=tmp_path / "artifacts",
        cohere_api_key="test-key",
    )


def runtime_mocks(monkeypatch: pytest.MonkeyPatch):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    sessions = MagicMock()
    monkeypatch.setattr("hybrid_rag_search.ingestion.runtime.create_async_engine", lambda _: engine)
    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.runtime.async_sessionmaker", lambda *_, **__: sessions
    )
    return engine, sessions


def test_configured_pipeline_version_is_stable_and_configuration_sensitive(tmp_path) -> None:
    first = settings(tmp_path)
    assert configured_pipeline_version(first) == configured_pipeline_version(first)
    changed = first.model_copy(update={"cohere_embed_model": "embed-other"})
    assert configured_pipeline_version(changed) != configured_pipeline_version(first)


@pytest.mark.anyio
async def test_configured_job_composes_pipeline_and_disposes_engine(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = runtime_mocks(monkeypatch)
    provider = MagicMock()

    @asynccontextmanager
    async def open_provider(_settings):
        yield provider

    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.runtime.configured_cohere_embeddings", open_provider
    )
    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.runtime.load_configured_tokenizer", MagicMock()
    )
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=JobRunResult.SUCCEEDED)
    pipeline_factory = MagicMock(return_value=pipeline)
    monkeypatch.setattr("hybrid_rag_search.ingestion.runtime.IngestionPipeline", pipeline_factory)

    job_id = uuid4()
    assert (
        await run_configured_job(
            settings(tmp_path),
            MagicMock(),
            FakeDocumentIndex(),
            job_id,
        )
        is JobRunResult.SUCCEEDED
    )
    pipeline.run.assert_awaited_once_with(job_id)
    assert len(pipeline_factory.call_args.args) == 7
    engine.dispose.assert_awaited_once_with()


@pytest.mark.anyio
async def test_configured_job_disposes_engine_when_provider_setup_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = runtime_mocks(monkeypatch)

    @asynccontextmanager
    async def failing_provider(_settings):
        raise ValueError("missing key")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.runtime.configured_cohere_embeddings",
        failing_provider,
    )
    with pytest.raises(ValueError, match="missing key"):
        await run_configured_job(
            settings(tmp_path),
            MagicMock(),
            FakeDocumentIndex(),
            uuid4(),
        )
    engine.dispose.assert_awaited_once_with()


@pytest.mark.anyio
async def test_configured_recovery_composes_services_and_disposes_engine(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = runtime_mocks(monkeypatch)
    recovery = MagicMock()
    recovery.run = AsyncMock(return_value=RecoveryRunResult(1, 2, 3))
    recovery_factory = MagicMock(return_value=recovery)
    monkeypatch.setattr("hybrid_rag_search.ingestion.runtime.IngestionRecovery", recovery_factory)
    assert await run_configured_recovery(settings(tmp_path), MagicMock()) == RecoveryRunResult(
        1, 2, 3
    )
    assert len(recovery_factory.call_args.args) == 2
    engine.dispose.assert_awaited_once_with()


@pytest.mark.anyio
async def test_configured_recovery_disposes_engine_after_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = runtime_mocks(monkeypatch)
    recovery = MagicMock()
    recovery.run = AsyncMock(side_effect=RuntimeError("database failed"))
    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.runtime.IngestionRecovery",
        MagicMock(return_value=recovery),
    )
    with pytest.raises(RuntimeError, match="database failed"):
        await run_configured_recovery(settings(tmp_path), MagicMock())
    engine.dispose.assert_awaited_once_with()
