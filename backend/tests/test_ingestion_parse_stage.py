import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    encode_parsed_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    LocalArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.parse_stage import (
    ParseSource,
    ParseStageRunner,
    PostgresParseSourceLoader,
    SourceReadError,
    SourceReadErrorCode,
)
from hybrid_rag_search.models.content import DocumentStatus, IngestionStage
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.storage import storage_key


def claimed(content: bytes = b"original") -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash=hashlib.sha256(content).hexdigest(),
        pipeline_version="pipe_" + "b" * 64,
        stage=IngestionStage.PARSE,
        attempts=1,
        max_attempts=3,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=MagicMock(),
    )


def parsed() -> ParsedDocument:
    return ParsedDocument(
        blocks=(ParsedBlock("Parsed text", SourceLocator(1)),),
        media_type="text/plain",
        parser_name="test-parser",
        parser_version="1",
    )


def identity(job: ClaimedJob) -> str:
    return artifact_identity(ArtifactKind.PARSED, job.content_hash, job.pipeline_version)


def key(job: ClaimedJob) -> str:
    return artifact_key(job.tenant_id, job.id, ArtifactKind.PARSED, identity(job))


def runner(
    tmp_path: Path,
) -> tuple[ParseStageRunner, AsyncMock, MagicMock, LocalArtifactStorage, AsyncMock]:
    loader = AsyncMock()
    parser = MagicMock()
    artifacts = LocalArtifactStorage(tmp_path)
    checkpointer = AsyncMock()
    return (
        ParseStageRunner(loader, parser, artifacts, checkpointer),
        loader,
        parser,
        artifacts,
        checkpointer,
    )


@pytest.mark.anyio
async def test_missing_artifact_is_parsed_saved_and_checkpointed(tmp_path: Path) -> None:
    job = claimed()
    service, loader, parser, artifacts, checkpointer = runner(tmp_path)
    source = ParseSource(uuid4(), b"original", "text/plain")
    loader.load.return_value = source
    parser.parse.return_value = parsed()
    advanced = replace(job, stage=IngestionStage.CHUNK)
    checkpointer.advance_artifact.return_value = advanced

    assert await service.run(job) == advanced

    loader.load.assert_awaited_once_with(job)
    request = parser.parse.call_args.args[0]
    assert request.content == source.content
    assert request.media_type == source.media_type
    reference = checkpointer.advance_artifact.await_args.args[1]
    assert reference == artifacts.lookup(key(job))


@pytest.mark.anyio
async def test_existing_valid_artifact_skips_source_and_parser(tmp_path: Path) -> None:
    job = claimed()
    service, loader, parser, artifacts, checkpointer = runner(tmp_path)
    reference = artifacts.save(key(job), encode_parsed_artifact(parsed(), identity(job)))
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.CHUNK)

    await service.run(job)

    loader.load.assert_not_awaited()
    parser.parse.assert_not_called()
    checkpointer.advance_artifact.assert_awaited_once_with(job, reference)


@pytest.mark.anyio
async def test_existing_invalid_artifact_fails_without_overwrite(tmp_path: Path) -> None:
    job = claimed()
    service, loader, parser, artifacts, checkpointer = runner(tmp_path)
    reference = artifacts.save(key(job), b"not a valid artifact")

    with pytest.raises(ArtifactContractError, match="Invalid parsed"):
        await service.run(job)

    assert artifacts.fetch(reference) == b"not a valid artifact"
    loader.load.assert_not_awaited()
    parser.parse.assert_not_called()
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_lost_lease_is_reported_by_none_result(tmp_path: Path) -> None:
    job = claimed()
    service, loader, parser, _, checkpointer = runner(tmp_path)
    loader.load.return_value = ParseSource(uuid4(), b"original", "text/plain")
    parser.parse.return_value = parsed()
    checkpointer.advance_artifact.return_value = None
    assert await service.run(job) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "job",
    ["job", replace(claimed(), stage=IngestionStage.CHUNK)],
)
async def test_runner_requires_claimed_parse_stage(tmp_path: Path, job: object) -> None:
    service, loader, parser, _, checkpointer = runner(tmp_path)
    with pytest.raises(ValueError, match="Parse stage|ClaimedJob"):
        await service.run(job)  # type: ignore[arg-type]
    loader.load.assert_not_awaited()
    parser.parse.assert_not_called()
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collection_id", "id", "collection ID"),
        ("content", "text", "content"),
        ("media_type", " ", "media type"),
    ],
)
def test_parse_source_validation(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "collection_id": uuid4(),
        "content": b"content",
        "media_type": "text/plain",
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ParseSource(**values)  # type: ignore[arg-type]


def loader(returned: dict[str, object] | None, original: bytes | Exception = b"original"):
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = returned
    session = AsyncMock()
    session.execute.return_value = result
    sessions = MagicMock()
    sessions.return_value.__aenter__.return_value = session
    originals = MagicMock()
    originals.fetch.side_effect = original if isinstance(original, Exception) else None
    originals.fetch.return_value = None if isinstance(original, Exception) else original
    return PostgresParseSourceLoader(sessions, originals), session, originals


def document_row(job: ClaimedJob, content: bytes = b"original") -> dict[str, object]:
    return {
        "collection_id": uuid4(),
        "storage_key": storage_key(job.tenant_id, job.document_id),
        "media_type": "text/plain",
        "content_hash": hashlib.sha256(content).hexdigest(),
        "status": DocumentStatus.PENDING,
    }


@pytest.mark.anyio
async def test_loader_fetches_scoped_original_and_verifies_hash() -> None:
    job = claimed()
    row = document_row(job)
    service, session, originals = loader(row)

    source = await service.load(job)

    assert source == ParseSource(row["collection_id"], b"original", "text/plain")
    originals.fetch.assert_called_once_with(storage_key(job.tenant_id, job.document_id))
    sql = str(session.execute.await_args.args[0])
    assert "documents.id" in sql and "documents.tenant_id" in sql


@pytest.mark.anyio
async def test_loader_requires_claimed_job() -> None:
    service, session, _ = loader(None)
    with pytest.raises(ValueError, match="ClaimedJob"):
        await service.load("job")  # type: ignore[arg-type]
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_loader_rejects_missing_or_deleted_document() -> None:
    job = claimed()
    service, _, _ = loader(None)
    with pytest.raises(DocumentNotFound):
        await service.load(job)
    for status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
        row = document_row(job)
        row["status"] = status
        service, _, originals = loader(row)
        with pytest.raises(DocumentUnavailable):
            await service.load(job)
        originals.fetch.assert_not_called()


@pytest.mark.anyio
async def test_loader_rejects_wrong_storage_key_or_database_hash() -> None:
    job = claimed()
    row = document_row(job)
    row["storage_key"] = f"{uuid4()}/{job.document_id}"
    service, _, originals = loader(row)
    with pytest.raises(SourceReadError) as wrong_key:
        await service.load(job)
    assert wrong_key.value.code is SourceReadErrorCode.INVALID_STORAGE_KEY
    assert wrong_key.value.retryable is False
    originals.fetch.assert_not_called()

    row = document_row(job)
    row["content_hash"] = "f" * 64
    service, _, originals = loader(row)
    with pytest.raises(SourceReadError) as wrong_hash:
        await service.load(job)
    assert wrong_hash.value.code is SourceReadErrorCode.CONTENT_HASH_MISMATCH
    assert wrong_hash.value.retryable is False
    originals.fetch.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (FileNotFoundError(), SourceReadErrorCode.ORIGINAL_MISSING),
        (OSError(), SourceReadErrorCode.STORAGE_UNAVAILABLE),
    ],
)
async def test_loader_classifies_storage_failures(
    error: Exception, code: SourceReadErrorCode
) -> None:
    job = claimed()
    service, _, _ = loader(document_row(job), error)
    with pytest.raises(SourceReadError) as caught:
        await service.load(job)
    assert caught.value.code is code
    assert caught.value.retryable is True


@pytest.mark.anyio
async def test_loader_rejects_original_bytes_with_wrong_hash() -> None:
    job = claimed()
    service, _, _ = loader(document_row(job), b"changed")
    with pytest.raises(SourceReadError) as caught:
        await service.load(job)
    assert caught.value.code is SourceReadErrorCode.CONTENT_HASH_MISMATCH
    assert caught.value.retryable is False
