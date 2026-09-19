import hashlib
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.chunking.contracts import ChunkingConfig
from hybrid_rag_search.chunking.identity import DocumentContentIdentity
from hybrid_rag_search.documents import DocumentNotFound, DocumentUnavailable
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    decode_chunks_artifact,
    encode_chunks_artifact,
    encode_parsed_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactRef,
    LocalArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.chunk_stage import (
    ChunkStageRunner,
    PostgresDocumentIdentityResolver,
)
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.models.content import DocumentStatus, IngestionStage
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.tokenization import TokenSpan

CONFIG = ChunkingConfig(max_tokens=4, overlap_tokens=1)


class WordTokenizer:
    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(
            TokenSpan(index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\S+", text))
        )


def claimed() -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash=hashlib.sha256(b"original").hexdigest(),
        pipeline_version="pipe_" + "b" * 64,
        stage=IngestionStage.CHUNK,
        attempts=1,
        max_attempts=3,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=MagicMock(),
    )


def parsed() -> ParsedDocument:
    return ParsedDocument(
        blocks=(ParsedBlock("one two three four five", SourceLocator(1)),),
        media_type="text/plain",
        parser_name="plain-text",
        parser_version="1",
    )


def parsed_identity(job: ClaimedJob) -> str:
    return artifact_identity(ArtifactKind.PARSED, job.content_hash, job.pipeline_version)


def prepare_parsed(storage: LocalArtifactStorage, job: ClaimedJob) -> ClaimedJob:
    identity = parsed_identity(job)
    key = artifact_key(job.tenant_id, job.id, ArtifactKind.PARSED, identity)
    reference = storage.save(key, encode_parsed_artifact(parsed(), identity))
    return replace(job, checkpoint={"parsed": reference.checkpoint_value()})


def runner(tmp_path: Path, document_id: str):
    resolver = AsyncMock()
    resolver.resolve.return_value = document_id
    tokenizer = WordTokenizer()
    artifacts = LocalArtifactStorage(tmp_path)
    checkpointer = AsyncMock()
    service = ChunkStageRunner(
        resolver,
        tokenizer,
        artifacts,
        checkpointer,
        config=CONFIG,
    )
    return service, resolver, tokenizer, artifacts, checkpointer


@pytest.mark.anyio
async def test_missing_chunks_are_built_saved_and_checkpointed(tmp_path: Path) -> None:
    document_id = "doc_" + "d" * 64
    service, resolver, _, artifacts, checkpointer = runner(tmp_path, document_id)
    job = prepare_parsed(artifacts, claimed())
    advanced = replace(job, stage=IngestionStage.EMBED)
    checkpointer.advance_artifact.return_value = advanced

    assert await service.run(job) == advanced

    resolver.resolve.assert_awaited_once_with(job)
    reference = checkpointer.advance_artifact.await_args.args[1]
    identity = artifact_identity(
        ArtifactKind.CHUNKS,
        ArtifactRef.from_checkpoint(job.checkpoint["parsed"]).sha256,
        document_id,
        CONFIG.version,
    )
    chunks = decode_chunks_artifact(artifacts.fetch(reference), identity)
    assert tuple(chunk.content_text for chunk in chunks) == (
        "one two three four",
        "four five",
    )


@pytest.mark.anyio
async def test_existing_valid_chunks_skip_parsed_fetch_and_chunking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = "doc_" + "d" * 64
    service, _, _, artifacts, checkpointer = runner(tmp_path, document_id)
    job = prepare_parsed(artifacts, claimed())
    parsed_reference = ArtifactRef.from_checkpoint(job.checkpoint["parsed"])
    identity = artifact_identity(
        ArtifactKind.CHUNKS,
        parsed_reference.sha256,
        document_id,
        CONFIG.version,
    )
    chunks_key = artifact_key(job.tenant_id, job.id, ArtifactKind.CHUNKS, identity)
    expected_chunks = (
        # Build once through the real path; the monkeypatch proves recovery skips it.
        __import__(
            "hybrid_rag_search.chunking.chunker", fromlist=["chunk_document"]
        ).chunk_document(
            document_id=document_id,
            document=parsed(),
            tokenizer=WordTokenizer(),
            config=CONFIG,
        )
    )
    reference = artifacts.save(chunks_key, encode_chunks_artifact(expected_chunks, identity))
    monkeypatch.setattr(
        "hybrid_rag_search.ingestion.chunk_stage.chunk_document",
        MagicMock(side_effect=AssertionError("must not chunk")),
    )
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.EMBED)

    await service.run(job)

    checkpointer.advance_artifact.assert_awaited_once_with(job, reference)


@pytest.mark.anyio
async def test_invalid_recovered_chunks_fail_without_checkpoint(tmp_path: Path) -> None:
    document_id = "doc_" + "d" * 64
    service, _, _, artifacts, checkpointer = runner(tmp_path, document_id)
    job = prepare_parsed(artifacts, claimed())
    parsed_reference = ArtifactRef.from_checkpoint(job.checkpoint["parsed"])
    identity = artifact_identity(
        ArtifactKind.CHUNKS,
        parsed_reference.sha256,
        document_id,
        CONFIG.version,
    )
    chunks_key = artifact_key(job.tenant_id, job.id, ArtifactKind.CHUNKS, identity)
    artifacts.save(chunks_key, b"invalid")
    with pytest.raises(ArtifactContractError, match="Invalid chunks"):
        await service.run(job)
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_recovered_chunks_must_match_resolved_document_and_config(tmp_path: Path) -> None:
    document_id = "doc_" + "d" * 64
    service, _, _, artifacts, checkpointer = runner(tmp_path, document_id)
    job = prepare_parsed(artifacts, claimed())
    parsed_reference = ArtifactRef.from_checkpoint(job.checkpoint["parsed"])
    identity = artifact_identity(
        ArtifactKind.CHUNKS,
        parsed_reference.sha256,
        document_id,
        CONFIG.version,
    )
    chunks_key = artifact_key(job.tenant_id, job.id, ArtifactKind.CHUNKS, identity)
    wrong_chunks = __import__(
        "hybrid_rag_search.chunking.chunker", fromlist=["chunk_document"]
    ).chunk_document(
        document_id="doc_" + "e" * 64,
        document=parsed(),
        tokenizer=WordTokenizer(),
        config=CONFIG,
    )
    artifacts.save(chunks_key, encode_chunks_artifact(wrong_chunks, identity))
    with pytest.raises(ArtifactContractError, match="does not match"):
        await service.run(job)
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "checkpoint",
    [
        {},
        {"parsed": "invalid"},
    ],
)
async def test_requires_valid_parsed_checkpoint(
    tmp_path: Path, checkpoint: dict[str, object]
) -> None:
    service, resolver, _, _, checkpointer = runner(tmp_path, "doc_" + "d" * 64)
    job = replace(claimed(), checkpoint=checkpoint)
    with pytest.raises(ArtifactContractError, match="checkpoint reference"):
        await service.run(job)
    resolver.resolve.assert_not_awaited()
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_parsed_checkpoint_must_belong_to_job(tmp_path: Path) -> None:
    service, resolver, _, artifacts, checkpointer = runner(tmp_path, "doc_" + "d" * 64)
    job = prepare_parsed(artifacts, claimed())
    reference = ArtifactRef.from_checkpoint(job.checkpoint["parsed"])
    other_key = artifact_key(uuid4(), job.id, ArtifactKind.PARSED, parsed_identity(job))
    job = replace(
        job,
        checkpoint={
            "parsed": ArtifactRef(
                other_key, reference.sha256, reference.size_bytes
            ).checkpoint_value()
        },
    )
    with pytest.raises(ArtifactContractError, match="does not match"):
        await service.run(job)
    resolver.resolve.assert_not_awaited()
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("job", ["job", replace(claimed(), stage=IngestionStage.EMBED)])
async def test_runner_requires_claimed_chunk_stage(tmp_path: Path, job: object) -> None:
    service, resolver, _, _, checkpointer = runner(tmp_path, "doc_" + "d" * 64)
    with pytest.raises(ValueError, match="Chunk stage|ClaimedJob"):
        await service.run(job)  # type: ignore[arg-type]
    resolver.resolve.assert_not_awaited()
    checkpointer.advance_artifact.assert_not_awaited()


def identity_resolver(returned: dict[str, object] | None):
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = returned
    session = AsyncMock()
    session.execute.return_value = result
    sessions = MagicMock()
    sessions.return_value.__aenter__.return_value = session
    return PostgresDocumentIdentityResolver(sessions), session


def document_row(job: ClaimedJob) -> dict[str, object]:
    return {
        "collection_id": uuid4(),
        "content_hash": job.content_hash,
        "status": DocumentStatus.PENDING,
    }


@pytest.mark.anyio
async def test_resolver_builds_collection_scoped_content_identity() -> None:
    job = claimed()
    row = document_row(job)
    resolver, session = identity_resolver(row)
    assert (
        await resolver.resolve(job)
        == DocumentContentIdentity(
            job.tenant_id, row["collection_id"], job.content_hash
        ).document_id
    )
    sql = str(session.execute.await_args.args[0])
    assert "documents.id" in sql and "documents.tenant_id" in sql


@pytest.mark.anyio
async def test_resolver_requires_claimed_job() -> None:
    resolver, session = identity_resolver(None)
    with pytest.raises(ValueError, match="ClaimedJob"):
        await resolver.resolve("job")  # type: ignore[arg-type]
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_resolver_rejects_missing_deleted_or_changed_document() -> None:
    job = claimed()
    resolver, _ = identity_resolver(None)
    with pytest.raises(DocumentNotFound):
        await resolver.resolve(job)
    for status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
        row = document_row(job)
        row["status"] = status
        resolver, _ = identity_resolver(row)
        with pytest.raises(DocumentUnavailable):
            await resolver.resolve(job)
    row = document_row(job)
    row["content_hash"] = "f" * 64
    resolver, _ = identity_resolver(row)
    with pytest.raises(ValueError, match="content no longer matches"):
        await resolver.resolve(job)
