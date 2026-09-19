import hashlib
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hybrid_rag_search.chunking.chunker import chunk_document
from hybrid_rag_search.chunking.contracts import Chunk, ChunkingConfig
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    EmbeddingsArtifact,
    decode_embeddings_artifact,
    encode_chunks_artifact,
    encode_embeddings_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactRef,
    LocalArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.embed_stage import EmbeddingStageRunner
from hybrid_rag_search.models.content import IngestionStage
from hybrid_rag_search.parsers.contracts import ParsedBlock, ParsedDocument, SourceLocator
from hybrid_rag_search.providers.embeddings import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
)
from hybrid_rag_search.providers.fake import FakeEmbeddingProvider
from hybrid_rag_search.tokenization import TokenSpan

CONFIG = ChunkingConfig(max_tokens=1, overlap_tokens=0)
DOCUMENT_ID = "doc_" + "d" * 64
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


class WordTokenizer:
    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(
            TokenSpan(index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\S+", text))
        )


class RecordingProvider:
    def __init__(
        self,
        *,
        dimensions: int = 3,
        fail_on_call: int | None = None,
        model: str = "embed-test",
        adapter: str = "cohere",
    ) -> None:
        self.dimensions = dimensions
        self.fail_on_call = fail_on_call
        self.model = model
        self.adapter = adapter
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.requests.append(request)
        if self.fail_on_call == len(self.requests):
            raise RuntimeError("provider unavailable")
        vectors = tuple(
            tuple(float(text_index + dimension) for dimension in range(self.dimensions))
            for text_index, _ in enumerate(request.texts, start=1)
        )
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=self.dimensions,
            purpose=request.purpose,
            adapter=self.adapter,  # type: ignore[arg-type]
        )


def chunks(count: int = 5) -> tuple[Chunk, ...]:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(" ".join(f"word-{index}" for index in range(count)), SourceLocator(1)),
        ),
        media_type="text/plain",
        parser_name="plain-text",
        parser_version="1",
    )
    return chunk_document(
        document_id=DOCUMENT_ID,
        document=document,
        tokenizer=WordTokenizer(),
        config=CONFIG,
    )


def claimed(storage: LocalArtifactStorage, source_chunks: tuple[Chunk, ...]) -> ClaimedJob:
    job = ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash=hashlib.sha256(b"original").hexdigest(),
        pipeline_version="pipe_" + "b" * 64,
        stage=IngestionStage.EMBED,
        attempts=1,
        max_attempts=3,
        checkpoint={},
        lease_token=uuid4(),
        lease_expires_at=NOW + timedelta(minutes=5),
    )
    identity = artifact_identity(ArtifactKind.CHUNKS, "parsed", CONFIG.version)
    key = artifact_key(job.tenant_id, job.id, ArtifactKind.CHUNKS, identity)
    reference = storage.save(key, encode_chunks_artifact(source_chunks, identity))
    return replace(job, checkpoint={"chunks": reference.checkpoint_value()})


def runner(
    tmp_path: Path,
    provider: RecordingProvider | FakeEmbeddingProvider,
    *,
    batch_size: int = 2,
):
    artifacts = LocalArtifactStorage(tmp_path)
    renewer = AsyncMock()
    renewer.renew.return_value = NOW + timedelta(minutes=10)
    checkpointer = AsyncMock()
    service = EmbeddingStageRunner(
        provider,
        artifacts,
        renewer,
        checkpointer,
        model=getattr(provider, "model", FakeEmbeddingProvider.MODEL),
        dimensions=provider.dimensions,
        chunk_config_version=CONFIG.version,
        batch_size=batch_size,
    )
    return service, artifacts, renewer, checkpointer


def final_reference(checkpointer: AsyncMock) -> ArtifactRef:
    return checkpointer.advance_artifact.await_args.args[1]


@pytest.mark.anyio
async def test_embeds_in_bounded_batches_and_publishes_ordered_final_artifact(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    service, artifacts, renewer, checkpointer = runner(tmp_path, provider)
    source_chunks = chunks()
    job = claimed(artifacts, source_chunks)
    advanced = replace(job, stage=IngestionStage.INDEX)
    checkpointer.advance_artifact.return_value = advanced

    assert await service.run(job) == advanced

    assert tuple(len(request.texts) for request in provider.requests) == (2, 2, 1)
    assert all(request.purpose is EmbeddingPurpose.DOCUMENT for request in provider.requests)
    assert renewer.renew.await_count == 3
    checkpoint_job = checkpointer.advance_artifact.await_args.args[0]
    assert checkpoint_job.lease_expires_at == NOW + timedelta(minutes=10)
    reference = final_reference(checkpointer)
    identity = reference.key.rsplit("/", 1)[1].removesuffix(".json")
    artifact = decode_embeddings_artifact(artifacts.fetch(reference), identity)
    assert tuple(item.chunk_id for item in artifact.embeddings) == tuple(
        chunk.chunk_id for chunk in source_chunks
    )


@pytest.mark.anyio
async def test_retry_reuses_completed_batches_after_provider_failure(tmp_path: Path) -> None:
    failing = RecordingProvider(fail_on_call=3)
    service, artifacts, _, checkpointer = runner(tmp_path, failing)
    source_chunks = chunks()
    job = claimed(artifacts, source_chunks)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await service.run(job)
    assert len(failing.requests) == 3
    checkpointer.advance_artifact.assert_not_awaited()

    resumed = RecordingProvider()
    service, _, renewer, checkpointer = runner(tmp_path, resumed)
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.INDEX)
    await service.run(job)
    assert tuple(len(request.texts) for request in resumed.requests) == (1,)
    assert renewer.renew.await_count == 3


@pytest.mark.anyio
async def test_final_artifact_recovery_skips_batches_and_provider(tmp_path: Path) -> None:
    first = RecordingProvider()
    service, artifacts, _, checkpointer = runner(tmp_path, first)
    job = claimed(artifacts, chunks())
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.INDEX)
    await service.run(job)
    reference = final_reference(checkpointer)

    recovered = RecordingProvider()
    service, _, renewer, checkpointer = runner(tmp_path, recovered)
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.INDEX)
    await service.run(job)
    assert recovered.requests == []
    renewer.renew.assert_not_awaited()
    checkpointer.advance_artifact.assert_awaited_once_with(job, reference)


@pytest.mark.anyio
async def test_lost_lease_stops_after_saving_current_batch(tmp_path: Path) -> None:
    provider = RecordingProvider()
    service, artifacts, renewer, checkpointer = runner(tmp_path, provider)
    job = claimed(artifacts, chunks())
    renewer.renew.return_value = None
    assert await service.run(job) is None
    assert len(provider.requests) == 1
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_invalid_recovered_batch_is_not_reembedded(tmp_path: Path) -> None:
    failing = RecordingProvider(fail_on_call=2)
    service, artifacts, _, _ = runner(tmp_path, failing)
    job = claimed(artifacts, chunks())
    with pytest.raises(RuntimeError):
        await service.run(job)
    batch_files = sorted(
        (artifacts.root / str(job.tenant_id) / str(job.id) / "embeddings").iterdir()
    )
    batch_files[0].write_bytes(b"invalid")

    provider = RecordingProvider()
    service, _, _, checkpointer = runner(tmp_path, provider)
    with pytest.raises(ArtifactContractError, match="Invalid embeddings"):
        await service.run(job)
    assert provider.requests == []
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_recovered_batch_must_match_expected_chunks(tmp_path: Path) -> None:
    provider = RecordingProvider(fail_on_call=2)
    service, artifacts, _, _ = runner(tmp_path, provider)
    job = claimed(artifacts, chunks())
    with pytest.raises(RuntimeError):
        await service.run(job)
    batch_file = sorted(
        (artifacts.root / str(job.tenant_id) / str(job.id) / "embeddings").iterdir()
    )[0]
    content: dict[str, Any] = __import__("json").loads(batch_file.read_bytes())
    content["payload"]["embeddings"].reverse()
    batch_file.write_text(__import__("json").dumps(content, separators=(",", ":")))

    service, _, _, _ = runner(tmp_path, RecordingProvider())
    with pytest.raises(ArtifactContractError, match="chunk order"):
        await service.run(job)


@pytest.mark.anyio
async def test_final_artifact_must_match_model_configuration(tmp_path: Path) -> None:
    provider = RecordingProvider()
    service, artifacts, _, checkpointer = runner(tmp_path, provider)
    source_chunks = chunks()
    job = claimed(artifacts, source_chunks)
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.INDEX)
    await service.run(job)
    reference = final_reference(checkpointer)
    identity = reference.key.rsplit("/", 1)[1].removesuffix(".json")
    stored = decode_embeddings_artifact(artifacts.fetch(reference), identity)
    artifacts.delete(reference)
    wrong = EmbeddingsArtifact(
        stored.document_id,
        "other-model",
        stored.dimensions,
        stored.adapter,
        stored.embeddings,
    )
    artifacts.save(reference.key, encode_embeddings_artifact(wrong, identity))

    service, _, _, _ = runner(tmp_path, RecordingProvider())
    with pytest.raises(ArtifactContractError, match="model configuration"):
        await service.run(job)


@pytest.mark.anyio
async def test_final_artifact_must_match_document_identity(tmp_path: Path) -> None:
    provider = RecordingProvider()
    service, artifacts, _, checkpointer = runner(tmp_path, provider)
    source_chunks = chunks()
    job = claimed(artifacts, source_chunks)
    checkpointer.advance_artifact.return_value = replace(job, stage=IngestionStage.INDEX)
    await service.run(job)
    reference = final_reference(checkpointer)
    identity = reference.key.rsplit("/", 1)[1].removesuffix(".json")
    stored = decode_embeddings_artifact(artifacts.fetch(reference), identity)
    artifacts.delete(reference)
    wrong = EmbeddingsArtifact(
        "doc_" + "e" * 64,
        stored.model,
        stored.dimensions,
        stored.adapter,
        stored.embeddings,
    )
    artifacts.save(reference.key, encode_embeddings_artifact(wrong, identity))

    service, _, _, _ = runner(tmp_path, RecordingProvider())
    with pytest.raises(ArtifactContractError, match="document identity"):
        await service.run(job)


@pytest.mark.anyio
async def test_batches_from_different_adapters_are_not_combined(tmp_path: Path) -> None:
    class AlternatingProvider(RecordingProvider):
        async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
            result = await super().embed(request)
            adapter = "fake" if len(self.requests) == 2 else "cohere"
            return replace(result, adapter=adapter)

    provider = AlternatingProvider()
    service, artifacts, _, checkpointer = runner(tmp_path, provider)
    job = claimed(artifacts, chunks())
    with pytest.raises(ArtifactContractError, match="different adapters"):
        await service.run(job)
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_provider_result_must_match_expected_model(tmp_path: Path) -> None:
    provider = RecordingProvider(model="wrong-model")
    artifacts = LocalArtifactStorage(tmp_path)
    renewer, checkpointer = AsyncMock(), AsyncMock()
    service = EmbeddingStageRunner(
        provider,
        artifacts,
        renewer,
        checkpointer,
        model="expected-model",
        dimensions=3,
        chunk_config_version=CONFIG.version,
        batch_size=2,
    )
    job = claimed(artifacts, chunks())
    with pytest.raises(ArtifactContractError, match="model configuration"):
        await service.run(job)
    renewer.renew.assert_not_awaited()


@pytest.mark.anyio
async def test_rejects_invalid_or_wrong_scoped_chunks_checkpoint(tmp_path: Path) -> None:
    service, artifacts, renewer, checkpointer = runner(tmp_path, RecordingProvider())
    job = replace(claimed(artifacts, chunks()), checkpoint={"chunks": "invalid"})
    with pytest.raises(ArtifactContractError, match="checkpoint reference"):
        await service.run(job)

    valid = claimed(artifacts, chunks())
    reference = ArtifactRef.from_checkpoint(valid.checkpoint["chunks"])
    wrong_key = artifact_key(uuid4(), valid.id, ArtifactKind.CHUNKS, "a" * 64)
    wrong = replace(
        valid,
        checkpoint={
            "chunks": ArtifactRef(
                wrong_key, reference.sha256, reference.size_bytes
            ).checkpoint_value()
        },
    )
    with pytest.raises(ArtifactContractError, match="does not match"):
        await service.run(wrong)
    renewer.renew.assert_not_awaited()
    checkpointer.advance_artifact.assert_not_awaited()


@pytest.mark.anyio
async def test_chunks_configuration_must_match_runner(tmp_path: Path) -> None:
    service, artifacts, _, _ = runner(tmp_path, RecordingProvider())
    job = claimed(artifacts, chunks())
    service.chunk_config_version = "chunkcfg_" + "f" * 64
    with pytest.raises(ArtifactContractError, match="configuration"):
        await service.run(job)


@pytest.mark.anyio
@pytest.mark.parametrize("job", ["job", None])
async def test_runner_requires_claimed_job(tmp_path: Path, job: object) -> None:
    service, _, _, _ = runner(tmp_path, RecordingProvider())
    with pytest.raises(ValueError, match="ClaimedJob"):
        await service.run(job)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_runner_requires_embed_stage(tmp_path: Path) -> None:
    service, artifacts, _, _ = runner(tmp_path, RecordingProvider())
    job = replace(claimed(artifacts, chunks()), stage=IngestionStage.INDEX)
    with pytest.raises(ValueError, match="embed stage"):
        await service.run(job)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": " "}, "model"),
        ({"dimensions": 0}, "dimensions"),
        ({"dimensions": True}, "dimensions"),
        ({"chunk_config_version": "invalid"}, "chunk configuration"),
        ({"batch_size": 0}, "batch size"),
        ({"batch_size": 97}, "batch size"),
        ({"batch_size": True}, "batch size"),
    ],
)
def test_runner_configuration_validation(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "provider": FakeEmbeddingProvider(3),
        "artifacts": LocalArtifactStorage(tmp_path),
        "lease_renewer": AsyncMock(),
        "checkpointer": AsyncMock(),
        "model": FakeEmbeddingProvider.MODEL,
        "dimensions": 3,
        "chunk_config_version": CONFIG.version,
        "batch_size": 2,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        EmbeddingStageRunner(**values)  # type: ignore[arg-type]
