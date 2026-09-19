"""Batch-resumable embedding-stage execution for a claimed ingestion job."""

import asyncio
from dataclasses import replace
from datetime import datetime
from typing import Protocol
from uuid import UUID

from hybrid_rag_search.chunking.contracts import Chunk
from hybrid_rag_search.ingestion.artifact_payloads import (
    ArtifactContractError,
    EmbeddingsArtifact,
    build_embeddings_artifact,
    decode_chunks_artifact,
    decode_embeddings_artifact,
    encode_embeddings_artifact,
)
from hybrid_rag_search.ingestion.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStorage,
    artifact_identity,
    artifact_key,
)
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.models.content import IngestionStage
from hybrid_rag_search.providers.embeddings import (
    EmbeddingProvider,
    EmbeddingPurpose,
    EmbeddingRequest,
)

COHERE_EMBEDDING_BATCH_LIMIT = 96


class LeaseRenewer(Protocol):
    async def renew(self, job_id: UUID, lease_token: UUID) -> datetime | None: ...


def _scoped_checkpoint_reference(
    claimed: ClaimedJob,
    checkpoint_name: str,
    kind: ArtifactKind,
) -> ArtifactRef:
    try:
        reference = ArtifactRef.from_checkpoint(claimed.checkpoint.get(checkpoint_name))
    except ValueError:
        raise ArtifactContractError(
            f"{checkpoint_name.capitalize()} checkpoint reference is invalid"
        ) from None
    expected_prefix = f"{claimed.tenant_id}/{claimed.id}/{kind.value}/"
    if not reference.key.startswith(expected_prefix):
        raise ArtifactContractError(
            f"{checkpoint_name.capitalize()} checkpoint reference does not match this job"
        )
    return reference


def _identity_from_reference(reference: ArtifactRef) -> str:
    return reference.key.rsplit("/", maxsplit=1)[1].removesuffix(".json")


def _validate_embeddings(
    artifact: EmbeddingsArtifact,
    chunks: tuple[Chunk, ...],
    *,
    model: str,
    dimensions: int,
) -> None:
    if artifact.document_id != chunks[0].document_id:
        raise ArtifactContractError("Embeddings artifact document identity does not match")
    if artifact.model != model or artifact.dimensions != dimensions:
        raise ArtifactContractError("Embeddings artifact model configuration does not match")
    if tuple(item.chunk_id for item in artifact.embeddings) != tuple(
        chunk.chunk_id for chunk in chunks
    ):
        raise ArtifactContractError("Embeddings artifact chunk order does not match")


class EmbeddingStageRunner:
    """Recover embedding batches and publish one ordered final artifact."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        artifacts: ArtifactStorage,
        lease_renewer: LeaseRenewer,
        checkpointer: JobCheckpointer,
        *,
        model: str,
        dimensions: int,
        chunk_config_version: str,
        batch_size: int = COHERE_EMBEDDING_BATCH_LIMIT,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Embedding stage model must be nonblank")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError("Embedding stage dimensions must be a positive integer")
        if not isinstance(chunk_config_version, str) or not chunk_config_version.startswith(
            "chunkcfg_"
        ):
            raise ValueError("Embedding stage chunk configuration is invalid")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= COHERE_EMBEDDING_BATCH_LIMIT
        ):
            raise ValueError("Embedding batch size must be between 1 and 96")
        self.provider = provider
        self.artifacts = artifacts
        self.lease_renewer = lease_renewer
        self.checkpointer = checkpointer
        self.model = model
        self.dimensions = dimensions
        self.chunk_config_version = chunk_config_version
        self.batch_size = batch_size

    def _batch_identity(
        self,
        chunks_reference: ArtifactRef,
        batch: tuple[Chunk, ...],
    ) -> str:
        return artifact_identity(
            ArtifactKind.EMBEDDINGS,
            "batch-v1",
            chunks_reference.sha256,
            self.model,
            str(self.dimensions),
            *(chunk.chunk_id for chunk in batch),
        )

    def _final_identity(self, chunks_reference: ArtifactRef) -> str:
        return artifact_identity(
            ArtifactKind.EMBEDDINGS,
            "final-v1",
            chunks_reference.sha256,
            self.model,
            str(self.dimensions),
            EmbeddingPurpose.DOCUMENT.value,
        )

    async def _recover_or_embed_batch(
        self,
        claimed: ClaimedJob,
        chunks_reference: ArtifactRef,
        batch: tuple[Chunk, ...],
    ) -> tuple[EmbeddingsArtifact, ArtifactRef]:
        identity = self._batch_identity(chunks_reference, batch)
        key = artifact_key(
            claimed.tenant_id,
            claimed.id,
            ArtifactKind.EMBEDDINGS,
            identity,
        )
        reference = await asyncio.to_thread(self.artifacts.lookup, key)
        if reference is not None:
            content = await asyncio.to_thread(self.artifacts.fetch, reference)
            artifact = decode_embeddings_artifact(content, identity)
        else:
            result = await self.provider.embed(
                EmbeddingRequest(
                    tuple(chunk.embedding_text for chunk in batch),
                    EmbeddingPurpose.DOCUMENT,
                )
            )
            artifact = build_embeddings_artifact(batch, result)
            _validate_embeddings(
                artifact,
                batch,
                model=self.model,
                dimensions=self.dimensions,
            )
            content = encode_embeddings_artifact(artifact, identity)
            reference = await asyncio.to_thread(self.artifacts.save, key, content)
        _validate_embeddings(
            artifact,
            batch,
            model=self.model,
            dimensions=self.dimensions,
        )
        return artifact, reference

    async def run(self, claimed: ClaimedJob) -> ClaimedJob | None:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Embedding stage requires a ClaimedJob")
        if claimed.stage is not IngestionStage.EMBED:
            raise ValueError("Embedding stage requires a job at the embed stage")
        chunks_reference = _scoped_checkpoint_reference(
            claimed,
            "chunks",
            ArtifactKind.CHUNKS,
        )
        chunks_identity = _identity_from_reference(chunks_reference)
        chunks_content = await asyncio.to_thread(self.artifacts.fetch, chunks_reference)
        chunks = decode_chunks_artifact(chunks_content, chunks_identity)
        if chunks[0].config_version != self.chunk_config_version:
            raise ArtifactContractError("Chunks artifact configuration does not match")

        final_identity = self._final_identity(chunks_reference)
        final_key = artifact_key(
            claimed.tenant_id,
            claimed.id,
            ArtifactKind.EMBEDDINGS,
            final_identity,
        )
        final_reference = await asyncio.to_thread(self.artifacts.lookup, final_key)
        if final_reference is not None:
            content = await asyncio.to_thread(self.artifacts.fetch, final_reference)
            final_artifact = decode_embeddings_artifact(content, final_identity)
            _validate_embeddings(
                final_artifact,
                chunks,
                model=self.model,
                dimensions=self.dimensions,
            )
            return await self.checkpointer.advance_artifact(claimed, final_reference)

        current = claimed
        batch_artifacts: list[EmbeddingsArtifact] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            artifact, _ = await self._recover_or_embed_batch(current, chunks_reference, batch)
            batch_artifacts.append(artifact)
            lease_expires_at = await self.lease_renewer.renew(current.id, current.lease_token)
            if lease_expires_at is None:
                return None
            current = replace(current, lease_expires_at=lease_expires_at)

        adapters = {artifact.adapter for artifact in batch_artifacts}
        if len(adapters) != 1:
            raise ArtifactContractError("Embedding batches use different adapters")
        final_artifact = EmbeddingsArtifact(
            document_id=chunks[0].document_id,
            model=self.model,
            dimensions=self.dimensions,
            adapter=adapters.pop(),
            embeddings=tuple(item for artifact in batch_artifacts for item in artifact.embeddings),
        )
        _validate_embeddings(
            final_artifact,
            chunks,
            model=self.model,
            dimensions=self.dimensions,
        )
        final_content = encode_embeddings_artifact(final_artifact, final_identity)
        final_reference = await asyncio.to_thread(
            self.artifacts.save,
            final_key,
            final_content,
        )
        return await self.checkpointer.advance_artifact(current, final_reference)
