"""Lease-protected ingestion checkpoint and stage advancement."""

from dataclasses import replace

from sqlalchemy import func, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hybrid_rag_search.ingestion.artifacts import ArtifactKind, ArtifactRef
from hybrid_rag_search.ingestion.claims import LEASE_DURATION_SECONDS, ClaimedJob
from hybrid_rag_search.ingestion.indexing import IndexReceipt
from hybrid_rag_search.models.content import (
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)

_ARTIFACT_TRANSITIONS = {
    IngestionStage.PARSE: (ArtifactKind.PARSED, IngestionStage.CHUNK),
    IngestionStage.CHUNK: (ArtifactKind.CHUNKS, IngestionStage.EMBED),
    IngestionStage.EMBED: (ArtifactKind.EMBEDDINGS, IngestionStage.INDEX),
}


class JobCheckpointer:
    """Record one immutable artifact and advance only the live lease owner."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def advance_artifact(
        self, claimed: ClaimedJob, reference: ArtifactRef
    ) -> ClaimedJob | None:
        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Checkpoint advancement requires a ClaimedJob")
        if not isinstance(reference, ArtifactRef):
            raise ValueError("Checkpoint advancement requires an ArtifactRef")
        transition = _ARTIFACT_TRANSITIONS.get(claimed.stage)
        if transition is None:
            raise ValueError("Current ingestion stage does not produce an artifact")
        kind, next_stage = transition
        expected_prefix = f"{claimed.tenant_id}/{claimed.id}/{kind.value}/"
        if not reference.key.startswith(expected_prefix):
            raise ValueError("Artifact reference does not match the claimed job and stage")

        reference_value = reference.checkpoint_value()
        existing = claimed.checkpoint.get(kind.value)
        if existing is not None and existing != reference_value:
            raise ValueError("Checkpoint already contains a different artifact reference")
        checkpoint = {**claimed.checkpoint, kind.value: reference_value}
        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.id == claimed.id,
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.stage == claimed.stage,
                IngestionJob.lease_token == claimed.lease_token,
                IngestionJob.lease_expires_at > func.now(),
            )
            .values(
                stage=next_stage,
                checkpoint=checkpoint,
                error_code=None,
                error_message=None,
                lease_expires_at=func.now() + text(f"INTERVAL '{LEASE_DURATION_SECONDS} seconds'"),
            )
            .returning(
                IngestionJob.stage,
                IngestionJob.checkpoint,
                IngestionJob.lease_expires_at,
            )
        )
        async with self.sessions.begin() as session:
            result = await session.execute(statement)
            row = result.mappings().one_or_none()
        if row is None:
            return None
        return replace(
            claimed,
            stage=IngestionStage(row["stage"]),
            checkpoint=dict(row["checkpoint"]),
            lease_expires_at=row["lease_expires_at"],
        )

    async def complete_index(self, claimed: ClaimedJob, receipt: IndexReceipt) -> bool:
        """Commit index success and document readiness under the live lease."""

        if not isinstance(claimed, ClaimedJob):
            raise ValueError("Index completion requires a ClaimedJob")
        if claimed.stage is not IngestionStage.INDEX:
            raise ValueError("Index completion requires a job at the index stage")
        if not isinstance(receipt, IndexReceipt):
            raise ValueError("Index completion requires an IndexReceipt")
        if (
            receipt.tenant_id != claimed.tenant_id
            or receipt.document_id != claimed.document_id
            or receipt.pipeline_version != claimed.pipeline_version
        ):
            raise ValueError("Index receipt does not match the claimed job")

        checkpoint = {**claimed.checkpoint, "index": receipt.checkpoint_value()}
        statement = (
            update(IngestionJob)
            .where(
                IngestionJob.id == claimed.id,
                IngestionJob.status == IngestionJobStatus.RUNNING,
                IngestionJob.stage == IngestionStage.INDEX,
                IngestionJob.lease_token == claimed.lease_token,
                IngestionJob.lease_expires_at > func.now(),
            )
            .values(
                status=IngestionJobStatus.SUCCEEDED,
                stage=IngestionStage.COMPLETE,
                checkpoint=checkpoint,
                error_code=None,
                error_message=None,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                finished_at=func.now(),
            )
            .returning(IngestionJob.document_id)
        )
        async with self.sessions.begin() as session:
            document_id = await session.scalar(statement)
            if document_id is None:
                return False
            await session.execute(
                update(Document)
                .where(
                    Document.id == claimed.document_id,
                    Document.tenant_id == claimed.tenant_id,
                    Document.content_hash == claimed.content_hash,
                )
                .values(
                    status=DocumentStatus.READY,
                    index_version=Document.index_version + 1,
                    error_message=None,
                )
            )
        return True
