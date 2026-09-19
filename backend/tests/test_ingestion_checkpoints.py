from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.artifacts import ArtifactKind, ArtifactRef, artifact_key
from hybrid_rag_search.ingestion.checkpoints import JobCheckpointer
from hybrid_rag_search.ingestion.claims import ClaimedJob
from hybrid_rag_search.ingestion.indexing import IndexReceipt
from hybrid_rag_search.models.content import IngestionStage

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def claimed(stage: IngestionStage = IngestionStage.PARSE) -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        tenant_id=uuid4(),
        document_id=uuid4(),
        content_hash="a" * 64,
        pipeline_version="pipe_" + "b" * 64,
        stage=stage,
        attempts=1,
        max_attempts=3,
        checkpoint={"retained": "metadata"},
        lease_token=uuid4(),
        lease_expires_at=NOW + timedelta(minutes=5),
    )


def reference(job: ClaimedJob, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(
        artifact_key(job.tenant_id, job.id, kind, "c" * 64),
        "d" * 64,
        123,
    )


def checkpointer(returned: dict[str, object] | None):
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = returned
    session = AsyncMock()
    session.execute.return_value = result
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    return JobCheckpointer(sessions), session


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stage", "kind", "next_stage"),
    [
        (IngestionStage.PARSE, ArtifactKind.PARSED, IngestionStage.CHUNK),
        (IngestionStage.CHUNK, ArtifactKind.CHUNKS, IngestionStage.EMBED),
        (IngestionStage.EMBED, ArtifactKind.EMBEDDINGS, IngestionStage.INDEX),
    ],
)
async def test_live_owner_records_reference_and_advances_stage(
    stage: IngestionStage, kind: ArtifactKind, next_stage: IngestionStage
) -> None:
    job = claimed(stage)
    artifact = reference(job, kind)
    expires_at = NOW + timedelta(minutes=10)
    expected_checkpoint = {
        "retained": "metadata",
        kind.value: artifact.checkpoint_value(),
    }
    service, session = checkpointer(
        {
            "stage": next_stage.value,
            "checkpoint": expected_checkpoint,
            "lease_expires_at": expires_at,
        }
    )

    advanced = await service.advance_artifact(job, artifact)

    assert advanced == replace(
        job,
        stage=next_stage,
        checkpoint=expected_checkpoint,
        lease_expires_at=expires_at,
    )
    statement = session.execute.await_args.args[0]
    compiled = statement.compile()
    sql = str(compiled)
    assert "ingestion_jobs.status = :status_1" in sql
    assert "ingestion_jobs.stage = :stage_1" in sql
    assert "ingestion_jobs.lease_token = :lease_token_1" in sql
    assert "ingestion_jobs.lease_expires_at > now()" in sql
    assert "error_code=:error_code" in sql
    assert "INTERVAL '300 seconds'" in sql
    assert stage in compiled.params.values()
    assert next_stage in compiled.params.values()
    assert expected_checkpoint in compiled.params.values()


@pytest.mark.anyio
async def test_lost_lease_or_changed_stage_cannot_advance() -> None:
    job = claimed()
    service, session = checkpointer(None)
    assert await service.advance_artifact(job, reference(job, ArtifactKind.PARSED)) is None
    session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_identical_existing_reference_is_idempotent() -> None:
    job = claimed()
    artifact = reference(job, ArtifactKind.PARSED)
    job = replace(job, checkpoint={"parsed": artifact.checkpoint_value()})
    service, session = checkpointer(None)
    assert await service.advance_artifact(job, artifact) is None
    session.execute.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("job_value", "reference_value", "message"),
    [
        ("invalid", "valid", "ClaimedJob"),
        ("valid", "invalid", "ArtifactRef"),
    ],
)
async def test_requires_contract_values(job_value: str, reference_value: str, message: str) -> None:
    job = claimed()
    artifact = reference(job, ArtifactKind.PARSED)
    service, session = checkpointer(None)
    with pytest.raises(ValueError, match=message):
        await service.advance_artifact(
            job if job_value == "valid" else "job",  # type: ignore[arg-type]
            artifact if reference_value == "valid" else "artifact",  # type: ignore[arg-type]
        )
    session.execute.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("stage", [IngestionStage.INDEX, IngestionStage.COMPLETE])
async def test_non_artifact_stage_cannot_use_artifact_transition(stage: IngestionStage) -> None:
    job = claimed(stage)
    service, session = checkpointer(None)
    with pytest.raises(ValueError, match="does not produce"):
        await service.advance_artifact(job, reference(job, ArtifactKind.PARSED))
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_reference_must_match_tenant_job_and_stage() -> None:
    job = claimed()
    service, session = checkpointer(None)
    mismatches = (
        ArtifactRef(artifact_key(uuid4(), job.id, ArtifactKind.PARSED, "c" * 64), "d" * 64, 1),
        ArtifactRef(
            artifact_key(job.tenant_id, uuid4(), ArtifactKind.PARSED, "c" * 64), "d" * 64, 1
        ),
        reference(job, ArtifactKind.CHUNKS),
    )
    for artifact in mismatches:
        with pytest.raises(ValueError, match="claimed job and stage"):
            await service.advance_artifact(job, artifact)
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_existing_different_reference_is_rejected() -> None:
    job = claimed()
    artifact = reference(job, ArtifactKind.PARSED)
    job = replace(job, checkpoint={"parsed": {"key": "different"}})
    service, session = checkpointer(None)
    with pytest.raises(ValueError, match="different artifact"):
        await service.advance_artifact(job, artifact)
    session.execute.assert_not_awaited()


def receipt(job: ClaimedJob) -> IndexReceipt:
    return IndexReceipt(
        "idxgen_" + "e" * 64,
        job.tenant_id,
        job.document_id,
        job.pipeline_version,
        2,
    )


@pytest.mark.anyio
async def test_index_completion_updates_job_and_matching_document() -> None:
    job = claimed(IngestionStage.INDEX)
    service, session = checkpointer(None)
    session.scalar.return_value = job.document_id

    assert await service.complete_index(job, receipt(job))
    job_statement = session.scalar.await_args.args[0]
    document_statement = session.execute.await_args.args[0]
    compiled_job = job_statement.compile()
    job_sql = str(compiled_job)
    document_sql = str(document_statement.compile(compile_kwargs={"literal_binds": True}))
    assert "ingestion_jobs.stage = :stage_1" in job_sql
    assert "ingestion_jobs.lease_expires_at > now()" in job_sql
    assert IngestionStage.INDEX in compiled_job.params.values()
    assert IngestionStage.COMPLETE in compiled_job.params.values()
    assert "documents.content_hash" in document_sql
    assert "index_version=(documents.index_version + 1)" in document_sql


@pytest.mark.anyio
async def test_index_completion_returns_false_after_lease_loss() -> None:
    job = claimed(IngestionStage.INDEX)
    service, session = checkpointer(None)
    session.scalar.return_value = None
    assert not await service.complete_index(job, receipt(job))
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_index_completion_validates_claim_stage_receipt_and_identity() -> None:
    job = claimed(IngestionStage.INDEX)
    service, session = checkpointer(None)
    with pytest.raises(ValueError, match="ClaimedJob"):
        await service.complete_index("job", receipt(job))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index stage"):
        await service.complete_index(replace(job, stage=IngestionStage.EMBED), receipt(job))
    with pytest.raises(ValueError, match="IndexReceipt"):
        await service.complete_index(job, "receipt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        await service.complete_index(
            job,
            replace(receipt(job), document_id=uuid4()),
        )
    session.scalar.assert_not_awaited()
