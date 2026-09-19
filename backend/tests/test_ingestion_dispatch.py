from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest

from hybrid_rag_search.ingestion.dispatch import DramatiqJobPublisher, RecoveryDispatcher


def dispatcher(job_ids=(), *, batch_size: int = 100):
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = list(job_ids)
    session.scalars.return_value = result
    sessions = MagicMock()
    sessions.begin.return_value.__aenter__.return_value = session
    publisher = AsyncMock()
    return RecoveryDispatcher(sessions, publisher, batch_size=batch_size), session, publisher


@pytest.mark.anyio
async def test_recovery_publishes_queued_job_ids_in_query_order() -> None:
    first, second = uuid4(), uuid4()
    recovery, session, publisher = dispatcher((first, second), batch_size=2)

    assert await recovery.dispatch_queued() == 2

    assert publisher.publish.await_args_list == [call(first), call(second)]
    statement = session.scalars.await_args.args[0]
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "ingestion_jobs.status = 'queued'" in compiled
    assert "ingestion_jobs.next_attempt_at IS NULL" in compiled
    assert "ingestion_jobs.next_attempt_at <= now()" in compiled
    assert "ORDER BY ingestion_jobs.created_at, ingestion_jobs.id" in compiled
    assert "LIMIT 2" in compiled


@pytest.mark.anyio
async def test_empty_recovery_batch_publishes_nothing() -> None:
    recovery, _, publisher = dispatcher()
    assert await recovery.dispatch_queued() == 0
    publisher.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_recovery_can_republish_same_queued_job() -> None:
    job_id = uuid4()
    recovery, _, publisher = dispatcher((job_id,))
    assert await recovery.dispatch_queued() == 1
    assert await recovery.dispatch_queued() == 1
    assert publisher.publish.await_args_list == [call(job_id), call(job_id)]


@pytest.mark.anyio
async def test_publish_failure_stops_batch_without_mutating_job_state() -> None:
    first, second = uuid4(), uuid4()
    recovery, session, publisher = dispatcher((first, second))
    publisher.publish.side_effect = RuntimeError("redis unavailable")
    with pytest.raises(RuntimeError, match="redis unavailable"):
        await recovery.dispatch_queued()
    publisher.publish.assert_awaited_once_with(first)
    session.execute.assert_not_awaited()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_recovery_requires_positive_integer_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dispatcher(batch_size=batch_size)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_dramatiq_adapter_sends_only_serialized_job_id() -> None:
    actor = MagicMock()
    publisher = DramatiqJobPublisher(actor)
    job_id = uuid4()
    await publisher.publish(job_id)
    actor.send.assert_called_once_with(str(job_id))


@pytest.mark.anyio
async def test_dramatiq_adapter_schedules_serialized_job_id() -> None:
    actor = MagicMock()
    publisher = DramatiqJobPublisher(actor)
    job_id = uuid4()
    await publisher.schedule(job_id, delay_ms=1250)
    actor.send_with_options.assert_called_once_with(args=(str(job_id),), delay=1250)


@pytest.mark.anyio
@pytest.mark.parametrize("delay", [-1, True, 1.5])
async def test_dramatiq_adapter_validates_delay(delay: object) -> None:
    publisher = DramatiqJobPublisher(MagicMock())
    with pytest.raises(ValueError, match="nonnegative integer"):
        await publisher.schedule(uuid4(), delay_ms=delay)  # type: ignore[arg-type]
