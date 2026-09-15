from unittest.mock import AsyncMock, Mock

import pytest

from hybrid_rag_search.config import Settings
from hybrid_rag_search.health import (
    check_opensearch,
    check_postgres,
    check_redis,
    dependency_status,
    run_check,
)
from hybrid_rag_search.worker import heartbeat


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_worker_actor_is_importable() -> None:
    assert heartbeat.fn() == "ok"


@pytest.mark.anyio
async def test_run_check_reports_success_and_failure() -> None:
    async def succeeds() -> None:
        return None

    async def fails() -> None:
        raise ConnectionError

    assert await run_check(succeeds) is True
    assert await run_check(fails) is False


@pytest.mark.anyio
async def test_postgres_check_executes_query_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock(fetchval=AsyncMock(), close=AsyncMock())
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr("hybrid_rag_search.health.asyncpg.connect", connect)

    await check_postgres(Settings())

    connection.fetchval.assert_awaited_once_with("SELECT 1")
    connection.close.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_check_pings_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(ping=AsyncMock(return_value=True), aclose=AsyncMock())
    monkeypatch.setattr("hybrid_rag_search.health.Redis.from_url", Mock(return_value=client))

    await check_redis(Settings())

    client.ping.assert_awaited_once()
    client.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_check_rejects_false_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(ping=AsyncMock(return_value=False), aclose=AsyncMock())
    monkeypatch.setattr("hybrid_rag_search.health.Redis.from_url", Mock(return_value=client))

    with pytest.raises(ConnectionError):
        await check_redis(Settings())

    client.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_opensearch_check_requests_cluster_health(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    client = AsyncMock()
    client.get.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    monkeypatch.setattr("hybrid_rag_search.health.httpx.AsyncClient", Mock(return_value=context))

    await check_opensearch(Settings())

    client.get.assert_awaited_once_with("http://localhost:9200/_cluster/health")
    response.raise_for_status.assert_called_once()


@pytest.mark.anyio
async def test_dependency_status_names_each_result(monkeypatch: pytest.MonkeyPatch) -> None:
    run = AsyncMock(side_effect=[True, False, True])
    monkeypatch.setattr("hybrid_rag_search.health.run_check", run)

    assert await dependency_status(Settings()) == {
        "postgres": True,
        "redis": False,
        "opensearch": True,
    }
