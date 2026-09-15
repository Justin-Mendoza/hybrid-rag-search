import asyncio
from collections.abc import Awaitable, Callable

import asyncpg  # type: ignore[import-untyped]
import httpx
from redis.asyncio import Redis

from hybrid_rag_search.config import Settings

Check = Callable[[], Awaitable[None]]


async def check_postgres(settings: Settings) -> None:
    connection = await asyncpg.connect(settings.database_url, timeout=2)
    try:
        await connection.fetchval("SELECT 1")
    finally:
        await connection.close()


async def check_redis(settings: Settings) -> None:
    client: Redis = Redis.from_url(settings.redis_url)
    try:
        if not await client.ping():
            raise ConnectionError("Redis did not respond to PING")
    finally:
        await client.aclose()


async def check_opensearch(settings: Settings) -> None:
    async with httpx.AsyncClient(timeout=2) as client:
        response = await client.get(f"{settings.opensearch_url}/_cluster/health")
        response.raise_for_status()


async def run_check(check: Check) -> bool:
    try:
        await check()
    except Exception:
        return False
    return True


async def dependency_status(settings: Settings) -> dict[str, bool]:
    results = await asyncio.gather(
        run_check(lambda: check_postgres(settings)),
        run_check(lambda: check_redis(settings)),
        run_check(lambda: check_opensearch(settings)),
    )
    return dict(zip(("postgres", "redis", "opensearch"), results, strict=True))
