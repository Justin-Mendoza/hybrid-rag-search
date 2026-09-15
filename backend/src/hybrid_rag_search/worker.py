import dramatiq
from dramatiq.brokers.redis import RedisBroker

from hybrid_rag_search.config import get_settings

settings = get_settings()
dramatiq.set_broker(RedisBroker(url=settings.redis_url))


@dramatiq.actor
def heartbeat() -> str:
    """Minimal actor proving that the worker can import application code."""
    return "ok"
