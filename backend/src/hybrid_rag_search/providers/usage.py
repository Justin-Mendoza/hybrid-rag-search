"""Reported token usage is distinct from billed usage and estimated cost."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class EmbeddingUsage:
    input_tokens: float | None = None
    billed_input_tokens: float | None = None
    latency_ms: float | None = None
    estimated_cost_usd: Decimal | None = None
    usd_per_million_tokens: Decimal | None = None


def estimate_embedding_cost(
    billed_input_tokens: float | None, usd_per_million_tokens: Decimal | None
) -> Decimal | None:
    """Estimate text embedding list cost; None means unknown, not zero."""
    if billed_input_tokens is None or usd_per_million_tokens is None:
        return None
    return Decimal(str(billed_input_tokens)) * usd_per_million_tokens / Decimal(1_000_000)
