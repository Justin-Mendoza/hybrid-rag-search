"""Application-owned embedding contract, independent of provider SDKs."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from hybrid_rag_search.providers.usage import EmbeddingUsage


class EmbeddingPurpose(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


@dataclass(frozen=True)
class EmbeddingRequest:
    """A nonempty batch of text for one retrieval purpose; preserve text as given."""

    texts: tuple[str, ...]
    purpose: EmbeddingPurpose

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, EmbeddingPurpose):
            raise ValueError("Embedding purpose must be DOCUMENT or QUERY")
        if not isinstance(self.texts, tuple) or not self.texts:
            raise ValueError("Embedding texts must be a nonempty tuple")
        if any(not isinstance(text, str) or not text.strip() for text in self.texts):
            raise ValueError("Embedding texts must contain nonblank strings")


@dataclass(frozen=True)
class EmbeddingResult:
    """One vector per input, in input order, identified by model and adapter.

    Implementations must return finite vectors of the declared dimension.
    Fake vectors exercise plumbing only; they do not measure semantic quality.
    """

    vectors: tuple[tuple[float, ...], ...]
    model: str
    dimensions: int
    purpose: EmbeddingPurpose
    adapter: Literal["fake", "cohere"]
    usage: EmbeddingUsage = EmbeddingUsage()


class EmbeddingProvider(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed a batch without changing its order or dropping inputs."""
        ...
