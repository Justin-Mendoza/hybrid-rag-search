"""Text generation with optional request-bound evidence and native citations."""

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Protocol


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class GenerationFinishReason(StrEnum):
    COMPLETE = "complete"
    MAX_TOKENS = "max_tokens"


@dataclass(frozen=True)
class GenerationMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise ValueError("Generation message role is invalid")
        if not self.content.strip():
            raise ValueError("Generation message content must be nonblank")


@dataclass(frozen=True)
class GenerationDocument:
    id: str
    text: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.text.strip():
            raise ValueError("Generation document ID and text must be nonblank")


@dataclass(frozen=True)
class GenerationCitation:
    start: int
    end: int
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[GenerationMessage, ...]
    max_output_tokens: int
    documents: tuple[GenerationDocument, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("Generation messages must be a nonempty tuple")
        if not all(isinstance(message, GenerationMessage) for message in self.messages):
            raise ValueError("Generation messages must contain GenerationMessage values")
        if not any(message.role == MessageRole.USER for message in self.messages):
            raise ValueError("Generation messages must contain a user message")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("Generation max_output_tokens must be a positive integer")
        if not isinstance(self.documents, tuple) or not all(
            isinstance(document, GenerationDocument) for document in self.documents
        ):
            raise ValueError("Generation documents must be a tuple of GenerationDocument values")
        if len({document.id for document in self.documents}) != len(self.documents):
            raise ValueError("Generation document IDs must be unique")


@dataclass(frozen=True)
class GenerationUsage:
    input_tokens: float | None = None
    output_tokens: float | None = None
    billed_input_tokens: float | None = None
    billed_output_tokens: float | None = None
    latency_ms: float | None = None
    estimated_cost_usd: Decimal | None = None
    input_usd_per_million_tokens: Decimal | None = None
    output_usd_per_million_tokens: Decimal | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: GenerationFinishReason
    model: str
    adapter: Literal["fake", "cohere"]
    usage: GenerationUsage = GenerationUsage()
    citations: tuple[GenerationCitation, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.model.strip():
            raise ValueError("Generation text and model must be nonblank")
        if not isinstance(self.finish_reason, GenerationFinishReason):
            raise ValueError("Generation finish reason is invalid")
        numeric_usage = (
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.billed_input_tokens,
            self.usage.billed_output_tokens,
            self.usage.latency_ms,
        )
        if any(
            value is not None and (not math.isfinite(value) or value < 0) for value in numeric_usage
        ):
            raise ValueError("Generation usage values must be nonnegative and finite")


class GenerationProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one complete text response for an ordered message history."""
        ...
