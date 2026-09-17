"""Stable application errors; never expose provider response bodies or headers."""

from typing import Literal

ProviderErrorCode = Literal[
    "timeout",
    "unavailable",
    "authentication",
    "rate_limited",
    "invalid_request",
    "invalid_response",
]


class ProviderError(Exception):
    def __init__(self, code: ProviderErrorCode, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"Model provider error: {code}")
