"""Shared parser contract, independent of any file-format library."""

from dataclasses import dataclass
from typing import Protocol


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class ParseRequest:
    """Immutable original bytes and their declared media type."""

    content: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise ValueError("Parse content must be bytes")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("Parse media type must be a nonblank string")


@dataclass(frozen=True)
class SourceLocator:
    """A user-facing location retained from the original document.

    Block numbers are always present and one-based. Format-specific parsers may
    additionally retain a one-based PDF page and/or a nested heading path.
    """

    block_number: int
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.block_number):
            raise ValueError("Block number must be a positive integer")
        if self.page_number is not None and not _is_positive_integer(self.page_number):
            raise ValueError("Page number must be a positive integer when present")
        if not isinstance(self.heading_path, tuple) or any(
            not isinstance(heading, str) or not heading.strip() for heading in self.heading_path
        ):
            raise ValueError("Heading path must be a tuple of nonblank strings")


@dataclass(frozen=True)
class ParsedBlock:
    """One ordered unit of normalized, citation-locatable text."""

    text: str
    locator: SourceLocator

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Parsed block text must be a nonblank string")
        if not isinstance(self.locator, SourceLocator):
            raise ValueError("Parsed block locator must be a SourceLocator")


@dataclass(frozen=True)
class ParsedDocument:
    """Deterministic parser output shared by chunking and later citations."""

    blocks: tuple[ParsedBlock, ...]
    media_type: str
    parser_name: str
    parser_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ValueError("Parsed document blocks must be a nonempty tuple")
        if any(not isinstance(block, ParsedBlock) for block in self.blocks):
            raise ValueError("Parsed document blocks must contain ParsedBlock values")
        expected_numbers = tuple(range(1, len(self.blocks) + 1))
        actual_numbers = tuple(block.locator.block_number for block in self.blocks)
        if actual_numbers != expected_numbers:
            raise ValueError("Parsed document block numbers must be consecutive and ordered")
        for label, value in (
            ("media type", self.media_type),
            ("parser name", self.parser_name),
            ("parser version", self.parser_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Parsed document {label} must be a nonblank string")


class DocumentParser(Protocol):
    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Return ordered normalized blocks or raise a stable parsing error."""
        ...
