"""Stable application errors for document parsing failures."""

from enum import StrEnum


class ParseErrorCode(StrEnum):
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    INVALID_ENCODING = "invalid_encoding"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"
    CORRUPT_DOCUMENT = "corrupt_document"


class ParseError(ValueError):
    """A safe parser failure that does not expose document contents."""

    def __init__(self, code: ParseErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)
