"""Deterministic UTF-8 plain-text parsing."""

from hybrid_rag_search.parsers.contracts import (
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.normalization import normalize_paragraphs

PARSER_NAME = "plain-text"
PARSER_VERSION = "1"


class PlainTextParser:
    """Parse UTF-8 text into paragraph blocks without encoding detection."""

    def parse(self, request: ParseRequest) -> ParsedDocument:
        media_type = request.media_type.partition(";")[0].strip().lower()
        if media_type != "text/plain":
            raise ParseError(
                ParseErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "Plain-text parser requires the text/plain media type",
            )
        try:
            text = request.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ParseError(
                ParseErrorCode.INVALID_ENCODING,
                "Plain-text content must be valid UTF-8",
            ) from None
        paragraphs = normalize_paragraphs(text)
        if not paragraphs:
            raise ParseError(
                ParseErrorCode.NO_EXTRACTABLE_TEXT,
                "Plain-text document contains no extractable text",
            )
        blocks = tuple(
            ParsedBlock(paragraph, SourceLocator(block_number=index))
            for index, paragraph in enumerate(paragraphs, start=1)
        )
        return ParsedDocument(
            blocks=blocks,
            media_type="text/plain",
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
