"""Media-type routing for application-owned document parsers."""

from collections.abc import Mapping

from hybrid_rag_search.parsers.contracts import DocumentParser, ParsedDocument, ParseRequest
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.html import HTMLDocumentParser
from hybrid_rag_search.parsers.markdown import MarkdownParser
from hybrid_rag_search.parsers.pdf import PDFParser
from hybrid_rag_search.parsers.plain_text import PlainTextParser


def base_media_type(media_type: str) -> str:
    return media_type.partition(";")[0].strip().lower()


class ParserDispatcher:
    """Select a parser without exposing format-specific classes to callers."""

    def __init__(self, parsers: Mapping[str, DocumentParser]) -> None:
        self._parsers = {
            base_media_type(media_type): parser for media_type, parser in parsers.items()
        }

    def parse(self, request: ParseRequest) -> ParsedDocument:
        parser = self._parsers.get(base_media_type(request.media_type))
        if parser is None:
            raise ParseError(
                ParseErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "No document parser supports the declared media type",
            )
        return parser.parse(request)


def default_parser_dispatcher() -> ParserDispatcher:
    """Build the supported local parser set with explicit media-type aliases."""

    markdown = MarkdownParser()
    html = HTMLDocumentParser()
    return ParserDispatcher(
        {
            "application/pdf": PDFParser(),
            "text/markdown": markdown,
            "text/x-markdown": markdown,
            "text/html": html,
            "application/xhtml+xml": html,
            "text/plain": PlainTextParser(),
        }
    )
