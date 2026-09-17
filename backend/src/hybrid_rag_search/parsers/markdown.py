"""CommonMark parsing into application-owned text blocks."""

from markdown_it import MarkdownIt
from markdown_it.token import Token

from hybrid_rag_search.parsers.contracts import (
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode

PARSER_NAME = "markdown-it-py"
PARSER_VERSION = "1"
SUPPORTED_MEDIA_TYPES = frozenset({"text/markdown", "text/x-markdown"})


def _inline_text(token: Token) -> str:
    parts: list[str] = []
    for child in token.children or ():
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif child.type == "image":
            parts.append(child.content)
    return "".join(parts).strip()


class MarkdownParser:
    """Parse UTF-8 CommonMark without rendering or retaining raw HTML."""

    def __init__(self) -> None:
        # Recognize raw HTML as dedicated tokens so extraction can omit it. We
        # parse tokens only and never invoke the HTML renderer.
        self._markdown = MarkdownIt("commonmark", {"html": True})

    def parse(self, request: ParseRequest) -> ParsedDocument:
        media_type = request.media_type.partition(";")[0].strip().lower()
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise ParseError(
                ParseErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "Markdown parser requires a supported Markdown media type",
            )
        try:
            source = request.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ParseError(
                ParseErrorCode.INVALID_ENCODING,
                "Markdown content must be valid UTF-8",
            ) from None

        tokens = self._markdown.parse(source)
        headings: list[str] = []
        extracted: list[tuple[str, tuple[str, ...]]] = []
        for index, token in enumerate(tokens):
            if token.type == "heading_open":
                level = int(token.tag[1:])
                text = _inline_text(tokens[index + 1])
                if text:
                    headings = headings[: level - 1]
                    headings.append(text)
                    extracted.append((text, tuple(headings)))
            elif token.type == "inline" and (
                index == 0 or tokens[index - 1].type != "heading_open"
            ):
                text = _inline_text(token)
                if text:
                    extracted.append((text, tuple(headings)))
            elif token.type in {"fence", "code_block"}:
                text = token.content.rstrip()
                if text.strip():
                    extracted.append((text, tuple(headings)))

        if not extracted:
            raise ParseError(
                ParseErrorCode.NO_EXTRACTABLE_TEXT,
                "Markdown document contains no extractable text",
            )
        blocks = tuple(
            ParsedBlock(
                text=text,
                locator=SourceLocator(block_number=number, heading_path=heading_path),
            )
            for number, (text, heading_path) in enumerate(extracted, start=1)
        )
        return ParsedDocument(
            blocks=blocks,
            media_type="text/markdown",
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
