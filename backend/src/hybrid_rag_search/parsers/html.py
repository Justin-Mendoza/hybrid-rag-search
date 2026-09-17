"""Standard-library HTML extraction into source-locatable text blocks."""

import re
from html.parser import HTMLParser

from hybrid_rag_search.parsers.contracts import (
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode

PARSER_NAME = "stdlib-html"
PARSER_VERSION = "1"
SUPPORTED_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_CONTENT_TAGS = frozenset({"p", "li", "pre", "blockquote", "dt", "dd", "td", "th"})
_BOUNDARY_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "body",
        "div",
        "footer",
        "header",
        "main",
        "nav",
        "ol",
        "section",
        "table",
        "tr",
        "ul",
    }
)
_IGNORED_TAGS = frozenset({"head", "script", "style", "noscript", "template"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_HEADING_TAGS = frozenset({f"h{level}" for level in range(1, 7)})


def _visible_text(parts: list[str], *, preserve_lines: bool = False) -> str:
    joined = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    if preserve_lines:
        return "\n".join(line.rstrip(" \t") for line in joined.split("\n")).strip()
    return re.sub(r"\s+", " ", joined).strip()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.extracted: list[tuple[str, tuple[str, ...]]] = []
        self._headings: list[str] = []
        self._loose: list[str] = []
        self._capture_tag: str | None = None
        self._capture_depth = 0
        self._capture_parts: list[str] = []
        self._ignored_depth = 0

    def _emit(self, text: str, tag: str | None = None) -> None:
        if not text:
            return
        if tag in _HEADING_TAGS:
            level = int(tag[1:])
            self._headings = self._headings[: level - 1]
            self._headings.append(text)
        self.extracted.append((text, tuple(self._headings)))

    def _flush_loose(self) -> None:
        self._emit(_visible_text(self._loose))
        self._loose = []

    def _flush_capture(self) -> None:
        self._emit(
            _visible_text(self._capture_parts, preserve_lines=self._capture_tag == "pre"),
            self._capture_tag,
        )
        self._capture_tag = None
        self._capture_depth = 0
        self._capture_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._ignored_depth:
            if tag not in _VOID_TAGS:
                self._ignored_depth += 1
            return
        attributes = {name.lower(): value for name, value in attrs}
        hidden = "hidden" in attributes or (attributes.get("aria-hidden") or "").lower() == "true"
        if tag in _IGNORED_TAGS or hidden:
            if tag not in _VOID_TAGS:
                self._ignored_depth = 1
            return
        if self._capture_tag is not None:
            if tag == "br":
                self._capture_parts.append("\n")
            elif tag not in _VOID_TAGS:
                if tag in _CONTENT_TAGS or tag in _BOUNDARY_TAGS or tag in _HEADING_TAGS:
                    self._capture_parts.append("\n")
                self._capture_depth += 1
            return
        if tag in _CONTENT_TAGS or tag in _HEADING_TAGS:
            self._flush_loose()
            self._capture_tag = tag
            self._capture_depth = 1
        elif tag == "br":
            self._loose.append("\n")
        elif tag in _BOUNDARY_TAGS:
            self._flush_loose()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._capture_tag is not None:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._flush_capture()
        elif tag.lower() in _BOUNDARY_TAGS:
            self._flush_loose()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._capture_tag is not None:
            self._capture_parts.append(data)
        else:
            self._loose.append(data)

    def finish(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if self._capture_tag is not None:
            self._flush_capture()
        self._flush_loose()
        return tuple(self.extracted)


class HTMLDocumentParser:
    """Extract visible UTF-8 HTML text without rendering or executing content."""

    def parse(self, request: ParseRequest) -> ParsedDocument:
        media_type = request.media_type.partition(";")[0].strip().lower()
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise ParseError(
                ParseErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "HTML parser requires a supported HTML media type",
            )
        try:
            source = request.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ParseError(
                ParseErrorCode.INVALID_ENCODING,
                "HTML content must be valid UTF-8",
            ) from None
        extractor = _HTMLTextExtractor()
        extractor.feed(source)
        extractor.close()
        extracted = extractor.finish()
        if not extracted:
            raise ParseError(
                ParseErrorCode.NO_EXTRACTABLE_TEXT,
                "HTML document contains no extractable text",
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
            media_type="text/html",
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
