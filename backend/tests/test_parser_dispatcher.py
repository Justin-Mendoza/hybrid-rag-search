from dataclasses import dataclass

import pytest

from hybrid_rag_search.parsers.contracts import (
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)
from hybrid_rag_search.parsers.dispatcher import ParserDispatcher, default_parser_dispatcher
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode


@dataclass
class RecordingParser:
    received: ParseRequest | None = None

    def parse(self, request: ParseRequest) -> ParsedDocument:
        self.received = request
        return ParsedDocument(
            blocks=(ParsedBlock("parsed", SourceLocator(1)),),
            media_type="application/example",
            parser_name="recording",
            parser_version="1",
        )


def test_dispatches_original_request_using_normalized_media_type() -> None:
    parser = RecordingParser()
    dispatcher = ParserDispatcher({"application/example": parser})
    request = ParseRequest(b"\x00original\xff", " Application/Example ; version=1")
    result = dispatcher.parse(request)
    assert result.parser_name == "recording"
    assert parser.received is request
    assert parser.received.content == b"\x00original\xff"


def test_dispatcher_copies_registration_mapping() -> None:
    parser = RecordingParser()
    registrations = {"application/example": parser}
    dispatcher = ParserDispatcher(registrations)
    registrations.clear()
    assert dispatcher.parse(ParseRequest(b"text", "application/example")).parser_name == "recording"


@pytest.mark.parametrize(
    ("media_type", "content", "parser_name"),
    [
        ("text/plain", b"Text", "plain-text"),
        ("text/markdown", b"# Text", "markdown-it-py"),
        ("text/x-markdown; charset=utf-8", b"# Text", "markdown-it-py"),
        ("text/html", b"<p>Text</p>", "stdlib-html"),
        ("application/xhtml+xml", b"<p>Text</p>", "stdlib-html"),
    ],
)
def test_default_dispatcher_routes_text_formats(
    media_type: str, content: bytes, parser_name: str
) -> None:
    result = default_parser_dispatcher().parse(ParseRequest(content, media_type))
    assert result.parser_name == parser_name


def test_default_dispatcher_routes_pdf(monkeypatch) -> None:
    class Page:
        def extract_text(self) -> str:
            return "PDF text"

    class Reader:
        is_encrypted = False
        pages = (Page(),)

    monkeypatch.setattr("hybrid_rag_search.parsers.pdf.PdfReader", lambda *args, **kwargs: Reader())
    result = default_parser_dispatcher().parse(ParseRequest(b"pdf", "application/pdf"))
    assert result.parser_name == "pypdf"
    assert result.blocks[0].locator.page_number == 1


def test_unsupported_media_type_has_stable_error() -> None:
    with pytest.raises(ParseError, match="No document parser") as caught:
        default_parser_dispatcher().parse(ParseRequest(b"data", "application/zip"))
    assert caught.value.code == ParseErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_parser_errors_propagate_unchanged() -> None:
    dispatcher = default_parser_dispatcher()
    with pytest.raises(ParseError, match="valid UTF-8") as caught:
        dispatcher.parse(ParseRequest(b"secret:\xff", "text/plain"))
    assert caught.value.code == ParseErrorCode.INVALID_ENCODING
