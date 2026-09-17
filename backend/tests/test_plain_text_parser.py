from pathlib import Path

import pytest

from hybrid_rag_search.parsers.contracts import DocumentParser, ParseRequest
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.plain_text import PlainTextParser

FIXTURES = Path(__file__).parent / "fixtures" / "parsers"


def parse(content: bytes, media_type: str = "text/plain"):
    parser: DocumentParser = PlainTextParser()
    return parser.parse(ParseRequest(content, media_type))


def test_fixture_produces_deterministic_paragraph_blocks() -> None:
    content = (FIXTURES / "plain_text.txt").read_bytes()
    first = parse(content)
    assert first == parse(content)
    assert first.media_type == "text/plain"
    assert first.parser_name == "plain-text"
    assert first.parser_version == "1"
    assert tuple(block.text for block in first.blocks) == (
        "Employee Handbook\nRemote Work Policy",
        (
            "Employees may work remotely three days per week.  Internal spacing stays intact.\n"
            "Manager approval is required."
        ),
        "Contact People Operations for exceptions.",
    )
    assert tuple(block.locator.block_number for block in first.blocks) == (1, 2, 3)
    assert all(block.locator.page_number is None for block in first.blocks)
    assert all(block.locator.heading_path == () for block in first.blocks)


def test_normalizes_all_line_endings_and_collapses_blank_lines() -> None:
    result = parse(b"First line  \rsecond line\r\n\r\n\r\nSecond paragraph\t\n")
    assert tuple(block.text for block in result.blocks) == (
        "First line\nsecond line",
        "Second paragraph",
    )


def test_accepts_utf8_bom_and_media_type_parameters() -> None:
    result = parse("\ufeffCaf\u00e9".encode(), " Text/Plain ; charset=utf-8")
    assert result.blocks[0].text == "Caf\u00e9"


def test_preserves_leading_indentation_and_internal_spacing() -> None:
    result = parse(b"  indented line\nspaces  between  words")
    assert result.blocks[0].text == "  indented line\nspaces  between  words"


@pytest.mark.parametrize("content", [b"", b" \t\r\n\n\t "])
def test_rejects_content_without_extractable_text(content: bytes) -> None:
    with pytest.raises(ParseError, match="no extractable text") as caught:
        parse(content)
    assert caught.value.code == ParseErrorCode.NO_EXTRACTABLE_TEXT


def test_rejects_invalid_utf8_without_exposing_bytes() -> None:
    with pytest.raises(ParseError, match="valid UTF-8") as caught:
        parse(b"secret:\xff")
    assert caught.value.code == ParseErrorCode.INVALID_ENCODING
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("media_type", ["text/html", "application/pdf", ""])
def test_rejects_unsupported_media_type(media_type: str) -> None:
    if media_type:
        with pytest.raises(ParseError, match="text/plain") as caught:
            parse(b"text", media_type)
        assert caught.value.code == ParseErrorCode.UNSUPPORTED_MEDIA_TYPE
    else:
        with pytest.raises(ValueError, match="media type"):
            parse(b"text", media_type)
