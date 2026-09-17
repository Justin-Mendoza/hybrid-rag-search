from pathlib import Path

import pytest

from hybrid_rag_search.parsers.contracts import DocumentParser, ParseRequest
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.markdown import MarkdownParser

FIXTURES = Path(__file__).parent / "fixtures" / "parsers"


def parse(content: bytes, media_type: str = "text/markdown"):
    parser: DocumentParser = MarkdownParser()
    return parser.parse(ParseRequest(content, media_type))


def test_fixture_produces_deterministic_blocks_and_heading_paths() -> None:
    content = (FIXTURES / "handbook.md").read_bytes()
    first = parse(content)
    assert first == parse(content)
    assert first.media_type == "text/markdown"
    assert first.parser_name == "markdown-it-py"
    assert first.parser_version == "1"
    assert tuple(block.text for block in first.blocks) == (
        "Employee Handbook",
        "Welcome to Acme. Read the full policy.",
        "Remote Work",
        "Employees may work remotely.",
        "Manager approval is required.",
        "Example",
        "days = 3",
        "Security",
        "Use MFA for all systems.  Internal spacing stays intact.",
        "Architecture diagram",
    )
    assert tuple(block.locator.heading_path for block in first.blocks) == (
        ("Employee Handbook",),
        ("Employee Handbook",),
        ("Employee Handbook", "Remote Work"),
        ("Employee Handbook", "Remote Work"),
        ("Employee Handbook", "Remote Work"),
        ("Employee Handbook", "Remote Work", "Example"),
        ("Employee Handbook", "Remote Work", "Example"),
        ("Employee Handbook", "Security"),
        ("Employee Handbook", "Security"),
        ("Employee Handbook", "Security"),
    )
    assert tuple(block.locator.block_number for block in first.blocks) == tuple(range(1, 11))
    assert "do_not_index" not in " ".join(block.text for block in first.blocks)


def test_skipped_heading_levels_retain_available_hierarchy() -> None:
    result = parse(b"# One\n\n### Three\n\nText")
    assert tuple(block.locator.heading_path for block in result.blocks) == (
        ("One",),
        ("One", "Three"),
        ("One", "Three"),
    )


def test_preserves_soft_and_hard_line_breaks() -> None:
    result = parse(b"first\nsecond  \nthird")
    assert result.blocks[0].text == "first\nsecond\nthird"


def test_accepts_bom_and_markdown_media_type_alias() -> None:
    result = parse("\ufeff# Caf\u00e9".encode(), " Text/X-Markdown ; charset=utf-8")
    assert result.blocks[0].text == "Caf\u00e9"


@pytest.mark.parametrize("content", [b"", b"  \n\t", b"<!-- comment only -->"])
def test_rejects_content_without_extractable_text(content: bytes) -> None:
    with pytest.raises(ParseError, match="no extractable text") as caught:
        parse(content)
    assert caught.value.code == ParseErrorCode.NO_EXTRACTABLE_TEXT


def test_rejects_invalid_utf8_without_exposing_bytes() -> None:
    with pytest.raises(ParseError, match="valid UTF-8") as caught:
        parse(b"secret:\xff")
    assert caught.value.code == ParseErrorCode.INVALID_ENCODING
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("media_type", ["text/plain", "text/html", "application/pdf"])
def test_rejects_unsupported_media_type(media_type: str) -> None:
    with pytest.raises(ParseError, match="Markdown media type") as caught:
        parse(b"text", media_type)
    assert caught.value.code == ParseErrorCode.UNSUPPORTED_MEDIA_TYPE
