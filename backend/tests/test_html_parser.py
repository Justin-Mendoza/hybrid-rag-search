from pathlib import Path

import pytest

from hybrid_rag_search.parsers.contracts import DocumentParser, ParseRequest
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.html import HTMLDocumentParser

FIXTURES = Path(__file__).parent / "fixtures" / "parsers"


def parse(content: bytes, media_type: str = "text/html"):
    parser: DocumentParser = HTMLDocumentParser()
    return parser.parse(ParseRequest(content, media_type))


def test_fixture_produces_visible_blocks_and_heading_paths() -> None:
    content = (FIXTURES / "handbook.html").read_bytes()
    first = parse(content)
    assert first == parse(content)
    assert first.media_type == "text/html"
    assert first.parser_name == "stdlib-html"
    assert first.parser_version == "1"
    assert tuple(block.text for block in first.blocks) == (
        "Employee & Contractor Handbook",
        "Welcome to Acme. Read the policy.",
        "Remote Work",
        "Employees may work remotely.",
        "Manager approval is required.",
        "Example",
        'days = 3\nlocation = "home"',
        "Security",
        "Loose visible text.",
    )
    assert tuple(block.locator.heading_path for block in first.blocks) == (
        ("Employee & Contractor Handbook",),
        ("Employee & Contractor Handbook",),
        ("Employee & Contractor Handbook", "Remote Work"),
        ("Employee & Contractor Handbook", "Remote Work"),
        ("Employee & Contractor Handbook", "Remote Work"),
        ("Employee & Contractor Handbook", "Remote Work", "Example"),
        ("Employee & Contractor Handbook", "Remote Work", "Example"),
        ("Employee & Contractor Handbook", "Security"),
        ("Employee & Contractor Handbook", "Security"),
    )
    assert tuple(block.locator.block_number for block in first.blocks) == tuple(range(1, 10))
    searchable = " ".join(block.text for block in first.blocks)
    for excluded in (
        "Hidden browser title",
        "do_not_index",
        "display: none",
        "hidden attribute text",
        "aria hidden text",
        "noscript fallback",
        "template content",
    ):
        assert excluded not in searchable


def test_nested_block_elements_emit_one_outer_block_without_duplicates() -> None:
    result = parse(b"<blockquote><p>One</p><p>Two</p></blockquote>")
    assert tuple(block.text for block in result.blocks) == ("One Two",)


def test_self_closing_and_void_elements_do_not_hide_following_text() -> None:
    result = parse(b"Before<br/>after<img src='x'>end<input hidden>visible")
    assert tuple(block.text for block in result.blocks) == ("Before afterendvisible",)


def test_unclosed_visible_block_is_flushed_deterministically() -> None:
    result = parse(b"<h1>Heading</h1><p>Unclosed")
    assert tuple(block.text for block in result.blocks) == ("Heading", "Unclosed")


def test_accepts_bom_and_xhtml_media_type() -> None:
    result = parse("\ufeff<h1>Caf\u00e9</h1>".encode(), " Application/XHTML+XML ; charset=utf-8")
    assert result.blocks[0].text == "Caf\u00e9"
    assert result.media_type == "text/html"


@pytest.mark.parametrize(
    "content",
    [b"", b" \n\t", b"<script>only script</script>", b"<div hidden>only hidden</div>"],
)
def test_rejects_content_without_extractable_text(content: bytes) -> None:
    with pytest.raises(ParseError, match="no extractable text") as caught:
        parse(content)
    assert caught.value.code == ParseErrorCode.NO_EXTRACTABLE_TEXT


def test_rejects_invalid_utf8_without_exposing_bytes() -> None:
    with pytest.raises(ParseError, match="valid UTF-8") as caught:
        parse(b"<p>secret:\xff</p>")
    assert caught.value.code == ParseErrorCode.INVALID_ENCODING
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("media_type", ["text/plain", "text/markdown", "application/pdf"])
def test_rejects_unsupported_media_type(media_type: str) -> None:
    with pytest.raises(ParseError, match="HTML media type") as caught:
        parse(b"<p>text</p>", media_type)
    assert caught.value.code == ParseErrorCode.UNSUPPORTED_MEDIA_TYPE
