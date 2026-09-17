from pathlib import Path

import pytest

from hybrid_rag_search.parsers.contracts import ParseRequest
from hybrid_rag_search.parsers.dispatcher import default_parser_dispatcher
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode

FIXTURES = Path(__file__).parent / "fixtures" / "parsers"


@pytest.mark.parametrize(
    ("filename", "media_type", "parser_name", "expected_text", "expected_heading"),
    [
        (
            "plain_text.txt",
            "text/plain",
            "plain-text",
            "Employee Handbook\nRemote Work Policy",
            (),
        ),
        (
            "handbook.md",
            "text/markdown",
            "markdown-it-py",
            "Employee Handbook",
            ("Employee Handbook",),
        ),
        (
            "handbook.html",
            "text/html",
            "stdlib-html",
            "Employee & Contractor Handbook",
            ("Employee & Contractor Handbook",),
        ),
    ],
)
def test_representative_file_is_deterministic_through_dispatcher(
    filename: str,
    media_type: str,
    parser_name: str,
    expected_text: str,
    expected_heading: tuple[str, ...],
) -> None:
    request = ParseRequest((FIXTURES / filename).read_bytes(), media_type)
    dispatcher = default_parser_dispatcher()
    first = dispatcher.parse(request)
    assert first == dispatcher.parse(request)
    assert first.parser_name == parser_name
    assert first.blocks[0].text == expected_text
    assert first.blocks[0].locator.heading_path == expected_heading
    assert tuple(block.locator.block_number for block in first.blocks) == tuple(
        range(1, len(first.blocks) + 1)
    )


def test_corrupt_document_does_not_affect_next_parse() -> None:
    dispatcher = default_parser_dispatcher()
    with pytest.raises(ParseError) as caught:
        dispatcher.parse(ParseRequest(b"%PDF-1.4 broken", "application/pdf"))
    assert caught.value.code == ParseErrorCode.CORRUPT_DOCUMENT

    recovered = dispatcher.parse(ParseRequest(b"A healthy document", "text/plain"))
    assert recovered.blocks[0].text == "A healthy document"
    assert recovered.parser_name == "plain-text"
