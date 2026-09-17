from typing import Any

import pytest

from hybrid_rag_search.parsers.contracts import (
    DocumentParser,
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)


def block(number: int = 1) -> ParsedBlock:
    return ParsedBlock("Normalized text", SourceLocator(block_number=number))


@pytest.mark.parametrize(
    ("content", "media_type", "message"),
    [
        ("not bytes", "text/plain", "content"),
        (b"text", "", "media type"),
        (b"text", " \n", "media type"),
        (b"text", 42, "media type"),
    ],
)
def test_parse_request_rejects_invalid_values(content: Any, media_type: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ParseRequest(content=content, media_type=media_type)


def test_parse_request_preserves_original_bytes() -> None:
    request = ParseRequest(content=b"\x00 unchanged \xff", media_type="application/pdf")
    assert request.content == b"\x00 unchanged \xff"
    assert request.media_type == "application/pdf"


@pytest.mark.parametrize("block_number", [0, -1, True, 1.5, "1"])
def test_source_locator_requires_positive_block_number(block_number: Any) -> None:
    with pytest.raises(ValueError, match="Block number"):
        SourceLocator(block_number=block_number)


@pytest.mark.parametrize("page_number", [0, -1, True, 1.5, "1"])
def test_source_locator_rejects_invalid_page_number(page_number: Any) -> None:
    with pytest.raises(ValueError, match="Page number"):
        SourceLocator(block_number=1, page_number=page_number)


@pytest.mark.parametrize("heading_path", [["Heading"], ("",), (" \n",), (42,), "Heading"])
def test_source_locator_rejects_invalid_heading_path(heading_path: Any) -> None:
    with pytest.raises(ValueError, match="Heading path"):
        SourceLocator(block_number=1, heading_path=heading_path)


def test_source_locator_retains_available_source_structure() -> None:
    locator = SourceLocator(
        block_number=3,
        page_number=2,
        heading_path=("Benefits", "Remote work"),
    )
    assert locator.block_number == 3
    assert locator.page_number == 2
    assert locator.heading_path == ("Benefits", "Remote work")


@pytest.mark.parametrize("text", ["", " \n", 42])
def test_parsed_block_requires_nonblank_text(text: Any) -> None:
    with pytest.raises(ValueError, match="text"):
        ParsedBlock(text=text, locator=SourceLocator(block_number=1))


def test_parsed_block_requires_source_locator() -> None:
    with pytest.raises(ValueError, match="locator"):
        ParsedBlock(text="Text", locator="page 1")  # type: ignore[arg-type]


@pytest.mark.parametrize("blocks", [(), [], ("text",)])
def test_parsed_document_requires_nonempty_block_tuple(blocks: Any) -> None:
    with pytest.raises(ValueError, match="blocks"):
        ParsedDocument(
            blocks=blocks,
            media_type="text/plain",
            parser_name="plain-text",
            parser_version="1",
        )


@pytest.mark.parametrize(
    "blocks",
    [
        (block(2),),
        (block(1), block(3)),
        (block(1), block(1)),
    ],
)
def test_parsed_document_requires_consecutive_ordered_block_numbers(
    blocks: tuple[ParsedBlock, ...],
) -> None:
    with pytest.raises(ValueError, match="consecutive and ordered"):
        ParsedDocument(
            blocks=blocks,
            media_type="text/plain",
            parser_name="plain-text",
            parser_version="1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("media_type", "", "media type"),
        ("media_type", 42, "media type"),
        ("parser_name", " ", "parser name"),
        ("parser_name", 42, "parser name"),
        ("parser_version", "", "parser version"),
        ("parser_version", 42, "parser version"),
    ],
)
def test_parsed_document_requires_parser_identity(field: str, value: Any, message: str) -> None:
    values: dict[str, Any] = {
        "blocks": (block(),),
        "media_type": "text/plain",
        "parser_name": "plain-text",
        "parser_version": "1",
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ParsedDocument(**values)


def test_document_parser_contract_is_structural() -> None:
    class ExampleParser:
        def parse(self, request: ParseRequest) -> ParsedDocument:
            return ParsedDocument(
                blocks=(ParsedBlock(request.content.decode(), SourceLocator(1)),),
                media_type=request.media_type,
                parser_name="example",
                parser_version="1",
            )

    parser: DocumentParser = ExampleParser()
    result = parser.parse(ParseRequest(b"Text", "text/plain"))
    assert result.blocks[0].text == "Text"
    assert result.blocks[0].locator.block_number == 1
    assert result.parser_name == "example"
