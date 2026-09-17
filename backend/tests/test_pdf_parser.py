from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from hybrid_rag_search.parsers.contracts import DocumentParser, ParseRequest
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.pdf import PDFParser


def pdf_fixture(*page_texts: str | None, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = content
    if password is not None:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def parse(content: bytes, media_type: str = "application/pdf"):
    parser: DocumentParser = PDFParser()
    return parser.parse(ParseRequest(content, media_type))


def test_pdf_fixture_produces_deterministic_page_locators() -> None:
    content = pdf_fixture("Page one policy", None, "Page three appendix")
    first = parse(content)
    assert first == parse(content)
    assert first.media_type == "application/pdf"
    assert first.parser_name == "pypdf"
    assert first.parser_version == "1"
    assert tuple(block.text for block in first.blocks) == (
        "Page one policy",
        "Page three appendix",
    )
    assert tuple(block.locator.block_number for block in first.blocks) == (1, 2)
    assert tuple(block.locator.page_number for block in first.blocks) == (1, 3)
    assert all(block.locator.heading_path == () for block in first.blocks)


def test_normalizes_extracted_page_text_into_paragraph_blocks(monkeypatch) -> None:
    class Page:
        def extract_text(self) -> str:
            return "First paragraph  \r\n\r\nSecond paragraph\t"

    class Reader:
        is_encrypted = False
        pages = (Page(),)

    monkeypatch.setattr("hybrid_rag_search.parsers.pdf.PdfReader", lambda *args, **kwargs: Reader())
    result = parse(b"synthetic")
    assert tuple(block.text for block in result.blocks) == (
        "First paragraph",
        "Second paragraph",
    )
    assert tuple(block.locator.page_number for block in result.blocks) == (1, 1)


def test_rejects_encrypted_pdf_without_exposing_password_or_content() -> None:
    content = pdf_fixture("confidential", password="secret-password")
    with pytest.raises(ParseError, match="Encrypted PDF") as caught:
        parse(content)
    assert caught.value.code == ParseErrorCode.CORRUPT_DOCUMENT
    assert "secret-password" not in str(caught.value)
    assert "confidential" not in str(caught.value)


@pytest.mark.parametrize("content", [b"", b"not a pdf", b"%PDF-1.4 broken"])
def test_rejects_corrupt_pdf_with_stable_safe_error(content: bytes) -> None:
    with pytest.raises(ParseError, match="could not be parsed") as caught:
        parse(content)
    assert caught.value.code == ParseErrorCode.CORRUPT_DOCUMENT
    assert repr(content) not in str(caught.value)


def test_rejects_pdf_without_extractable_text_and_explains_ocr_boundary() -> None:
    with pytest.raises(ParseError, match="OCR is not supported") as caught:
        parse(pdf_fixture(None))
    assert caught.value.code == ParseErrorCode.NO_EXTRACTABLE_TEXT


def test_accepts_case_and_parameters_in_pdf_media_type() -> None:
    result = parse(pdf_fixture("Text"), " Application/PDF ; version=1.7")
    assert result.blocks[0].text == "Text"


@pytest.mark.parametrize("media_type", ["text/plain", "text/html", "text/markdown"])
def test_rejects_unsupported_media_type(media_type: str) -> None:
    with pytest.raises(ParseError, match="application/pdf") as caught:
        parse(b"not inspected", media_type)
    assert caught.value.code == ParseErrorCode.UNSUPPORTED_MEDIA_TYPE
