"""Page-aware PDF text extraction using pypdf."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from hybrid_rag_search.parsers.contracts import (
    ParsedBlock,
    ParsedDocument,
    ParseRequest,
    SourceLocator,
)
from hybrid_rag_search.parsers.errors import ParseError, ParseErrorCode
from hybrid_rag_search.parsers.normalization import normalize_paragraphs

PARSER_NAME = "pypdf"
PARSER_VERSION = "1"


class PDFParser:
    """Extract embedded text while retaining one-based source page numbers."""

    def parse(self, request: ParseRequest) -> ParsedDocument:
        media_type = request.media_type.partition(";")[0].strip().lower()
        if media_type != "application/pdf":
            raise ParseError(
                ParseErrorCode.UNSUPPORTED_MEDIA_TYPE,
                "PDF parser requires the application/pdf media type",
            )
        try:
            reader = PdfReader(BytesIO(request.content), strict=False)
            if reader.is_encrypted:
                raise ParseError(
                    ParseErrorCode.CORRUPT_DOCUMENT,
                    "Encrypted PDF documents are not supported",
                )
            extracted = tuple(
                (page_number, paragraph)
                for page_number, page in enumerate(reader.pages, start=1)
                for paragraph in normalize_paragraphs(page.extract_text() or "")
            )
        except PyPdfError:
            raise ParseError(
                ParseErrorCode.CORRUPT_DOCUMENT,
                "PDF document could not be parsed",
            ) from None
        if not extracted:
            raise ParseError(
                ParseErrorCode.NO_EXTRACTABLE_TEXT,
                "PDF document contains no extractable text; OCR is not supported",
            )
        blocks = tuple(
            ParsedBlock(
                text=text,
                locator=SourceLocator(block_number=block_number, page_number=page_number),
            )
            for block_number, (page_number, text) in enumerate(extracted, start=1)
        )
        return ParsedDocument(
            blocks=blocks,
            media_type="application/pdf",
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
        )
