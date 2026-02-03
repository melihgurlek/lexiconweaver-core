"""Load document content from text, EPUB, and PDF files."""

from html.parser import HTMLParser
from pathlib import Path

from lexiconweaver.exceptions import ValidationError
from lexiconweaver.utils.validators import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    validate_document_file,
)


class _HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, ignoring tags."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _load_txt(file_path: Path, encoding: str) -> str:
    """Load plain text file."""
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def _load_epub(file_path: Path) -> str:
    """Load EPUB and return concatenated plain text from all document items."""
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as e:
        raise ValidationError(
            "EPUB support requires the ebooklib package. "
            "Install it with: pip install ebooklib"
        ) from e

    book = epub.read_epub(str(file_path))
    parts: list[str] = []

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        raw = item.get_content()
        if not raw:
            continue
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = raw.decode("latin-1", errors="replace")
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text().strip()
        if text:
            parts.append(text)

    if not parts:
        raise ValidationError(f"No text content found in EPUB: {file_path}")

    return "\n\n".join(parts)


def _load_pdf(file_path: Path) -> str:
    """Load PDF and return extracted text from all pages."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValidationError(
            "PDF support requires the pypdf package. "
            "Install it with: pip install pypdf"
        ) from e

    reader = PdfReader(str(file_path))
    parts: list[str] = []

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            parts.append(text.strip())

    if not parts:
        raise ValidationError(f"No text content found in PDF: {file_path}")

    return "\n\n".join(parts)


def load_document(file_path: Path) -> str:
    """
    Load document content from a file.

    Supports:
    - .txt  – plain text (encoding auto-detected for validation)
    - .epub – EPUB e-books (requires ebooklib)
    - .pdf  – PDF documents (requires pypdf)

    Raises ValidationError if the file is invalid or format is unsupported.
    """
    path, encoding = validate_document_file(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _load_txt(path, encoding or "utf-8")
    if suffix == ".epub":
        return _load_epub(path)
    if suffix == ".pdf":
        return _load_pdf(path)

    raise ValidationError(
        f"Unsupported document format: {suffix}. "
        f"Supported: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}"
    )
