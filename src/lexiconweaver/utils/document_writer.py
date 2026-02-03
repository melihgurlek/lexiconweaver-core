"""Write document content to text, PDF, and EPUB files."""

import html
from pathlib import Path

from lexiconweaver.exceptions import ValidationError

SUPPORTED_EXPORT_EXTENSIONS = (".txt", ".pdf", ".epub")


def _write_txt(path: Path, text: str) -> None:
    """Write plain text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_pdf(path: Path, text: str, title: str = "Translation") -> None:
    """Write PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise ValidationError(
            "PDF export requires the fpdf2 package. "
            "Install it with: pip install fpdf2"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_title(title[:255] if title else "Translation")

    # Split into paragraphs and write (multi_cell for wrapping)
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        block = block.strip()
        if not block:
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 6, block)
        pdf.ln(2)

    pdf.output(str(path))


def _write_epub(path: Path, text: str, title: str = "Translation") -> None:
    """Write EPUB using ebooklib."""
    try:
        from ebooklib import epub
    except ImportError as e:
        raise ValidationError(
            "EPUB export requires the ebooklib package. "
            "Install it with: pip install ebooklib"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier("translation-1")
    book.set_title(title[:255] if title else "Translation")
    book.set_language("en")

    # Build XHTML content from paragraphs
    paragraphs = [
        p.strip()
        for p in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
        if p.strip()
    ]
    body_parts = [
        f"<p>{html.escape(p)}</p>" for p in paragraphs
    ]
    content = "<!DOCTYPE html><html><head><meta charset='utf-8'/></head><body>\n" + "\n".join(body_parts) + "\n</body></html>"

    chapter = epub.EpubHtml(
        title=title[:255] if title else "Content",
        file_name="chapter.xhtml",
        lang="en",
    )
    chapter.content = content
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [chapter]

    epub.write_epub(str(path), book)


def write_document(
    path: Path,
    text: str,
    title: str = "Translation",
) -> None:
    """
    Write text to a file in the given format.

    Format is inferred from path suffix:
    - .txt  – plain text (UTF-8)
    - .pdf  – PDF (requires fpdf2)
    - .epub – EPUB (requires ebooklib)

    Raises ValidationError if the format is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_EXTENSIONS:
        raise ValidationError(
            f"Unsupported export format: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXPORT_EXTENSIONS)}"
        )

    if suffix == ".txt":
        _write_txt(path, text)
    elif suffix == ".pdf":
        _write_pdf(path, text, title=title)
    elif suffix == ".epub":
        _write_epub(path, text, title=title)
