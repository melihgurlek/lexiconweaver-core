import html
import os
from pathlib import Path

from lexiconweaver.exceptions import ValidationError
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXPORT_EXTENSIONS = (".txt", ".pdf", ".epub")

def _get_unicode_font_path() -> Path | None:
    """
    Return path to a Unicode-supporting TTF (Noto Sans).
    Priority:
    1. Bundled project asset (../assets/fonts/NotoSans-Regular.ttf)
    2. System font locations
    """
    base_dir = Path(__file__).resolve().parent.parent
    bundled_font = base_dir / "assets" / "fonts" / "NotoSans-VariableFont_wdth,wght.ttf"
    
    if bundled_font.is_file():
        return bundled_font

    search_dirs: list[Path] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        search_dirs.append(Path(windir) / "Fonts")
    else:
        search_dirs.extend([
            Path("/usr/share/fonts/google-noto-vf"),
            Path("/usr/share/fonts/google-noto-sans"),
            Path("/usr/share/fonts/google-noto"),
            Path("/usr/share/fonts/noto"),
            Path("/usr/share/fonts/truetype/noto"),
            Path("/usr/share/fonts/TTF"),
            Path.home() / ".local" / "share" / "fonts",
        ])

    font_names = (
        "NotoSans[wght].ttf",
        "NotoSans-Regular.ttf",
        "NotoSans.ttf",
        "DejaVuSans.ttf",
        "NotoSans-VariableFont_wdth,wght.ttf",
    )

    for directory in search_dirs:
        if not directory.is_dir():
            continue

        for name in font_names:
            path = directory / name
            if path.is_file():
                return path

        for child in directory.rglob("*.ttf"):
            if child.name in font_names:
                return child
            if child.stem in ("NotoSans", "NotoSans-Regular"):
                return child

    return None


def _write_txt(path: Path, text: str) -> None:
    """Write plain text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_pdf(path: Path, text: str, title: str = "Translation") -> None:
    """Write PDF using fpdf2. Uses Noto Sans/DejaVu for Unicode support."""
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

    font_path = _get_unicode_font_path()
    
    # Check if text contains non-ASCII characters
    has_unicode = any(ord(c) > 127 for c in text)
    
    if font_path:
        try:
            pdf.add_font("BodyFont", style="", fname=str(font_path), uni=True)
            pdf.set_font("BodyFont", size=12)
            logger.debug("Using Unicode font for PDF export", font_path=str(font_path))
        except Exception as e:
            if has_unicode:
                raise ValidationError(
                    f"Failed to load Unicode font for PDF export. "
                    f"Font path: {font_path}, Error: {e}. "
                    f"Please ensure Noto Sans or DejaVu Sans is installed."
                ) from e
            logger.warning("Failed to load Unicode font, falling back to Helvetica", font_path=str(font_path), error=str(e))
            pdf.set_font("Helvetica", size=12)
    else:
        if has_unicode:
            raise ValidationError(
                "PDF export requires a Unicode font (Noto Sans or DejaVu Sans) for non-ASCII characters. "
                "Please install Noto Sans fonts or use TXT export instead."
            )
        logger.warning("No Unicode font found. PDF export may fail on special characters.")
        pdf.set_font("Helvetica", size=12)

    pdf.set_font_size(16)
    clean_title = title[:255] if title else "Translation"
    pdf.cell(0, 10, clean_title, ln=True, align='C')
    pdf.ln(5)

    pdf.set_font_size(12)

    content = text.replace("\r\n", "\n").replace("\r", "\n")
    
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        
        pdf.multi_cell(0, 6, block)
        pdf.ln(2)

    pdf.output(str(path))


def _write_epub(path: Path, text: str, title: str = "Translation", chapters: list[tuple[str, str]] | None = None) -> None:
    """
    Write EPUB using ebooklib with optional multi-chapter support.
    
    Args:
        path: Output file path
        text: Content (used if chapters not provided)
        title: Book title
        chapters: Optional list of (chapter_title, chapter_content) tuples for multi-chapter EPUB
    """
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
    
    style_css = """
    .chapter-title {
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
        margin: 2em 0 1.5em 0;
    }
    .chapter-content {
        text-align: justify;
        line-height: 1.6;
    }
    p {
        margin: 0.8em 0;
        text-indent: 1.5em;
    }
    """
    
    css = epub.EpubItem(
        uid="style",
        file_name="style.css",
        media_type="text/css",
        content=style_css.encode('utf-8')
    )
    book.add_item(css)
    
    chapter_objects = []
    
    if chapters:
        for idx, (ch_title, ch_content) in enumerate(chapters, 1):
            chapter_obj = _create_epub_chapter(
                idx, ch_title, ch_content, css
            )
            book.add_item(chapter_obj)
            chapter_objects.append(chapter_obj)
    else:
        paragraphs = [
            p.strip()
            for p in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
            if p.strip()
        ]
        body_parts = [
            f"<p>{html.escape(p)}</p>" for p in paragraphs
        ]
        
        content = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'/>"
            "<link rel='stylesheet' type='text/css' href='style.css'/>"
            "</head><body>\n"
            "<div class='chapter-content'>\n" + "\n".join(body_parts) + "\n</div>"
            "\n</body></html>"
        )

        chapter = epub.EpubHtml(
            title=title[:255] if title else "Content",
            file_name="chapter.xhtml",
            lang="en",
        )
        chapter.content = content
        book.add_item(chapter)
        chapter_objects.append(chapter)
    
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    book.spine = ["nav"] + chapter_objects
    
    book.toc = chapter_objects

    epub.write_epub(str(path), book)


def _create_epub_chapter(
    chapter_num: int,
    chapter_title: str,
    chapter_content: str,
    css
):
    """
    Create a single EPUB chapter with styled title.
    
    Args:
        chapter_num: Chapter number
        chapter_title: Chapter title
        chapter_content: Chapter content
        css: CSS style item (EpubItem)
        
    Returns:
        EpubHtml chapter object
    """
    from ebooklib import epub
    
    paragraphs = [
        p.strip()
        for p in chapter_content.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
        if p.strip()
    ]
    
    body_parts = [f"<p>{html.escape(p)}</p>" for p in paragraphs]
    
    content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'/>
    <link rel='stylesheet' type='text/css' href='style.css'/>
    <title>{html.escape(chapter_title)}</title>
</head>
<body>
    <h1 class='chapter-title'>{html.escape(chapter_title)}</h1>
    <div class='chapter-content'>
        {chr(10).join(body_parts)}
    </div>
</body>
</html>"""
    
    chapter = epub.EpubHtml(
        title=chapter_title[:255],
        file_name=f"chapter_{chapter_num:03d}.xhtml",
        lang="en",
    )
    chapter.content = content
    chapter.add_item(css)
    
    return chapter


def write_document(
    path: Path,
    text: str = "",
    title: str = "Translation",
    chapters: list[tuple[str, str]] | None = None
) -> None:
    """
    Write text to a file in the given format.

    Format is inferred from path suffix:
    - .txt  – plain text (UTF-8)
    - .pdf  – PDF (requires fpdf2 + Unicode font)
    - .epub – EPUB (requires ebooklib)
    
    Args:
        path: Output file path
        text: Content (used for single-content mode or if chapters not provided)
        title: Document title
        chapters: Optional list of (chapter_title, chapter_content) tuples for multi-chapter EPUB

    Raises ValidationError if the format is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_EXTENSIONS:
        raise ValidationError(
            f"Unsupported export format: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXPORT_EXTENSIONS)}"
        )

    if suffix == ".txt":
        if chapters:
            parts = []
            for ch_title, ch_content in chapters:
                parts.append(f"{'=' * 60}\n{ch_title}\n{'=' * 60}\n\n{ch_content}")
            text = "\n\n\n".join(parts)
        _write_txt(path, text)
    elif suffix == ".pdf":
        if chapters:
            parts = []
            for ch_title, ch_content in chapters:
                parts.append(f"{ch_title}\n\n{ch_content}")
            text = "\n\n\n".join(parts)
        _write_pdf(path, text, title=title)
    elif suffix == ".epub":
        _write_epub(path, text, title=title, chapters=chapters)