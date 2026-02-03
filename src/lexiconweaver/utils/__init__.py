"""Utility functions for text processing and validation."""

from lexiconweaver.utils.document_loader import load_document
from lexiconweaver.utils.document_writer import write_document
from lexiconweaver.utils.highlighting import highlight_terms, LongestMatchHighlighter
from lexiconweaver.utils.text_processor import extract_paragraphs, normalize_text
from lexiconweaver.utils.validators import (
    validate_document_file,
    validate_encoding,
    validate_text_file,
)

__all__ = [
    "extract_paragraphs",
    "normalize_text",
    "highlight_terms",
    "LongestMatchHighlighter",
    "load_document",
    "validate_document_file",
    "validate_encoding",
    "validate_text_file",
    "write_document",
]
