"""Text panel widget for displaying chapter text with highlighting."""

from textual.widgets import Static

from lexiconweaver.utils.highlighting import HighlightSpan


class TextPanel(Static):
    """Panel for displaying text with term highlighting."""

    DEFAULT_CSS = """
    TextPanel {
        border: solid $primary;
        padding: 1;
        scrollbar-gutter: stable;
    }
    """

    def __init__(
        self,
        text: str = "",
        highlights: list[HighlightSpan] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the text panel."""
        super().__init__(*args, **kwargs)
        self._text = text
        self._highlights = highlights or []
        # Update display with initial content
        if text:
            self._update_display()
        else:
            # Show placeholder if no text
            self.update("No text loaded. Press 'r' to run Scout or load a file.")

    def set_text(self, text: str) -> None:
        """Set the text content."""
        self._text = text
        self._update_display()

    def set_highlights(self, highlights: list[HighlightSpan]) -> None:
        """Set the highlight spans."""
        self._highlights = highlights
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed text with highlights."""
        if not self._highlights:
            self.update(self._text)
            return

        # Build highlighted text
        parts: list[tuple[str, str]] = []  # (text, style)
        last_pos = 0

        # Sort highlights by position
        sorted_highlights = sorted(self._highlights, key=lambda x: x.start)

        for span in sorted_highlights:
            # Add text before highlight
            if span.start > last_pos:
                parts.append((self._text[last_pos : span.start], ""))

            # Add highlighted text
            highlight_text = self._text[span.start : span.end]
            style = "bold green" if span.is_confirmed else "bold yellow"
            parts.append((highlight_text, style))

            last_pos = span.end

        # Add remaining text
        if last_pos < len(self._text):
            parts.append((self._text[last_pos:], ""))

        # Render with Rich markup
        from rich.text import Text

        rendered = Text()
        for text_part, style in parts:
            if style:
                rendered.append(text_part, style=style)
            else:
                rendered.append(text_part)

        self.update(rendered)
