"""Translation panel widget for displaying translated text with highlighting."""

from textual.containers import ScrollableContainer
from textual.widgets import Static

from lexiconweaver.utils.highlighting import HighlightSpan
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)


class TranslationPanel(Static):
    """Panel for displaying translated text with term highlighting."""

    DEFAULT_CSS = """
    TranslationPanel {
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
        """Initialize the translation panel."""
        super().__init__(*args, **kwargs)
        self._text = text
        self._highlights = highlights or []
        if text:
            self._update_display()
        else:
            self.update("Translation will appear here...")

    def append_text(self, text: str) -> None:
        """Append text to the current content (for streaming)."""
        if not text:
            return
        self._text += text
        self._update_display()
        # Force a refresh to ensure the update is visible
        self.refresh()

    def set_text(self, text: str) -> None:
        """Set the text content."""
        self._text = text
        self._update_display()

    def set_highlights(self, highlights: list[HighlightSpan]) -> None:
        """Set the highlight spans."""
        self._highlights = highlights
        self._update_display()

    def get_text(self) -> str:
        """Get the current text content."""
        return self._text

    def scroll_to_position(self, position: int) -> None:
        """Scroll to a specific position in the text.
        
        Args:
            position: Character position in the text to scroll to
        """
        if not self._text or position < 0 or position >= len(self._text):
            return
        
        try:
            parent = self.parent
            scroll_container = None
            while parent is not None:
                if isinstance(parent, ScrollableContainer):
                    scroll_container = parent
                    break
                parent = parent.parent
            
            if not scroll_container:
                screen = self.screen
                if screen:
                    try:
                        scroll_container = screen.query_one("#translation_scroll_container", ScrollableContainer)
                    except Exception:
                        pass
            
            if scroll_container:
                lines_before = self._text[:position].count('\n')
                
                visible_height = scroll_container.size.height if scroll_container.size.height > 0 else 10
                if visible_height <= 0:
                    visible_height = 10
                
                target_scroll = max(0, lines_before - visible_height // 2)
                
                self.call_after_refresh(
                    lambda: scroll_container.scroll_to(y=target_scroll, animate=True)
                )
        except Exception as e:
            logger.debug(f"Error scrolling to position {position}: {e}")

    def _update_display(self) -> None:
        """Update the displayed text with highlights."""
        if not self._highlights:
            self.update(self._text)
            return

        parts: list[tuple[str, str]] = []
        last_pos = 0

        sorted_highlights = sorted(self._highlights, key=lambda x: x.start)

        for span in sorted_highlights:
            if span.start > last_pos:
                parts.append((self._text[last_pos : span.start], ""))

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
