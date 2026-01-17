"""Text panel widget for displaying chapter text with highlighting."""

from textual.containers import ScrollableContainer
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

    def scroll_to_position(self, position: int) -> None:
        """Scroll to a specific position in the text.
        
        Args:
            position: Character position in the text to scroll to
        """
        if not self._text or position < 0 or position >= len(self._text):
            return
        
        try:
            # Try to find the ScrollableContainer by walking up the parent chain
            parent = self.parent
            scroll_container = None
            while parent is not None:
                if isinstance(parent, ScrollableContainer):
                    scroll_container = parent
                    break
                parent = parent.parent
            
            if not scroll_container:
                # Fallback: try to query by ID from the screen
                screen = self.screen
                if screen:
                    try:
                        scroll_container = screen.query_one("#text_scroll_container", ScrollableContainer)
                    except Exception:
                        pass
            
            if scroll_container:
                # Calculate which line the position is on
                lines_before = self._text[:position].count('\n')
                
                # Get the visible height in lines
                visible_height = scroll_container.size.height if scroll_container.size.height > 0 else 10
                if visible_height <= 0:
                    visible_height = 10
                
                # Calculate target scroll position (center the target line if possible)
                # ScrollableContainer scrolls by pixel/line offset
                target_scroll = max(0, lines_before - visible_height // 2)
                
                # Scroll the container - use call_after_refresh to ensure widget is ready
                self.call_after_refresh(
                    lambda: scroll_container.scroll_to(y=target_scroll, animate=True)
                )
        except Exception as e:
            # If scrolling fails, log but don't crash
            import logging
            logging.debug(f"Error scrolling to position {position}: {e}")

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
