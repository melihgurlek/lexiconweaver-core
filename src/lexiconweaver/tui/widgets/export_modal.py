"""Modal dialog for choosing export format and path."""

from pathlib import Path

from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Select, Static

EXPORT_FORMATS = [
    ("txt", "TXT (plain text)"),
    ("pdf", "PDF"),
    ("epub", "EPUB"),
]


class ExportModal(Container):
    """Modal for selecting export format and output path."""

    DEFAULT_CSS = """
    ExportModal {
        width: 70;
        height: auto;
        max-height: 20;
        border: thick $primary;
        background: $surface;
    }

    ExportModal Vertical {
        padding: 1;
    }

    ExportModal Horizontal {
        height: 3;
        align: center middle;
    }

    ExportModal Label {
        width: 100%;
    }

    ExportModal Input {
        width: 1fr;
        margin: 1 0;
    }

    ExportModal Select {
        width: 1fr;
        margin: 1 0;
    }

    ExportModal Button {
        margin: 0 1;
        width: 12;
    }

    ExportModal .action-buttons {
        height: 5;
        align: center middle;
    }
    """

    class ExportRequested(Message):
        """User confirmed export to the given path."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class Cancelled(Message):
        """User cancelled export."""

    def __init__(
        self,
        default_path: Path,
        default_format: str = "txt",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._default_path = default_path
        self._default_format = default_format
        self._path_input: Input | None = None
        self._format_select: Select | None = None

    def _path_for_format(self, fmt: str) -> Path:
        """Return default path with the given extension."""
        p = self._default_path
        stem = p.stem
        if stem.endswith("_translated"):
            base = stem
        else:
            base = f"{stem}_translated"
        return (p.parent if p.parent else Path.cwd()) / f"{base}.{fmt}"

    def compose(self):
        """Compose the modal widgets."""
        with Vertical():
            yield Label("Export format:", classes="section_label")
            yield Select(
                options=EXPORT_FORMATS,
                value=self._default_format,
                id="export_format",
            )
            yield Label("Output path:", classes="section_label")
            initial_path = self._path_for_format(self._default_format)
            yield Input(
                value=str(initial_path),
                placeholder="Path to save translation...",
                id="export_path",
            )
            with Horizontal(classes="action-buttons"):
                yield Button("Export", variant="primary", id="export_button")
                yield Button("Cancel", variant="default", id="cancel_button")

    def on_mount(self) -> None:
        """Called when the modal is mounted."""
        self._path_input = self.query_one("#export_path", Input)
        self._format_select = self.query_one("#export_format", Select)
        self._path_input.focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update path extension when format changes."""
        if event.select.id != "export_format" or not self._path_input:
            return
        fmt = event.value
        if fmt:
            new_path = self._path_for_format(fmt)
            self._path_input.value = str(new_path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "export_button" and self._path_input:
            path = Path(self._path_input.value.strip())
            if not path.suffix:
                ext = self._format_select.value if self._format_select else "txt"
                path = path.with_suffix(f".{ext}")
            self.post_message(self.ExportRequested(path))
            self.remove()
        elif event.button.id == "cancel_button":
            self.post_message(self.Cancelled())
            self.remove()
