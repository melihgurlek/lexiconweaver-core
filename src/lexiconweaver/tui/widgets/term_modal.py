"""Modal dialog for editing term definitions."""

from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Select

TERM_CATEGORIES = [
    ("Person", "Person"),
    ("Location", "Location"),
    ("Skill", "Skill"),
    ("Item", "Item"),
    ("Other", "Other"),
]


class TermModal(Container):
    """Modal dialog for adding/editing a term."""

    DEFAULT_CSS = """
    TermModal {
        width: 80;
        height: 20;
        border: thick $primary;
        background: $surface;
    }

    TermModal Vertical {
        padding: 1;
    }

    TermModal Horizontal {
        height: 3;
        align: center middle;
    }

    TermModal Input {
        width: 1fr;
        margin: 1;
    }

    TermModal Select {
        width: 1fr;
        margin: 1;
    }

    TermModal Button {
        margin: 1;
        width: 12;
    }
    """

    def __init__(
        self,
        source_term: str,
        target_term: str = "",
        category: str = "",
        is_regex: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the term modal."""
        super().__init__(*args, **kwargs)
        self.source_term = source_term
        self._target_term_input: Input | None = None
        self._category_select: Select | None = None
        self._is_regex = is_regex

    def compose(self):
        """Compose the modal widgets."""
        with Vertical():
            yield Label(f"Source Term: {self.source_term}", id="source_label")
            yield Label("Target Term:", classes="label")
            yield Input(
                value="",
                placeholder="Enter translation...",
                id="target_input",
            )
            yield Label("Category:", classes="label")
            yield Select(
                options=TERM_CATEGORIES,
                allow_blank=True,
                id="category_select",
            )
            yield Label("Scope: Global (all chapters)", id="scope_label")

            with Horizontal():
                yield Button("Save", variant="primary", id="save_button")
                yield Button("Cancel", variant="default", id="cancel_button")

    def on_mount(self) -> None:
        """Called when the modal is mounted."""
        self._target_term_input = self.query_one("#target_input", Input)
        self._category_select = self.query_one("#category_select", Select)
        self._target_term_input.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_button":
            self._save_term()
        elif event.button.id == "cancel_button":
            self._cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        if event.input.id == "target_input":
            self._save_term()

    def _save_term(self) -> None:
        """Save the term and close modal."""
        target_term = self._target_term_input.value if self._target_term_input else ""
        category = ""
        if self._category_select and self._category_select.value:
            # When allow_blank=True, value is a tuple (label, value) when selected, or Select.BLANK when blank
            if isinstance(self._category_select.value, tuple):
                category = self._category_select.value[0]

        self.post_message(
            self.TermSaved(
                source_term=self.source_term, target_term=target_term, category=category, is_regex=self._is_regex
            )
        )
        self.remove()

    def _cancel(self) -> None:
        """Cancel and close modal."""
        self.post_message(self.Cancelled())
        self.remove()

    class TermSaved(Message):
        """Message sent when term is saved."""

        def __init__(
            self, source_term: str, target_term: str, category: str, is_regex: bool
        ) -> None:
            super().__init__()
            self.source_term = source_term
            self.target_term = target_term
            self.category = category
            self.is_regex = is_regex

    class Cancelled(Message):
        """Message sent when modal is cancelled."""

        def __init__(self) -> None:
            super().__init__()
