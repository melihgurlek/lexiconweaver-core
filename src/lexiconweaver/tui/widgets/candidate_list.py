"""Candidate list widget for displaying potential terms."""

from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widgets import OptionList
from rich.text import Text

from lexiconweaver.engines.scout import CandidateTerm


class CandidateList(OptionList):
    """List widget for displaying candidate terms sorted by confidence."""

    BINDINGS = [
        Binding("enter", "select", "Edit/Confirm", show=True),
        Binding("delete", "ignore", "Ignore", show=True),
        Binding("s", "skip", "Skip", show=True),
    ]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the candidate list."""
        super().__init__(*args, **kwargs)
        self.candidates: list[CandidateTerm] = []
        self._terms_with_translations: set[str] = set()
        self._pending_keypress = False  # Track if Enter key was just pressed
        self._processing_selection = False  # Prevent double-calling action_select

    def set_candidates(
        self, candidates: list[CandidateTerm], terms_with_translations: set[str] | None = None
    ) -> None:
        """Set the candidate terms to display.
        
        Args:
            candidates: List of candidate terms to display
            terms_with_translations: Set of terms that already have translations in the database
        """
        # 1. Store the raw objects
        self.candidates = candidates
        self._terms_with_translations = terms_with_translations or set()
        
        # 2. Clear the UI
        self.clear_options()

        # 3. Add them to the UI (In the exact same order)
        for candidate in candidates:
            # Create label with Rich Text for coloring
            if candidate.term in self._terms_with_translations:
                # Use Rich Text to color translated terms green
                label = Text()
                label.append(
                    f"{candidate.term} (conf: {candidate.confidence:.2f}, freq: {candidate.frequency})",
                    style="green"
                )
            else:
                # Regular text for terms without translations
                label = (
                    f"{candidate.term} "
                    f"(conf: {candidate.confidence:.2f}, "
                    f"freq: {candidate.frequency})"
                )
            self.add_option(label) 

    def get_selected_candidate(self) -> CandidateTerm | None:
        """Get the currently selected candidate.
        
        The OptionList index matches the self.candidates index.
        """
        if self.highlighted is None:
            return None

        if 0 <= self.highlighted < len(self.candidates):
            return self.candidates[self.highlighted]

        return None

    def on_key(self, event: Key) -> None:
        """Track Enter key press."""
        if event.key == "enter":
            self._pending_keypress = True
        # Don't call super().on_key() - OptionList handles keys via action methods

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection (click or keyboard)."""
        # Stop the default behavior
        event.stop()
        
        candidate = self.get_selected_candidate()
        if not candidate:
            return
            
        # If Enter key was pressed, trigger the Selected message (opens modal)
        if self._pending_keypress and not self._processing_selection:
            # Don't reset _pending_keypress here - let action_select handle it
            # This ensures action_select can verify it was from a keypress
            self.action_select()
        else:
            # This was a click - send a Highlighted message to navigate to term location
            if not self._processing_selection:
                self.post_message(self.Highlighted(candidate))

    def action_select(self) -> None:
        """Handle Enter key press - allow selection."""
        # Only allow selection if it was triggered by Enter key
        # If _pending_keypress is False, this was called from a click - ignore it
        if not self._pending_keypress:
            return
        
        # Prevent double-calling if already processing
        if self._processing_selection:
            return
        
        self._processing_selection = True
        try:
            candidate = self.get_selected_candidate()
            if candidate:
                self.post_message(self.Selected(candidate))
        finally:
            self._processing_selection = False
            self._pending_keypress = False

    def action_ignore(self) -> None:
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Ignored(candidate))

    def action_skip(self) -> None:
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Skipped(candidate))

    class Selected(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Ignored(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Skipped(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Highlighted(Message):
        """Message sent when a candidate is clicked (not Enter key)."""
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate