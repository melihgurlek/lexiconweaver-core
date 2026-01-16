"""Candidate list widget for displaying potential terms."""

from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widgets import OptionList

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
        self._pending_keypress = False  # Track if Enter key was just pressed
        self._processing_selection = False  # Prevent double-calling action_select

    def set_candidates(self, candidates: list[CandidateTerm]) -> None:
        """Set the candidate terms to display."""
        # 1. Store the raw objects
        self.candidates = candidates
        
        # 2. Clear the UI
        self.clear_options()

        # 3. Add them to the UI (In the exact same order)
        for candidate in candidates:
            label = (
                f"{candidate.term} "
                f"(conf: {candidate.confidence:.2f}, "
                f"freq: {candidate.frequency})"
            )
            # FIX: Do not pass 'id'. Just pass the label.
            self.add_option(label) 

    def get_selected_candidate(self) -> CandidateTerm | None:
        """Get the currently selected candidate."""
        # FIX: Simplified logic. 
        # The OptionList index matches the self.candidates index perfectly.
        
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
        """Override default click behavior - prevent automatic selection on click."""
        # Always stop the default behavior
        event.stop()
        # Only trigger selection if Enter key was pressed
        # (The binding system will also call action_select, but our guard prevents double-calls)
        if self._pending_keypress and not self._processing_selection:
            # Don't reset _pending_keypress here - let action_select handle it
            # This ensures action_select can verify it was from a keypress
            self.action_select()

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