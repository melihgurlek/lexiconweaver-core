"""Main screen for the LexiconWeaver TUI."""

from pathlib import Path
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project
from lexiconweaver.engines.scout import CandidateTerm, Scout
from lexiconweaver.logging_config import get_logger
from lexiconweaver.tui.widgets.candidate_list import CandidateList
from lexiconweaver.tui.widgets.term_modal import TermModal
from lexiconweaver.tui.widgets.text_panel import TextPanel
from lexiconweaver.utils.cache import get_cache
from lexiconweaver.utils.highlighting import highlight_terms

logger = get_logger(__name__)


class MainScreen(Screen):
    """Main screen displaying text panel and candidate list."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_scout", "Run Scout"),
        ("t", "translate", "Translate"),
    ]

    DEFAULT_CSS = """
    MainScreen {
        background: $surface;
        height: 100%;
    }

    Horizontal {
        height: 1fr;
        min-height: 10;
    }

    Vertical {
        width: 1fr;
        min-height: 5;
    }

    #text_container {
        width: 2fr;
        border: solid $primary;
        min-height: 5;
    }

    #text_scroll_container {
        height: 1fr;
    }

    #candidate_container {
        width: 1fr;
        border: solid $primary;
        min-height: 5;
    }

    #status_bar {
        height: 1;
        dock: bottom;
        background: $panel;
    }
    """

    def __init__(
        self,
        config: Config,
        project: Project,
        text: str = "",
        text_file: Path | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the main screen."""
        super().__init__(*args, **kwargs)
        self.config = config
        self.project = project
        self._text = text
        self._text_file = text_file
        self._scout: Scout | None = None
        self._candidates: list[CandidateTerm] = []
        self._confirmed_terms: set[str] = set()
        self._candidate_terms: set[str] = set()
        self._cache = get_cache()

    def compose(self):
        """Compose the screen widgets."""
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="text_container"):
                yield Static("Chapter Text", classes="section_title")
                with ScrollableContainer(id="text_scroll_container"):
                    yield TextPanel(self._text, id="text_panel")
            with Vertical(id="candidate_container"):
                yield Static("Candidate Terms", classes="section_title")
                yield CandidateList(id="candidate_list")
        yield Static("Ready", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        logger.debug("MainScreen mounted, deferring initialization until widgets are ready")
        # Use call_after_refresh to ensure widgets are fully ready before accessing them
        self.call_after_refresh(self._initialize_screen)

    def _initialize_screen(self) -> None:
        """Initialize screen after widgets are ready."""
        logger.debug("Initializing screen widgets")
        try:
            self._load_text()
            self._load_confirmed_terms()
            self._update_highlights()
            self._initialize_scout()
            logger.debug("Screen initialization completed successfully")
        except Exception as e:
            logger.exception("Error initializing screen", error=str(e))
            # Try to update status if possible
            self._safe_update_status(f"Error: {e}")

    def _load_text(self) -> None:
        """Load text from file or use provided text."""
        logger.debug("Loading text", text_file=str(self._text_file) if self._text_file else None)
        if self._text_file and self._text_file.exists():
            try:
                with open(self._text_file, "r", encoding="utf-8") as f:
                    self._text = f.read()
                logger.debug("Text loaded from file", length=len(self._text))
            except Exception as e:
                logger.exception("Error loading file", file=str(self._text_file), error=str(e))
                self._safe_update_status(f"Error loading file: {e}")
                return

        try:
            text_panel = self.query_one("#text_panel", TextPanel)
            text_panel.set_text(self._text)
            logger.debug("Text panel updated", text_length=len(self._text))
        except Exception as e:
            logger.exception("Error updating text panel", error=str(e))
            raise

    def _load_confirmed_terms(self) -> None:
        """Load confirmed terms from database, using cache if available."""
        def _fetch_confirmed_terms(project: Project | None) -> set[str]:
            """Fetch confirmed terms from database."""
            try:
                terms = GlossaryTerm.select().where(
                    GlossaryTerm.project == project
                )
                return {term.source_term for term in terms}
            except Exception as e:
                logger.exception("Error loading confirmed terms", error=str(e))
                return set()
        
        self._confirmed_terms = self._cache.get_confirmed_terms(self.project, _fetch_confirmed_terms)
        logger.debug("Loaded confirmed terms", count=len(self._confirmed_terms))

    def _get_terms_with_translations(self, term_list: list[str]) -> set[str]:
        """Get set of terms that already have translations in the database.
        
        Uses cache to avoid repeated database queries.
        
        Args:
            term_list: List of source terms to check
            
        Returns:
            Set of source terms that have translations
        """
        if not term_list:
            return set()
        
        def _fetch_terms_with_translations(project: Project | None, terms: list[str]) -> set[str]:
            """Fetch terms with translations from database."""
            try:
                existing_terms = GlossaryTerm.select(GlossaryTerm.source_term).where(
                    GlossaryTerm.project == project,
                    GlossaryTerm.source_term.in_(terms)
                )
                return {term.source_term for term in existing_terms}
            except Exception as e:
                logger.debug("Error checking existing translations", error=str(e))
                return set()
        
        return self._cache.get_terms_with_translations(
            self.project, term_list, _fetch_terms_with_translations
        )

    def _update_highlights(self) -> None:
        """Update text highlights based on confirmed and candidate terms."""
        try:
            text_panel = self.query_one("#text_panel", TextPanel)
            highlights = highlight_terms(
                self._text, self._confirmed_terms, self._candidate_terms
            )
            text_panel.set_highlights(highlights)
            logger.debug("Highlights updated", highlight_count=len(highlights))
        except Exception as e:
            logger.exception("Error updating highlights", error=str(e))
            # Don't raise - highlights are not critical for basic functionality

    def _initialize_scout(self) -> None:
        """Initialize the Scout engine."""
        try:
            self._scout = Scout(self.config, self.project)
            logger.debug("Scout engine initialized")
        except Exception as e:
            logger.exception("Error initializing Scout", error=str(e))
            # Don't raise - Scout can be initialized later if needed

    def action_run_scout(self) -> None:
        """Run the Scout to find candidate terms."""
        if self._scout is None:
            self._safe_update_status("Scout not initialized")
            logger.warning("Scout not initialized when run_scout was called")
            return

        self._safe_update_status("Running Scout...")
        logger.debug("Running Scout engine")
        try:
            self._candidates = self._scout.process(self._text)
            self._candidate_terms = {c.term for c in self._candidates}

            # Get terms that already have translations in the database
            terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])

            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates, terms_with_translations)

            self._update_highlights()
            self._safe_update_status(f"Found {len(self._candidates)} candidate terms")
            logger.debug("Scout completed", candidate_count=len(self._candidates))
        except Exception as e:
            logger.exception("Scout error", error=str(e))
            self._safe_update_status(f"Scout error: {e}")

    def action_translate(self) -> None:
        """Start translation process."""
        self._safe_update_status("Translation not yet implemented in TUI")

    def _safe_update_status(self, message: str) -> None:
        """Update the status bar safely, handling cases where it may not be ready."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
            logger.debug("Status updated", message=message)
        except Exception as e:
            # Status bar not ready or query failed, log but don't raise
            logger.debug("Could not update status bar", message=message, error=str(e))

    def on_candidate_list_selected(self, message: CandidateList.Selected) -> None:
        """Handle candidate selection."""
        candidate = message.candidate
        self._show_term_modal(candidate.term)

    def on_candidate_list_ignored(self, message: CandidateList.Ignored) -> None:
        """Handle candidate ignore."""
        candidate = message.candidate
        try:
            from lexiconweaver.database.models import IgnoredTerm

            IgnoredTerm.get_or_create(
                project=self.project, term=candidate.term
            )
            self._cache.invalidate_ignored_terms(self.project)
            
            # Remove from candidate list
            self._candidates = [c for c in self._candidates if c.term != candidate.term]
            
            # Get terms that still have translations (for green coloring)
            terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])
            
            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates, terms_with_translations)
            self._safe_update_status(f"Ignored term: {candidate.term}")
            logger.debug("Term ignored", term=candidate.term)
        except Exception as e:
            logger.exception("Error ignoring term", term=candidate.term, error=str(e))
            self._safe_update_status(f"Error ignoring term: {e}")

    def on_candidate_list_skipped(self, message: CandidateList.Skipped) -> None:
        """Handle candidate skip."""
        candidate = message.candidate
        self._safe_update_status(f"Skipped: {candidate.term}")
        logger.debug("Term skipped", term=candidate.term)

    def on_candidate_list_highlighted(self, message: CandidateList.Highlighted) -> None:
        """Handle candidate click - navigate to term location in text."""
        candidate = message.candidate
        term = candidate.term
        
        # Find the first occurrence of the term in the text
        # Try case-sensitive first, then case-insensitive
        position = self._text.find(term)
        if position == -1:
            position = self._text.lower().find(term.lower())
        
        if position != -1:
            try:
                # Calculate which line the position is on (counting newlines)
                lines_before = self._text[:position].count('\n')
                line_number = lines_before + 1
                
                # Use call_after_refresh to ensure widgets are ready
                def do_scroll():
                    try:
                        scroll_container = self.query_one("#text_scroll_container", ScrollableContainer)
                        target_y = max(0, lines_before - 2)  # Show a few lines before target
                        
                        # Try different ways to scroll
                        if hasattr(scroll_container, 'scroll_y'):
                            scroll_container.scroll_y = target_y
                        elif hasattr(scroll_container, 'scroll_to'):
                            scroll_container.scroll_to(y=target_y, animate=False)
                        elif hasattr(scroll_container, 'scroll_relative'):
                            current = getattr(scroll_container, 'scroll_y', 0) or 0
                            scroll_container.scroll_relative(y=target_y - current, animate=False)
                        
                        self._safe_update_status(f"Found {term} at line {line_number}")
                    except Exception as e:
                        self._safe_update_status(f"Scroll error: {str(e)[:40]}")
                
                self.call_after_refresh(do_scroll)
                
            except Exception as e:
                logger.exception("Error scrolling to term", term=term, error=str(e))
                self._safe_update_status(f"Error: {str(e)[:60]}")
        else:
            self._safe_update_status(f"Term '{term}' not found in text")
            logger.debug("Term not found in text", term=term)

    def _show_term_modal(self, source_term: str) -> None:
        """Show the term editing modal."""
        # Check if a modal is already open to prevent duplicates
        existing_modals = self.query(TermModal)
        if existing_modals:
            logger.debug("Modal already open, ignoring duplicate request", term=source_term)
            return
        
        # Check if term already exists in database to load its data
        target_term = ""
        category = ""
        is_regex = False
        try:
            existing_term = GlossaryTerm.get_or_none(
                project=self.project,
                source_term=source_term
            )
            if existing_term:
                target_term = existing_term.target_term or ""
                category = existing_term.category or ""
                is_regex = existing_term.is_regex or False
        except Exception as e:
            logger.debug("Error loading existing term data", term=source_term, error=str(e))
        
        modal = TermModal(
            source_term=source_term,
            target_term=target_term,
            category=category,
            is_regex=is_regex
        )
        self.mount(modal)

    def on_term_modal_term_saved(self, message: TermModal.TermSaved) -> None:
        """Handle term save from modal."""
        try:
            # Use update_or_create to handle both new and existing terms
            # This ensures category updates work for existing terms
            term, created = GlossaryTerm.get_or_create(
                project=self.project,
                source_term=message.source_term,
                defaults={
                    "target_term": message.target_term,
                    "category": message.category if message.category else None,
                    "is_regex": message.is_regex,
                },
            )
            # Update existing term if it already exists
            if not created:
                term.target_term = message.target_term
                term.category = message.category if message.category else None
                term.is_regex = message.is_regex
                term.save()
            
            self._cache.invalidate_glossary_terms(self.project)
            
            self._confirmed_terms.add(message.source_term)
            self._candidate_terms.discard(message.source_term)
            self._update_highlights()
            
            # Refresh candidate list to show updated translation status (green color)
            if self._candidates:
                terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])
                candidate_list = self.query_one("#candidate_list", CandidateList)
                candidate_list.set_candidates(self._candidates, terms_with_translations)
            
            self._safe_update_status(f"Saved term: {message.source_term} -> {message.target_term}")
            logger.debug("Term saved", source=message.source_term, target=message.target_term, category=message.category)
        except Exception as e:
            logger.exception("Error saving term", source=message.source_term, error=str(e))
            self._safe_update_status(f"Error saving term: {e}")

    def on_term_modal_cancelled(self, message: TermModal.Cancelled) -> None:
        """Handle modal cancellation."""
        pass
