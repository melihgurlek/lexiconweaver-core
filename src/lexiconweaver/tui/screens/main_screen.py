"""Main screen for the LexiconWeaver TUI."""

import asyncio
from pathlib import Path

from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, IgnoredTerm, ProposedTerm, Project
from lexiconweaver.engines.scout import CandidateTerm, Scout
from lexiconweaver.engines.scout_refiner import RefinedTerm, ScoutRefiner
from lexiconweaver.logging_config import get_logger
from lexiconweaver.tui.screens.translation_screen import TranslationScreen
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
        ("s", "run_smart_scout", "Smart Scout"),
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

    #candidate_search {
        margin: 1;
        width: 1fr;
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
        self._scout_refiner: ScoutRefiner | None = None
        self._candidates: list[CandidateTerm] = []
        self._refined_terms: list[RefinedTerm] = []
        self._confirmed_terms: set[str] = set()
        self._candidate_terms: set[str] = set()
        self._cache = get_cache()
        self._smart_scout_running = False

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
                yield Input(
                    placeholder="Search candidates...",
                    id="candidate_search",
                )
                yield CandidateList(id="candidate_list")
        yield Static("Ready | r: Scout | s: Smart Scout | t: Translate", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        logger.debug("MainScreen mounted, deferring initialization until widgets are ready")
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
        """Get set of terms that already have translations in the database."""
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

    def _initialize_scout(self) -> None:
        """Initialize the Scout engine."""
        try:
            self._scout = Scout(self.config, self.project)
            logger.debug("Scout engine initialized")
        except Exception as e:
            logger.exception("Error initializing Scout", error=str(e))

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
            self._refined_terms = []  # Clear any refined terms

            terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])

            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates, terms_with_translations)

            self._update_highlights()
            self._safe_update_status(f"Found {len(self._candidates)} candidate terms")
            logger.debug("Scout completed", candidate_count=len(self._candidates))
        except Exception as e:
            logger.exception("Scout error", error=str(e))
            self._safe_update_status(f"Scout error: {e}")

    def action_run_smart_scout(self) -> None:
        """Run the Smart Scout (two-pass with LLM refinement)."""
        if self._smart_scout_running:
            self._safe_update_status("Smart Scout is already running...")
            return

        if not self._text:
            self._safe_update_status("No text to analyze")
            return

        self._smart_scout_running = True
        self._safe_update_status("Smart Scout: Initializing LLM...")
        
        # Run the async task
        asyncio.create_task(self._run_smart_scout_async())

    async def _run_smart_scout_async(self) -> None:
        """Run the Smart Scout asynchronously."""
        try:
            self._scout_refiner = ScoutRefiner(self.config, self.project)

            def on_progress(msg: str) -> None:
                self._safe_update_status(msg)

            self._refined_terms = await self._scout_refiner.refine_text(
                self._text, progress_callback=on_progress
            )

            self._safe_update_status("Smart Scout: Saving proposals...")
            await asyncio.sleep(0)  # Let UI update
            saved_count = await self._scout_refiner.save_proposals(self._refined_terms)

            self._candidates = [
                CandidateTerm(
                    term=rt.source_term,
                    confidence=0.9,
                    frequency=1,
                    context_pattern="llm_refined",
                    context_snippet=rt.context_snippet,
                )
                for rt in self._refined_terms
                if rt.is_valid
            ]
            self._candidate_terms = {c.term for c in self._candidates}

            terms_with_translations = self._get_terms_with_translations(
                [c.term for c in self._candidates]
            )

            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(
                self._candidates, terms_with_translations, has_proposals=True
            )

            self._update_highlights()
            self._safe_update_status(
                f"Smart Scout complete — {len(self._candidates)} terms, {saved_count} proposals saved."
            )

        except Exception as e:
            logger.exception("Smart Scout error", error=str(e))
            self._safe_update_status(f"Smart Scout error: {str(e)[:60]}")
        finally:
            self._smart_scout_running = False

    def action_translate(self) -> None:
        """Start translation process."""
        if not self._text:
            self._safe_update_status("No text to translate")
            return
        
        translation_screen = TranslationScreen(
            config=self.config,
            project=self.project,
            text=self._text,
            text_file=self._text_file,
        )
        self.app.push_screen(translation_screen)

    def _safe_update_status(self, message: str) -> None:
        """Update the status bar safely."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
            logger.debug("Status updated", message=message)
        except Exception as e:
            logger.debug("Could not update status bar", message=message, error=str(e))

    def on_candidate_list_selected(self, message: CandidateList.Selected) -> None:
        """Handle candidate selection."""
        candidate = message.candidate
        self._show_term_modal(candidate.term)

    def on_candidate_list_ignored(self, message: CandidateList.Ignored) -> None:
        """Handle candidate ignore."""
        candidate = message.candidate
        try:
            IgnoredTerm.get_or_create(
                project=self.project, term=candidate.term
            )
            self._cache.invalidate_ignored_terms(self.project)
            
            try:
                proposed = ProposedTerm.get_or_none(
                    ProposedTerm.project == self.project,
                    ProposedTerm.source_term == candidate.term,
                )
                if proposed:
                    proposed.status = "rejected"
                    proposed.save()
            except Exception:
                pass
            
            self._candidates = [c for c in self._candidates if c.term != candidate.term]
            
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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "candidate_search":
            search_query = event.value
            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.filter_candidates(search_query)

    def on_candidate_list_highlighted(self, message: CandidateList.Highlighted) -> None:
        """Handle candidate click - navigate to term location in text."""
        candidate = message.candidate
        term = candidate.term
        
        position = self._text.find(term)
        if position == -1:
            position = self._text.lower().find(term.lower())
        
        if position != -1:
            try:
                lines_before = self._text[:position].count('\n')
                line_number = lines_before + 1
                
                def do_scroll():
                    try:
                        scroll_container = self.query_one("#text_scroll_container", ScrollableContainer)
                        target_y = max(0, lines_before - 2)
                        
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
        existing_modals = self.query(TermModal)
        if existing_modals:
            logger.debug("Modal already open, ignoring duplicate request", term=source_term)
            return
        
        # Check for LLM proposal first
        proposed_translation = None
        proposed_category = None
        llm_reasoning = None
        
        try:
            proposal = ProposedTerm.get_or_none(
                ProposedTerm.project == self.project,
                ProposedTerm.source_term == source_term,
                ProposedTerm.status == "pending",
            )
            if proposal:
                proposed_translation = proposal.proposed_translation
                proposed_category = proposal.proposed_category
                llm_reasoning = proposal.llm_reasoning
        except Exception as e:
            logger.debug("Error loading proposal", term=source_term, error=str(e))
        
        # Check for existing glossary term
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
            is_regex=is_regex,
            proposed_translation=proposed_translation,
            proposed_category=proposed_category,
            llm_reasoning=llm_reasoning,
        )
        self.mount(modal)

    def on_term_modal_term_saved(self, message: TermModal.TermSaved) -> None:
        """Handle term save from modal."""
        try:
            term, created = GlossaryTerm.get_or_create(
                project=self.project,
                source_term=message.source_term,
                defaults={
                    "target_term": message.target_term,
                    "category": message.category if message.category else None,
                    "is_regex": message.is_regex,
                },
            )
            if not created:
                term.target_term = message.target_term
                term.category = message.category if message.category else None
                term.is_regex = message.is_regex
                term.save()
            
            self._cache.invalidate_glossary_terms(self.project)
            
            if message.was_proposal:
                try:
                    proposed = ProposedTerm.get_or_none(
                        ProposedTerm.project == self.project,
                        ProposedTerm.source_term == message.source_term,
                    )
                    if proposed:
                        if message.action == "approve":
                            proposed.status = "approved"
                        elif message.action == "modify":
                            proposed.status = "modified"
                            proposed.user_translation = message.target_term
                            proposed.user_category = message.category
                        proposed.save()
                except Exception as e:
                    logger.debug("Error updating proposal status", error=str(e))
            
            self._confirmed_terms.add(message.source_term)
            self._candidate_terms.discard(message.source_term)
            self._update_highlights()
            
            if self._candidates:
                terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])
                candidate_list = self.query_one("#candidate_list", CandidateList)
                candidate_list.set_candidates(self._candidates, terms_with_translations)
            
            action_str = f"({message.action})" if message.was_proposal else ""
            self._safe_update_status(f"Saved {action_str}: {message.source_term} -> {message.target_term}")
            logger.debug("Term saved", source=message.source_term, target=message.target_term, action=message.action)
        except Exception as e:
            logger.exception("Error saving term", source=message.source_term, error=str(e))
            self._safe_update_status(f"Error saving term: {e}")

    def on_term_modal_term_rejected(self, message: TermModal.TermRejected) -> None:
        """Handle term rejection from modal."""
        try:
            proposed = ProposedTerm.get_or_none(
                ProposedTerm.project == self.project,
                ProposedTerm.source_term == message.source_term,
            )
            if proposed:
                proposed.status = "rejected"
                proposed.save()
            
            self._candidates = [c for c in self._candidates if c.term != message.source_term]
            self._candidate_terms.discard(message.source_term)
            
            terms_with_translations = self._get_terms_with_translations([c.term for c in self._candidates])
            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates, terms_with_translations)
            
            self._safe_update_status(f"Rejected: {message.source_term}")
            logger.debug("Term rejected", term=message.source_term)
        except Exception as e:
            logger.exception("Error rejecting term", term=message.source_term, error=str(e))
            self._safe_update_status(f"Error rejecting term: {e}")

    def on_term_modal_cancelled(self, message: TermModal.Cancelled) -> None:
        """Handle modal cancellation."""
        pass
