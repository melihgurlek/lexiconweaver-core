"""Translation screen for side-by-side translation view."""

import asyncio
from pathlib import Path

from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project
from lexiconweaver.engines.weaver import Weaver
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers import LLMProviderManager
from lexiconweaver.tui.widgets.export_modal import ExportModal
from lexiconweaver.tui.widgets.text_panel import TextPanel
from lexiconweaver.tui.widgets.translation_panel import TranslationPanel
from lexiconweaver.utils.document_writer import write_document
from lexiconweaver.utils.highlighting import HighlightSpan, highlight_terms
from lexiconweaver.utils.text_processor import batch_paragraphs_smart, split_into_sentences

logger = get_logger(__name__)


class TranslationScreen(Screen):
    """Screen displaying source and translated text side-by-side."""

    BINDINGS = [
        ("q", "quit", "Back to Main"),
        ("e", "export", "Export"),
        ("s", "sync_scroll", "Sync Scroll"),
    ]

    DEFAULT_CSS = """
    TranslationScreen {
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

    #source_container {
        width: 1fr;
        border: solid $primary;
        min-height: 5;
    }

    #translation_container {
        width: 1fr;
        border: solid $primary;
        min-height: 5;
    }

    #source_scroll_container {
        height: 1fr;
    }

    #translation_scroll_container {
        height: 1fr;
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
        """Initialize the translation screen."""
        super().__init__(*args, **kwargs)
        self.config = config
        self.project = project
        self._text = text
        self._text_file = text_file
        self._weaver: Weaver | None = None
        self._translated_text = ""
        self._is_translating = False
        self._target_terms: set[str] = set()

    def compose(self):
        """Compose the screen widgets."""
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="source_container"):
                yield Static("Source Text", classes="section_title")
                with ScrollableContainer(id="source_scroll_container"):
                    yield TextPanel(self._text, id="source_panel")
            with Vertical(id="translation_container"):
                yield Static("Translation", classes="section_title", id="translation_title")
                with ScrollableContainer(id="translation_scroll_container"):
                    yield TranslationPanel(id="translation_panel")
        yield Static("Ready", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        logger.debug("TranslationScreen mounted")
        self.call_after_refresh(self._initialize_screen)

    def _initialize_screen(self) -> None:
        """Initialize screen after widgets are ready."""
        try:
            # Load source text highlights
            self._load_source_highlights()
            
            # Load target terms for translation highlighting
            self._load_target_terms()
            
            # Initialize Weaver
            self._weaver = Weaver(self.config, self.project)
            
            # Start translation automatically (connection test happens in worker)
            self._start_translation()
            
            logger.debug("TranslationScreen initialization completed")
        except Exception as e:
            logger.exception("Error initializing translation screen", error=str(e))
            self._safe_update_status(f"Error: {e}")

    def _load_source_highlights(self) -> None:
        """Load and apply highlights to source text panel."""
        try:
            from lexiconweaver.database.models import GlossaryTerm
            from lexiconweaver.utils.cache import get_cache
            
            cache = get_cache()
            
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
            
            confirmed_terms = cache.get_confirmed_terms(self.project, _fetch_confirmed_terms)
            
            source_panel = self.query_one("#source_panel", TextPanel)
            highlights = highlight_terms(self._text, confirmed_terms, set())
            source_panel.set_highlights(highlights)
            
            logger.debug("Source highlights loaded", count=len(highlights))
        except Exception as e:
            logger.exception("Error loading source highlights", error=str(e))

    def _load_target_terms(self) -> None:
        """Load target terms from glossary for translation highlighting."""
        try:
            terms = GlossaryTerm.select().where(
                GlossaryTerm.project == self.project
            )
            self._target_terms = {term.target_term for term in terms if term.target_term}
            logger.debug("Target terms loaded", count=len(self._target_terms))
        except Exception as e:
            logger.exception("Error loading target terms", error=str(e))
            self._target_terms = set()

    def _set_translation_title(self, title: str) -> None:
        """Update the translation panel section title."""
        try:
            title_widget = self.query_one("#translation_title", Static)
            title_widget.update(title)
        except Exception:
            pass

    async def _test_provider_connection(self) -> None:
        """Test LLM provider connection (Ollama or DeepSeek)."""
        try:
            self._safe_update_status("Testing connection...")
            await asyncio.sleep(0)
            manager = LLMProviderManager(self.config)
            provider = await manager.get_available_provider()
            self._safe_update_status(f"Connected ({provider.name}). Starting translation...")
            await asyncio.sleep(0)
        except Exception as e:
            logger.exception("Provider connection test failed", error=str(e))
            raise

    def _update_translation_highlights(self) -> None:
        """Update highlights in translation panel."""
        try:
            translation_panel = self.query_one("#translation_panel", TranslationPanel)
            # Highlight target terms in translated text
            highlights = highlight_terms(self._translated_text, self._target_terms, set())
            translation_panel.set_highlights(highlights)
        except Exception as e:
            logger.debug("Error updating translation highlights", error=str(e))

    def _start_translation(self) -> None:
        """Start the translation process."""
        if self._is_translating:
            logger.warning("Translation already in progress")
            return
        
        if not self._weaver:
            self._safe_update_status("Weaver not initialized")
            return
        
        self._is_translating = True
        self._safe_update_status("Starting translation...")
        
        self._set_translation_title("Translation (in progress…)")
        # Use run_worker to run async translation in background
        async def test_and_translate():
            try:
                await self._test_provider_connection()
                await self._translate_text()
            except Exception as e:
                logger.exception("Translation worker error", error=str(e))
                self._safe_update_status(f"Error: {e}")
                self._set_translation_title("Translation")
                self._is_translating = False
        
        self.run_worker(
            test_and_translate(),
            name="translation",
            exclusive=True,
        )

    async def _translate_text(self) -> None:
        """Translate the text using smart batching."""
        try:
            if not self._text or not self._text.strip():
                self._safe_update_status("No text to translate")
                self._set_translation_title("Translation")
                logger.warning("Attempted to translate empty text")
                return

            max_chars = self.config.weaver.translation_batch_max_chars
            context_sentences = self.config.weaver.translation_context_sentences

            batches = batch_paragraphs_smart(self._text, max_chars, context_sentences)
            total_batches = len(batches)

            if total_batches == 0:
                self._safe_update_status("No batches to translate")
                self._set_translation_title("Translation")
                logger.warning("No batches created from text")
                return

            self._safe_update_status(f"Translating batch 1/{total_batches}…")
            await asyncio.sleep(0)
            logger.info("Starting translation", total_batches=total_batches, text_length=len(self._text))

            translation_panel = self.query_one("#translation_panel", TranslationPanel)
            previous_context = ""

            for batch_idx, batch in enumerate(batches, 1):
                if not batch or not batch.strip():
                    logger.warning(f"Skipping empty batch {batch_idx}")
                    continue

                self._safe_update_status(
                    f"Translating batch {batch_idx}/{total_batches}… (streaming)"
                )
                await asyncio.sleep(0)
                logger.debug(f"Translating batch {batch_idx}", batch_length=len(batch))

                batch_translation = ""
                chunk_count = 0
                try:
                    async for chunk in self._weaver.translate_batch_streaming(
                        batch, previous_context
                    ):
                        if chunk:
                            batch_translation += chunk
                            chunk_count += 1
                            translation_panel.append_text(chunk)
                            if len(batch_translation) % 200 == 0:
                                self._translated_text = translation_panel.get_text()
                                self._update_translation_highlights()
                                await asyncio.sleep(0)
                except Exception as batch_error:
                    error_msg = str(batch_error)
                    logger.exception(
                        f"Error translating batch {batch_idx}", error=error_msg
                    )
                    self._safe_update_status(
                        f"Batch {batch_idx} error: {error_msg[:60]}"
                    )
                    batch_translation = (
                        f"\n[ERROR in batch {batch_idx}: {error_msg[:50]}...]\n"
                    )
                    translation_panel.append_text(batch_translation)
                    continue

                logger.debug(
                    f"Batch {batch_idx} completed",
                    chunks=chunk_count,
                    translation_length=len(batch_translation),
                )

                if not batch_translation:
                    logger.warning(f"Batch {batch_idx} produced no translation")
                    batch_translation = f"[Translation failed for batch {batch_idx}]"
                    translation_panel.append_text(batch_translation)

                self._translated_text = translation_panel.get_text()
                self._update_translation_highlights()

                if batch_idx < total_batches:
                    self._safe_update_status(
                        f"Batch {batch_idx}/{total_batches} done. Next…"
                    )
                    await asyncio.sleep(0)

                if batch_translation:
                    context_sentences_list = split_into_sentences(batch_translation)
                    if (
                        context_sentences_list
                        and len(context_sentences_list) >= context_sentences
                    ):
                        previous_context = "\n\n".join(
                            context_sentences_list[-context_sentences:]
                        )
                    else:
                        context_length = min(200, len(batch_translation))
                        previous_context = (
                            batch_translation[-context_length:]
                            if batch_translation
                            else ""
                        )

            self._translated_text = translation_panel.get_text()
            self._update_translation_highlights()

            self._set_translation_title("Translation (complete)")
            if self.config.weaver.translation_auto_export:
                self._export_translation(auto=True)
                self._safe_update_status(
                    "Translation complete. Exported. Press 'e' to export again."
                )
            else:
                self._safe_update_status(
                    "Translation complete. Press 'e' to export."
                )

            logger.debug("Translation completed", length=len(self._translated_text))

        except Exception as e:
            logger.exception("Error during translation", error=str(e))
            self._safe_update_status(f"Translation error: {e}")
            self._set_translation_title("Translation")
        finally:
            self._is_translating = False

    def _safe_update_status(self, message: str) -> None:
        """Update the status bar safely."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
            logger.debug("Status updated", message=message)
        except Exception as e:
            logger.debug("Could not update status bar", message=message, error=str(e))

    def action_quit(self) -> None:
        """Handle quit action - return to main screen."""
        # Cancel any running workers
        if self._is_translating:
            # The worker will be cancelled when screen is dismissed
            self._is_translating = False
        self.dismiss()

    def action_export(self) -> None:
        """Handle export action: show format/path modal or export to default."""
        if not self._translated_text:
            self._safe_update_status("No translation to export")
            return
        default_path = (
            self._text_file.parent / f"{self._text_file.stem}_translated.txt"
            if self._text_file
            else Path.cwd() / "translation.txt"
        )
        existing = self.query(ExportModal)
        if not existing:
            self.mount(
                ExportModal(default_path=default_path, default_format="txt")
            )

    def _export_translation(self, auto: bool = False) -> None:
        """Export translation to default .txt file (used for auto-export)."""
        if not self._translated_text:
            return
        default_path = (
            self._text_file.parent / f"{self._text_file.stem}_translated.txt"
            if self._text_file
            else Path.cwd() / "translation.txt"
        )
        try:
            write_document(
                default_path,
                self._translated_text,
                title=self._text_file.stem if self._text_file else "Translation",
            )
            self._safe_update_status(f"Translation exported to {default_path}")
            logger.info("Translation exported", file=str(default_path))
        except Exception as e:
            logger.exception("Error exporting translation", error=str(e))
            self._safe_update_status(f"Export error: {e}")

    def on_export_modal_export_requested(
        self, event: ExportModal.ExportRequested
    ) -> None:
        """Handle export from modal: write to chosen path and update status."""
        path = event.path
        try:
            title = self._text_file.stem if self._text_file else "Translation"
            write_document(path, self._translated_text, title=title)
            self._safe_update_status(f"Translation exported to {path}")
            logger.info("Translation exported", file=str(path))
        except Exception as e:
            logger.exception("Error exporting translation", error=str(e))
            self._safe_update_status(f"Export error: {e}")

    def on_export_modal_cancelled(self, _event: ExportModal.Cancelled) -> None:
        """Handle export modal cancelled."""
        self._safe_update_status("Export cancelled")

    def action_sync_scroll(self) -> None:
        """Synchronize scroll positions between panels."""
        try:
            source_container = self.query_one("#source_scroll_container", ScrollableContainer)
            translation_container = self.query_one("#translation_scroll_container", ScrollableContainer)
            
            # Get source scroll position
            source_scroll_y = getattr(source_container, "scroll_y", 0) or 0
            source_max_scroll = getattr(source_container, "scroll_max_y", 0) or 0
            
            # Calculate percentage
            if source_max_scroll > 0:
                scroll_percentage = source_scroll_y / source_max_scroll
                
                # Apply to translation panel
                translation_max_scroll = getattr(translation_container, "scroll_max_y", 0) or 0
                target_scroll = int(scroll_percentage * translation_max_scroll)
                
                if hasattr(translation_container, "scroll_to"):
                    translation_container.scroll_to(y=target_scroll, animate=False)
                elif hasattr(translation_container, "scroll_y"):
                    translation_container.scroll_y = target_scroll
                
                self._safe_update_status("Scroll synchronized")
            else:
                self._safe_update_status("Cannot sync: no scrollable content")
                
        except Exception as e:
            logger.debug("Error syncing scroll", error=str(e))
            self._safe_update_status("Scroll sync failed")

