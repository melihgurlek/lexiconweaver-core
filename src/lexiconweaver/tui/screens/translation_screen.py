"""Translation screen for side-by-side translation view."""

from pathlib import Path

from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project
from lexiconweaver.engines.weaver import Weaver
from lexiconweaver.logging_config import get_logger
from lexiconweaver.tui.widgets.text_panel import TextPanel
from lexiconweaver.tui.widgets.translation_panel import TranslationPanel
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
                yield Static("Translation", classes="section_title")
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

    async def _test_ollama_connection(self) -> None:
        """Test Ollama connection and model availability."""
        import httpx
        
        try:
            self._safe_update_status("Testing Ollama connection...")
            url = f"{self.config.ollama.url}/api/tags"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned status {response.status_code}")
                
                models_data = response.json()
                available_models = [model.get("name", "") for model in models_data.get("models", [])]
                
                # Check if configured model is available
                configured_model = self.config.ollama.model
                if configured_model not in available_models:
                    model_list = ", ".join(available_models) if available_models else "none"
                    raise Exception(
                        f"Model '{configured_model}' not found. Available models: {model_list}. "
                        f"Install with: ollama pull {configured_model}"
                    )
                
                logger.info("Ollama connection test passed", model=configured_model)
                self._safe_update_status(f"Connected to Ollama (model: {configured_model})")
        except httpx.RequestError as e:
            raise Exception(f"Cannot connect to Ollama at {self.config.ollama.url}. Is Ollama running?") from e
        except Exception as e:
            logger.exception("Ollama connection test failed", error=str(e))
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
        
        # Use run_worker to run async translation in background
        # Also run connection test in the same worker
        async def test_and_translate():
            try:
                await self._test_ollama_connection()
                await self._translate_text()
            except Exception as e:
                logger.exception("Translation worker error", error=str(e))
                self._safe_update_status(f"Error: {e}")
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
                logger.warning("Attempted to translate empty text")
                return
            
            max_chars = self.config.weaver.translation_batch_max_chars
            context_sentences = self.config.weaver.translation_context_sentences
            
            # Batch paragraphs
            batches = batch_paragraphs_smart(self._text, max_chars, context_sentences)
            total_batches = len(batches)
            
            if total_batches == 0:
                self._safe_update_status("No batches to translate")
                logger.warning("No batches created from text")
                return
            
            self._safe_update_status(f"Translating batch 1/{total_batches}...")
            logger.info("Starting translation", total_batches=total_batches, text_length=len(self._text))
            
            translation_panel = self.query_one("#translation_panel", TranslationPanel)
            previous_context = ""
            
            for batch_idx, batch in enumerate(batches, 1):
                if not batch or not batch.strip():
                    logger.warning(f"Skipping empty batch {batch_idx}")
                    continue
                    
                self._safe_update_status(f"Translating batch {batch_idx}/{total_batches}...")
                logger.debug(f"Translating batch {batch_idx}", batch_length=len(batch))
                
                # Translate batch with context
                # Note: context is included in prompt for coherence, but we'll accept
                # that some repetition may occur in the output
                batch_translation = ""
                chunk_count = 0
                try:
                    async for chunk in self._weaver.translate_batch_streaming(batch, previous_context):
                        if chunk:
                            batch_translation += chunk
                            chunk_count += 1
                            # Append to panel for streaming display
                            translation_panel.append_text(chunk)
                            # Update highlights periodically (every 200 chars to avoid performance issues)
                            if len(batch_translation) % 200 == 0:
                                self._translated_text = translation_panel.get_text()
                                self._update_translation_highlights()
                except Exception as batch_error:
                    error_msg = str(batch_error)
                    logger.exception(f"Error translating batch {batch_idx}", error=error_msg)
                    self._safe_update_status(f"Batch {batch_idx} error: {error_msg[:60]}")
                    # Add error marker to translation
                    batch_translation = f"\n[ERROR in batch {batch_idx}: {error_msg[:50]}...]\n"
                    translation_panel.append_text(batch_translation)
                    # Continue with next batch
                    continue
                
                logger.debug(f"Batch {batch_idx} completed", chunks=chunk_count, translation_length=len(batch_translation))
                
                if not batch_translation:
                    logger.warning(f"Batch {batch_idx} produced no translation")
                    batch_translation = f"[Translation failed for batch {batch_idx}]"
                    translation_panel.append_text(batch_translation)
                
                # Update translated text from panel (single source of truth)
                self._translated_text = translation_panel.get_text()
                
                # Update highlights after each batch
                self._update_translation_highlights()
                
                # Extract context for next batch from the translated batch
                # Use the last N sentences from the current batch translation
                if batch_translation:
                    context_sentences_list = split_into_sentences(batch_translation)
                    if context_sentences_list and len(context_sentences_list) >= context_sentences:
                        previous_context = "\n\n".join(context_sentences_list[-context_sentences:])
                    else:
                        # Fallback: use last portion of batch_translation
                        # Take last ~200 chars or entire batch if shorter
                        context_length = min(200, len(batch_translation))
                        previous_context = batch_translation[-context_length:] if batch_translation else ""
            
            # Final highlight update
            self._translated_text = translation_panel.get_text()
            self._update_translation_highlights()
            
            # Auto-export if enabled
            if self.config.weaver.translation_auto_export:
                self._export_translation(auto=True)
            else:
                self._safe_update_status("Translation complete! Press 'e' to export.")
            
            logger.debug("Translation completed", length=len(self._translated_text))
            
        except Exception as e:
            logger.exception("Error during translation", error=str(e))
            self._safe_update_status(f"Translation error: {e}")
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
        """Handle export action."""
        self._export_translation(auto=False)

    def _export_translation(self, auto: bool = False) -> None:
        """Export translation to file."""
        if not self._translated_text:
            self._safe_update_status("No translation to export")
            return
        
        try:
            # Determine output filename
            if self._text_file:
                output_file = self._text_file.parent / f"{self._text_file.stem}_translated{self._text_file.suffix}"
            else:
                output_file = Path.cwd() / "translation.txt"
            
            # If not auto-export, we could show a modal for filename input
            # For now, just use the default
            if not auto:
                # In a future enhancement, we could add a filename input modal
                pass
            
            # Write translation to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(self._translated_text)
            
            self._safe_update_status(f"Translation exported to {output_file}")
            logger.info("Translation exported", file=str(output_file))
            
        except Exception as e:
            logger.exception("Error exporting translation", error=str(e))
            self._safe_update_status(f"Export error: {e}")

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

