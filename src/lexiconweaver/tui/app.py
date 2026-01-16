"""Main Textual application for LexiconWeaver."""

from pathlib import Path
from typing import Optional

from textual.app import App

from lexiconweaver.config import Config
from lexiconweaver.database.models import Project
from lexiconweaver.logging_config import configure_logging, get_logger
from lexiconweaver.tui.screens.main_screen import MainScreen

logger = get_logger(__name__)


class LexiconWeaverApp(App):
    """Main TUI application."""

    CSS_PATH = None  # Using inline CSS in screens
    TITLE = "LexiconWeaver"
    SUB_TITLE = "Web Novel Translation Framework"

    def __init__(
        self,
        config: Config,
        project: Project,
        text: str = "",
        text_file: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the application."""
        super().__init__(*args, **kwargs)
        self.config = config
        self.project = project
        self._text = text
        self._text_file = text_file

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        logger.info("LexiconWeaver TUI started", project=self.project.title)
        # Push the main screen explicitly
        self.push_screen(
            MainScreen(
                config=self.config,
                project=self.project,
                text=self._text,
                text_file=self._text_file,
            )
        )

    def action_quit(self) -> None:
        """Handle quit action."""
        self.exit()
