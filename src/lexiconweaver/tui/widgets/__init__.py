"""TUI widgets for LexiconWeaver."""

from lexiconweaver.tui.widgets.candidate_list import CandidateList
from lexiconweaver.tui.widgets.export_modal import ExportModal
from lexiconweaver.tui.widgets.term_modal import TermModal
from lexiconweaver.tui.widgets.text_panel import TextPanel
from lexiconweaver.tui.widgets.translation_panel import TranslationPanel

__all__ = ["TextPanel", "TranslationPanel", "CandidateList", "TermModal", "ExportModal"]
