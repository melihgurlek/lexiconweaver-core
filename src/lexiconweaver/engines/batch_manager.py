"""Batch Manager for orchestrating multi-chapter translation workflow."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from lexiconweaver.config import Config
from lexiconweaver.database.models import ChapterMetadata, Project
from lexiconweaver.engines.global_scout import Chapter, GlobalScout
from lexiconweaver.engines.weaver import TranslatedChapter, Weaver
from lexiconweaver.exceptions import ValidationError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.utils.cost_estimator import calculate_dry_run_summary, format_dry_run_summary
from lexiconweaver.utils.document_writer import write_document
from lexiconweaver.utils.folder_manager import (
    ChapterFile,
    discover_chapters,
    get_output_filename,
    get_translated_chapters,
    setup_workspace,
)

logger = get_logger(__name__)


@dataclass
class BatchProgress:
    """Progress information for batch processing."""
    
    stage: Literal["scout", "translate", "merge"]
    current: int
    total: int
    message: str


class BatchManager:
    """
    Orchestrates multi-chapter translation workflow.
    
    Manages:
    - Chapter discovery and organization
    - Global scouting across chapters
    - Parallel translation with checkpointing
    - Merging and TOC generation
    """
    
    def __init__(self, config: Config, project: Project, workspace_path: Path):
        """
        Initialize Batch Manager.
        
        Args:
            config: Application configuration
            project: Project context
            workspace_path: Base path for workspace folders
        """
        self.config = config
        self.project = project
        self.workspace_path = workspace_path
        
        self.folders = setup_workspace(workspace_path)
        
        self.weaver = Weaver(config, project)
        self.global_scout = GlobalScout(config, project)
    
    async def process_folder(
        self,
        mode: Literal["full", "scout_only", "translate_only"] = "full",
        resume: bool = True,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        auto_approve_terms: bool = False
    ) -> dict:
        """
        Process all chapters in input folder.
        
        Args:
            mode: Processing mode
                - "full": Scout + translate + merge
                - "scout_only": Only run global scout
                - "translate_only": Only translate (assumes scouting done)
            resume: Skip already-translated chapters (default: True)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with results summary
        """
        logger.info(f"Starting batch processing in {mode} mode", workspace=str(self.workspace_path))
        
        chapters = discover_chapters(self.folders["input"])
        logger.info(f"Discovered {len(chapters)} chapters")
        
        results = {
            "chapters_found": len(chapters),
            "chapters_scouted": 0,
            "chapters_translated": 0,
            "terms_discovered": 0,
            "merged_output": None
        }
        
        if mode in ["full", "scout_only"]:
            scout_results = await self.run_global_scout(chapters, progress_callback)
            results["chapters_scouted"] = scout_results["chapters_analyzed"]
            results["terms_discovered"] = scout_results["terms_found"]
            results["proposals_saved"] = scout_results.get("proposals_saved", 0)
            
            if mode == "scout_only":
                logger.info("Scout complete, stopping for user review")
                return results
            
            if mode == "full" and not auto_approve_terms and results.get("proposals_saved", 0) > 0:
                from lexiconweaver.database.models import ProposedTerm

                pending_count = ProposedTerm.select().where(
                    ProposedTerm.project == self.project,
                    ProposedTerm.status == "pending"
                ).count()
                if pending_count > 0:
                    results["pending_review"] = True
                    results["pending_count"] = pending_count
                    logger.info(
                        "Stopping before translation: %s term(s) pending review. "
                        "Review with approve-terms, then run with --mode translate_only",
                        pending_count,
                    )
                    return results

            if mode == "full" and auto_approve_terms and results.get("proposals_saved", 0) > 0:
                from lexiconweaver.database.models import ProposedTerm, GlossaryTerm

                pending = ProposedTerm.select().where(
                    ProposedTerm.project == self.project,
                    ProposedTerm.status == "pending"
                )

                approved_count = 0
                for proposal in pending:
                    try:
                        existing = GlossaryTerm.get_or_none(
                            GlossaryTerm.project == self.project,
                            GlossaryTerm.source_term == proposal.source_term
                        )
                        
                        if not existing:
                            GlossaryTerm.create(
                                project=self.project,
                                source_term=proposal.source_term,
                                target_term=proposal.proposed_translation,
                                category=proposal.proposed_category or "General",
                                notes=f"Auto-approved from scout: {proposal.llm_reasoning}"
                            )
                        
                        proposal.status = "approved"
                        proposal.save()
                        approved_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to auto-approve '{proposal.source_term}': {e}")
                        continue
                
                logger.info(f"Auto-approved {approved_count} terms before translation")
                results["auto_approved"] = approved_count
        
        if mode in ["full", "translate_only"]:
            translate_results = await self.translate_all_chapters(
                chapters,
                resume=resume,
                progress_callback=progress_callback
            )
            results["chapters_translated"] = translate_results["chapters_completed"]
            results["chapters_skipped"] = translate_results["chapters_skipped"]
            
            if translate_results["chapters_completed"] > 0:
                merge_results = await self.merge_chapters(
                    translate_results["translated_chapters"],
                    progress_callback=progress_callback
                )
                results["merged_output"] = merge_results["output_path"]
        
        logger.info("Batch processing completed", results=results)
        return results
    
    async def run_global_scout(
        self,
        chapter_files: list[ChapterFile],
        progress_callback: Callable[[BatchProgress], None] | None = None
    ) -> dict:
        """
        Run global scout across all chapters.
        
        Args:
            chapter_files: List of chapter files
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with scouting results
        """
        logger.info("Starting global scout", chapters=len(chapter_files))
        
        if progress_callback:
            progress_callback(BatchProgress(
                stage="scout",
                current=0,
                total=len(chapter_files),
                message="Loading chapters..."
            ))
        
        chapters = []
        for ch_file in chapter_files:
            text = ch_file.path.read_text(encoding='utf-8')
            chapters.append(Chapter(
                number=ch_file.number,
                text=text,
                filename=ch_file.filename
            ))
        
        def scout_progress(msg: str):
            if progress_callback:
                progress_callback(BatchProgress(
                    stage="scout",
                    current=0,
                    total=len(chapter_files),
                    message=msg
                ))
        
        candidates = self.global_scout.analyze_chapters(
            chapters,
            min_frequency=self.config.batch.scout_min_frequency,
            window_size=self.config.batch.scout_burst_window_size,
            burst_threshold=self.config.batch.scout_burst_threshold,
            progress_callback=scout_progress
        )
        
        logger.info(f"Global scout found {len(candidates)} candidate terms")
        
        if progress_callback:
            progress_callback(BatchProgress(
                stage="scout",
                current=len(chapter_files),
                total=len(chapter_files),
                message="Refining candidates with LLM..."
            ))
        
        refined_terms = await self.global_scout.refine_with_llm(
            candidates,
            top_percent=self.config.batch.scout_llm_refine_top_percent,
            progress_callback=scout_progress
        )
        
        burst_saved = self.global_scout.save_burst_terms_to_db(candidates[:len(refined_terms)])
        
        proposals_saved = self.global_scout.save_refined_terms_as_proposals(refined_terms)
        
        logger.info(
            "Global scout completed",
            burst_terms=burst_saved,
            proposals=proposals_saved
        )
        
        return {
            "chapters_analyzed": len(chapters),
            "candidates_found": len(candidates),
            "terms_refined": len(refined_terms),
            "terms_found": burst_saved,
            "proposals_saved": proposals_saved
        }
    
    async def translate_all_chapters(
        self,
        chapter_files: list[ChapterFile],
        resume: bool = True,
        progress_callback: Callable[[BatchProgress], None] | None = None
    ) -> dict:
        """
        Translate all chapters with checkpoint/resume support.
        
        Args:
            chapter_files: List of chapter files
            resume: Skip already-translated chapters (default: True)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with translation results
        """
        logger.info("Starting chapter translation", chapters=len(chapter_files), resume=resume)
        
        output_folder = self.folders["output"]
        
        translated_chapters_nums = get_translated_chapters(output_folder) if resume else set()
        
        chapters_to_translate = []
        chapters_skipped = []
        
        for ch_file in chapter_files:
            if resume and ch_file.number in translated_chapters_nums:
                logger.info(f"Skipping chapter {ch_file.number} (already translated)")
                chapters_skipped.append(ch_file.number)
            else:
                chapters_to_translate.append(ch_file)
        
        if not chapters_to_translate:
            logger.info("All chapters already translated")
            return {
                "chapters_completed": 0,
                "chapters_skipped": len(chapters_skipped),
                "translated_chapters": []
            }
        
        logger.info(f"Translating {len(chapters_to_translate)} chapters (skipped {len(chapters_skipped)})")
        
        chapters_data = []
        for ch_file in chapters_to_translate:
            text = ch_file.path.read_text(encoding='utf-8')
            chapters_data.append((text, ch_file.number, ch_file.filename))
        
        completed = 0
        total = len(chapters_data)
        
        def update_progress():
            if progress_callback:
                progress_callback(BatchProgress(
                    stage="translate",
                    current=completed,
                    total=total,
                    message=f"Translating {completed}/{total} chapters..."
                ))
        
        update_progress()
        
        max_parallel = self.config.batch.max_parallel_chapters
        translated_chapters = await self.weaver.translate_chapters_parallel(
            chapters_data,
            max_concurrency=max_parallel,
            extract_titles=self.config.batch.extract_chapter_titles
        )
        
        for translated_ch in translated_chapters:
            output_file = output_folder / get_output_filename(
                ChapterFile(
                    path=Path(translated_ch.source_filename),
                    filename=translated_ch.source_filename,
                    number=translated_ch.number
                )
            )
            
            output_file.write_text(translated_ch.content, encoding='utf-8')
            logger.info(f"Saved chapter {translated_ch.number} to {output_file.name}")
            
            try:
                ChapterMetadata.create(
                    project=self.project,
                    chapter_num=translated_ch.number,
                    filename=translated_ch.source_filename,
                    title=translated_ch.title,
                    word_count=len(translated_ch.content.split()),
                    char_count=len(translated_ch.content),
                    translated=True
                )
            except Exception as e:
                logger.warning(f"Failed to save chapter metadata: {e}")
            
            completed += 1
            update_progress()
        
        return {
            "chapters_completed": len(translated_chapters),
            "chapters_skipped": len(chapters_skipped),
            "translated_chapters": translated_chapters
        }
    
    async def merge_chapters(
        self,
        translated_chapters: list[TranslatedChapter],
        output_format: str = "epub",
        progress_callback: Callable[[BatchProgress], None] | None = None
    ) -> dict:
        """
        Merge translated chapters into a single output file.
        
        Args:
            translated_chapters: List of translated chapters
            output_format: Output format ("epub", "txt", "pdf")
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with merge results
        """
        logger.info(f"Merging {len(translated_chapters)} chapters", format=output_format)
        
        if progress_callback:
            progress_callback(BatchProgress(
                stage="merge",
                current=0,
                total=1,
                message="Merging chapters..."
            ))
        
        sorted_chapters = sorted(translated_chapters, key=lambda ch: ch.number)
        
        merged_folder = self.folders["merged"]
        output_file = merged_folder / f"complete.{output_format}"
        
        if output_format == "epub":
            chapters_data = [(ch.title, ch.content) for ch in sorted_chapters]
            write_document(
                output_file,
                title=f"{self.project.title} - Complete Translation",
                chapters=chapters_data
            )
        else:
            chapters_data = [(ch.title, ch.content) for ch in sorted_chapters]
            write_document(
                output_file,
                title=f"{self.project.title} - Complete Translation",
                chapters=chapters_data
            )
        
        logger.info(f"Merged output saved to {output_file}")
        
        if progress_callback:
            progress_callback(BatchProgress(
                stage="merge",
                current=1,
                total=1,
                message=f"Merge complete: {output_file.name}"
            ))
        
        return {
            "output_path": str(output_file),
            "chapters_merged": len(sorted_chapters),
            "output_format": output_format
        }
    
    def dry_run(self, provider: str | None = None) -> str:
        """
        Calculate cost estimate without running translation.
        
        Args:
            provider: Provider name (default: from config)
            
        Returns:
            Formatted summary string
        """
        if provider is None:
            provider = self.config.provider.primary
        
        estimate = calculate_dry_run_summary(
            self.folders["input"],
            provider=provider,
            max_parallel=self.config.batch.max_parallel_chapters
        )
        
        return format_dry_run_summary(estimate)
