"""Weaver engine for LLM-based translation with glossary enforcement."""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from flashtext import KeywordProcessor
from peewee import IntegrityError

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project, TranslationCache
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.exceptions import ProviderError, TranslationError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers import LLMProviderManager
from lexiconweaver.utils.text_processor import extract_paragraphs, generate_hash, split_into_sentences

logger = get_logger(__name__)


@dataclass
class TranslatedChapter:
    """Result of chapter translation."""
    
    number: int
    title: str
    content: str
    source_filename: str


class Weaver(BaseEngine):
    """Generation engine that interfaces with LLM providers for translation."""

    def __init__(self, config: Config, project: Project) -> None:
        """Initialize the Weaver engine.
        
        Args:
            config: Application configuration.
            project: The project context for glossary terms.
        """
        self.config = config
        self.project = project
        self.provider_manager = LLMProviderManager(config)
        self.cache_enabled = config.cache.enabled

    def _prepare_messages(
        self, text: str, mini_glossary: dict[str, str], context: str = ""
    ) -> list[dict[str, str]]:
        """Construct the chat messages for the LLM."""

        # 1. Prepare glossary block (strict mapping format).
        glossary_block = "No specific terms."
        if mini_glossary:
            glossary_lines = [
                f"- {src} -> {tgt}" for src, tgt in mini_glossary.items()
            ]
            glossary_block = "\n".join(glossary_lines)

        # 2. System prompt (strict rules).
        system_content = (
        "You are an expert literary translator specializing in Wuxia/Xianxia fantasy novels. "
        "Target Language: Turkish.\n\n"

        "PRIMARY OBJECTIVE: Produce a translation that reads naturally in Turkish, adhering to the logic of the Turkish language (agglutinative suffixes, SOV order) rather than mimicking English structure.\n\n"

        "CRITICAL GRAMMAR RULES:\n"
        "1. **Possessive & Suffix Logic (HIGHEST PRIORITY):**\n"
        "   - **Rule:** Ensure possessive suffixes match the intended subject, even if words are far apart.\n"
        "   - *Source:* 'I will be your teacher for this class.'\n"
        "   - *Bad:* 'Bu dersin öğretmeniniz olacağım' (Broken suffix chain)\n"
        "   - *Good:* 'Bu dersteki öğretmeniniz ben olacağım'\n\n"

        "2. **Context-Aware Case Selection (Magical Context):**\n"
        "   - **Rule:** For abstract nouns (safety, permission), ask: Is this a physical location or a functional state?\n"
        "   - *Source:* '...in the safety of his room' (Implies magical wards/protection)\n"
        "   - *Bad:* 'Odasının güvenliğinde' (Locative - implies just a place)\n"
        "   - *Good:* 'Odasının korunaklı ortamında' OR 'Odasının güvenliği sayesinde'\n"
        "   - *Source:* 'Zorian was allowed'\n"
        "   - *Bad:* 'Zorian izin verildi'\n"
        "   - *Good:* 'Zorian'a izin verildi' (Dative recipient)\n\n"

        "3. **Turkish Syntax (SOV Reordering):**\n"
        "   - Move verbs to the end.\n"
        "   - Move time expressions (now, then) to the start.\n"
        "   - *Source:* 'He quickly ate the apple.' -> 'Elmayı hızlıca yedi.'\n\n"

        "4. **Idiomatic Localization:**\n"
        "   - *Source:* 'something to eat' -> 'yiyecek bir şeyler' (NOT 'yemek için bir şey')\n"
        "   - *Source:* 'you had all the time in the world' -> 'bol bol vaktin vardı'\n"
        "   - *Source:* 'whose fault is that?' -> 'suç kimde peki?'\n"
        "   - *Source:* 'sorry for you' -> 'senin adına üzüldüm'\n\n"

        "5. **Formatting & Terms:**\n"
        "   - **START IMMEDIATELY:** Do not repeat the input headers ('### SÖZLÜK', '### BAĞLAM', etc.).\n"
        "   - Use glossary Root Forms but apply correct suffixes (e.g., 'Qi' -> 'Qi'yi').\n"
        "   - **Contextual translations:** If a glossary entry shows alternatives separated by ' / ' (e.g., 'A -> X / Y'), choose the ONE that fits the context best. Output only that option, not the full string.\n"
        "   - **NO** markdown artifacts ('***', '-break-', '-ara-').\n"
        "   - Output **ONLY** the Turkish translation. If you output headers, you FAIL."
    )
        user_content = (
            "### GLOSSARY (Use as Roots):\n"
            f"{glossary_block}\n\n"
            "### CONTEXT (PREVIOUSLY TRANSLATED - DO NOT TRANSLATE):\n"
            f'{context if context else "No previous context."}\n\n'
            "### CURRENT CHUNK (TRANSLATE THIS):\n"
            f"{text}\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    async def translate_paragraph(
        self, paragraph: str, use_cache: bool = True
    ) -> str:
        """Translate a single paragraph."""
        if use_cache and self.cache_enabled:
            cached = self._get_from_cache(paragraph)
            if cached is not None:
                logger.debug("Cache hit for paragraph", hash=generate_hash(paragraph)[:8])
                return cached

        mini_glossary = self._build_mini_glossary(paragraph)

        messages = self._prepare_messages(paragraph, mini_glossary)

        try:
            translation = await self.provider_manager.generate(messages)
        except ProviderError as e:
            raise TranslationError(
                f"Translation failed: {e.message}",
                details=f"Provider: {e.provider}",
            ) from e

        self._verify_translation(paragraph, translation, mini_glossary)

        if use_cache and self.cache_enabled:
            self._store_in_cache(paragraph, translation)

        return translation

    async def translate_streaming(
        self, paragraph: str
    ) -> AsyncIterator[str]:
        """Translate a paragraph with streaming response."""
        mini_glossary = self._build_mini_glossary(paragraph)

        messages = self._prepare_messages(paragraph, mini_glossary)

        full_translation = ""
        try:
            async for chunk in self.provider_manager.generate_streaming(messages):
                full_translation += chunk
                yield chunk
        except ProviderError as e:
            raise TranslationError(
                f"Streaming translation failed: {e.message}",
                details=f"Provider: {e.provider}",
            ) from e

        self._verify_translation(paragraph, full_translation, mini_glossary)

        if self.cache_enabled:
            self._store_in_cache(paragraph, full_translation)

    async def translate_batch_streaming(
        self, batch: str, context: str = ""
    ) -> AsyncIterator[str]:
        """Translate a batch with streaming response, optionally including context.
        
        Args:
            batch: The text batch to translate (ONLY new text, not merged with context)
            context: Optional context from previous batch (e.g., last 2 sentences)
            
        Yields:
            Translation chunks as they arrive from the LLM
        """
        text_to_translate = batch
        
        mini_glossary = self._build_mini_glossary(text_to_translate)
        
        messages = self._prepare_messages(text_to_translate, mini_glossary, context)
        
        full_translation = ""
        try:
            async for chunk in self.provider_manager.generate_streaming(messages):
                full_translation += chunk
                yield chunk
        except ProviderError as e:
            raise TranslationError(
                f"Batch streaming translation failed: {e.message}",
                details=f"Provider: {e.provider}",
            ) from e
        
        self._verify_translation(text_to_translate, full_translation, mini_glossary)
        
        if self.cache_enabled:
            self._store_in_cache(batch, full_translation)

    async def translate_text(
        self, text: str, use_cache: bool = True
    ) -> str:
        """Translate full text by processing paragraphs."""
        paragraphs = extract_paragraphs(text)
        translated_paragraphs: list[str] = []

        for i, paragraph in enumerate(paragraphs):
            logger.info("Translating paragraph", paragraph=i + 1, total=len(paragraphs))
            try:
                translation = await self.translate_paragraph(paragraph, use_cache)
                translated_paragraphs.append(translation)
            except TranslationError as e:
                logger.error(
                    "Translation failed for paragraph",
                    paragraph=i + 1,
                    error=str(e),
                )
                # Mark as unstable by adding marker
                translated_paragraphs.append(f"[UNSTABLE] {paragraph}")
            except Exception as e:
                logger.error(
                    "Unexpected error translating paragraph",
                    paragraph=i + 1,
                    error=str(e),
                )
                translated_paragraphs.append(f"[ERROR] {paragraph}")

        return "\n\n".join(translated_paragraphs)

    async def translate_chapter(
        self,
        text: str,
        chapter_num: int,
        extract_title: bool = False,
        source_filename: str = ""
    ) -> TranslatedChapter:
        """
        Translate a complete chapter with optional title extraction.
        
        Args:
            text: Chapter text to translate
            chapter_num: Chapter number
            extract_title: If True, extract chapter title from first batch
            source_filename: Original filename for reference
            
        Returns:
            TranslatedChapter with title and translated content
        """
        from lexiconweaver.utils.text_processor import batch_paragraphs_smart
        
        max_chars = self.config.weaver.translation_batch_max_chars
        context_sentences = self.config.weaver.translation_context_sentences
        batches = batch_paragraphs_smart(text, max_chars, context_sentences)
        
        if not batches:
            logger.warning(f"Chapter {chapter_num} produced no batches")
            return TranslatedChapter(
                number=chapter_num,
                title=f"Chapter {chapter_num}",
                content="",
                source_filename=source_filename
            )
        
        logger.info(f"Translating chapter {chapter_num}", batches=len(batches))
        
        title = None
        translated_batches = []
        previous_context = ""
        
        for batch_idx, batch in enumerate(batches, 1):
            if not batch or not batch.strip():
                continue
            
            logger.debug(f"Translating batch {batch_idx}/{len(batches)} of chapter {chapter_num}")
            
            if batch_idx == 1 and extract_title:
                title, translation = await self._translate_batch_with_title_extraction(
                    batch, chapter_num
                )
            else:
                mini_glossary = self._build_mini_glossary(batch)
                messages = self._prepare_messages(batch, mini_glossary, previous_context)
                
                try:
                    translation = await self.provider_manager.generate(messages)
                except ProviderError as e:
                    raise TranslationError(
                        f"Translation failed for chapter {chapter_num}, batch {batch_idx}: {e.message}",
                        details=f"Provider: {e.provider}"
                    ) from e
                
                self._verify_translation(batch, translation, mini_glossary)
            
            translated_batches.append(translation)
            
            if translation:
                context_sentences_list = split_into_sentences(translation)
                if context_sentences_list and len(context_sentences_list) >= context_sentences:
                    previous_context = "\n\n".join(context_sentences_list[-context_sentences:])
                else:
                    context_length = min(200, len(translation))
                    previous_context = translation[-context_length:] if translation else ""
        
        if not title or not title.strip():
            if source_filename:
                from pathlib import Path
                stem = Path(source_filename).stem
                title = stem.replace("_", " ").replace("-", " ").strip()
                if not title:
                    title = f"Chapter {chapter_num}"
            else:
                title = f"Chapter {chapter_num}"
            
            logger.info(f"Using fallback title for chapter {chapter_num}: {title}")
        elif not title.startswith(f"Chapter {chapter_num}"):
            title = f"Chapter {chapter_num}: {title}"
        
        content = "\n\n".join(translated_batches)
        
        return TranslatedChapter(
            number=chapter_num,
            title=title,
            content=content,
            source_filename=source_filename
        )
    
    async def _translate_batch_with_title_extraction(
        self, batch: str, chapter_num: int
    ) -> tuple[str, str]:
        """
        Translate first batch and extract chapter title using metadata header format.
        
        Args:
            batch: Text batch to translate
            chapter_num: Chapter number for fallback
            
        Returns:
            Tuple of (title, translation)
        """
        mini_glossary = self._build_mini_glossary(batch)
        
        messages = self._prepare_messages_with_title_extraction(
            batch, mini_glossary, chapter_num
        )
        
        try:
            response = await self.provider_manager.generate(messages)
        except ProviderError as e:
            raise TranslationError(
                f"Title extraction failed: {e.message}",
                details=f"Provider: {e.provider}"
            ) from e
        
        title, translation = self._extract_title_and_translation(response, chapter_num)
        
        self._verify_translation(batch, translation, mini_glossary)
        
        return title, translation
    
    def _prepare_messages_with_title_extraction(
        self, text: str, mini_glossary: dict[str, str], chapter_num: int
    ) -> list[dict[str, str]]:
        """Prepare messages with title extraction instruction."""
        glossary_block = "No specific terms."
        if mini_glossary:
            glossary_lines = [
                f"- {src} -> {tgt}" for src, tgt in mini_glossary.items()
            ]
            glossary_block = "\n".join(glossary_lines)
        
        system_content = (
            "You are an expert literary translator specializing in Wuxia/Xianxia fantasy novels. "
            "Target Language: Turkish.\n\n"
            
            "PRIMARY OBJECTIVE: Produce a translation that reads naturally in Turkish, adhering to the logic of the Turkish language (agglutinative suffixes, SOV order) rather than mimicking English structure.\n\n"
            
            "CRITICAL GRAMMAR RULES:\n"
            "1. **Possessive & Suffix Logic (HIGHEST PRIORITY):**\n"
            "   - **Rule:** Ensure possessive suffixes match the intended subject, even if words are far apart.\n"
            "   - *Source:* 'I will be your teacher for this class.'\n"
            "   - *Bad:* 'Bu dersin öğretmeniniz olacağım' (Broken suffix chain)\n"
            "   - *Good:* 'Bu dersteki öğretmeniniz ben olacağım'\n\n"
            
            "2. **Context-Aware Case Selection (Magical Context):**\n"
            "   - **Rule:** For abstract nouns (safety, permission), ask: Is this a physical location or a functional state?\n"
            "   - *Source:* '...in the safety of his room' (Implies magical wards/protection)\n"
            "   - *Bad:* 'Odasının güvenliğinde' (Locative - implies just a place)\n"
            "   - *Good:* 'Odasının korunaklı ortamında' OR 'Odasının güvenliği sayesinde'\n"
            "   - *Source:* 'Zorian was allowed'\n"
            "   - *Bad:* 'Zorian izin verildi'\n"
            "   - *Good:* 'Zorian'a izin verildi' (Dative recipient)\n\n"
            
            "3. **Turkish Syntax (SOV Reordering):**\n"
            "   - Move verbs to the end.\n"
            "   - Move time expressions (now, then) to the start.\n"
            "   - *Source:* 'He quickly ate the apple.' -> 'Elmayı hızlıca yedi.'\n\n"
            
            "4. **Idiomatic Localization:**\n"
            "   - *Source:* 'something to eat' -> 'yiyecek bir şeyler' (NOT 'yemek için bir şey')\n"
            "   - *Source:* 'you had all the time in the world' -> 'bol bol vaktin vardı'\n"
            "   - *Source:* 'whose fault is that?' -> 'suç kimde peki?'\n"
            "   - *Source:* 'sorry for you' -> 'senin adına üzüldüm'\n\n"
            
            "5. **Formatting & Terms:**\n"
            "   - Use glossary Root Forms but apply correct suffixes (e.g., 'Qi' -> 'Qi'yi').\n"
            "   - **Contextual translations:** If a glossary entry shows alternatives separated by ' / ' (e.g., 'A -> X / Y'), choose the ONE that fits the context best. Output only that option, not the full string.\n"
            "   - **NO** markdown artifacts ('***', '-break-', '-ara-').\n"
        )
        
        user_content = (
            "### ADDITIONAL TASK (CRITICAL): This is the beginning of a chapter. Extract the chapter title from the first line/heading.\n"
            "Output format:\n"
            f"TITLE: Chapter {chapter_num}: [Extracted Chapter Title]\n"
            "---\n"
            "[Translation text here]\n\n"
            
            f"If no clear title exists, use \"TITLE: Chapter {chapter_num}\".\n"
            "CRITICAL: Include the chapter number in the title. Start with \"TITLE:\" on the first line, then \"---\" separator, then the translation.\n\n"
            
            "### GLOSSARY (Use as Roots):\n"
            f"{glossary_block}\n\n"
            
            "### CURRENT CHUNK (TRANSLATE THIS):\n"
            f"{text}\n"
        )
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    
    def _extract_title_and_translation(
        self, llm_response: str, chapter_num: int
    ) -> tuple[str, str]:
        """
        Extract title and translation from metadata header format.
        
        Args:
            llm_response: LLM response with TITLE:/--- format
            chapter_num: Chapter number for fallback
            
        Returns:
            Tuple of (title, translation)
        """
        lines = llm_response.strip().split('\n')
        
        title = ""
        translation_lines = []
        found_separator = False
        
        for i, line in enumerate(lines[:5]):
            if line.startswith("TITLE:"):
                title = line[6:].strip()
            elif line.strip() == "---":
                found_separator = True
                translation_lines = lines[i+1:]
                break
        
        if not found_separator:
            logger.warning("No TITLE/--- separator found, treating as plain translation")
            translation_lines = lines
        
        if title and title.strip():
            # Avoid duplicating "Chapter N" if LLM already included it
            if title.strip().startswith(f"Chapter {chapter_num}"):
                full_title = title.strip()
            else:
                full_title = f"Chapter {chapter_num}: {title.strip()}"
        else:
            full_title = f"Chapter {chapter_num}"
            logger.info(f"No title found, using fallback: {full_title}")
        
        translation = '\n'.join(translation_lines).strip()
        return full_title, translation
    
    async def translate_chapters_parallel(
        self,
        chapters: list[tuple[str, int, str]],
        max_concurrency: int = 5,
        extract_titles: bool = True
    ) -> list[TranslatedChapter]:
        """
        Translate multiple chapters in parallel with rate limiting.
        
        Args:
            chapters: List of (text, chapter_num, filename) tuples
            max_concurrency: Maximum concurrent translations (default: 5)
            extract_titles: Whether to extract chapter titles (default: True)
            
        Returns:
            List of TranslatedChapter objects
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def translate_with_semaphore(
            text: str, chapter_num: int, filename: str
        ) -> TranslatedChapter:
            async with semaphore:
                logger.info(f"Starting chapter {chapter_num}")
                result = await self.translate_chapter(
                    text, chapter_num, extract_title=extract_titles, source_filename=filename
                )
                await asyncio.sleep(0.1)
                return result
        
        tasks = [
            translate_with_semaphore(text, num, filename)
            for text, num, filename in chapters
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        translated_chapters = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chapter {chapters[i][1]} translation failed", error=str(result))
            else:
                translated_chapters.append(result)
        
        return translated_chapters

    def _build_mini_glossary(self, text: str) -> dict[str, str]:
        """Build a mini-glossary containing only terms that appear in the text."""
        glossary_terms = GlossaryTerm.select().where(
            GlossaryTerm.project == self.project
        )

        keyword_processor = KeywordProcessor(case_sensitive=False)
        term_map: dict[str, str] = {}
        
        for term in glossary_terms:
            if term.is_regex:
                continue

            source_term = term.source_term
            target_term = term.target_term

            keyword_processor.add_keyword(source_term)
            term_map[source_term.lower()] = target_term

        match_text = text.replace("\u00a0", " ")
        found_list = keyword_processor.extract_keywords(match_text)
        found_terms: set[str] = set(found_list) if found_list else set()
        
        text_lower = match_text.lower()
        for source_term_lower, target_term in term_map.items():
            if source_term_lower in text_lower:
                source_variants = [source_term_lower, source_term_lower.capitalize(), source_term_lower.title()]
                for variant in source_variants:
                    if variant in match_text or variant.lower() in text_lower:
                        found_terms.add(variant)
                        break
        
        mini_glossary: dict[str, str] = {}
        for found_term in found_terms:
            target = term_map.get(found_term.lower())
            if target:
                mini_glossary[found_term] = target

        return mini_glossary

    def _verify_translation(
        self, source: str, translation: str, mini_glossary: dict[str, str]
    ) -> None:
        """Verify that all glossary terms in source appear in translation.
        
        Uses lenient matching to account for Turkish suffixes and variations.
        """
        missing_terms: list[str] = []

        source_lower = source.lower()
        translation_lower = translation.lower()

        for source_term, target_term in mini_glossary.items():
            source_term_lower = source_term.lower()
            target_term_lower = target_term.lower()

            if source_term_lower not in source_lower:
                continue

            if target_term_lower in translation_lower:
                continue


            found = False
            if len(target_term_lower) >= 4:
                root_len = int(len(target_term_lower) * 0.7)
                root_guess = target_term_lower[:root_len]
                
                if root_guess in translation_lower:
                    found = True
                else:
                    translation_words = translation_lower.split()
                    for word in translation_words:
                        if root_guess in word or word.startswith(root_guess):
                            found = True
                            break

            if not found:
                missing_terms.append(source_term)

        if missing_terms:
            if len(missing_terms) > 2:
                logger.warning(
                    "Translation verification: multiple terms not found",
                    missing_terms=missing_terms[:5],  # Show max 5
                    count=len(missing_terms),
                    source_hash=generate_hash(source)[:8],
                )
            else:
                logger.debug(
                    "Translation verification: minor term variations",
                    missing_terms=missing_terms,
                    source_hash=generate_hash(source)[:8],
                )

    def _get_from_cache(self, paragraph: str) -> str | None:
        """Get translation from cache if available."""
        try:
            project_id = getattr(self.project, "id", None)
            cache_hash = (
                generate_hash(f"{project_id}:{paragraph}") if project_id is not None else generate_hash(paragraph)
            )

            cached = TranslationCache.get_or_none(
                TranslationCache.hash == cache_hash,
                TranslationCache.project == self.project,
            )

            if cached is None:
                legacy_hash = generate_hash(paragraph)
                cached = TranslationCache.get_or_none(
                    TranslationCache.hash == legacy_hash,
                    TranslationCache.project == self.project,
                )
            return cached.translation if cached else None
        except Exception as e:
            logger.warning("Failed to get from cache", error=str(e))
            return None

    def _store_in_cache(self, paragraph: str, translation: str) -> None:
        """Store translation in cache."""
        try:
            project_id = getattr(self.project, "id", None)
            cache_hash = (
                generate_hash(f"{project_id}:{paragraph}") if project_id is not None else generate_hash(paragraph)
            )

            try:
                TranslationCache.create(
                    hash=cache_hash,
                    project=self.project,
                    translation=translation,
                )
            except IntegrityError:
                (
                    TranslationCache.update(translation=translation, project=self.project)
                    .where(TranslationCache.hash == cache_hash)
                    .execute()
                )
        except Exception as e:
            logger.warning("Failed to store in cache", error=str(e))

    def process(self, text: str) -> str:
        """Process text synchronously (wrapper for async method)."""
        return asyncio.run(self.translate_text(text))
