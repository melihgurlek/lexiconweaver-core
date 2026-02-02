"""Scout Refiner engine for LLM-based term cleaning and proposal."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, ProposedTerm, Project
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.engines.scout import CandidateTerm, Scout
from lexiconweaver.exceptions import ProviderError, ScoutError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers import LLMProviderManager

logger = get_logger(__name__)


@dataclass
class RefinedTerm:
    """A term that has been refined by the LLM."""

    source_term: str
    proposed_translation: str
    proposed_category: str | None
    reasoning: str | None
    context_snippet: str | None
    is_valid: bool  # Whether LLM considers this a valid term


class ScoutRefiner(BaseEngine):
    """LLM-based refinement engine for Scout candidates.
    
    This engine takes raw candidates from Scout and:
    1. Filters false positives (Clean Pass)
    2. Identifies missed terms (Find Missed Pass)
    3. Proposes translations and categories for valid terms
    
    All operations use batching for performance.
    """

    # Categories for term classification
    CATEGORIES = ["Person", "Location", "Skill", "Clan", "Item", "Other"]

    def __init__(self, config: Config, project: Project) -> None:
        """Initialize the Scout Refiner.
        
        Args:
            config: Application configuration.
            project: The project context.
        """
        self.config = config
        self.project = project
        self.provider_manager = LLMProviderManager(config)
        self.batch_size = config.scout.llm_batch_size
        self.scout = Scout(config, project)

    def process(self, text: str) -> list[RefinedTerm]:
        """Process text synchronously (wrapper for async method)."""
        return asyncio.run(self.refine_text(text))

    async def refine_text(
        self, text: str, progress_callback: Callable[[str], None] | None = None
    ) -> list[RefinedTerm]:
        """Run the full two-pass refinement on text.
        
        Args:
            text: The source text to analyze.
            progress_callback: Optional callback for progress updates (message string).
        
        Returns:
            List of refined terms with proposed translations.
        """
        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        existing_terms = self._get_existing_terms()

        _progress("Smart Scout: Pass 1 — heuristic scout...")
        await asyncio.sleep(0)
        logger.info("Running heuristic scout (Pass 1)")
        candidates = self.scout.process(text)
        logger.info("Scout found candidates", count=len(candidates))

        new_candidates = [
            c for c in candidates
            if c.term.lower() not in existing_terms
        ]
        logger.info(
            "Filtered existing terms",
            before=len(candidates),
            after=len(new_candidates),
        )

        if not new_candidates:
            logger.info("No new candidates to refine")
            return []

        _progress("Smart Scout: Pass 2a — cleaning candidates with LLM...")
        await asyncio.sleep(0)
        logger.info("Cleaning candidates with LLM (Pass 2a)")
        cleaned = await self._clean_candidates(new_candidates, text)
        logger.info("LLM validated candidates", count=len(cleaned))

        _progress("Smart Scout: Pass 2b — finding missed terms...")
        await asyncio.sleep(0)
        logger.info("Finding missed terms with LLM (Pass 2b)")
        missed = await self._find_missed_terms(text, cleaned)
        logger.info("LLM found missed terms", count=len(missed))

        all_terms = cleaned + missed

        if not all_terms:
            logger.info("No valid terms found after LLM refinement")
            return []

        _progress("Smart Scout: Pass 2c — proposing translations...")
        await asyncio.sleep(0)
        logger.info("Proposing translations with LLM (Pass 2c)")
        refined = await self._propose_translations(all_terms, text)
        logger.info("Generated proposals", count=len(refined))

        return refined

    async def _clean_candidates(
        self, candidates: list[CandidateTerm], text: str
    ) -> list[CandidateTerm]:
        """Use LLM to filter false positives from candidates.
        
        Args:
            candidates: Raw candidates from Scout.
            text: The source text for context.
        
        Returns:
            Filtered list of valid candidates.
        """
        if not candidates:
            return []

        batches = [
            candidates[i:i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        valid_terms: set[str] = set()

        for batch_idx, batch in enumerate(batches):
            logger.debug(
                "Processing clean batch",
                batch=batch_idx + 1,
                total=len(batches),
                size=len(batch),
            )

            terms_with_context = []
            for candidate in batch:
                context = candidate.context_snippet or ""
                terms_with_context.append({
                    "term": candidate.term,
                    "context": context[:200] if context else "",
                })

            prompt = self._build_clean_prompt(terms_with_context)

            try:
                response = await self.provider_manager.generate(prompt)
                valid_in_batch = self._parse_clean_response(response, batch)
                valid_terms.update(valid_in_batch)
            except ProviderError as e:
                logger.warning(
                    "LLM clean batch failed, keeping all candidates",
                    batch=batch_idx + 1,
                    error=str(e),
                )
                valid_terms.update(c.term.lower() for c in batch)

        return [c for c in candidates if c.term.lower() in valid_terms]

    async def _find_missed_terms(
        self, text: str, found_terms: list[CandidateTerm]
    ) -> list[CandidateTerm]:
        """Use LLM to find terms that Scout missed.
        
        Args:
            text: The source text.
            found_terms: Terms already found.
        
        Returns:
            List of newly discovered candidates.
        """
        found_set = {t.term.lower() for t in found_terms}
        existing = self._get_existing_terms()

        prompt = self._build_find_missed_prompt(text, list(found_set))

        try:
            response = await self.provider_manager.generate(prompt)
            missed = self._parse_find_missed_response(response, text)

            new_missed = []
            for term in missed:
                if term.term.lower() not in found_set and term.term.lower() not in existing:
                    new_missed.append(term)

            return new_missed

        except ProviderError as e:
            logger.warning("LLM find missed failed", error=str(e))
            return []

    async def _propose_translations(
        self, candidates: list[CandidateTerm], text: str
    ) -> list[RefinedTerm]:
        """Use LLM to propose translations and categories.
        
        Args:
            candidates: Valid candidates to translate.
            text: Source text for context.
        
        Returns:
            List of refined terms with proposals.
        """
        if not candidates:
            return []

        batches = [
            candidates[i:i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        results: list[RefinedTerm] = []

        for batch_idx, batch in enumerate(batches):
            logger.debug(
                "Processing translation batch",
                batch=batch_idx + 1,
                total=len(batches),
                size=len(batch),
            )

            terms_with_context = []
            for candidate in batch:
                context = candidate.context_snippet or self._extract_context(candidate.term, text)
                terms_with_context.append({
                    "term": candidate.term,
                    "context": context[:200] if context else "",
                })

            prompt = self._build_translation_prompt(terms_with_context)

            try:
                response = await self.provider_manager.generate(prompt)
                batch_results = self._parse_translation_response(response, batch)
                results.extend(batch_results)
            except ProviderError as e:
                logger.warning(
                    "LLM translation batch failed",
                    batch=batch_idx + 1,
                    error=str(e),
                )
                if len(batch) > 10:
                    smaller_batches = [
                        batch[i:i + 25]
                        for i in range(0, len(batch), 25)
                    ]
                    for small_batch in smaller_batches:
                        try:
                            small_terms = [
                                {"term": c.term, "context": c.context_snippet or ""}
                                for c in small_batch
                            ]
                            small_prompt = self._build_translation_prompt(small_terms)
                            small_response = await self.provider_manager.generate(small_prompt)
                            small_results = self._parse_translation_response(small_response, small_batch)
                            results.extend(small_results)
                        except ProviderError:
                            pass

        return results

    def _build_clean_prompt(
        self, terms_with_context: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Build the prompt for cleaning candidates."""
        terms_list = "\n".join(
            f"- Term: \"{t['term']}\"\n  Context: \"{t['context']}\""
            for t in terms_with_context
        )

        system_content = (
            "You are an expert in Wuxia/Xianxia fantasy literature terminology. "
            "Your task is to identify which candidate terms are actual proper nouns, "
            "skills, items, locations, cultivation terms, or other important terms "
            "that should be consistently translated.\n\n"
            "CRITERIA FOR VALID TERMS:\n"
            "- Names of people (e.g., Fang Yuan, Li Qing)\n"
            "- Names of places (e.g., Qing Mao Mountain, Blue Wave Sect)\n"
            "- Cultivation terms (e.g., Golden Core, Nascent Soul)\n"
            "- Skills/techniques (e.g., Sword Drawing Art, Heaven Swallowing)\n"
            "- Items (e.g., Spirit Stone, Gu worm names)\n"
            "- Clans/Sects (e.g., Gu Yue Clan, Bai Clan)\n"
            "- Titles (e.g., Gu Master, Demon Lord)\n\n"
            "INVALID TERMS (should be rejected):\n"
            "- Common English words incorrectly flagged\n"
            "- Generic descriptions (e.g., 'The Mountain', 'His Technique')\n"
            "- Sentence fragments\n"
            "- Partial terms\n"
        )

        user_content = (
            "Analyze these candidate terms and determine which are valid:\n\n"
            f"{terms_list}\n\n"
            "Return a JSON array of valid term strings (only the term names):\n"
            "Example: [\"Fang Yuan\", \"Golden Core\", \"Qing Mao Mountain\"]\n\n"
            "Return ONLY the JSON array, no explanation."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_find_missed_prompt(
        self, text: str, found_terms: list[str]
    ) -> list[dict[str, str]]:
        """Build the prompt for finding missed terms."""
        text_sample = text[:3000] if len(text) > 3000 else text
        found_list = ", ".join(found_terms[:50]) if found_terms else "None yet"

        system_content = (
            "You are an expert in Wuxia/Xianxia fantasy literature terminology. "
            "Your task is to identify important terms that should be consistently "
            "translated but may have been missed by our automated detection.\n\n"
            "Look for:\n"
            "- Character names (Chinese-style names)\n"
            "- Location names (mountains, valleys, cities, sects)\n"
            "- Cultivation terms and realms\n"
            "- Skills, techniques, and arts\n"
            "- Important items (Gu, pills, artifacts)\n"
            "- Titles and honorifics\n"
        )

        user_content = (
            f"TEXT SAMPLE:\n{text_sample}\n\n"
            f"ALREADY FOUND TERMS: {found_list}\n\n"
            "Find any important terms that were MISSED. "
            "Return a JSON array of missed term strings:\n"
            "Example: [\"Chen Wei\", \"Immortal Crane Valley\"]\n\n"
            "Return ONLY new terms not in the 'already found' list.\n"
            "Return ONLY the JSON array, no explanation.\n"
            "If no terms were missed, return: []"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_translation_prompt(
        self, terms_with_context: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Build the prompt for proposing translations."""
        terms_list = "\n".join(
            f"{i+1}. Term: \"{t['term']}\"\n   Context: \"{t['context']}\""
            for i, t in enumerate(terms_with_context)
        )

        categories_str = ", ".join(self.CATEGORIES)

        system_content = (
            "You are an expert translator specializing in Wuxia/Xianxia fantasy novels "
            "from English to Turkish. Your task is to propose Turkish translations "
            "for the given terms.\n\n"
            "TRANSLATION GUIDELINES:\n"
            "1. PRESERVE character names (Chinese names stay as-is: Fang Yuan → Fang Yuan)\n"
            "2. TRANSLATE generic parts of location names (Mountain → Dağı, Valley → Vadisi)\n"
            "3. Use appropriate Turkish fantasy terms (Clan → Klan, Sect → Tarikat)\n"
            "4. For cultivation terms, use meaningful Turkish equivalents\n"
            "5. For Gu/insects, keep 'Gu' but translate the descriptor\n"
            f"6. Categories: {categories_str}\n"
        )

        user_content = (
            "Provide translations for these terms:\n\n"
            f"{terms_list}\n\n"
            "Return a JSON array with objects containing:\n"
            "- term: original term\n"
            "- translation: proposed Turkish translation\n"
            "- category: one of Person, Location, Skill, Clan, Item, Other\n"
            "- reasoning: brief explanation (1 sentence)\n\n"
            "Example:\n"
            "[{\"term\": \"Fang Yuan\", \"translation\": \"Fang Yuan\", "
            "\"category\": \"Person\", \"reasoning\": \"Character name, kept unchanged\"}]\n\n"
            "Return ONLY the JSON array."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_clean_response(
        self, response: str, batch: list[CandidateTerm]
    ) -> set[str]:
        """Parse the LLM response for clean pass."""
        valid_terms: set[str] = set()

        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                terms = json.loads(json_match.group())
                if isinstance(terms, list):
                    valid_terms = {t.lower() for t in terms if isinstance(t, str)}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse clean response", error=str(e))
            valid_terms = {c.term.lower() for c in batch}

        return valid_terms

    def _parse_find_missed_response(
        self, response: str, text: str
    ) -> list[CandidateTerm]:
        """Parse the LLM response for find missed pass."""
        missed: list[CandidateTerm] = []

        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                terms = json.loads(json_match.group())
                if isinstance(terms, list):
                    for term in terms:
                        if isinstance(term, str) and term.strip():
                            context = self._extract_context(term, text)
                            missed.append(CandidateTerm(
                                term=term.strip(),
                                confidence=0.8,
                                frequency=text.lower().count(term.lower()),
                                context_pattern="llm_discovered",
                                context_snippet=context,
                            ))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse find missed response", error=str(e))

        return missed

    def _parse_translation_response(
        self, response: str, batch: list[CandidateTerm]
    ) -> list[RefinedTerm]:
        """Parse the LLM response for translation proposals."""
        results: list[RefinedTerm] = []
        batch_map = {c.term.lower(): c for c in batch}

        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                proposals = json.loads(json_match.group())
                if isinstance(proposals, list):
                    for prop in proposals:
                        if isinstance(prop, dict):
                            term = prop.get("term", "")
                            candidate = batch_map.get(term.lower())
                            
                            results.append(RefinedTerm(
                                source_term=term,
                                proposed_translation=prop.get("translation", term),
                                proposed_category=prop.get("category"),
                                reasoning=prop.get("reasoning"),
                                context_snippet=candidate.context_snippet if candidate else None,
                                is_valid=True,
                            ))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse translation response", error=str(e))

        return results

    def _extract_context(self, term: str, text: str) -> str | None:
        """Extract a context sentence containing the term."""
        term_lower = term.lower()
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if term_lower in sentence.lower():
                if len(sentence) > 200:
                    idx = sentence.lower().find(term_lower)
                    start = max(0, idx - 80)
                    end = min(len(sentence), idx + len(term) + 80)
                    snippet = sentence[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(sentence):
                        snippet = snippet + "..."
                    return snippet
                return sentence
        
        return None

    def _get_existing_terms(self) -> set[str]:
        """Get existing glossary terms for this project."""
        try:
            terms = GlossaryTerm.select().where(
                GlossaryTerm.project == self.project
            )
            return {t.source_term.lower() for t in terms}
        except Exception:
            return set()

    async def save_proposals(self, refined_terms: list[RefinedTerm]) -> int:
        """Save refined terms as proposals in the database.
        
        Args:
            refined_terms: List of refined terms to save.
        
        Returns:
            Number of proposals saved.
        """
        saved = 0
        for term in refined_terms:
            if not term.is_valid:
                continue

            try:
                ProposedTerm.get_or_create(
                    project=self.project,
                    source_term=term.source_term,
                    defaults={
                        "proposed_translation": term.proposed_translation,
                        "proposed_category": term.proposed_category,
                        "llm_reasoning": term.reasoning,
                        "context_snippet": term.context_snippet,
                        "status": "pending",
                    },
                )
                saved += 1
            except Exception as e:
                logger.warning(
                    "Failed to save proposal",
                    term=term.source_term,
                    error=str(e),
                )

        logger.info("Saved proposals", count=saved)
        return saved
