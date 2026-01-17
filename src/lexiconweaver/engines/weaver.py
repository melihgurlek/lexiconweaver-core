"""Weaver engine for LLM-based translation with glossary enforcement."""

import asyncio
import json
from typing import AsyncIterator

import httpx
from flashtext import KeywordProcessor

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project, TranslationCache
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.exceptions import TranslationError, WeaverError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.utils.text_processor import extract_paragraphs, generate_hash

logger = get_logger(__name__)


class Weaver(BaseEngine):
    """Generation engine that interfaces with Ollama for translation."""

    def __init__(self, config: Config, project: Project) -> None:
        """Initialize the Weaver engine."""
        self.config = config
        self.project = project
        self.ollama_url = config.ollama.url
        self.model = config.ollama.model
        self.timeout = config.ollama.timeout
        self.max_retries = config.ollama.max_retries
        self.cache_enabled = config.cache.enabled

    async def translate_paragraph(
        self, paragraph: str, use_cache: bool = True
    ) -> str:
        """Translate a single paragraph."""
        # Check cache first
        if use_cache and self.cache_enabled:
            cached = self._get_from_cache(paragraph)
            if cached is not None:
                logger.debug("Cache hit for paragraph", hash=generate_hash(paragraph)[:8])
                return cached

        # Build mini-glossary for this paragraph
        mini_glossary = self._build_mini_glossary(paragraph)

        # Build prompt
        prompt = self._build_prompt(paragraph, mini_glossary)

        # Call Ollama
        translation = await self._call_ollama(prompt)

        # Verify translation
        self._verify_translation(paragraph, translation, mini_glossary)

        # Store in cache
        if use_cache and self.cache_enabled:
            self._store_in_cache(paragraph, translation)

        return translation

    async def translate_streaming(
        self, paragraph: str
    ) -> AsyncIterator[str]:
        """Translate a paragraph with streaming response."""
        # Build mini-glossary
        mini_glossary = self._build_mini_glossary(paragraph)

        # Build prompt
        prompt = self._build_prompt(paragraph, mini_glossary)

        # Stream from Ollama
        full_translation = ""
        async for chunk in self._call_ollama_streaming(prompt):
            full_translation += chunk
            yield chunk

        # Verify after streaming completes
        self._verify_translation(paragraph, full_translation, mini_glossary)

        # Store in cache
        if self.cache_enabled:
            self._store_in_cache(paragraph, full_translation)

    async def translate_batch_streaming(
        self, batch: str, context: str = ""
    ) -> AsyncIterator[str]:
        """Translate a batch with streaming response, optionally including context.
        
        Args:
            batch: The text batch to translate
            context: Optional context from previous batch (e.g., last 2 sentences)
            
        Yields:
            Translation chunks as they arrive from the LLM
        """
        text_to_translate = f"{context}\n\n{batch}" if context else batch
        
        mini_glossary = self._build_mini_glossary(text_to_translate)
        
        prompt = self._build_prompt_with_context(text_to_translate, mini_glossary, context)
        
        full_translation = ""
        async for chunk in self._call_ollama_streaming(prompt):
            full_translation += chunk
            yield chunk
        
        # Extract only the new translation (remove context translation if present)
        # This is a simple heuristic: if context was provided, the translation
        # might include it. We'll return the full translation and let the caller
        # handle context removal if needed.
        
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

    def _build_mini_glossary(self, text: str) -> dict[str, str]:
        """Build a mini-glossary containing only terms that appear in the text."""
        # Get all glossary terms for this project
        glossary_terms = GlossaryTerm.select().where(
            GlossaryTerm.project == self.project
        )

        # Use FlashText for efficient matching
        keyword_processor = KeywordProcessor(case_sensitive=False)
        term_map: dict[str, str] = {}

        for term in glossary_terms:
            if term.is_regex:
                # For regex terms, we'd need pattern matching
                # For now, skip regex terms in mini-glossary (they're handled in verification)
                continue

            source_term = term.source_term
            target_term = term.target_term

            # Add to keyword processor
            keyword_processor.add_keyword(source_term)
            term_map[source_term.lower()] = target_term

        # Find all terms in the text
        found_terms = keyword_processor.extract_keywords(text.lower())
        mini_glossary = {
            term: term_map[term.lower()] for term in found_terms if term.lower() in term_map
        }

        return mini_glossary

    def _build_prompt(self, paragraph: str, mini_glossary: dict[str, str]) -> str:
        """Build the translation prompt for Ollama."""
        glossary_block = ""
        if mini_glossary:
            glossary_lines = [
                f"- {source}: {target}" for source, target in mini_glossary.items()
            ]
            glossary_block = "Glossary (use these exact translations):\n" + "\n".join(glossary_lines) + "\n\n"

        prompt = f"""You are a professional translator specializing in fantasy and web novels.

Your task is to translate the following text into Turkish while maintaining consistency with specialized terminology.

{glossary_block}Translate the following paragraph. You must use the glossary terms exactly as provided. Maintain the original meaning, tone, and style.

Paragraph to translate:
{paragraph}

Translation:"""

        return prompt

    def _build_prompt_with_context(
        self, text: str, mini_glossary: dict[str, str], context: str = ""
    ) -> str:
        """Build the translation prompt with context awareness."""
        glossary_block = ""
        if mini_glossary:
            glossary_lines = [
                f"- {source}: {target}" for source, target in mini_glossary.items()
            ]
            glossary_block = "Glossary (use these exact translations):\n" + "\n".join(glossary_lines) + "\n\n"

        context_note = ""
        if context:
            context_note = f"""Note: The text below includes context from the previous section for coherence. Translate the entire text, maintaining continuity with the previous translation.

Context (already translated, for reference only):
{context}

"""

        prompt = f"""You are a professional translator specializing in fantasy and web novels.

Your task is to translate the following text into Turkish while maintaining consistency with specialized terminology.

{glossary_block}{context_note}Translate the following text. You must use the glossary terms exactly as provided. Maintain the original meaning, tone, and style. Ensure the translation flows naturally with the context provided above.

Text to translate:
{text}

Translation:"""

        return prompt

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for translation with retry logic."""
        url = f"{self.ollama_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

                    if response.status_code != 200:
                        raise WeaverError(
                            f"Ollama API returned status {response.status_code}: {response.text}"
                        )

                    result = response.json()
                    return result.get("response", "").strip()

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        "Ollama timeout, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise TranslationError(
                        f"Ollama request timed out after {self.max_retries + 1} attempts"
                    ) from e

            except httpx.RequestError as e:
                raise TranslationError(
                    f"Failed to connect to Ollama at {self.ollama_url}: {e}"
                ) from e

            except Exception as e:
                raise TranslationError(f"Unexpected error calling Ollama: {e}") from e

        raise TranslationError("Failed to get translation after retries")

    async def _call_ollama_streaming(
        self, prompt: str
    ) -> AsyncIterator[str]:
        """Call Ollama API with streaming response."""
        url = f"{self.ollama_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        raise WeaverError(
                            f"Ollama API returned status {response.status_code}: {text.decode()}"
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            raise TranslationError(f"Ollama streaming request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TranslationError(
                f"Failed to connect to Ollama at {self.ollama_url}: {e}"
            ) from e
        except Exception as e:
            raise TranslationError(f"Unexpected error streaming from Ollama: {e}") from e

    def _verify_translation(
        self, source: str, translation: str, mini_glossary: dict[str, str]
    ) -> None:
        """Verify that all glossary terms in source appear in translation."""
        # This is a basic check - in a real implementation, you might want
        # more sophisticated verification (fuzzy matching, etc.)
        missing_terms: list[str] = []

        source_lower = source.lower()
        translation_lower = translation.lower()

        for source_term, target_term in mini_glossary.items():
            source_term_lower = source_term.lower()
            target_term_lower = target_term.lower()

            # Check if source term appears in source text
            if source_term_lower not in source_lower:
                continue

            # Check if target term appears in translation
            if target_term_lower not in translation_lower:
                missing_terms.append(source_term)

        if missing_terms:
            logger.warning(
                "Translation verification failed",
                missing_terms=missing_terms,
                source_hash=generate_hash(source)[:8],
            )
            # Don't raise error, just log - the UI will handle marking as [UNSTABLE]

    def _get_from_cache(self, paragraph: str) -> str | None:
        """Get translation from cache if available."""
        try:
            cache_hash = generate_hash(paragraph)
            cached = TranslationCache.get_or_none(
                TranslationCache.hash == cache_hash,
                TranslationCache.project == self.project,
            )
            return cached.translation if cached else None
        except Exception as e:
            logger.warning("Failed to get from cache", error=str(e))
            return None

    def _store_in_cache(self, paragraph: str, translation: str) -> None:
        """Store translation in cache."""
        try:
            cache_hash = generate_hash(paragraph)
            TranslationCache.create(
                hash=cache_hash,
                project=self.project,
                translation=translation,
            )
        except Exception as e:
            logger.warning("Failed to store in cache", error=str(e))
            # Don't raise - caching is not critical

    def process(self, text: str) -> str:
        """Process text synchronously (wrapper for async method)."""
        return asyncio.run(self.translate_text(text))
