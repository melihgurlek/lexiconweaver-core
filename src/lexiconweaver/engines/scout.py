"""Scout engine for discovering potential terms in text."""

import re
from collections import Counter
from typing import NamedTuple

from lexiconweaver.config import Config
from lexiconweaver.database.models import IgnoredTerm, Project
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.exceptions import ScoutError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.utils.text_processor import extract_ngrams

logger = get_logger(__name__)


class CandidateTerm(NamedTuple):
    """Represents a candidate term with confidence score."""

    term: str
    confidence: float
    frequency: int
    context_pattern: str | None


class Scout(BaseEngine):
    """Discovery engine that identifies potential terms using heuristics."""

    # Common English stopwords (subset, expandable)
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "could", "may", "might", "must", "can", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "what",
        "which", "who", "whom", "where", "when", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "now", "then", "here", "there", "when", "where", "why", "how",
    }

    # Definition patterns that indicate a term
    DEFINITION_PATTERNS = [
        (r"called\s+([A-Z][\w\s]+)", "called"),
        (r"known\s+as\s+([A-Z][\w\s]+)", "known as"),
        (r"rank\s+(\d+|\w+)", "rank"),
        (r'["\'`]([A-Z][\w\s]+)["\'`]', "quoted"),
        (r"the\s+([A-Z][\w]+(?:\s+[A-Z][\w]+)*)", "the capitalized"),
    ]

    def __init__(self, config: Config, project: Project | None = None) -> None:
        """Initialize the Scout engine."""
        self.config = config
        self.project = project
        self.min_confidence = config.scout.min_confidence
        self.max_ngram_size = config.scout.max_ngram_size

    def process(self, text: str) -> list[CandidateTerm]:
        """Process text and return candidate terms with confidence scores."""
        try:
            # Get ignored terms for this project
            ignored_terms = self._get_ignored_terms()

            # Extract all potential terms
            candidates = self._extract_candidates(text)

            # Filter out ignored terms and stopwords
            filtered = self._filter_candidates(candidates, ignored_terms)

            # Calculate confidence scores
            scored = self._score_candidates(filtered, text)

            # Filter by minimum confidence
            final = [
                c for c in scored if c.confidence >= self.min_confidence
            ]

            # Sort by confidence (descending)
            final.sort(key=lambda x: x.confidence, reverse=True)

            logger.info(
                "Scout processed text",
                total_candidates=len(candidates),
                filtered=len(filtered),
                final=len(final),
            )

            return final

        except Exception as e:
            raise ScoutError(f"Failed to process text: {e}") from e

    def _extract_candidates(self, text: str) -> list[str]:
        """Extract candidate terms from text using multiple strategies."""
        candidates: set[str] = set()

        # Strategy 1: Extract from definition patterns
        for pattern, _ in self.DEFINITION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                if term:
                    candidates.add(term)

        # Strategy 2: Extract capitalized phrases (Proper Nouns)
        capitalized_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.finditer(capitalized_pattern, text)
        for match in matches:
            term = match.group(1).strip()
            if len(term.split()) <= self.max_ngram_size:
                candidates.add(term)

        # Strategy 3: Extract N-grams (up to max_ngram_size)
        for n in range(1, self.max_ngram_size + 1):
            for ngram, _ in extract_ngrams(text, n, min_length=3):
                # Only add if it contains at least one capitalized word
                words = ngram.split()
                if any(w[0].isupper() for w in words if w):
                    candidates.add(ngram.title())

        return list(candidates)

    def _filter_candidates(
        self, candidates: list[str], ignored_terms: set[str]
    ) -> list[str]:
        """Filter out stopwords and ignored terms."""
        filtered = []

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Skip if in ignored terms
            if candidate_lower in ignored_terms:
                continue

            # Skip if all words are stopwords
            words = candidate_lower.split()
            if all(word in self.STOPWORDS for word in words):
                continue

            # Skip single-letter or very short terms
            if len(candidate.strip()) < 2:
                continue

            filtered.append(candidate)

        return filtered

    def _score_candidates(
        self, candidates: list[str], text: str
    ) -> list[CandidateTerm]:
        """Calculate confidence scores for candidates."""
        scored: list[CandidateTerm] = []

        # Count frequencies
        text_lower = text.lower()
        frequencies = Counter()
        capitalization_counts = Counter()
        pattern_matches: dict[str, str | None] = {}

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Count frequency
            count = text_lower.count(candidate_lower)
            frequencies[candidate] = count

            # Count capitalization occurrences
            capitalized_pattern = re.escape(candidate)
            caps_matches = len(re.findall(rf"\b{capitalized_pattern}\b", text))
            capitalization_counts[candidate] = caps_matches

            # Check definition patterns
            matched_pattern = None
            for pattern, pattern_name in self.DEFINITION_PATTERNS:
                if re.search(pattern.replace(r"([A-Z][\w\s]+)", re.escape(candidate)), text, re.IGNORECASE):
                    matched_pattern = pattern_name
                    break
            pattern_matches[candidate] = matched_pattern

        # Calculate scores
        max_freq = max(frequencies.values()) if frequencies else 1
        max_caps = max(capitalization_counts.values()) if capitalization_counts else 1

        for candidate in candidates:
            freq = frequencies[candidate]
            caps = capitalization_counts[candidate]
            has_pattern = pattern_matches[candidate] is not None

            # Frequency weight (30%)
            freq_score = min(freq / max_freq, 1.0) * 0.3

            # Capitalization weight (30%)
            caps_score = min(caps / max_caps, 1.0) * 0.3

            # Structural context weight (40%)
            pattern_score = 1.0 * 0.4 if has_pattern else 0.0

            confidence = freq_score + caps_score + pattern_score

            scored.append(
                CandidateTerm(
                    term=candidate,
                    confidence=confidence,
                    frequency=freq,
                    context_pattern=pattern_matches.get(candidate),
                )
            )

        return scored

    def _get_ignored_terms(self) -> set[str]:
        """Get ignored terms for the current project."""
        # #region agent log
        import json; log_data = {"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"scout.py:212","message":"_get_ignored_terms entry","data":{"project_id":str(self.project.id) if self.project else None},"timestamp":int(__import__("time").time()*1000)}; open("/home/melihgurlek/Code/WeaveCodex/.cursor/debug.log","a").write(json.dumps(log_data)+"\n")
        # #endregion
        if self.project is None:
            # #region agent log
            log_data = {"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"scout.py:215","message":"_get_ignored_terms no project","data":{},"timestamp":int(__import__("time").time()*1000)}; open("/home/melihgurlek/Code/WeaveCodex/.cursor/debug.log","a").write(json.dumps(log_data)+"\n")
            # #endregion
            return set()

        try:
            ignored = IgnoredTerm.select().where(
                IgnoredTerm.project == self.project
            )
            result = {term.term.lower() for term in ignored}
            # #region agent log
            log_data = {"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"scout.py:221","message":"_get_ignored_terms success","data":{"count":str(len(result))},"timestamp":int(__import__("time").time()*1000)}; open("/home/melihgurlek/Code/WeaveCodex/.cursor/debug.log","a").write(json.dumps(log_data)+"\n")
            # #endregion
            return result
        except Exception as e:
            # #region agent log
            log_data = {"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"scout.py:223","message":"_get_ignored_terms error","data":{"error":str(e),"error_type":str(type(e).__name__)},"timestamp":int(__import__("time").time()*1000)}; open("/home/melihgurlek/Code/WeaveCodex/.cursor/debug.log","a").write(json.dumps(log_data)+"\n")
            # #endregion
            logger.warning("Failed to load ignored terms", error=str(e))
            return set()
