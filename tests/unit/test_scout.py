"""Unit tests for Scout engine."""

import pytest

from lexiconweaver.config import Config
from lexiconweaver.database.models import Project
from lexiconweaver.engines.scout import Scout
from tests.conftest import initialized_db, sample_text


def test_scout_initialization(config: Config, initialized_db: Project) -> None:
    """Test Scout initialization."""
    scout = Scout(config, initialized_db)
    assert scout.config == config
    assert scout.project == initialized_db
    assert scout.min_confidence == config.scout.min_confidence


def test_scout_extract_candidates(config: Config, initialized_db: Project, sample_text: str) -> None:
    """Test candidate extraction."""
    scout = Scout(config, initialized_db)
    candidates = scout._extract_candidates(sample_text)

    assert len(candidates) > 0
    assert any("golden core" in c.lower() for c in candidates)
    assert any("void step" in c.lower() for c in candidates)


def test_scout_filter_candidates(config: Config, initialized_db: Project) -> None:
    """Test candidate filtering."""
    scout = Scout(config, initialized_db)
    candidates = ["Golden Core", "the", "a", "Void Step", "x"]

    ignored = {"the"}
    filtered = scout._filter_candidates(candidates, ignored)

    assert "the" not in filtered  # Ignored
    assert "a" not in filtered  # Stopword
    assert "x" not in filtered  # Too short
    assert "Golden Core" in filtered or any("golden" in c.lower() for c in filtered)


def test_scout_score_candidates(config: Config, initialized_db: Project, sample_text: str) -> None:
    """Test candidate scoring."""
    scout = Scout(config, initialized_db)
    candidates = ["Golden Core", "Void Step"]

    scored = scout._score_candidates(candidates, sample_text)

    assert len(scored) == len(candidates)
    for candidate in scored:
        assert 0.0 <= candidate.confidence <= 1.0
        assert candidate.frequency >= 0


def test_scout_process(config: Config, initialized_db: Project, sample_text: str) -> None:
    """Test full Scout processing."""
    scout = Scout(config, initialized_db)
    scout.min_confidence = 0.1  # Lower threshold for testing

    candidates = scout.process(sample_text)

    assert isinstance(candidates, list)
    # Should find some terms from sample text
    assert len(candidates) > 0
