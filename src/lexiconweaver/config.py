"""Configuration management for LexiconWeaver."""

import os
from pathlib import Path
from typing import Literal, Optional

import toml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lexiconweaver.exceptions import ConfigurationError


class OllamaConfig(BaseSettings):
    """Configuration for Ollama LLM connection."""

    model_config = SettingsConfigDict(extra="allow")

    url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="llama3.1:latest", description="Model name to use")
    timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries")


class DeepSeekConfig(BaseSettings):
    """Configuration for DeepSeek API connection."""

    model_config = SettingsConfigDict(extra="allow")

    api_key: str = Field(
        default="",
        description="DeepSeek API key (set via LEXICONWEAVER_DEEPSEEK__API_KEY)",
    )
    model: str = Field(default="deepseek-chat", description="DeepSeek model name (V3)")
    base_url: str = Field(
        default="https://api.deepseek.com", description="DeepSeek API base URL"
    )
    timeout: int = Field(default=120, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries")


class ProviderConfig(BaseSettings):
    """Configuration for LLM provider selection."""

    model_config = SettingsConfigDict(extra="allow")

    primary: Literal["ollama", "deepseek"] = Field(
        default="ollama", description="Primary LLM provider"
    )
    fallback: Literal["ollama", "deepseek", "none"] = Field(
        default="none", description="Fallback provider when primary fails"
    )
    fallback_on_error: bool = Field(
        default=True, description="Use fallback on provider errors"
    )


class DatabaseConfig(BaseSettings):
    """Configuration for database."""

    model_config = SettingsConfigDict(extra="allow")

    path: str = Field(default="", description="Path to SQLite database file")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(extra="allow")

    level: str = Field(default="INFO", description="Log level")
    json_logging: bool = Field(default=False, description="Enable JSON logging")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class CacheConfig(BaseSettings):
    """Configuration for translation cache."""

    model_config = SettingsConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable caching")
    max_size: int = Field(default=10000, ge=0, description="Maximum cache size")


class ScoutConfig(BaseSettings):
    """Configuration for Scout engine."""

    model_config = SettingsConfigDict(extra="allow")

    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    max_ngram_size: int = Field(
        default=4, ge=1, le=10, description="Maximum N-gram size"
    )
    use_pos_filter: bool = Field(
        default=True, description="Enable POS tagging filter"
    )
    use_llm_refinement: bool = Field(
        default=True, description="Enable two-pass LLM refinement for Smart Scout"
    )
    llm_batch_size: int = Field(
        default=50, ge=1, le=100, description="Terms per LLM API call for batching"
    )


class WeaverConfig(BaseSettings):
    """Configuration for Weaver engine."""

    model_config = SettingsConfigDict(extra="allow")

    parallel_paragraphs: int = Field(
        default=0, ge=0, description="Number of parallel paragraphs (0 = sequential)"
    )
    streaming: bool = Field(
        default=True, description="Enable streaming responses"
    )
    max_context_tokens: int = Field(
        default=4000, ge=1000, description="Maximum context window size"
    )
    translation_batch_max_chars: int = Field(
        default=2000, ge=500, description="Maximum characters per translation batch"
    )
    translation_context_sentences: int = Field(
        default=2, ge=0, le=10, description="Number of sentences to keep as context between batches"
    )
    translation_auto_export: bool = Field(
        default=True, description="Automatically export translation when complete"
    )


class BatchConfig(BaseSettings):
    """Configuration for batch translation workflow."""

    model_config = SettingsConfigDict(extra="allow")

    max_parallel_chapters: int = Field(
        default=5, ge=1, le=20, description="Maximum chapters to translate in parallel"
    )
    max_parallel_batches: int = Field(
        default=1, ge=1, le=10, description="Maximum batches per chapter in parallel (future)"
    )
    
    rate_limit_rpm: int = Field(
        default=60, ge=1, description="Estimated API rate limit (requests per minute)"
    )
    rate_limit_backoff_base: int = Field(
        default=2, ge=2, le=10, description="Exponential backoff base (2^attempt seconds)"
    )
    rate_limit_max_wait: int = Field(
        default=60, ge=10, le=300, description="Maximum wait time for rate limit backoff (seconds)"
    )
    
    scout_min_frequency: int = Field(
        default=3, ge=1, description="Minimum frequency for term candidates"
    )
    scout_burst_threshold: int = Field(
        default=5, ge=1, description="Minimum frequency in window for burst detection"
    )
    scout_burst_window_size: int = Field(
        default=3, ge=1, le=10, description="Sliding window size for burst detection"
    )
    scout_llm_refine_top_percent: int = Field(
        default=20, ge=1, le=100, description="Percentage of top candidates to refine with LLM"
    )
    
    extract_chapter_titles: bool = Field(
        default=True, description="Extract chapter titles from content"
    )
    fallback_to_filename: bool = Field(
        default=True, description="Use filename as fallback if title extraction fails"
    )
    
    input_folder: str = Field(
        default="input", description="Input folder for raw chapters"
    )
    output_folder: str = Field(
        default="output", description="Output folder for translated chapters"
    )
    merged_folder: str = Field(
        default="merged", description="Folder for merged output files"
    )
    metadata_folder: str = Field(
        default=".weavecodex", description="Folder for metadata and cache"
    )


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_prefix="LEXICONWEAVER_",
        env_nested_delimiter="__",
        extra="allow",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scout: ScoutConfig = Field(default_factory=ScoutConfig)
    weaver: WeaverConfig = Field(default_factory=WeaverConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file and environment variables."""
        if config_path is None:
            config_path = cls._get_default_config_path()

        config_data: dict = {}

        # Load from TOML file if it exists
        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = toml.load(f)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load config file {config_path}: {e}"
                )

        # Create config instance (environment variables will override)
        return cls(**config_data)

    @staticmethod
    def get_config_path() -> Path:
        """Return the path where config is read from (and init writes to)."""
        return Config._get_default_config_path()

    @staticmethod
    def _get_default_config_path() -> Path:
        """Get the default configuration file path."""
        if os.name == "nt":  # Windows
            config_dir = Path(os.getenv("APPDATA", "")) / "lexiconweaver"
        else:  # Unix-like
            config_dir = Path.home() / ".config" / "lexiconweaver"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.toml"

    @staticmethod
    def get_default_database_path() -> Path:
        """Get the default database path."""
        if os.name == "nt":  # Windows
            data_dir = Path(os.getenv("LOCALAPPDATA", "")) / "lexiconweaver"
        else:  # Unix-like
            data_dir = Path.home() / ".local" / "share" / "lexiconweaver"

        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "lexiconweaver.db"

    @staticmethod
    def get_default_log_file_path() -> Path:
        """Get the default log file path."""
        if os.name == "nt":  # Windows
            data_dir = Path(os.getenv("LOCALAPPDATA", "")) / "lexiconweaver"
        else:  # Unix-like
            data_dir = Path.home() / ".local" / "share" / "lexiconweaver"

        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "lexiconweaver.log"

    def get_database_path(self) -> Path:
        """Get the resolved database path."""
        db_path_str = self.database.path
        
        # Detect and reject system PATH environment variable
        # System PATH on Unix contains ':' separators, on Windows ';' separators
        # A valid database path should not look like a PATH variable
        if db_path_str:
            path_separator = ";" if os.name == "nt" else ":"
            if path_separator in db_path_str:
                # This is likely the system PATH, not a database path - use default instead
                db_path_str = ""
        
        if not db_path_str:
            return self.get_default_database_path()

        db_path = Path(db_path_str)
        if db_path.is_absolute():
            return db_path
        # If relative, resolve relative to current working directory
        return Path.cwd() / db_path

