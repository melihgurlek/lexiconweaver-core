"""Logging configuration for LexiconWeaver."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog.stdlib import LoggerFactory

from lexiconweaver.config import Config


def configure_logging(config: Config, quiet: bool = False) -> None:
    """Configure structured logging based on configuration.
    
    Args:
        config: Configuration object
        quiet: If True, suppress console output (useful for TUI mode)
    """
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Configure standard library logging
    # In quiet mode (TUI), don't output to stdout to avoid conflicts with TUI
    if not quiet:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
            force=True,  # Allow reconfiguration
        )
    else:
        # In quiet mode, configure logging without console output
        # Remove any existing handlers first
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure root logger without stream handlers
        root_logger.setLevel(log_level)

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # In quiet mode, use JSON renderer (which goes through logging handlers)
    # or a null renderer, so console output is suppressed
    if quiet:
        # Use JSON renderer in quiet mode - it will respect logging handlers
        # Since we removed stdout handlers, this won't output to console
        processors.append(structlog.processors.JSONRenderer())
    elif config.logging.json_logging:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure file logging - use default path if not specified
    log_file_path: Path | None = None
    if config.logging.log_file:
        log_file_path = Path(config.logging.log_file)
    else:
        # Use default log file path if not configured
        log_file_path = Config.get_default_log_file_path()
    
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)

        if config.logging.json_logging:
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
