"""Database manager with singleton pattern for connection management."""

import threading
from pathlib import Path
from typing import Optional

from peewee import SqliteDatabase

from lexiconweaver.config import Config
from lexiconweaver.database.models import create_tables, db_proxy
from lexiconweaver.exceptions import DatabaseError


class DatabaseManager:
    """Singleton database manager for SQLite connections."""

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DatabaseManager":
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize database manager."""
        if hasattr(self, "_initialized"):
            return

        self._db: Optional[SqliteDatabase] = None
        self._config: Optional[Config] = None
        self._initialized = False

    def initialize(self, config: Config, db_path: Optional[Path] = None) -> None:
        """Initialize the database connection."""
        if self._initialized and self._db is not None:
            return

        self._config = config

        if db_path is None:
            db_path = config.get_database_path()

        try:
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create database connection with connection pooling
            self._db = SqliteDatabase(
                str(db_path),
                pragmas={
                    "foreign_keys": 1,  # Enable foreign key constraints
                    "journal_mode": "wal",  # Write-Ahead Logging for better concurrency
                    "cache_size": -64000,  # 64MB cache
                    "synchronous": 1,  # Safe mode
                },
            )

            # Test connection
            self._db.connect(reuse_if_open=True)

            # Initialize database proxy for models
            db_proxy.initialize(self._db)

            # Create tables if they don't exist
            create_tables()

            self._initialized = True

        except Exception as e:
            raise DatabaseError(f"Failed to initialize database at {db_path}: {e}") from e

    def get_connection(self) -> SqliteDatabase:
        """Get the database connection."""
        if not self._initialized or self._db is None:
            if self._config is None:
                raise DatabaseError("Database not initialized. Call initialize() first.")
            self.initialize(self._config)

        return self._db

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None and not self._db.is_closed():
            try:
                self._db.close()
            except Exception:
                pass  # Ignore errors on close
        self._initialized = False

    def execute_in_transaction(self, func):
        """Execute a function within a transaction."""
        db = self.get_connection()
        with db.atomic():
            return func()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(config: Config, db_path: Optional[Path] = None) -> None:
    """Initialize the global database connection."""
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()
    _db_manager.initialize(config, db_path)


def close_database() -> None:
    """Close the global database connection."""
    global _db_manager

    if _db_manager is not None:
        _db_manager.close()
    _db_manager = None
