"""SQLite initialization for APAR's small mutable metadata index."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_DATABASE_PATH = Path(".apar/state.db")


def connect_database(path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a configured SQLite connection, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Apply the version-1 registry schema idempotently."""
    with connect_database(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY
            )
            """
        )
        applied_versions = {
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        unsupported_versions = applied_versions - {SCHEMA_VERSION}
        if unsupported_versions:
            versions = ", ".join(str(version) for version in sorted(unsupported_versions))
            raise RuntimeError(f"unsupported database schema version: {versions}")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS threat_cards (
                threat_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                confidence REAL NOT NULL,
                implementation_status TEXT NOT NULL,
                card_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_threat_cards_threat_id ON threat_cards(threat_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_threat_cards_family ON threat_cards(family)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_threat_cards_confidence ON threat_cards(confidence)"
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_threat_cards_implementation_status
            ON threat_cards(implementation_status)
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,)
        )
