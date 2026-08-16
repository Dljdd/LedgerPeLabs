"""SQLite-backed metadata repository for validated threat cards."""

import json
from pathlib import Path

from apar.registry.models import ThreatCard
from apar.storage.database import DEFAULT_DATABASE_PATH, connect_database, initialize_database


def _canonical_json(card: ThreatCard) -> str:
    return json.dumps(
        card.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ThreatRepository:
    """Maintain the current validated threat-card index in SQLite."""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = database_path
        initialize_database(database_path)

    def upsert(self, card: ThreatCard) -> None:
        """Insert or replace a card and its queryable metadata atomically."""
        validated = ThreatCard.model_validate(card)
        with connect_database(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO threat_cards(
                    threat_id, family, confidence, implementation_status, card_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(threat_id) DO UPDATE SET
                    family = excluded.family,
                    confidence = excluded.confidence,
                    implementation_status = excluded.implementation_status,
                    card_json = excluded.card_json
                """,
                (
                    validated.threat_id,
                    validated.family,
                    validated.confidence,
                    validated.implementation_status,
                    _canonical_json(validated),
                ),
            )

    def get(self, threat_id: str) -> ThreatCard | None:
        """Return the current card for a stable evidence slug, if registered."""
        with connect_database(self.database_path) as connection:
            row = connection.execute(
                "SELECT card_json FROM threat_cards WHERE threat_id = ?", (threat_id,)
            ).fetchone()
        if row is None:
            return None
        return ThreatCard.model_validate_json(row["card_json"])

    def list(self) -> list[ThreatCard]:
        """Return all current cards in stable threat-ID order."""
        with connect_database(self.database_path) as connection:
            rows = connection.execute(
                "SELECT card_json FROM threat_cards ORDER BY threat_id"
            ).fetchall()
        return [ThreatCard.model_validate_json(row["card_json"]) for row in rows]
