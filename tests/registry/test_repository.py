import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from apar.registry.models import EvidenceRecord, ThreatCard
from apar.registry.repository import ThreatRepository
from tests.factories import make_threat_card


def test_repository_round_trip_preserves_the_validated_card(tmp_path: Path) -> None:
    """Catches lossy persistence of evidence or typed scenario configuration."""
    threat_card = make_threat_card(title="AI-personalized APP scam — reviewed")
    repository = ThreatRepository(tmp_path / "state.db")

    repository.upsert(threat_card)

    assert repository.get(threat_card.threat_id) == threat_card


def test_repository_initialization_is_idempotent_and_records_migration_one(
    tmp_path: Path,
) -> None:
    """Catches lifespan restarts reapplying or omitting the registry schema."""
    database_path = tmp_path / "nested" / "state.db"
    ThreatRepository(database_path)
    ThreatRepository(database_path)

    with sqlite3.connect(database_path) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()

    assert versions == [(1,)]


def test_upsert_replaces_card_and_its_indexed_metadata(tmp_path: Path) -> None:
    """Catches metadata indexes drifting from the canonical current card payload."""
    database_path = tmp_path / "state.db"
    repository = ThreatRepository(database_path)
    repository.upsert(make_threat_card())
    replacement = make_threat_card(
        family="social_engineering",
        confidence=0.75,
        implementation_status="mapped",
    )

    repository.upsert(replacement)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT family, confidence, implementation_status
            FROM threat_cards
            WHERE threat_id = ?
            """,
            (replacement.threat_id,),
        ).fetchone()
    assert row == ("social_engineering", 0.75, "mapped")
    assert repository.list() == [replacement]


def test_repository_stores_canonical_utf8_json(tmp_path: Path) -> None:
    """Catches nondeterministic JSON or escaped Unicode crossing the SQLite boundary."""
    database_path = tmp_path / "state.db"
    repository = ThreatRepository(database_path)
    card = make_threat_card(title="Fraude personnalisée — révisée")

    repository.upsert(card)

    with sqlite3.connect(database_path) as connection:
        payload = connection.execute(
            "SELECT card_json FROM threat_cards WHERE threat_id = ?", (card.threat_id,)
        ).fetchone()[0]
    assert "Fraude personnalisée — révisée" in payload
    assert ": " not in payload
    assert ", " not in payload
    assert payload == json.dumps(
        card.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_repository_returns_none_for_unknown_threat(tmp_path: Path) -> None:
    """Catches unknown IDs being confused with malformed or placeholder cards."""
    repository = ThreatRepository(tmp_path / "state.db")

    assert repository.get("not-registered") is None


@pytest.mark.parametrize("model_type", [ThreatCard, EvidenceRecord])
def test_registry_records_reject_unknown_major_schema_versions(
    model_type: type[ThreatCard] | type[EvidenceRecord],
) -> None:
    """Catches incompatible evidence or threat records entering the registry boundary."""
    card = make_threat_card()
    candidate = card if model_type is ThreatCard else card.evidence[0]

    with pytest.raises(ValidationError, match="unsupported schema major"):
        model_type.model_validate(candidate.model_copy(update={"schema_version": "9.0.0"}))
