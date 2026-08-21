"""Contracts describing verified corpora and evaluator-only truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from apar.contracts._validation import ExternalContract
from apar.defense.contracts import ObservedEvent

Family = Literal[
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
]

_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


class EvaluationTruthRow(ExternalContract):
    """Evaluator-only labels and lifecycle outcomes for one payment opening."""

    event_id: str
    payment_id: str
    campaign_id: str
    family: Family
    viewpoint: Literal["development", "hidden"]
    is_fraud: bool
    label_source: Literal["population_truth", "integrity_truth", "hidden_truth"]
    label_mature_at: datetime
    first_settlement_at: datetime | None
    net_settled_value: Decimal
    lifecycle_event_ids: tuple[str, ...]


class CorpusProfile(ExternalContract):
    """Closed development-corpus selection and label-delay policy."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    profile_id: str = "development-fixture-v1"
    families: tuple[Family, ...] = _FAMILIES
    label_delay_days: int = Field(default=7, ge=0)
    fixture_only: bool = True

    @classmethod
    def fixture(cls) -> CorpusProfile:
        """Return the deliberately small profile used by unit and integration fixtures."""
        return cls()


class CorpusManifest(ExternalContract):
    """Immutable provenance summary for one assembled verified corpus."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    profile_id: str
    run_ids: tuple[str, ...]
    run_lineage_digests: tuple[str, ...]
    observation_count: int = Field(ge=0)
    truth_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class FrozenCorpus:
    """Separately addressed observations, truth, and corpus provenance."""

    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]
    manifest: CorpusManifest
