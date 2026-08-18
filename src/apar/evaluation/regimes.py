"""Deterministic, provenance-bound robustness corpus transformations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow, FrozenCorpus
from apar.runs.wire import canonical_json_bytes

_TRANSFORMER_VERSION: Literal["1.0.0"] = "1.0.0"


class RegimeKind(StrEnum):
    PREVALENCE_DILUTION = "prevalence_dilution"
    MISSING_OPTIONAL = "missing_optional"
    AVAILABILITY_DELAY = "availability_delay"
    COMPRESSED_BURSTS = "compressed_bursts"
    BENIGN_AMOUNT_SHIFT = "benign_amount_shift"
    COLD_ID_REMAP = "cold_id_remap"


class RegimeSpec(ExternalContract):
    """Closed kind-specific parameters for one declared corpus transform."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: RegimeKind
    control_campaign_ids: tuple[str, ...] | None = None
    delay_seconds: int | None = None
    compression_numerator: int | None = None
    compression_denominator: int | None = None
    scale: Decimal | None = None
    salt: str | None = None

    @field_validator("control_campaign_ids", mode="before")
    @classmethod
    def campaign_ids_are_an_exact_tuple(cls, value: object) -> object:
        if value is not None and type(value) is not tuple:
            raise ValueError("control_campaign_ids must be an exact tuple")
        return value

    @model_validator(mode="after")
    def parameters_match_kind(self) -> RegimeSpec:
        relevant = {
            RegimeKind.PREVALENCE_DILUTION: {"control_campaign_ids"},
            RegimeKind.MISSING_OPTIONAL: set(),
            RegimeKind.AVAILABILITY_DELAY: {"delay_seconds"},
            RegimeKind.COMPRESSED_BURSTS: {
                "compression_numerator",
                "compression_denominator",
            },
            RegimeKind.BENIGN_AMOUNT_SHIFT: {"scale"},
            RegimeKind.COLD_ID_REMAP: {"salt"},
        }[self.kind]
        supplied = {
            name
            for name in (
                "control_campaign_ids",
                "delay_seconds",
                "compression_numerator",
                "compression_denominator",
                "scale",
                "salt",
            )
            if getattr(self, name) is not None
        }
        if supplied != relevant:
            raise ValueError("regime has missing or irrelevant parameters")
        if self.kind is RegimeKind.PREVALENCE_DILUTION:
            ids = self.control_campaign_ids
            assert ids is not None
            if not ids or any(type(item) is not str or not item for item in ids):
                raise ValueError("control campaign IDs must be nonempty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("control campaign IDs must be unique")
            if ids != tuple(sorted(ids)):
                raise ValueError("control campaign IDs must be an exact sorted tuple")
        elif self.kind is RegimeKind.AVAILABILITY_DELAY:
            if type(self.delay_seconds) is not int or self.delay_seconds <= 0:
                raise ValueError("delay_seconds must be a positive integer")
        elif self.kind is RegimeKind.COMPRESSED_BURSTS:
            numerator = self.compression_numerator
            denominator = self.compression_denominator
            if type(numerator) is not int or type(denominator) is not int:
                raise ValueError("compression values must be positive integers")
            if numerator <= 0 or denominator <= 0:
                raise ValueError("compression values must be positive integers")
            if numerator >= denominator:
                raise ValueError("compression numerator must be less than denominator")
        elif self.kind is RegimeKind.BENIGN_AMOUNT_SHIFT:
            if self.scale is None or not self.scale.is_finite() or self.scale <= 0:
                raise ValueError("scale must be finite and greater than zero")
        elif self.kind is RegimeKind.COLD_ID_REMAP:
            if type(self.salt) is not str or not self.salt:
                raise ValueError("salt must be a nonempty string")
        return self

    @classmethod
    def prevalence_dilution(cls, control_campaign_ids: tuple[str, ...]) -> RegimeSpec:
        return cls(kind=RegimeKind.PREVALENCE_DILUTION, control_campaign_ids=control_campaign_ids)

    @classmethod
    def missing_optional(cls) -> RegimeSpec:
        return cls(kind=RegimeKind.MISSING_OPTIONAL)

    @classmethod
    def availability_delay(cls, delay_seconds: int = 300) -> RegimeSpec:
        return cls(kind=RegimeKind.AVAILABILITY_DELAY, delay_seconds=delay_seconds)

    @classmethod
    def compressed_bursts(
        cls,
        compression_numerator: int = 1,
        compression_denominator: int = 4,
    ) -> RegimeSpec:
        return cls(
            kind=RegimeKind.COMPRESSED_BURSTS,
            compression_numerator=compression_numerator,
            compression_denominator=compression_denominator,
        )

    @classmethod
    def benign_amount_shift(cls, scale: Decimal = Decimal("1.25")) -> RegimeSpec:
        return cls(kind=RegimeKind.BENIGN_AMOUNT_SHIFT, scale=scale)

    @classmethod
    def cold_id_remap(cls, salt: str = "defense-v1-cold-remap") -> RegimeSpec:
        return cls(kind=RegimeKind.COLD_ID_REMAP, salt=salt)


class DerivedRegimeManifest(ExternalContract):
    """Lineage and canonical result proof for one derived corpus."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    parent_corpus_digest: str
    transformer: RegimeKind
    transformer_version: Literal["1.0.0"] = _TRANSFORMER_VERSION
    parameters: dict[str, str | int | tuple[str, ...]]
    output_corpus_digest: str
    truth_bytes_unchanged: bool


def frozen_corpus_digest(corpus: FrozenCorpus) -> str:
    """Hash the complete frozen corpus in canonical JSON form."""
    if type(corpus) is not FrozenCorpus:
        raise TypeError("corpus must be an exact FrozenCorpus")
    return hashlib.sha256(canonical_json_bytes(_corpus_document(corpus))).hexdigest()


def derive_regime(
    corpus: FrozenCorpus,
    spec: RegimeSpec,
    *,
    control_corpus: FrozenCorpus | None = None,
) -> tuple[FrozenCorpus, DerivedRegimeManifest]:
    """Derive a new corpus without ever materializing or mutating feature matrices."""
    if type(corpus) is not FrozenCorpus:
        raise TypeError("corpus must be an exact FrozenCorpus")
    if type(spec) is not RegimeSpec:
        raise TypeError("spec must be an exact RegimeSpec")
    if spec.kind is not RegimeKind.PREVALENCE_DILUTION and control_corpus is not None:
        raise ValueError("control_corpus is irrelevant to this regime")

    if spec.kind is RegimeKind.PREVALENCE_DILUTION:
        changed = _prevalence_dilution(corpus, spec, control_corpus)
    elif spec.kind is RegimeKind.MISSING_OPTIONAL:
        changed = FrozenCorpus(
            observations=tuple(
                row.model_copy(update={"optional_refs": {}}) for row in corpus.observations
            ),
            truth=corpus.truth,
            manifest=corpus.manifest,
        )
    elif spec.kind is RegimeKind.AVAILABILITY_DELAY:
        assert spec.delay_seconds is not None
        delta = timedelta(seconds=spec.delay_seconds)
        changed = FrozenCorpus(
            observations=tuple(
                row
                if row.is_decision_point
                else row.model_copy(update={"available_at": row.available_at + delta})
                for row in corpus.observations
            ),
            truth=corpus.truth,
            manifest=corpus.manifest,
        )
    elif spec.kind is RegimeKind.COMPRESSED_BURSTS:
        changed = _compressed_bursts(corpus, spec)
    elif spec.kind is RegimeKind.BENIGN_AMOUNT_SHIFT:
        changed = _benign_amount_shift(corpus, spec)
    else:
        changed = _cold_id_remap(corpus, spec)

    truth_unchanged = _truth_bytes(changed) == _truth_bytes(corpus)
    manifest = DerivedRegimeManifest(
        parent_corpus_digest=frozen_corpus_digest(corpus),
        transformer=spec.kind,
        parameters=_parameters(spec),
        output_corpus_digest=frozen_corpus_digest(changed),
        truth_bytes_unchanged=truth_unchanged,
    )
    return changed, manifest


def _prevalence_dilution(
    corpus: FrozenCorpus,
    spec: RegimeSpec,
    control_corpus: FrozenCorpus | None,
) -> FrozenCorpus:
    if type(control_corpus) is not FrozenCorpus:
        raise ValueError("prevalence dilution requires an exact control_corpus")
    assert spec.control_campaign_ids is not None
    requested = set(spec.control_campaign_ids)
    available = {row.campaign_id for row in control_corpus.truth}
    missing = requested - available
    if missing:
        raise ValueError("control_corpus is missing a requested campaign")
    selected_truth = tuple(row for row in control_corpus.truth if row.campaign_id in requested)
    if any(row.is_fraud for row in selected_truth):
        raise ValueError("prevalence controls must be entirely benign")
    selected_payments = {row.payment_id for row in selected_truth}
    if len(selected_payments) != len(selected_truth):
        raise ValueError("control truth contains duplicate payment IDs")
    selected_observations = tuple(
        row for row in control_corpus.observations if row.payment_id in selected_payments
    )
    observations_by_payment: dict[str, set[str]] = {payment: set() for payment in selected_payments}
    for observation in selected_observations:
        observations_by_payment[observation.payment_id].add(observation.event_id)
    for truth in selected_truth:
        if observations_by_payment[truth.payment_id] != set(truth.lifecycle_event_ids):
            raise ValueError("selected observation does not resolve to selected truth lifecycle")

    base_event_ids = {row.event_id for row in corpus.observations}
    selected_event_ids = {row.event_id for row in selected_observations}
    if len(selected_event_ids) != len(selected_observations) or base_event_ids & selected_event_ids:
        raise ValueError("event ID collision while merging controls")
    base_payments = {row.payment_id for row in corpus.truth}
    if base_payments & selected_payments:
        raise ValueError("payment ID collision while merging controls")
    base_campaigns = {row.campaign_id for row in corpus.truth}
    if base_campaigns & requested:
        raise ValueError("campaign ID collision while merging controls")

    observations = tuple(
        sorted(
            (*corpus.observations, *selected_observations),
            key=lambda row: row.event_id,
        )
    )
    merged_truth = tuple(sorted((*corpus.truth, *selected_truth), key=lambda row: row.event_id))
    manifest = CorpusManifest(
        profile_id=corpus.manifest.profile_id,
        run_ids=(*corpus.manifest.run_ids, *control_corpus.manifest.run_ids),
        run_lineage_digests=(
            *corpus.manifest.run_lineage_digests,
            *control_corpus.manifest.run_lineage_digests,
        ),
        observation_count=len(observations),
        truth_count=len(merged_truth),
    )
    return FrozenCorpus(observations=observations, truth=merged_truth, manifest=manifest)


def _compressed_bursts(corpus: FrozenCorpus, spec: RegimeSpec) -> FrozenCorpus:
    assert spec.compression_numerator is not None
    assert spec.compression_denominator is not None
    numerator = spec.compression_numerator
    denominator = spec.compression_denominator
    truth_by_payment = _truth_by_payment(corpus.truth)
    campaign_by_payment = {
        payment_id: truth.campaign_id for payment_id, truth in truth_by_payment.items()
    }
    anchors: dict[str, datetime] = {}
    for observation in corpus.observations:
        try:
            campaign = campaign_by_payment[observation.payment_id]
        except KeyError as error:
            raise ValueError("observation does not resolve to evaluator truth") from error
        anchor = anchors.get(campaign)
        if anchor is None or observation.event_time < anchor:
            anchors[campaign] = observation.event_time

    def shifted(value: datetime, campaign: str) -> datetime:
        anchor = anchors[campaign]
        micros = _timedelta_microseconds(value - anchor)
        scaled = (Decimal(micros) * Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal("1"), rounding=ROUND_HALF_EVEN
        )
        return anchor + timedelta(microseconds=int(scaled))

    observations = tuple(
        row.model_copy(
            update={
                "event_time": shifted(row.event_time, campaign_by_payment[row.payment_id]),
                "available_at": shifted(row.available_at, campaign_by_payment[row.payment_id]),
                "decision_at": (
                    shifted(row.decision_at, campaign_by_payment[row.payment_id])
                    if row.decision_at is not None
                    else None
                ),
            }
        )
        for row in corpus.observations
    )
    truth = tuple(
        row.model_copy(
            update={
                "first_settlement_at": (
                    shifted(row.first_settlement_at, row.campaign_id)
                    if row.first_settlement_at is not None
                    else None
                ),
                "label_mature_at": shifted(row.label_mature_at, row.campaign_id),
            }
        )
        for row in corpus.truth
    )
    return FrozenCorpus(observations=observations, truth=truth, manifest=corpus.manifest)


def _benign_amount_shift(corpus: FrozenCorpus, spec: RegimeSpec) -> FrozenCorpus:
    assert spec.scale is not None
    scale = spec.scale
    benign_payments = {row.payment_id for row in corpus.truth if not row.is_fraud}

    def scaled(value: Decimal) -> Decimal:
        return (value * scale).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    observations = tuple(
        row.model_copy(update={"amount": scaled(row.amount)})
        if row.payment_id in benign_payments
        else row
        for row in corpus.observations
    )
    truth = tuple(
        row.model_copy(update={"net_settled_value": scaled(row.net_settled_value)})
        if not row.is_fraud
        else row
        for row in corpus.truth
    )
    return FrozenCorpus(observations=observations, truth=truth, manifest=corpus.manifest)


def _cold_id_remap(corpus: FrozenCorpus, spec: RegimeSpec) -> FrozenCorpus:
    assert spec.salt is not None
    identities = {
        identity
        for row in corpus.observations
        for identity in (row.actor_id, row.counterparty_id, *row.optional_refs.values())
    }
    mapping = {
        identity: str(uuid5(NAMESPACE_URL, f"apar:{spec.salt}:{identity}"))
        for identity in sorted(identities)
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("cold-ID remapping is not bijective")
    observations = tuple(
        row.model_copy(
            update={
                "actor_id": mapping[row.actor_id],
                "counterparty_id": mapping[row.counterparty_id],
                "optional_refs": {
                    name: mapping[value] for name, value in sorted(row.optional_refs.items())
                },
            }
        )
        for row in corpus.observations
    )
    return FrozenCorpus(observations=observations, truth=corpus.truth, manifest=corpus.manifest)


def _truth_by_payment(
    truth: tuple[EvaluationTruthRow, ...],
) -> dict[str, EvaluationTruthRow]:
    result: dict[str, EvaluationTruthRow] = {}
    for row in truth:
        if row.payment_id in result:
            raise ValueError("truth contains duplicate payment IDs")
        result[row.payment_id] = row
    return result


def _timedelta_microseconds(value: timedelta) -> int:
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds


def _corpus_document(corpus: FrozenCorpus) -> dict[str, object]:
    return {
        "observations": [row.model_dump(mode="json") for row in corpus.observations],
        "truth": [row.model_dump(mode="json") for row in corpus.truth],
        "manifest": corpus.manifest.model_dump(mode="json"),
    }


def _truth_bytes(corpus: FrozenCorpus) -> bytes:
    return canonical_json_bytes([row.model_dump(mode="json") for row in corpus.truth])


def _parameters(spec: RegimeSpec) -> dict[str, str | int | tuple[str, ...]]:
    if spec.kind is RegimeKind.PREVALENCE_DILUTION:
        assert spec.control_campaign_ids is not None
        return {"control_campaign_ids": spec.control_campaign_ids}
    if spec.kind is RegimeKind.MISSING_OPTIONAL:
        return {}
    if spec.kind is RegimeKind.AVAILABILITY_DELAY:
        assert spec.delay_seconds is not None
        return {"delay_seconds": spec.delay_seconds}
    if spec.kind is RegimeKind.COMPRESSED_BURSTS:
        assert spec.compression_numerator is not None
        assert spec.compression_denominator is not None
        return {
            "compression_denominator": spec.compression_denominator,
            "compression_numerator": spec.compression_numerator,
        }
    if spec.kind is RegimeKind.BENIGN_AMOUNT_SHIFT:
        assert spec.scale is not None
        return {"scale": str(spec.scale)}
    assert spec.salt is not None
    return {"salt": spec.salt}


__all__ = [
    "DerivedRegimeManifest",
    "RegimeKind",
    "RegimeSpec",
    "derive_regime",
    "frozen_corpus_digest",
]
