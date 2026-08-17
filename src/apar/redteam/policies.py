"""Closed public contracts and bounded, decision-only attack policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException, localcontext
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, Self, cast

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.generators import CampaignParameterError, CampaignParams

PUBLIC_REASON_FAMILIES = frozenset(
    {
        "amount",
        "approved",
        "authentication",
        "authorization",
        "entity",
        "evaluation_failure",
        "integrity",
        "invalid_candidate",
        "other",
        "policy",
        "recovery",
        "velocity",
    }
)

_ADAPTIVE_FIELDS = frozenset(
    {
        "merchant_concentration",
        "device_reuse_rate",
        "retry_intensity",
        "mule_count",
        "mule_layers",
        "mule_fanout",
        "cash_out_fraction",
        "cash_out_strategy",
        "cash_out_delay_seconds",
        "recovery_probability",
        "agentic_mutations",
        "agentic_attack_mix",
    }
)
_FAMILY_FIELDS = MappingProxyType(
    {
        "app_scam_mule": frozenset(
            {
                "cash_out_delay_seconds",
                "cash_out_fraction",
                "cash_out_strategy",
                "mule_count",
                "mule_fanout",
                "mule_layers",
                "recovery_probability",
            }
        ),
        "card_testing_cnp": frozenset(
            {"device_reuse_rate", "merchant_concentration", "retry_intensity"}
        ),
        "synthetic_merchant_refund": frozenset({"recovery_probability"}),
        "agentic_intent_abuse": frozenset(
            {"agentic_attack_mix", "agentic_mutations"}
        ),
    }
)
_HEX = frozenset("0123456789abcdef")


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _exact_non_negative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    return value


def _domain_int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("domain value must be an exact integer")
    return value


def _domain_text(value: object) -> str:
    if type(value) is not str:
        raise TypeError("domain value must be an exact string")
    return value


def _domain_decimal(value: object) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError("domain value must be an exact finite Decimal")
    return value


def _canonical_value(value: object) -> object:
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("domain Decimal values must be finite")
        return str(value)
    if type(value) in {int, str}:
        return value
    if type(value) is tuple and all(type(item) is str for item in value):
        return list(value)
    raise TypeError("adaptive values must be exact int, string, Decimal, or string tuple")


def _same_adaptive_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is Decimal:
        right_decimal = cast(Decimal, right)
        if not left.is_finite() or not right_decimal.is_finite():
            return False
        return left.as_tuple() == right_decimal.as_tuple()
    return left == right


def _is_no_op_alias(left: object, right: object) -> bool:
    if type(left) is Decimal and type(right) is Decimal:
        return (
            left.is_finite()
            and right.is_finite()
            and left == right
        )
    return _same_adaptive_value(left, right)


def _canonical_params_document(params: CampaignParams) -> dict[str, object]:
    return {
        item.name: _canonical_value(getattr(params, item.name))
        for item in fields(CampaignParams)
    }


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _ratio(numerator: int, denominator: int, *, precision: int = 18) -> Decimal:
    if denominator <= 0:
        raise ValueError("canonical ratio denominator must be positive")
    with localcontext() as context:
        context.prec = max(precision + 8, 36)
        return (Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal(1).scaleb(-precision),
            rounding=ROUND_HALF_EVEN,
        )


def _canonical_recovery_probability(recovery_count: int, eligible_count: int) -> Decimal:
    compatible = int(Decimal(eligible_count) * Decimal("0.25"))
    if Decimal(eligible_count) * Decimal("0.25") > compatible:
        compatible += 1
    if recovery_count == compatible:
        return Decimal("0.25")
    with localcontext() as context:
        context.prec = 50
        return (Decimal(recovery_count) / Decimal(eligible_count)).quantize(
            Decimal("0.000000000000000001"),
            rounding=ROUND_HALF_EVEN,
        )


def _recovery_values(eligible_count: int, *, require_unrecovered: bool) -> tuple[Decimal, ...]:
    maximum = eligible_count - 1 if require_unrecovered else eligible_count
    if maximum < 1:
        raise CampaignParameterError("recovery domain has no feasible level")
    return tuple(
        sorted(
            {
                _canonical_recovery_probability(count, eligible_count)
                for count in range(1, maximum + 1)
            }
        )
    )


def _cash_fraction_values(total: Decimal) -> tuple[Decimal, ...]:
    total_cents = int(total / Decimal("0.01"))
    values: set[Decimal] = set()
    # Task 5 proves exactly these two externally observable APP cash-out levels.
    for basis_points in (2000, 3000):
        cash = (total * Decimal(basis_points) / Decimal(10000)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN
        )
        compatible_cash = (total * Decimal("0.30")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN
        )
        value = (
            Decimal("0.30")
            if cash == compatible_cash
            else _ratio(int(cash / Decimal("0.01")), total_cents)
        )
        if Decimal(0) < value <= Decimal(1):
            values.add(value)
    return tuple(sorted(values))


def _log_values(minimum: int, maximum: int, default: int) -> tuple[int, ...]:
    if not minimum <= default <= maximum:
        raise CampaignParameterError("log-domain default is outside its bounds")
    values = {minimum, maximum, default}
    current = max(2, minimum)
    while current < maximum:
        values.add(current)
        current *= 3
    return tuple(sorted(values))


class DomainKind(StrEnum):
    """Declared sampling semantics for a finite canonical domain."""

    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    LINEAR = "linear"
    LOG = "log"


class ParameterDomain(ExternalContract):
    """One exact, finite parameter domain with deterministic ordering."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    name: str
    kind: DomainKind
    values: tuple[object, ...]

    @field_validator("name", mode="before")
    @classmethod
    def name_is_exact(cls, value: object) -> object:
        return _exact_text("domain name", value)

    @field_validator("kind", mode="before")
    @classmethod
    def kind_is_exact(cls, value: object) -> object:
        if type(value) is not DomainKind:
            raise TypeError("domain kind must be an exact DomainKind")
        return value

    @field_validator("values", mode="before")
    @classmethod
    def values_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise TypeError("domain values must be a non-empty exact tuple")
        for item in value:
            _canonical_value(item)
        canonical = tuple(
            json.dumps(_canonical_value(item), sort_keys=True, separators=(",", ":"))
            for item in value
        )
        if len(set(canonical)) != len(canonical):
            raise ValueError("domain values must be unique")
        if any(
            _is_no_op_alias(left, right)
            for index, left in enumerate(value)
            for right in value[index + 1 :]
        ):
            raise ValueError("domain values contain numerically equal no-op aliases")
        return value

    def sample(self, rng: np.random.Generator) -> object:
        if type(rng) is not np.random.Generator:
            raise TypeError("rng must be an exact numpy.random.Generator")
        return self.values[int(rng.integers(0, len(self.values)))]

    def alternatives(self, current: object) -> tuple[object, ...]:
        return tuple(
            value for value in self.values if not _same_adaptive_value(value, current)
        )


class ParameterBounds(ExternalContract):
    """Family-specific Task 5 adaptive bounds and their immutable template."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    family: str
    template: CampaignParams
    domains: tuple[ParameterDomain, ...]

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        checked = _exact_text("family", value)
        if checked not in _FAMILY_FIELDS:
            raise ValueError("unsupported campaign family")
        return checked

    @field_validator("template", mode="before")
    @classmethod
    def template_is_exact(cls, value: object) -> object:
        if type(value) is not CampaignParams:
            raise TypeError("template must be an exact CampaignParams")
        return value

    @field_validator("domains", mode="before")
    @classmethod
    def domains_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not ParameterDomain for item in value):
            raise TypeError("domains must be an exact tuple of exact ParameterDomain records")
        return value

    @model_validator(mode="after")
    def domains_are_closed(self) -> Self:
        expected = _FAMILY_FIELDS[self.family]
        names = tuple(domain.name for domain in self.domains)
        if names != tuple(sorted(expected)):
            raise ValueError("domains must declare exactly the sorted family adaptive fields")
        if not set(names) <= _ADAPTIVE_FIELDS:
            raise ValueError("bounds contain a non-adaptive field")
        for domain in self.domains:
            default = getattr(self.template, domain.name)
            if not any(_same_adaptive_value(default, value) for value in domain.values):
                raise ValueError(f"default for {domain.name} is absent from its domain")
            expected_type = type(default)
            if any(type(value) is not expected_type for value in domain.values):
                raise TypeError(f"domain {domain.name} contains a non-exact value type")
        self.validate_params(self.template)
        return self

    @classmethod
    def for_campaign(cls, family: str, template: CampaignParams) -> ParameterBounds:
        if type(template) is not CampaignParams:
            raise TypeError("template must be an exact CampaignParams")
        checked_family = _exact_text("family", family)
        try:
            domains: tuple[ParameterDomain, ...]
            illicit_count = round(Decimal(template.payment_count) * template.target_illicit_rate)
            if checked_family == "card_testing_cnp":
                retry_values = tuple(range(1, min(10, illicit_count - 1) + 1))
                domains = (
                    ParameterDomain(
                        name="device_reuse_rate",
                        kind=DomainKind.LINEAR,
                        values=tuple(sorted({Decimal(0), Decimal("0.60"), Decimal(1)})),
                    ),
                    ParameterDomain(
                        name="merchant_concentration",
                        kind=DomainKind.LINEAR,
                        values=tuple(sorted({Decimal(0), Decimal("0.70"), Decimal(1)})),
                    ),
                    ParameterDomain(
                        name="retry_intensity",
                        kind=DomainKind.DISCRETE,
                        values=retry_values,
                    ),
                )
            elif checked_family == "app_scam_mule":
                if template.mule_count != template.mule_layers + 1:
                    raise CampaignParameterError("APP template topology is not canonical")
                maximum_fanout = min(16, illicit_count - template.mule_layers - 2)
                if maximum_fanout < 2:
                    raise CampaignParameterError("APP adaptive topology has no feasible fan-out")
                fanouts = tuple(range(2, maximum_fanout + 1))
                recovery_values = tuple(
                    sorted(
                        {
                            value
                            for fanout in fanouts
                            for value in _recovery_values(
                                fanout, require_unrecovered=False
                            )
                        }
                    )
                )
                domains = (
                    ParameterDomain(
                        name="cash_out_delay_seconds",
                        kind=DomainKind.LOG,
                        values=_log_values(
                            template.min_delay_seconds,
                            template.max_delay_seconds,
                            template.cash_out_delay_seconds,
                        ),
                    ),
                    ParameterDomain(
                        name="cash_out_fraction",
                        kind=DomainKind.LINEAR,
                        values=_cash_fraction_values(template.target_value_total),
                    ),
                    ParameterDomain(
                        name="cash_out_strategy",
                        kind=DomainKind.CATEGORICAL,
                        values=("burst", "delayed", "staged"),
                    ),
                    ParameterDomain(
                        name="mule_count",
                        kind=DomainKind.DISCRETE,
                        values=(template.mule_count,),
                    ),
                    ParameterDomain(
                        name="mule_fanout",
                        kind=DomainKind.DISCRETE,
                        values=fanouts,
                    ),
                    ParameterDomain(
                        name="mule_layers",
                        kind=DomainKind.DISCRETE,
                        values=(template.mule_layers,),
                    ),
                    ParameterDomain(
                        name="recovery_probability",
                        kind=DomainKind.DISCRETE,
                        values=recovery_values,
                    ),
                )
            elif checked_family == "synthetic_merchant_refund":
                domains = (
                    ParameterDomain(
                        name="recovery_probability",
                        kind=DomainKind.DISCRETE,
                        values=_recovery_values(illicit_count, require_unrecovered=True),
                    ),
                )
            elif checked_family == "agentic_intent_abuse":
                mixes: list[Decimal] = []
                for count in range(23, template.payment_count - 1):
                    with localcontext() as context:
                        context.prec = 28
                        mix = Decimal(count) / Decimal(template.payment_count)
                    if abs(mix - template.target_illicit_rate) <= template.class_rate_tolerance:
                        mixes.append(mix)
                mutation_values = (
                    template.agentic_mutations,
                    *(tuple([mutation]) for mutation in template.agentic_mutations[1:]),
                )
                domains = (
                    ParameterDomain(
                        name="agentic_attack_mix",
                        kind=DomainKind.DISCRETE,
                        values=tuple(mixes),
                    ),
                    ParameterDomain(
                        name="agentic_mutations",
                        kind=DomainKind.CATEGORICAL,
                        values=mutation_values,
                    ),
                )
            else:
                raise CampaignParameterError("unsupported campaign family")
            return cls(family=checked_family, template=template, domains=domains)
        except (ArithmeticError, DecimalException, TypeError, ValueError) as error:
            if isinstance(error, CampaignParameterError):
                raise
            raise CampaignParameterError(str(error)) from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(domain.name for domain in self.domains)

    @property
    def has_non_no_op_mutation(self) -> bool:
        """Whether Task 5 exposes at least one canonical alternative value."""
        if self.family == "agentic_intent_abuse":
            return any(
                self._agentic_additional_attacks(_domain_decimal(value)) > 0
                for value in self.domain("agentic_attack_mix").values
            )
        return any(len(domain.values) > 1 for domain in self.domains)

    def domain(self, name: str) -> ParameterDomain:
        checked = _exact_text("domain name", name)
        for domain in self.domains:
            if domain.name == checked:
                return domain
        raise CampaignParameterError(f"undeclared parameter: {checked}")

    def defaults_document(self) -> dict[str, object]:
        return {
            name: _canonical_value(getattr(self.template, name)) for name in self.names
        }

    def schema_document(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": [
                {
                    "name": domain.name,
                    "kind": domain.kind.value,
                    "allowed_values": [_canonical_value(value) for value in domain.values],
                }
                for domain in self.domains
            ],
            "required": list(self.names),
            "additional_parameters": False,
        }

    def decode_updates(self, document: object) -> dict[str, object]:
        if type(document) is not dict:
            raise CampaignParameterError("planner params must be an exact object")
        unknown = set(document) - set(self.names)
        missing = set(self.names) - set(document)
        if unknown:
            raise CampaignParameterError(f"undeclared parameter: {sorted(unknown)}")
        if missing:
            raise CampaignParameterError(f"missing parameter: {sorted(missing)}")
        decoded: dict[str, object] = {}
        for name in self.names:
            raw = document[name]
            expected = getattr(self.template, name)
            if type(expected) is Decimal:
                if type(raw) is not str:
                    raise CampaignParameterError(f"{name} must be a canonical decimal string")
                try:
                    value: object = Decimal(raw)
                except DecimalException as error:
                    raise CampaignParameterError(f"{name} is not a finite decimal") from error
            elif type(expected) is tuple:
                if type(raw) is not list or any(type(item) is not str for item in raw):
                    raise CampaignParameterError(f"{name} must be an exact string array")
                value = tuple(raw)
            elif type(raw) is not type(expected):
                raise CampaignParameterError(f"{name} has the wrong exact type")
            else:
                value = raw
            decoded[name] = value
        return decoded

    def with_updates(
        self,
        base: CampaignParams,
        updates: dict[str, object],
    ) -> CampaignParams:
        if type(base) is not CampaignParams:
            raise CampaignParameterError("base must be an exact CampaignParams")
        if type(updates) is not dict:
            raise CampaignParameterError("updates must be an exact mapping")
        unknown = set(updates) - set(self.names)
        if unknown:
            raise CampaignParameterError(
                f"non-adaptive or undeclared fields: {sorted(unknown)}"
            )
        self.validate_params(base)
        for name, value in updates.items():
            if not any(
                _same_adaptive_value(value, item)
                for item in self.domain(name).values
            ):
                raise CampaignParameterError(f"{name} is outside its canonical domain")
        try:
            values = {
                item.name: getattr(base, item.name) for item in fields(CampaignParams)
            }
            values.update(updates)
            candidate = CampaignParams.from_mapping(values)
        except (DecimalException, TypeError, ValueError) as error:
            raise CampaignParameterError(str(error)) from error
        self.validate_params(candidate)
        return candidate

    def validate_params(self, params: CampaignParams) -> None:
        if type(params) is not CampaignParams:
            raise CampaignParameterError("candidate params must be exact CampaignParams")
        for item in fields(CampaignParams):
            if item.name not in self.names and getattr(params, item.name) != getattr(
                self.template, item.name
            ):
                raise CampaignParameterError(f"non-adaptive field changed: {item.name}")
        for domain in self.domains:
            value = getattr(params, domain.name)
            if not any(_same_adaptive_value(value, item) for item in domain.values):
                raise CampaignParameterError(
                    f"{domain.name} is outside its canonical domain"
                )
        if self.family == "app_scam_mule":
            if params.mule_count != params.mule_layers + 1:
                raise CampaignParameterError("APP topology must bind mule count to layers")
            if (
                round(Decimal(params.payment_count) * params.target_illicit_rate)
                - params.mule_layers
                - params.mule_fanout
                < 2
            ):
                raise CampaignParameterError("APP topology leaves fewer than two incoming flows")
            if params.recovery_probability not in _recovery_values(
                params.mule_fanout, require_unrecovered=False
            ):
                raise CampaignParameterError(
                    "APP recovery probability is not canonical for fan-out"
                )
            if params.cash_out_strategy == "burst":
                if params.cash_out_delay_seconds != params.min_delay_seconds:
                    raise CampaignParameterError("burst delay must equal the minimum delay")
            elif params.cash_out_delay_seconds <= params.min_delay_seconds:
                raise CampaignParameterError("staged/delayed cash-out needs a distinct delay")
        elif self.family == "agentic_intent_abuse":
            additional = self._agentic_additional_attacks(params.agentic_attack_mix)
            if additional == 0 and params.agentic_mutations != self.template.agentic_mutations:
                raise CampaignParameterError("agentic mutations have no adaptive slot")
            if (
                additional > 0
                and params.agentic_mutations != self.template.agentic_mutations
                and len(params.agentic_mutations) > additional
            ):
                raise CampaignParameterError("agentic mutations contain unused values")
            extras = tuple(
                params.agentic_mutations[index % len(params.agentic_mutations)]
                for index in range(additional)
            )
            defaults = tuple(
                self.template.agentic_mutations[
                    index % len(self.template.agentic_mutations)
                ]
                for index in range(additional)
            )
            if (
                params.agentic_mutations != self.template.agentic_mutations
                and extras == defaults
            ):
                raise CampaignParameterError("agentic mutations alias the default sequence")

    def _agentic_additional_attacks(self, mix: Decimal) -> int:
        with localcontext() as context:
            context.prec = 28
            attack_count = int(mix * Decimal(self.template.payment_count))
            if Decimal(attack_count) / Decimal(self.template.payment_count) != mix:
                raise CampaignParameterError("agentic attack mix is not canonical")
        return attack_count - 23

    def changed_field_count(self, left: CampaignParams, right: CampaignParams) -> int:
        self.validate_params(left)
        self.validate_params(right)
        return sum(getattr(left, name) != getattr(right, name) for name in self.names)

    def sample_params(self, rng: np.random.Generator) -> CampaignParams:
        if type(rng) is not np.random.Generator:
            raise TypeError("rng must be an exact numpy.random.Generator")
        if self.family == "agentic_intent_abuse":
            mix = _domain_decimal(self.domain("agentic_attack_mix").sample(rng))
            additional = self._agentic_additional_attacks(mix)
            mutations = tuple(
                value
                for value in self.domain("agentic_mutations").values
                if (
                    value == self.template.agentic_mutations
                    if additional == 0
                    else (
                        value == self.template.agentic_mutations
                        or (
                            type(value) is tuple
                            and len(value) <= additional
                            and tuple(
                                value[index % len(value)] for index in range(additional)
                            )
                            != tuple(
                                self.template.agentic_mutations[
                                    index % len(self.template.agentic_mutations)
                                ]
                                for index in range(additional)
                            )
                        )
                    )
                )
            )
            return self.with_updates(
                self.template,
                {
                    "agentic_attack_mix": mix,
                    "agentic_mutations": mutations[
                        int(rng.integers(0, len(mutations)))
                    ],
                },
            )
        if self.family != "app_scam_mule":
            return self.with_updates(
                self.template,
                {domain.name: domain.sample(rng) for domain in self.domains},
            )
        fanout = _domain_int(self.domain("mule_fanout").sample(rng))
        recovery = self.domain("recovery_probability")
        recovery_values = tuple(
            value
            for value in recovery.values
            if value in _recovery_values(fanout, require_unrecovered=False)
        )
        strategy = _domain_text(self.domain("cash_out_strategy").sample(rng))
        delay_domain = self.domain("cash_out_delay_seconds")
        delay_values = tuple(
            _domain_int(value)
            for value in delay_domain.values
            if (
                _domain_int(value) == self.template.min_delay_seconds
                if strategy == "burst"
                else _domain_int(value) > self.template.min_delay_seconds
            )
        )
        updates = {
            domain.name: domain.sample(rng)
            for domain in self.domains
            if domain.name
            not in {
                "mule_fanout",
                "recovery_probability",
                "cash_out_strategy",
                "cash_out_delay_seconds",
            }
        }
        updates.update(
            {
                "mule_fanout": fanout,
                "recovery_probability": recovery_values[
                    int(rng.integers(0, len(recovery_values)))
                ],
                "cash_out_strategy": strategy,
                "cash_out_delay_seconds": delay_values[
                    int(rng.integers(0, len(delay_values)))
                ],
            }
        )
        return self.with_updates(self.template, updates)

    def mutate_params(
        self,
        base: CampaignParams,
        rng: np.random.Generator,
    ) -> CampaignParams:
        self.validate_params(base)
        if type(rng) is not np.random.Generator:
            raise TypeError("rng must be an exact numpy.random.Generator")
        groups: list[tuple[str, ...]]
        if self.family == "app_scam_mule":
            groups = [
                ("cash_out_fraction",),
                ("mule_fanout", "recovery_probability"),
                ("recovery_probability",),
                ("cash_out_strategy", "cash_out_delay_seconds"),
                ("cash_out_delay_seconds",),
            ]
        elif self.family == "agentic_intent_abuse":
            groups = [("agentic_attack_mix", "agentic_mutations")]
        else:
            mutable = [
                domain.name
                for domain in self.domains
                if domain.alternatives(getattr(base, domain.name))
            ]
            groups = []
            for count in range(1, min(3, len(mutable)) + 1):
                for start in range(len(mutable)):
                    selected = tuple(
                        mutable[(start + offset) % len(mutable)] for offset in range(count)
                    )
                    if len(set(selected)) == count and selected not in groups:
                        groups.append(selected)
        viable = [
            group
            for group in groups
            if any(
                self.domain(name).alternatives(getattr(base, name)) for name in group
            )
        ]
        if not viable:
            raise CampaignParameterError("adaptive domain has no non-no-op mutation")
        for _ in range(100):
            group = viable[int(rng.integers(0, len(viable)))]
            try:
                if self.family == "app_scam_mule":
                    candidate = self._mutate_app(base, group, rng)
                else:
                    updates: dict[str, object] = {}
                    for name in group:
                        alternatives = self.domain(name).alternatives(getattr(base, name))
                        if alternatives:
                            updates[name] = alternatives[
                                int(rng.integers(0, len(alternatives)))
                            ]
                    if not updates:
                        continue
                    candidate = self.with_updates(base, updates)
            except CampaignParameterError:
                continue
            changed = self.changed_field_count(base, candidate)
            if 1 <= changed <= 3:
                return candidate
        raise CampaignParameterError("no canonical non-no-op mutation exists")

    def _mutate_app(
        self,
        base: CampaignParams,
        group: tuple[str, ...],
        rng: np.random.Generator,
    ) -> CampaignParams:
        updates: dict[str, object] = {}
        if "mule_fanout" in group:
            alternatives = self.domain("mule_fanout").alternatives(base.mule_fanout)
            fanout = _domain_int(
                alternatives[int(rng.integers(0, len(alternatives)))]
            )
            recoveries = _recovery_values(fanout, require_unrecovered=False)
            updates["mule_fanout"] = fanout
            updates["recovery_probability"] = recoveries[
                int(rng.integers(0, len(recoveries)))
            ]
        elif "recovery_probability" in group:
            alternatives = tuple(
                value
                for value in _recovery_values(
                    base.mule_fanout, require_unrecovered=False
                )
                if value != base.recovery_probability
            )
            if not alternatives:
                raise CampaignParameterError("no alternate recovery level")
            updates["recovery_probability"] = alternatives[
                int(rng.integers(0, len(alternatives)))
            ]
        if "cash_out_strategy" in group:
            alternatives = self.domain("cash_out_strategy").alternatives(
                base.cash_out_strategy
            )
            strategy = _domain_text(
                alternatives[int(rng.integers(0, len(alternatives)))]
            )
            delay_values = tuple(
                _domain_int(value)
                for value in self.domain("cash_out_delay_seconds").values
                if (
                    _domain_int(value) == base.min_delay_seconds
                    if strategy == "burst"
                    else _domain_int(value) > base.min_delay_seconds
                )
            )
            updates["cash_out_strategy"] = strategy
            updates["cash_out_delay_seconds"] = delay_values[
                int(rng.integers(0, len(delay_values)))
            ]
        elif "cash_out_delay_seconds" in group:
            alternatives = tuple(
                _domain_int(value)
                for value in self.domain("cash_out_delay_seconds").alternatives(
                    base.cash_out_delay_seconds
                )
                if (
                    _domain_int(value) == base.min_delay_seconds
                    if base.cash_out_strategy == "burst"
                    else _domain_int(value) > base.min_delay_seconds
                )
            )
            if not alternatives:
                raise CampaignParameterError("no alternate compatible cash-out delay")
            updates["cash_out_delay_seconds"] = alternatives[
                int(rng.integers(0, len(alternatives)))
            ]
        if "cash_out_fraction" in group:
            alternatives = self.domain("cash_out_fraction").alternatives(
                base.cash_out_fraction
            )
            updates["cash_out_fraction"] = alternatives[
                int(rng.integers(0, len(alternatives)))
            ]
        return self.with_updates(base, updates)


class AttackCandidate(ExternalContract):
    """A canonical candidate containing no evaluator-owned evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    params: CampaignParams
    parent_id: str | None
    generation: int

    @field_validator("params", mode="before")
    @classmethod
    def params_are_exact(cls, value: object) -> object:
        if type(value) is not CampaignParams:
            raise TypeError("params must be an exact CampaignParams")
        return value

    @field_validator("parent_id", mode="before")
    @classmethod
    def parent_is_canonical(cls, value: object) -> object:
        if value is None:
            return value
        text = _exact_text("parent_id", value)
        if len(text) != 64 or not set(text) <= _HEX:
            raise ValueError("parent_id must be a lowercase SHA-256 hex digest")
        return text

    @field_validator("generation", mode="before")
    @classmethod
    def generation_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("generation", value)

    @property
    def fingerprint(self) -> str:
        return _digest(_canonical_params_document(self.params))

    @property
    def candidate_id(self) -> str:
        return _digest(
            {
                "fingerprint": self.fingerprint,
                "generation": self.generation,
                "parent_id": self.parent_id,
            }
        )


class Feedback(ExternalContract):
    """The complete and intentionally coarse attacker observation."""

    action: Action
    reason_family: str
    realized_value: Decimal | None

    @field_validator("action", mode="before")
    @classmethod
    def action_is_exact(cls, value: object) -> object:
        if type(value) is not Action:
            raise ValueError("action must be an exact Action")
        return value

    @field_validator("reason_family", mode="before")
    @classmethod
    def reason_is_public(cls, value: object) -> object:
        checked = _exact_text("reason_family", value)
        if checked not in PUBLIC_REASON_FAMILIES:
            raise ValueError("reason_family is not in the public allowlist")
        return checked

    @field_validator("realized_value", mode="before")
    @classmethod
    def realized_value_is_exact(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not Decimal:
            raise TypeError("realized_value must be an exact Decimal or None")
        if not value.is_finite() or value < 0:
            raise ValueError("realized_value must be finite and non-negative")
        try:
            quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        except DecimalException as error:
            raise ValueError("realized_value exceeds canonical money bounds") from error
        if value.as_tuple().exponent != -2 or value != quantized:
            raise ValueError("realized_value must be canonically quantized")
        return value

    @model_validator(mode="after")
    def action_and_reason_are_consistent(self) -> Self:
        if (self.action is Action.APPROVE) != (self.reason_family == "approved"):
            raise ValueError("approved reason_family must exactly match approve action")
        if (
            self.reason_family in {"evaluation_failure", "invalid_candidate"}
            and self.action is not Action.DECLINE
        ):
            raise ValueError("failure reason families must decline")
        return self


def visible_objective(feedback: Feedback) -> Decimal:
    """Compute the only objective value permitted in policy-visible history."""
    if type(feedback) is not Feedback:
        raise TypeError("feedback must be an exact Feedback")
    penalty = {
        Action.APPROVE: Decimal(0),
        Action.CHALLENGE: Decimal("0.25"),
        Action.DECLINE: Decimal(1),
    }[feedback.action]
    return (feedback.realized_value or Decimal(0)) - penalty


class VisibleTrial(ExternalContract):
    """Past-only policy history; no score, label, or hidden reason can fit."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    candidate: AttackCandidate
    feedback: Feedback
    objective_value: Decimal

    @field_validator("candidate", mode="before")
    @classmethod
    def candidate_is_exact(cls, value: object) -> object:
        if type(value) is not AttackCandidate:
            raise TypeError("candidate must be an exact AttackCandidate")
        return value

    @field_validator("feedback", mode="before")
    @classmethod
    def feedback_is_exact(cls, value: object) -> object:
        if type(value) is not Feedback:
            raise TypeError("feedback must be an exact Feedback")
        return value

    @field_validator("objective_value", mode="before")
    @classmethod
    def objective_is_exact(cls, value: object) -> object:
        if type(value) is not Decimal or not value.is_finite():
            raise TypeError("objective_value must be an exact finite Decimal")
        return value

    @model_validator(mode="after")
    def objective_is_derived_from_public_feedback(self) -> Self:
        if self.objective_value != visible_objective(self.feedback):
            raise ValueError("objective_value must be derived only from public feedback")
        return self


class Policy(Protocol):
    """A policy consumes only visible history, public bounds, and a local RNG."""

    policy_name: str

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate: ...


def _validate_policy_inputs(
    history: tuple[VisibleTrial, ...],
    bounds: ParameterBounds,
    rng: np.random.Generator,
) -> None:
    if type(history) is not tuple or any(type(item) is not VisibleTrial for item in history):
        raise TypeError("history must be an exact tuple of exact VisibleTrial records")
    if type(bounds) is not ParameterBounds:
        raise TypeError("bounds must be an exact ParameterBounds")
    if type(rng) is not np.random.Generator:
        raise TypeError("rng must be an exact numpy.random.Generator")


class FixedPolicy:
    """Repeat the declared Task 5 defaults as the static control."""

    policy_name = "fixed"

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate:
        _validate_policy_inputs(history, bounds, rng)
        return AttackCandidate(params=bounds.template, parent_id=None, generation=len(history))


class RandomPolicy:
    """Sample the exact finite linear/log/discrete/categorical domains."""

    policy_name = "random"

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate:
        _validate_policy_inputs(history, bounds, rng)
        return AttackCandidate(
            params=bounds.sample_params(rng),
            parent_id=None,
            generation=len(history),
        )


class AdaptiveTournamentPolicy:
    """Order-invariant tournament selection with one-to-three bounded mutations."""

    policy_name = "adaptive"

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate:
        _validate_policy_inputs(history, bounds, rng)
        if not bounds.has_non_no_op_mutation:
            return AttackCandidate(
                params=bounds.template,
                parent_id=None,
                generation=len(history),
            )
        if not history:
            params = bounds.mutate_params(bounds.template, rng)
            return AttackCandidate(params=params, parent_id=None, generation=0)
        eligible = tuple(
            sorted(
                (
                    trial
                    for trial in history
                    if trial.feedback.reason_family
                    not in {"evaluation_failure", "invalid_candidate"}
                ),
                key=lambda trial: trial.candidate.candidate_id,
            )
        )
        if not eligible:
            params = bounds.mutate_params(bounds.template, rng)
            return AttackCandidate(params=params, parent_id=None, generation=len(history))
        global_best = min(
            eligible,
            key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id),
        )
        remaining = tuple(trial for trial in eligible if trial is not global_best)
        sampled = (
            ()
            if not remaining
            else tuple(
                remaining[int(index)]
                for index in rng.choice(
                    len(remaining),
                    size=min(2, len(remaining)),
                    replace=False,
                )
            )
        )
        contenders = (global_best, *sampled)
        parent = min(
            contenders,
            key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id),
        )
        seen = {trial.candidate.fingerprint for trial in history}
        params = self._feedback_mutation(parent, bounds, rng)
        for _ in range(99):
            candidate = AttackCandidate(
                params=params,
                parent_id=parent.candidate.candidate_id,
                generation=len(history),
            )
            if candidate.fingerprint not in seen:
                return candidate
            params = self._feedback_mutation(parent, bounds, rng)
        return AttackCandidate(
            params=params,
            parent_id=parent.candidate.candidate_id,
            generation=len(history),
        )

    @staticmethod
    def _feedback_mutation(
        parent: VisibleTrial,
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> CampaignParams:
        current = parent.candidate.params
        if bounds.family == "card_testing_cnp":
            if parent.feedback.reason_family == "velocity":
                lower = tuple(
                    _domain_int(value)
                    for value in bounds.domain("retry_intensity").values
                    if _domain_int(value) < current.retry_intensity
                )
                if lower:
                    return bounds.with_updates(current, {"retry_intensity": min(lower)})
            if parent.feedback.action is Action.APPROVE:
                names = ("device_reuse_rate", "merchant_concentration")
                name = names[int(rng.integers(0, len(names)))]
                alternatives = bounds.domain(name).alternatives(getattr(current, name))
                if alternatives:
                    return bounds.with_updates(
                        current,
                        {name: alternatives[int(rng.integers(0, len(alternatives)))]},
                    )
        if bounds.family == "app_scam_mule":
            if parent.feedback.reason_family == "amount":
                minimum = min(
                    _domain_decimal(value)
                    for value in bounds.domain("cash_out_fraction").values
                )
                if minimum != current.cash_out_fraction:
                    return bounds.with_updates(current, {"cash_out_fraction": minimum})
            if parent.feedback.action is Action.APPROVE:
                for _ in range(100):
                    candidate = bounds.mutate_params(current, rng)
                    if candidate.cash_out_fraction == current.cash_out_fraction:
                        return candidate
        return bounds.mutate_params(parent.candidate.params, rng)


__all__ = [
    "PUBLIC_REASON_FAMILIES",
    "AdaptiveTournamentPolicy",
    "AttackCandidate",
    "DomainKind",
    "Feedback",
    "FixedPolicy",
    "ParameterBounds",
    "ParameterDomain",
    "Policy",
    "RandomPolicy",
    "VisibleTrial",
    "visible_objective",
]
