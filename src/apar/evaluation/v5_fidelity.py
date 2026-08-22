"""Behavioral fidelity auditor for Defend v5."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from apar.evaluation.v5_population import V5Corpus


class FidelityDimension(StrEnum):
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    ECONOMIC = "economic"


class FidelityCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: FidelityDimension
    name: str
    observed: float
    reference_min: float
    reference_max: float
    passed: bool


class FidelityAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    checks: tuple[FidelityCheck, ...]
    overall_status: str


def _check(name: str, dim: FidelityDimension, obs: float, lo: float, hi: float) -> FidelityCheck:
    return FidelityCheck(
        dimension=dim, name=name, observed=obs,
        reference_min=lo, reference_max=hi, passed=(lo <= obs <= hi),
    )


def audit_v5_fidelity(corpus: V5Corpus) -> FidelityAudit:
    """Run all mandatory fidelity checks on the corpus."""
    checks: list[FidelityCheck] = []

    for name, partition in corpus.partitions.items():
        rows = partition.decisions
        if not rows:
            continue
        fraud = [r for r in rows if r.is_fraud]
        benign = [r for r in rows if not r.is_fraud]

        # Statistical
        benign_amounts = sorted(float(r.amount) for r in benign)
        fraud_amounts = sorted(float(r.amount) for r in fraud)
        if benign_amounts:
            median_benign = benign_amounts[len(benign_amounts) // 2]
            checks.append(
                _check(f"benign_median_amount_{name}", FidelityDimension.STATISTICAL,
                       median_benign, 0.0, 10000.0)
            )
        if fraud_amounts:
            median_fraud = fraud_amounts[len(fraud_amounts) // 2]
            checks.append(
                _check(f"fraud_median_amount_{name}", FidelityDimension.STATISTICAL,
                       median_fraud, 0.0, 50000.0)
            )

        # Temporal
        if fraud:
            times = sorted(r.decision_at.hour for r in fraud)
            mean_hour = sum(times) / len(times)
            checks.append(
                _check(f"fraud_mean_hour_{name}", FidelityDimension.TEMPORAL,
                       mean_hour, 0.0, 24.0)
            )

        # Relational
        actors = {r.actor_id for r in rows}
        counterparties = {r.counterparty_id for r in rows}
        degree_ratio = len(rows) / max(len(actors), 1)
        checks.append(
            _check(f"actor_degree_{name}", FidelityDimension.RELATIONAL,
                   degree_ratio, 1.0, 50.0)
        )
        cp_degree = len(rows) / max(len(counterparties), 1)
        checks.append(
            _check(f"counterparty_degree_{name}", FidelityDimension.RELATIONAL,
                   cp_degree, 1.0, 100.0)
        )

        # Economic
        total_fraud_value = sum(float(r.amount) for r in fraud)
        total_all_value = sum(float(r.amount) for r in rows)
        fraud_share = total_fraud_value / total_all_value if total_all_value > 0 else 0.0
        checks.append(
            _check(f"fraud_value_share_{name}", FidelityDimension.ECONOMIC,
                   fraud_share, 0.0, 1.0)
        )

    overall = "pass" if all(c.passed for c in checks) else "fail"
    return FidelityAudit(checks=tuple(checks), overall_status=overall)


__all__ = ["FidelityAudit", "FidelityCheck", "FidelityDimension", "audit_v5_fidelity"]
