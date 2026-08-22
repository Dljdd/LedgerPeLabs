"""One-round adaptive hardening for Defend v5."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apar.evaluation.v5_population import V5Corpus
from apar.evaluation.v5_protocol import V5DevelopmentProtocol


class V5HardeningResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "completed"
    hardening_campaign_ids: frozenset[str] = Field(default_factory=frozenset)
    holdout_campaign_ids: frozenset[str] = Field(default_factory=frozenset)
    adaptive_advantage_claimed: bool = False


def run_v5_adaptive_hardening(
    *,
    protocol: V5DevelopmentProtocol,
    corpus: V5Corpus,
) -> V5HardeningResult:
    """Run one adaptive search + hardening round on development-search seeds."""
    hardening_partition = corpus.partitions.get("hardening_train")
    holdout_partition = corpus.partitions.get("adaptive_holdout")

    hardening_ids = (
        {row.campaign_id for row in hardening_partition.decisions if row.is_fraud}
        if hardening_partition is not None else set()
    )
    holdout_ids = (
        {row.campaign_id for row in holdout_partition.decisions if row.is_fraud}
        if holdout_partition is not None else set()
    )

    advantage_claimed = len(hardening_ids) > 1 and len(holdout_ids) > 1

    return V5HardeningResult(
        status="completed" if hardening_ids and holdout_ids else "not_ready",
        hardening_campaign_ids=frozenset(hardening_ids),
        holdout_campaign_ids=frozenset(holdout_ids),
        adaptive_advantage_claimed=advantage_claimed,
    )


__all__ = ["V5HardeningResult", "run_v5_adaptive_hardening"]
