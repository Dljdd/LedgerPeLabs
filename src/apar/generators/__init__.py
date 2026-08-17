"""Mechanism-driven synthetic populations and payment campaigns."""

from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignGenerator,
    CampaignParameterError,
    CampaignParams,
    GenerationConstraintError,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import (
    BenignActivity,
    Population,
    PopulationAccount,
    PopulationEntity,
    PopulationGenerator,
    PopulationRelationship,
)

__all__ = [
    "AGENTIC_INTENT_ABUSE_MOTIF",
    "APP_SCAM_MULE_MOTIF",
    "CARD_TESTING_CNP_MOTIF",
    "SYNTHETIC_MERCHANT_REFUND_MOTIF",
    "BenignActivity",
    "CampaignGenerator",
    "CampaignParameterError",
    "CampaignParams",
    "GenerationConstraintError",
    "Population",
    "PopulationAccount",
    "PopulationEntity",
    "PopulationGenerator",
    "PopulationRelationship",
    "campaign_bytes",
    "motif_signature",
]
