"""Mechanism-driven synthetic populations and payment campaigns."""

from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    AgenticFixture,
    CampaignDependency,
    CampaignEvidence,
    CampaignGenerator,
    CampaignParams,
    GenerationConstraintError,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import (
    Population,
    PopulationEntity,
    PopulationGenerator,
    PopulationRelationship,
)

__all__ = [
    "AGENTIC_INTENT_ABUSE_MOTIF",
    "APP_SCAM_MULE_MOTIF",
    "CARD_TESTING_CNP_MOTIF",
    "SYNTHETIC_MERCHANT_REFUND_MOTIF",
    "AgenticFixture",
    "CampaignDependency",
    "CampaignEvidence",
    "CampaignGenerator",
    "CampaignParams",
    "GenerationConstraintError",
    "Population",
    "PopulationEntity",
    "PopulationGenerator",
    "PopulationRelationship",
    "campaign_bytes",
    "motif_signature",
]
